import os
# transformers-Fortschrittsbalken/Füll-Logs aus der Konsole fernhalten (vor import!).
os.environ.setdefault("TF_VERBOSITY", "error")
import cv2
import time
import torch
import threading
import collections
import numpy as np
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "models/gemma-4-e4b-it"   # lokal im Projektordner (aus HF-Cache kopiert; kein Download/Token nötig)
YOLO_WEIGHTS = "models/yolo26l.pt"   # lokal im Projektordner
DEFAULT_PROMPT = "Beschreibe, was du auf diesem Bild siehst. Antworte auf Deutsch, konkret und knapp, in 2 bis 3 kurzen Sätzen."
MAX_NEW_TOKENS = 96
IDLE_DELAY_SECONDS = 1.0
YOLO_CONF = float(os.environ.get("GEMMA_YOLO_CONF", "0.60"))
LOG_FILE = "log/app8.log"
CAM_WIDTH = int(os.environ.get("GEMMA_CAM_W", "1280"))
CAM_HEIGHT = int(os.environ.get("GEMMA_CAM_H", "720"))


def _is_junk_text(text):
    """True = Mist (Meta-Text/Prompt-Echo/leer) -> Cache darf den Text NICHT speichern."""
    t = (text or "").strip()
    if not t or t == "Leere Antwort.":
        return True
    if len(t) < 25:
        return True
    low = t.lower()
    markers = (
        "deine aufgabe",
        "schreibe die",
        "beschreibung:",
        "**beschreibung**",
        "schon einmal geliefert",
        "hier ist eine beschreibung dessen",
        "beschreibe das bild, das du siehst",
    )
    if any(m in low for m in markers):
        return True
    return False


def _fingerprint(detections):
    """Stabile Signatur eines Bildzustands: (Personenanzahl, sortierte Label-/Position-Zellen).

    Objekte werden auf ein grobes 64-px-Positionsraster quantisiert: eine Person,
    die ein paar Pixel wandert (Box-Jitter), bleibt in derselben Zelle -> zählt
    NICHT als "wesentliche Änderung". Klar verschobene Objekte fallen in neue
    Zellen -> Bildzustand "geändert".
    """
    cells = {}
    for label, (x0, y0, x1, y1), _conf in detections:
        key = (label, ((x0 + x1) // 2) // 64, ((y0 + y1) // 2) // 64)
        cells[key] = cells.get(key, 0) + 1
    return (sum(1 for d in detections if d[0] == "person"), tuple(sorted(cells.items())))


class App8:
    """ThREADED webcam analysis.

    Design notes:
      * analysis is ADAPTIV SCHEDULED: the next frame is analyzed ~1 s after
        the previous inference finished (not every Nth frame, which would be
        skipped forever because inference is much slower than the frame rate).
      * the UI thread never blocks on the inference thread; state is shared
        through a small lock-protected dict.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.debug_log = os.environ.get("GEMMA_DEBUG_LOG") or LOG_FILE
        self.prompt_text = os.environ.get("GEMMA_PROMPT", DEFAULT_PROMPT)
        self.prompt_display = self.prompt_text
        self.yolo_conf = YOLO_CONF
        self.state = {
            "busy": False,
            "started_at": None,
            "status": "ready",
            "text": "Warte auf ersten Frame...",
            "last_seconds": None,
            "last_tok_s": None,
            "ready_after": 0.0,
            "fps": None,
            "camera_res": "",
            "detections": [],
            "yolo_summary": "",
            "streak": 0,
            "stable": False,
        }
        # Stabilitäts-Merkzettel: letzter Bildzustand + wie oft er wiederholte.
        self._last_fingerprint = None
        self._streak = 0
        # Letzter erzeugter Beschreibungstext -> Cache-Quelle (stabil) / Prompt-Input (geändert).
        self._last_text = None
        self._last_text_valid = False
        self.model = None
        self.processor = None
        self.device = None
        self.yolo = None
        # Default AUS -> schneller Start; beide werden erst beim Einschalten (Slide) geladen.
        self.yolo_enabled = False
        self.gemma_enabled = False
        # Gemma wird im Hintergrund geladen: UI + YOLO laufen währenddessen.
        self._gemma_loaded = False
        self._gemma_load_error = None

    # ---------------------------------------------------------------- state
    def _log(self, msg):
        """Alle Ausgaben (inkl. Start-/Fehler-Meldungen) nur ins Logfile, NICHT in die Konsole."""
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        if self.debug_log:
            path = self.debug_log
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            with open(path, "a") as f:
                f.write(line + "\n")

    def _snapshot(self):
        with self._lock:
            return dict(self.state)

    def _update(self, **kwargs):
        with self._lock:
            self.state.update(kwargs)
            self._log(f"STATE UPDATE: {kwargs}")

    # ------------------------------------------------------------- inference
    def _build_prompt(self, detections, streak, stable, last_text=None):
        """YOLO-Ergebnis + Stabilitäts-Anteil + zuletzt erzeugter Text in den Prompt.

        * YOLO liefert Objekte + Anzahl (z.B. Personen) -> Gemma bewertet/verifiziert.
        * stabiles Bild (gleiche Signatur wie vorher): Gemma wird aufgefordert,
          den zuletzt erzeugten Text im Wesentlichen unverändert wiederzugeben,
          damit das Gesamtbild ruhig bleibt.
        * Wesentliche Änderung (neue Signatur): letzter Text als Orientierung,
          aber den NEUEN Zustand anpassen beschreiben.
        """
        parts = [self.prompt_text]
        if detections:
            counts = {}
            for label, _, conf in detections:
                counts[label] = counts.get(label, 0) + 1
            yolo_list = ", ".join(f"{label} x{n}" for label, n in
                                  sorted(counts.items(), key=lambda kv: -kv[1]))
            parts.append(
                f"Objekterkennung (YOLO26l, Konfidenz >= {self.yolo_conf:.2f}): {yolo_list}. "
                "Nutze diese Erkennungen: stimme die genannte Personenanzahl mit dem Bild ab "
                "(korrigiere sie, wenn YOLO daneben liegt) und erwähne sie."
            )
        else:
            parts.append("Objekterkennung (YOLO26l): keine Objekte erkannt – prüfe das Bild selbst.")
        if not stable:
            if last_text:
                parts.append(
                    f"Das Bild hat sich geändert (neue oder verschobene Objekte). Deine letzte "
                    f"Beschreibung war: \"{last_text}\". Beschreibe jetzt den NEUEN Zustand – "
                    "passe den Text an die sichtbare Änderung an."
                )
            else:
                parts.append("Das ist ein neuer Bildzustand (oder der erste): beschreibe ihn neu und frisch.")
        else:
            # Stabil, aber kein Cache-Text verfügbar (letzter Text war Mist):
            # NEU beschreiben – aber OHNE den alten Text einzufüttern, sonst
            # spiegelt Gemma ihn (s. Schleifen mit Meta-Text im Log).
            parts.append(
                "Hinweis: die Szene ist gleich wie im vorherigen Durchlauf (Schleife Nr. "
                f"{max(streak, 2)}). Beschreibe sie konkret und knapp in 2-3 Sätzen – "
                "nichts weiter."
            )
        return " ".join(parts)

    def _classify_stability(self, detections):
        """Vergleicht die aktuelle YOLO-Signatur mit der vorherigen -> (stable, streak)."""
        fp = _fingerprint(detections)
        stable = (fp == self._last_fingerprint)
        self._streak = (self._streak + 1) if stable else 0
        self._streak = max(0, min(self._streak, 50))
        self._last_fingerprint = fp
        return stable, self._streak

    def _valid_text(self, text):
        """Echtem Beschreibungstext? (Modul-Standards + Prompt-Spiegelung prüfen.)"""
        if _is_junk_text(text):
            return False
        low = (text or "").strip().lower()
        if low.startswith(self.prompt_text.lower().strip()[:40]):
            return False
        return True

    def _run_inference(self, rgb_image, det_frame):
        self._log(f"INFERENZ START: frame {rgb_image.shape[:2][::-1]}")
        try:
            self._update(busy=True, status="thinking", started_at=time.time())
            detections = []
            if self.yolo_enabled:
                detections = self.detect(det_frame)
                counts = {}
                for label, _, conf in detections:
                    counts[label] = counts.get(label, 0) + 1
                yolo_summary = (
                    ", ".join(f"{l} x{c}" for l, c in sorted(counts.items(), key=lambda kv: -kv[1]))
                    if detections else "keine Objekte"
                )
                stable, streak = self._classify_stability(detections)
                self._log(f"YOLO: {yolo_summary} | stabil={stable} streak={streak}")
                self._update(detections=detections, yolo_summary=yolo_summary,
                             stable=stable, streak=streak)
            else:
                stable, streak = False, 0
                self._update(detections=[], yolo_summary="", stable=False, streak=0)

            if not self.gemma_enabled:
                if self.yolo_enabled:
                    text = (f"YOLO ({self.yolo_conf:.2f}): {', '.join(f'{l} x{c}' for l, c in sorted(counts.items(), key=lambda kv: -kv[1]))}"
                            if detections else f"YOLO ({self.yolo_conf:.2f}): keine Objekte erkannt.")
                else:
                    text = "Alle Module AUS – nur Livebild. Taster 'YOLO' / 'Gemma' zum Einschalten."
                self._last_text = None
                self._last_text_valid = False
                self._update(
                    busy=False,
                    status="ok",
                    text=text,
                    last_seconds=None,
                    last_tok_s=None,
                    ready_after=time.time() + IDLE_DELAY_SECONDS,
                )
                return
            if self._gemma_state() == "error":
                self._log(f"Gemma nicht verfügbar: {self._gemma_load_error}")
                self._update(
                    busy=False,
                    status="error",
                    text=f"Gemma-Modell nicht verfügbar: {self._gemma_load_error}",
                    ready_after=time.time() + IDLE_DELAY_SECONDS,
                )
                return
            if self._gemma_state() != "ready":
                # Gemma laedt im Hintergrund: YOLO laeuft weiter, naechster Zyklus
                # folgt nach IDLE_DELAY – die Auswertung beginnt, sobald es da ist.
                self._log("YOLO-Zyklus ohne Gemma (laedt noch im Hintergrund)...")
                self._update(
                    busy=False,
                    status="ok",
                    text="KI-Modell (Gemma) laedt noch – YOLO-Erkennung laeuft schon.",
                    last_seconds=None,
                    last_tok_s=None,
                    ready_after=time.time() + IDLE_DELAY_SECONDS,
                )
                return

            if stable and self._last_text_valid and self._last_text:
                # HARTE CACHE-REGEL: gleiche Signatur wie vorher -> letzter gueltiger
                # Text wird DIREKT wiedergegeben, Gemma wird NICHT aufgerufen.
                # (Prompt-Tricks "wiederhole den Text" erzeugten bei Gemma Meta-Mist,
                #  leere Antworten und Prompt-Echos -> Cache koennte sich verselbstaendigen.)
                self._log(f"CACHE HIT: Schleife {streak} (Modell uebersprungen) -> {self._last_text!r}")
                self._update(
                    busy=False,
                    status="ok",
                    text=self._last_text,
                    last_seconds=None,
                    last_tok_s=None,
                    ready_after=time.time() + IDLE_DELAY_SECONDS,
                )
                return

            prompt = self._build_prompt(
                detections, streak, stable,
                last_text=self._last_text if self._last_text_valid else None,
            )
            self._log(f"PROMPT: {prompt}")
            t0 = time.time()
            inputs = self.processor(
                images=rgb_image, text=prompt, return_tensors="pt"
            ).to(self.device)
            prompt_len = inputs["input_ids"].shape[-1]
            with torch.no_grad():
                generated = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
            dt = time.time() - t0
            new_ids = generated[0, prompt_len:]
            text = self.processor.decode(new_ids, skip_special_tokens=True).strip()
            if not text:
                text = "Leere Antwort."
            tok_s = max((new_ids.numel() - 1), 0) / dt if dt > 0 else 0.0
            self._log(f"INFERENZ OK: {dt:.2f} s, {new_ids.numel()} neue tokens, Text: {text!r}")
            if self._valid_text(text):
                self._last_text = text
                self._last_text_valid = True
                self._update(
                    busy=False,
                    status="ok",
                    text=text,
                    last_seconds=round(dt, 2),
                    last_tok_s=round(tok_s, 1),
                    ready_after=time.time() + IDLE_DELAY_SECONDS,
                )
            else:
                # Ungueltiger Text (leer/Meta-Text/Prompt-Echo): NICHT anzeigen und
                # NICHT als letzten Text merken -> der zuletzt gueltige Text bleibt
                # im Overlay stehen, bis naechsten Zyklus.
                if self._last_text_valid and self._last_text:
                    self._log(f"TEXT UNGUETIG -> vorheriger Text bleibt sichtbar: {text!r}")
                    self._update(
                        busy=False,
                        status="ok",
                        text=self._last_text,
                        last_seconds=round(dt, 2),
                        last_tok_s=round(tok_s, 1),
                        ready_after=time.time() + IDLE_DELAY_SECONDS,
                    )
                else:
                    self._log(f"TEXT UNGUETIG und kein vorheriger Text: {text!r}")
                    self._update(
                        busy=False,
                        status="ok",
                        text="Keine gültige Bildbeschreibung (KI-Antwort leer oder ungültig).",
                        last_seconds=round(dt, 2),
                        last_tok_s=round(tok_s, 1),
                        ready_after=time.time() + IDLE_DELAY_SECONDS,
                    )
        except Exception as e:
            self._log(f"INFERENZ FEHLER: {e!r}")
            self._update(
                busy=False,
                status="error",
                text=f"Fehler in KI-Thread: {e}",
                ready_after=time.time() + 10.0,
            )

    # ------------------------------------------------------------------ load
    def _gemma_state(self):
        """'disabled' | 'loading' | 'ready' | 'error' — Status von Gemma."""
        if not self.gemma_enabled:
            return "disabled"
        if self._gemma_load_error is not None:
            return "error"
        if self._gemma_loaded:
            return "ready"
        return "loading"

    def _start_model_loading(self):
        if not os.path.isdir(MODEL_ID):
            self._log(f"[FEHLER] Gemma-Modell nicht gefunden unter '{MODEL_ID}' – "
                      "YOLO-Livebild läuft weiter, keine Gemma-Auswertung.")
            self._gemma_load_error = f"Gemma-Modell nicht gefunden unter '{MODEL_ID}'"
            self._gemma_loaded = True
            return
        self._log("Gemma-Modell wird im Hintergrund geladen (Livebild + YOLO laufen schon)…")
        def _worker():
            try:
                self._load_model()
                self._log("Gemma-Modell einsatzbereit – Gemma-Auswertung aktiv.")
            except Exception as e:
                self._log(f"[FEHLER] Gemma-Modell konnte nicht geladen werden: {e!r}")
                self._gemma_load_error = str(e)
            finally:
                self._gemma_loaded = True
        threading.Thread(target=_worker, daemon=True).start()

    def _load_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._log(f"[1/3] Lade {MODEL_ID} ({device.upper()}, {str(dtype)[6:]}) ...")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=dtype)
        self.device = device
        self.model.to(device)
        self.model.eval()
        self._log("[2/3] Modell geladen.")

        tpl = self.processor.tokenizer
        self.prompt_text = tpl.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.prompt_text},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        self.device = device
        self._log(f"[3/3] Gerät: {device.upper()}")

    # ------------------------------------------------------------- detection
    def _ensure_detector(self):
        """Lädt den YOLO26l-Detektor lazy (nicht bei App-Start), einmalig.

        Liefert das Gewicht lokal aus models/ wenn vorhanden, sonst lädt Ultralytics
        es automatisch herunter (läuft danach ins ultralytics-Verzeichnis).
        """
        if self.yolo is not None:
            return
        if os.path.isfile(YOLO_WEIGHTS):
            self.yolo = YOLO(YOLO_WEIGHTS)
        else:
            self.yolo = YOLO("yolo26l.pt")
            self.yolo.to(self.device)
        self._log(f"[4/4] Objektdetektor geladen (YOLO26l, {YOLO_WEIGHTS}).")

    def detect(self, frame):
        """YOLO-Detection: Liste von (label, (x0,y0,x1,y1), conf), sortiert nach conf absteigend."""
        self._ensure_detector()
        with torch.no_grad():
            results = self.yolo.predict(frame, conf=self.yolo_conf, verbose=False)
        detections = []
        for cls_id, x0, y0, x1, y1, conf in zip(
            results[0].boxes.cls, *results[0].boxes.xyxy.cpu().numpy().T, results[0].boxes.conf.cpu().numpy()
        ):
            label = self.yolo.names[int(cls_id)]
            detections.append((label, (int(x0), int(y0), int(x1), int(y1)), float(conf)))
        detections.sort(key=lambda d: d[2], reverse=True)
        return detections

    # --------------------------------------------------------------------- ui
    @staticmethod
    def _wrap(text, font, scale, thickness, limit):
        words = text.split()
        lines, cur = [], ""
        for w in words:
            trial = (cur + " " + w).strip()
            if cv2.getTextSize(trial, font, scale, thickness)[0][0] <= limit or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    def _draw_overlay(self, frame):
        h, w = frame.shape[:2]
        state = self._snapshot()
        fscale, fthick = 0.55, 2
        lh = int(26 * fscale) + 10
        fps_txt = f"   |  Kamera {state['camera_res']}@{state['fps']:.0f} fps" if state.get("fps") else (
            f"   |  Kamera {state['camera_res']}" if state.get("camera_res") else "")
        status_txt = f"   |  Stabil (Schleife {state['streak']})" if state.get("stable") and state.get("streak", 0) >= 1 else ""
        if not self.yolo_enabled:
            yolo_txt = "   |  YOLO: AUS"
        elif state.get("yolo_summary"):
            yolo_txt = f"   |  YOLO: {state['yolo_summary']}"
        else:
            yolo_txt = ""
        has_result = state["text"] not in ("", "Warte auf ersten Frame...")
        content_lines = [f"Prompt: {self.prompt_display}{fps_txt}{status_txt}{yolo_txt}"]
        if state["status"] == "error":
            content_lines.append(state["text"])
        elif has_result:
            meta = ""
            if state["last_seconds"] is not None:
                meta = f"   [{state['last_seconds']} s | {state['last_tok_s']} tok/s]"
            content_lines.append(state["text"] + meta)
            if state["status"] == "thinking":
                elapsed = time.time() - (state["started_at"] or time.time())
                content_lines.append(f"Naechste Analyse laeuft... ({elapsed:.0f} s)")
        elif state["status"] == "thinking":
            elapsed = time.time() - (state["started_at"] or time.time())
            content_lines.append(f"Analysiere... ({elapsed:.0f} s)")
        else:
            content_lines.append("Warte auf ersten Frame...")
        gstate = self._gemma_state()
        if gstate == "disabled":
            content_lines.append("KI-Modell (Gemma): AUS – Taster 'Gemma' zum Einschalten.")
        elif gstate == "loading":
            content_lines.append("KI-Modell (Gemma) laedt noch im Hintergrund – Livebild + YOLO laufen schon.")
        elif gstate == "error":
            content_lines.append("KI-Modell (Gemma) ist nicht erreichbar – nur YOLO-Overlay.")
        content_lines = [line
             for ln in content_lines
             for line in self._wrap(ln, cv2.FONT_HERSHEY_SIMPLEX, fscale, fthick, w - 30)]
        key = (status_key := state["status"], tuple(l[:40] for l in content_lines))
        if key != getattr(self, "_last_overlay_key", None):
            self._last_overlay_key = key
            self._log(f"OVERLAY: {content_lines}")
        panel_h = 24 + len(content_lines) * lh + 12
        for label, (x0, y0, x1, y1), conf in (state.get("detections") or []):
            if label == "person":
                color = (0, 255, 0)
            else:
                color = (255, 165, 0)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x0, max(y0 - 6, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        panel = np.zeros((panel_h, w, 3), np.uint8)
        y = 20
        for i, line in enumerate(content_lines):
            color = (255, 255, 255) if i else (0, 200, 255)
            cv2.putText(panel, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, fscale, color, fthick, cv2.LINE_AA)
            y += lh
        return np.vstack((frame, panel))

    # -------------------------------------------------------------------- main
    def _open_camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self._log("[FEHLER] Kamera konnte nicht geöffnet werden.")
            return None
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Höhere Auflösung erzwingen (Kamera liefert oft nur 640x480 als Default).
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        w2 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h2 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._update(camera_res=f"{w2}x{h2}")
        self._log(f"Kamera: {w}x{h} -> {w2}x{h2} (Ziel {CAM_WIDTH}x{CAM_HEIGHT})")
        time.sleep(1.0)
        return cap

    def run(self):
        # Schnellstart: beides startet AUS – YOLO/Gemma werden erst beim Einschalten
        # (Taster unten) geladen, damit der Start ohne 15-GB-Modell schnell bleibt.
        cap = self._open_camera()
        if cap is None:
            return
        self._log("Stream läuft!  Tasten: 's' = Frame+Text speichern  |  'q' im Videofenster beenden.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Taster (0/AUS – 1/EIN) + Live-Slider für die YOLO-Konfidenz;
        # wirken sofort im nächsten Analysezyklus (detect() liest self.yolo_conf).
        cv2.namedWindow("Gemma 4 Live-Videoanalyse", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("YOLO", "Gemma 4 Live-Videoanalyse", 0, 1, self._toggle_yolo)
        cv2.createTrackbar("Gemma", "Gemma 4 Live-Videoanalyse", 0, 1, self._toggle_gemma)
        cv2.createTrackbar("YOLO Conf", "Gemma 4 Live-Videoanalyse",
                           int(round(self.yolo_conf * 100)), 100,
                           self._set_yolo_conf)

        # Frame-Rate der Kamera (UI-Loop) messen — rolling window über ~30 Frames.
        frame_times = collections.deque(maxlen=30)

        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            now = time.time()
            frame_times.append(now)
            if len(frame_times) >= 2:
                span = now - frame_times[0]
                if span > 0:
                    self._update(fps=round((len(frame_times) - 1) / span, 1))

            state = self._snapshot()
            if not state["busy"] and time.time() >= state["ready_after"]:
                # ready_after wird am ENDE der Inferenz gesetzt (s. _run_inference),
                # damit Abstand zwischen Ende eines Laufs und Start des naechsten >= IDLE_DELAY bleibt.
                self._update(ready_after=time.time() + IDLE_DELAY_SECONDS)
                # Eigene Kopien: die UI zeichnet Boxen in `frame` hinein (in-place),
                # der Inferenz-Thread muesste parallel den Frame detektieren —
                # ohne Kopie wuerden Boxen-Pixel die Erkennung verfälschen.
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                det_frame = frame.copy()
                t = threading.Thread(
                    target=self._run_inference,
                    args=(rgb_image, det_frame),
                    daemon=True,
                )
                t.start()

            display = self._draw_overlay(frame)
            cv2.imshow("Gemma 4 Live-Videoanalyse", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                self._save_frame(frame, self._snapshot())

        self._log("Schließe Kamera und Fenster...")
        cap.release()
        cv2.destroyAllWindows()
        self._log("Programm erfolgreich beendet.")

    def _set_yolo_conf(self, value):
        """Slider-Callback: YOLO-Konfidenz live anpassen (0–100 % -> 0.0–1.0)."""
        self.yolo_conf = max(0.01, min(1.0, value / 100.0))
        self._log(f"YOLO-Conf (Slider) -> {self.yolo_conf:.2f}")

    def _toggle_yolo(self, value):
        """Taster 'YOLO' (0=AUS, 1=EIN): Detection + Box-Overlay an/aus."""
        on = value >= 1
        if on != self.yolo_enabled:
            self.yolo_enabled = on
            if on and self.yolo is None:
                self._log("YOLO TASTER EINS -> Detektor wird im ersten Zyklus geladen…")
            else:
                self._log(f"YOLO TASTER -> {'EIN' if on else 'AUS'}")

    def _toggle_gemma(self, value):
        """Taster 'Gemma' (0=AUS, 1=EIN): Bildbeschreibung an/aus.

        Beim Einschalten startet das Hintergrund-Loading; beim Ausschalten wird
        der Cache verworfen (kein 'Cachetext' mehr aus alten Zyklen).
        """
        on = value >= 1
        if on != self.gemma_enabled:
            self.gemma_enabled = on
            if on:
                self._last_text = None
                self._last_text_valid = False
                if not self._gemma_loaded and self._gemma_load_error is None:
                    self._log("GEMMA TASTER EINS -> Modell wird im Hintergrund geladen…")
                    self._start_model_loading()
            self._log(f"GEMMA TASTER -> {'EIN' if on else 'AUS'}")

    # ------------------------------------------------------------ folder mode
    _IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

    def _folder_images(self, folder):
        paths = [
            os.path.join(folder, name)
            for name in sorted(os.listdir(folder))
            if name.lower().endswith(self._IMG_EXTS)
        ]
        if not paths:
            self._log(f"[FEHLER] Keine Bilder gefunden in '{folder}' "
                      f"(Erwarte: {', '.join(self._IMG_EXTS)})")
        return paths

    def run_folder(self, folder, out_path):
        """Bildordner-Modus: YOLO + Gemma pro Bild, Stabilitäts-Cache wie im Live-Betrieb.

        Schreiben der Zusammenfassung (Markdown, kompakt):
          * Bild mit GEÄNDERT-Zustand  -> Eintrag mit Zeitstempel + Beschreibung.
          * Bild mit GLEICHER Zustand  -> wird NICHT ausgegeben (nur gezählt).
        """
        paths = self._folder_images(folder)
        if not paths:
            return
        # Ordner-Modus: default EINS (Batch-Analyse); --no-yolo/--no-gemma deaktivieren.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.yolo_enabled:
            self._ensure_detector()
        if self.gemma_enabled:
            self._start_model_loading()
        self._log(f"ORDNER-MODUS: {len(paths)} Bilder aus '{folder}' -> '{out_path}'")

        def write(out_lines):
            os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
            with open(out_path, "w") as f:
                f.write("\n".join(out_lines) + "\n")

        header = [
            f"# Zusammenfassung: {os.path.basename(os.path.abspath(folder))}",
            "",
            f"- Erstellt: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- Bilder: {len(paths)} | YOLO-Schwelle: {self.yolo_conf:.2f}",
            f"- Nur Zustandsänderungen (gleiche Bilder ohne Eintrag)",
            "",
        ]
        out_lines = list(header)

        skipped, errors = 0, 0
        for idx, path in enumerate(paths, 1):
            img = cv2.imread(path)
            if img is None:
                self._log(f"[FEHLER] Bild nicht lesbar: {path}")
                errors += 1
                continue
            t0 = time.time()
            name = os.path.basename(path)
            dt = time.time() - t0
            try:
                if self.yolo_enabled:
                    detections = self.detect(img)
                    yolo_summary = "keine Objekte"
                    if detections:
                        counts = {}
                        for label, _, _ in detections:
                            counts[label] = counts.get(label, 0) + 1
                        yolo_summary = ", ".join(
                            f"{l} x{c}" for l, c in sorted(counts.items(), key=lambda kv: -kv[1]))
                    stable, _streak = self._classify_stability(detections)
                else:
                    detections, yolo_summary, stable, _streak = [], "(aus)", False, 0
                if stable and self._last_text_valid and self._last_text:
                    text, dt = self._last_text, time.time() - t0
                    self._log(f"ORDNER {idx}/{len(paths)} {name}: GLEICH wie vorher -> kein Eintrag")
                    skipped += 1
                    continue
                if self._gemma_state() == "ready":
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    prompt = self._build_prompt(
                        detections, _streak, stable,
                        last_text=self._last_text if self._last_text_valid else None)
                    with torch.no_grad():
                        inputs = self.processor(images=rgb, text=prompt,
                                                return_tensors="pt").to(self.device)
                        prompt_len = inputs["input_ids"].shape[-1]
                        generated = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
                        new_ids = generated[0, prompt_len:]
                        text = self.processor.decode(new_ids, skip_special_tokens=True).strip()
                    dt = time.time() - t0
                else:
                    gs = self._gemma_state()
                    gs_txt = {"disabled": "aus", "loading": "lädt noch"}.get(gs, "nicht verfügbar")
                    text = f"(Gemma {gs_txt}) – YOLO: {yolo_summary}"
            except Exception as e:
                self._log(f"ORDNER {idx}/{len(paths)} {name}: FEHLER {e!r}")
                errors += 1
                continue

            self._log(f"ORDNER {idx}/{len(paths)} {name}: YOLO {yolo_summary} "
                      f"| stabil={stable} | {dt:.2f} s | Text: {text!r}")
            if self._valid_text(text):
                self._last_text = text
                self._last_text_valid = True
            if not stable:
                out_lines.append(
                    f"## {time.strftime('%Y-%m-%d %H:%M:%S')} — Bild {idx} (`{name}`)\n\n"
                    f"{text}\n"
                    f"- YOLO: {yolo_summary}")
            else:
                skipped += 1

        tail = [
            "",
            f"- Ergebnis: {len(paths) - errors - skipped} Änderung(en) ausgegeben, "
            f"{skipped} gleiche/gecachte Bild(er), {errors} Fehler",
        ]
        out_lines.extend(tail)
        write(out_lines)
        self._log(f"Ordner-Modus fertig: {out_path} ({len(paths) - errors - skipped} Einträge, "
                  f"{skipped} identisch, {errors} Fehler)")

    def _save_frame(self, frame, state):
        """Speichert das aktuelle Frame + Metadaten (s Taste)."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs("tmp", exist_ok=True)
        img_path = f"tmp/save_{ts}.png"
        meta_path = f"tmp/save_{ts}.txt"
        cv2.imwrite(img_path, frame)
        with open(meta_path, "w") as f:
            f.write(f"Zeitpunkt: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Auflösung: {state.get('camera_res', '?')}\n")
            f.write(f"Stabil: {state['stable']} | Schleife: {state['streak']}\n")
            f.write(f"YOLO: {state['yolo_summary']}\n")
            f.write(f"--- Beschreibung ---\n")
            f.write(f"{state['text']}\n")
        self._log(f"Gespeichert: {img_path}  |  {meta_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Gemma-4-Live-Videoanalyse / Bildordner-Analyse")
    ap.add_argument("--images", metavar="ORDNER",
                    help="Bildordner analysieren (statt Kamera-Live) und kompakte "
                         "MD-Zusammenfassung schreiben (nur Zustandsänderungen, mit Zeitstempel)")
    ap.add_argument("--out", default="tmp/bilder_zusammenfassung.md",
                    help="Ziel-Datei für --images (Default: tmp/bilder_zusammenfassung.md)")
    ap.add_argument("--conf", type=float, default=None, metavar="0.0-1.0",
                    help="YOLO-Konfidenzschwelle (Default: 0.60, live per Slider änderbar)")
    ap.add_argument("--yolo", action="store_true",
                    help="YOLO-Erkennung aktivieren (Live-Modus default AUS; Ordner-Modus default EIN)")
    ap.add_argument("--no-yolo", dest="yolo", action="store_false",
                    help="YOLO-Erkennung deaktivieren")
    ap.add_argument("--gemma", action="store_true",
                    help="Gemma-Bildbeschreibung aktivieren (Live-Modus default AUS; Ordner-Modus default EIN)")
    ap.add_argument("--no-gemma", dest="gemma", action="store_false",
                    help="Gemma-Bildbeschreibung deaktivieren")
    ap.set_defaults(yolo=None, gemma=None)
    args = ap.parse_args()

    app = App8()
    if args.conf is not None:
        app.yolo_conf = max(0.01, min(1.0, float(args.conf)))
    if args.images:
        app.run_folder(args.images, args.out)
    else:
        # Live-Modus: default AUS (schneller Start); --yolo/--gemma erzwingen EIN.
        if args.yolo is not None:
            app.yolo_enabled = args.yolo
        if args.gemma is not None:
            app.gemma_enabled = bool(args.gemma)
            if app.gemma_enabled:
                app._start_model_loading()
        app.run()
