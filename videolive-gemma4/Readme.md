# Echtzeit-Kamera-Analyse mit Google Gemma 4 (E4B)

Dieses Projekt demonstriert die Echtzeit-Auswertung einer Laptop-Kamera in Python. Es verwendet das multimodale Open-Weights-Modell **Gemma 4 E4B** von Google DeepMind sowie **YOLO26l** für Objekterkennung. Beide Modellgewichte liegen **lokal im Projektordner `models/`** — der Start läuft daher **ohne Internetzugriff und ohne Hugging Face Token**.

## Voraussetzungen

1. **Hardware**
   * Eine funktionierende Webcam / Laptop-Kamera.
   * **Empfohlen:** Eine Nvidia-GPU (CUDA) oder Apple Silicon (Mac), um akzeptable Latenzen zu erzielen. Auf reinen x86-CPUs erfolgt die Auswertung stark verzögert.

2. **Abhängigkeiten**
   ```bash
   pip install -r requirements.txt
   ```

## Modellgewichte (lokal)

Die Gewichte liegen im Ordner `models/`:

```
models/
├── gemma-4-e4b-it/    # Gemma 4 E4B (model.safetensors ~15 GB + Configs/Tokenizer)
└── yolo26l.pt         # YOLO26l Objekt-Detektor (~51 MB)
```

* `app8.py` lädt Gemma direkt aus `models/gemma-4-e4b-it` und YOLO aus `models/yolo26l.pt`.
* **Kein HF_TOKEN, kein Internet nötig** (sowohl Gemma als auch YOLO sind öffentlich).
* `app8.py` prüft nur, dass `models/gemma-4-e4b-it` existiert.
* **Alle Ausgaben (Start, Fehler, Inferenz, Speichern) gehen in `log/app8.log`, nicht in die Konsole** (override: `GEMMA_DEBUG_LOG`).

### Gewichte einmalig herunterladen

Am einfachsten per Script (nutzt `curl` oder `wget`, loggt nach `log/`):

```bash
./download_weights.sh
```

Es lädt die zwei Quellen direkt in `models/`:
* **YOLO26l:** `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt`
* **Gemma 4 (7 Dateien, u. a. model.safetensors ~15 GB):** `https://huggingface.co/google/gemma-4-e4b-it/resolve/main/<datei>`

Manuell mit `curl` (äquivalent, ohne Script):

```bash
curl -fSL -o models/yolo26l.pt \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt

for f in config.json generation_config.json processor_config.json \
         tokenizer.json tokenizer_config.json chat_template.jinja model.safetensors; do
  curl -fSL -o "models/gemma-4-e4b-it/$f" \
    "https://huggingface.co/google/gemma-4-e4b-it/resolve/main/$f"
done
```

* Das Script ist idempotent: vorhandene Dateien werden übersprungen. `FORCE=1` erzwingt Neu-Download, `ONLY_SMALL=1` lädt beim Gemma nur die kleine Konfiguration/Tokenizer (ohne die ~15 GB), `SKIP_YOLO=1` / `SKIP_GEMMA=1` überspringen eine der beiden.
* Protokoll: `log/download_weights.log` · Staging: `tmp/downloads/` (wird nach dem Download nach `models/` verschoben).

## Starten der Anwendung

```bash
python app8.py
```

* **Eigene Anweisung (optional):** `GEMMA_PROMPT="Benenne das prominenteste Objekt."`
* **Höhere Kamera-Auflösung (optional):** `GEMMA_CAM_W=1920 GEMMA_CAM_H=1080`
* **Beenden:** Taste **'q'** im OpenCV-Fenster; **'s'** speichert Frame + Text nach `tmp/`.

## Verhalten (kurz)

* **Adaptive Analyse:** die nächste Auswertung startet ~1 s nach Ende der vorherigen (keine feste Bildrate), da die Inferenz langsamer ist als der Frame-Stream.
* **Stabile Szene (Harter Cache):** bleibt die Szene (Objektanzahl + Positionen) unverändert, wird der zuletzt erzeugte Text direkt angezeigt – **Gemma wird übersprungen** (Log-Zeile `CACHE HIT`). Damit bleibt die Beschreibung bei einer statischen Szene stabil und ändert sich nicht.
* **Geänderte Szene:** neue/verschobene Objekte -> Gemma beschreibt den neuen Zustand.
