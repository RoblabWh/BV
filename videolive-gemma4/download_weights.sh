#!/usr/bin/env bash
#
# download_weights.sh — lädt die beiden Modellgewichte aus dem Internet in models/.
#
#   YOLO26l:  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt
#              -> models/yolo26l.pt                              (53211173 Byte, wird geprüft)
#   Gemma 4:  https://huggingface.co/google/gemma-4-e4b-it/resolve/main/<datei>
#              -> models/gemma-4-e4b-it/<datei>  (7 Dateien, model.safetensors ~15 GB)
#
# Konvention des Projekts:
#   - Arbeits-/Staging-Ort : tmp/downloads/   (vorübergehend, danach mv nach models/)
#   - Protokoll            : log/download_weights.log
#   - Zielfolder           : models/
#
# Steuerung per Umgebungsvariablen:
#   FORCE=1          Neu laden, auch wenn die Datei schon da ist.
#   ONLY_SMALL=1     Beim Gemma nur die kleinen Dateien (Config/Tokenizer),
#                    model.safetensors (~15 GB) wird übersprungen.
#   SKIP_YOLO=1      YOLO-Gewicht überspringen.
#   SKIP_GEMMA=1     Gemma überspringen.
#
# Beispiel:  FORCE=1 ONLY_SMALL=1 ./download_weights.sh   (Gemma-Konfig neu, ohne 15 GB)

set -euo pipefail
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$ROOT/models"
TMP_DIR="$ROOT/tmp/downloads"
LOG_DIR="$ROOT/log"
LOG_FILE="$LOG_DIR/download_weights.log"

YOLO_URL="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt"
YOLO_DEST="$MODELS_DIR/yolo26l.pt"

GEMMA_REPO="google/gemma-4-e4b-it"
GEMMA_BASE="https://huggingface.co/${GEMMA_REPO}/resolve/main"
GEMMA_DEST="$MODELS_DIR/gemma-4-e4b-it"
GEMMA_FILES=(
  "config.json"
  "generation_config.json"
  "processor_config.json"
  "tokenizer.json"
  "tokenizer_config.json"
  "chat_template.jinja"
  "model.safetensors"
)

mkdir -p "$MODELS_DIR" "$TMP_DIR" "$LOG_DIR"
: > "$LOG_FILE"   # Log je Lauf neu starten (überschreiben)

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG_FILE"; }

# curl bevorzugen, sonst wget.
if command -v curl >/dev/null 2>&1; then
  DL="curl"
elif command -v wget >/dev/null 2>&1; then
  DL="wget"
else
  log "FEHLER: weder 'curl' noch 'wget' gefunden."
  exit 1
fi

# download <url> <dest> <expected_size|->
# Lädt nach tmp/downloads/<name>.part, prüft Größe (optional), dann mv nach dest.
fetch_file() {
  local url="$1" dest="$2" expected="${3:--}"
  local name part
  name="$(basename "$dest")"
  part="$TMP_DIR/$name.part"

  if [[ "${FORCE:-0}" != "1" && -s "$dest" ]]; then
    log "SKIP (existiert bereits): $dest"
    return 0
  fi

  log "GET  $url"
  rm -f "$part"
  local rc=0
  if [[ "$DL" == "curl" ]]; then
    # -f HTTP-Fehler -> Exit-Fehler   -L Redirects   -C - Fortsetzung
    # -s leise (kein Progress in Konsole)   -S Fehler trotzdem auf stderr -> Log
    curl -fsSL --retry 4 --retry-delay 2 -C - -o "$part" "$url" 2>>"$LOG_FILE" || rc=$?
  else
    wget -q --tries=4 -c -O "$part" "$url" 2>>"$LOG_FILE" || rc=$?
  fi
  local ok=0
  if [[ "$rc" == "0" ]]; then ok=1; fi

  if [[ "$ok" != "1" ]]; then
    log "FEHLER: Download fehlgeschlagen: $name  ($url)"
    rm -f "$part"
    return 1
  fi

  if [[ ! -s "$part" ]]; then
    log "FEHLER: Datei leer nach Download: $name"
    rm -f "$part"
    return 1
  fi

  if [[ -n "$expected" && "$expected" != "-" ]]; then
    local got
    got="$(stat -c%s "$part" 2>/dev/null || stat -f%z "$part")"
    if [[ "$got" != "$expected" ]]; then
      log "FEHLER: Größen-Mismatch $name  (erhalten $got, erwartet $expected)"
      rm -f "$part"
      return 1
    fi
  fi

  local dir
  dir="$(dirname "$dest")"
  mkdir -p "$dir"
  mv "$part" "$dest"
  local hum
  hum="$(du -h "$dest" | cut -f1)"
  log "OK   $dest  ($hum)"
}

# ---------------------------------------------------------------- YOLO
if [[ "${SKIP_YOLO:-0}" != "1" ]]; then
  fetch_file "$YOLO_URL" "$YOLO_DEST" "53211173"
else
  log "SKIP (SKIP_YOLO=1): YOLO"
fi

# ---------------------------------------------------------------- GEMMA
if [[ "${SKIP_GEMMA:-0}" != "1" ]]; then
  for f in "${GEMMA_FILES[@]}"; do
    if [[ "$f" == "model.safetensors" && "${ONLY_SMALL:-0}" == "1" ]]; then
      log "SKIP (ONLY_SMALL=1): $f  (~15 GB)"
      continue
    fi
    # Gemma: keine harte Größenprüfung (Datei wird von der App beim Laden validiert).
    fetch_file "$GEMMA_BASE/$f" "$GEMMA_DEST/$f" "-"
  done
else
  log "SKIP (SKIP_GEMMA=1): Gemma"
fi

# ---------------------------------------------------------------- FERTIG
if [[ -s "$YOLO_DEST" ]]; then
  log "YOLO  vorhanden: $(du -h "$YOLO_DEST" | cut -f1)  ($YOLO_DEST)"
fi
if [[ -s "$GEMMA_DEST/model.safetensors" ]]; then
  log "GEMMA vorhanden: $(du -sh "$GEMMA_DEST" | cut -f1)  ($GEMMA_DEST)"
fi
log "FERTIG. Protokoll: $LOG_FILE"
exit 0
