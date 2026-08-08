#!/usr/bin/env bash
# Wait for JC promo final mp4, then send via Telegram (text + video).
set -euo pipefail
set -a
# shellcheck disable=SC1091
source /home/keyspark/.hermes/.env
set +a

LOG=/tmp/jc_promo_30s/run.log
OUT=/home/keyspark/Videos/jc_promo_30s/jc_promo_ablit_h3_turbo_30s.mp4
CHAT="${TELEGRAM_ALLOWED_USERS%%,*}"
TOKEN="$TELEGRAM_BOT_TOKEN"
API="https://api.telegram.org/bot${TOKEN}"
STATE=/tmp/jc_promo_tg_sent.flag

if [ -f "$STATE" ]; then
  echo "already sent, exit"
  exit 0
fi

send_text() {
  curl -sf -X POST "$API/sendMessage" \
    -d "chat_id=$CHAT" \
    --data-urlencode "text=$1" \
    -d "disable_web_page_preview=true" >/dev/null || true
}

send_video() {
  local f="$1" cap="$2"
  local sendf="$f"
  local sz
  sz=$(stat -c%s "$f")
  if [ "$sz" -gt 48000000 ]; then
    sendf=/tmp/jc_promo_tg_send.mp4
    ffmpeg -y -i "$f" -c:v libx264 -crf 23 -preset fast -c:a aac -b:a 128k \
      -movflags +faststart "$sendf" >/dev/null 2>&1
  fi
  curl -sf -X POST "$API/sendVideo" \
    -F "chat_id=$CHAT" \
    -F "video=@${sendf}" \
    -F "caption=${cap}" \
    -F "supports_streaming=true" \
    -o /tmp/jc_promo_tg_video_reply.json || true
  if ! grep -q '"ok":true' /tmp/jc_promo_tg_video_reply.json 2>/dev/null; then
    curl -sf -X POST "$API/sendDocument" \
      -F "chat_id=$CHAT" \
      -F "document=@${sendf}" \
      -F "caption=${cap}" \
      -o /tmp/jc_promo_tg_video_reply.json || true
  fi
  cat /tmp/jc_promo_tg_video_reply.json >> /tmp/jc_promo_tg_watch.log 2>/dev/null || true
  echo >> /tmp/jc_promo_tg_watch.log
}

for i in $(seq 1 540); do
  if grep -q "Done in" "$LOG" 2>/dev/null && [ -f "$OUT" ]; then
    WALL=$(grep "Done in" "$LOG" | tail -1)
    DUR=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUT" 2>/dev/null || echo "?")
    SIZE=$(ls -lh "$OUT" | awk '{print $5}')
    CAP="✅ JC promo 30s READY
${WALL}
${DUR}s · ${SIZE}
dual-H3 FLF multishot + Turbo · 576×768
keys + tonyd2wild ablit/heretic DSV4F DSpark + MiniMax H3 Turbo parallel 2× DGX Spark"
    send_text "🎬 Final 30s video is ready — uploading to Telegram now…"
    send_video "$OUT" "$CAP"
    date > "$STATE"
    echo "VIDEO_SENT $(date)" >> /tmp/jc_promo_tg_watch.log
    exit 0
  fi
  if ! pgrep -f "python3 jc_promo_30s.py" >/dev/null 2>&1; then
    if ! grep -q "Done in" "$LOG" 2>/dev/null; then
      send_text "⚠️ JC promo process exited without finishing. Check /tmp/jc_promo_30s/run.log"
      exit 1
    fi
  fi
  sleep 20
done
send_text "⏱️ JC promo still not done after long wait — check /tmp/jc_promo_30s/run.log"
exit 1
