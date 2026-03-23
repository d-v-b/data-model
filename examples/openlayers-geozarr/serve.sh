#!/usr/bin/env bash
# Usage:
#   ./serve.sh <zarr-url-or-local-path> [multiscales-group]
#
# Examples:
#   ./serve.sh /tmp/s2_converted.zarr measurements/reflectance
#   ./serve.sh https://example.com/store.zarr measurements/reflectance
#
# The multiscales group defaults to "measurements/reflectance".

set -euo pipefail

ZARR_PATH="${1:?Usage: ./serve.sh <zarr-url-or-local-path> [multiscales-group]}"
GROUP="${2:-measurements/reflectance}"
FILE_SERVER_PORT=8000
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cleanup() {
  if [[ -n "${FILE_SERVER_PID:-}" ]]; then
    kill "$FILE_SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# If the path is local, start a CORS file server
if [[ ! "$ZARR_PATH" =~ ^https?:// ]]; then
  ZARR_PATH="$(cd "$ZARR_PATH" && pwd)"
  echo "Starting file server for $ZARR_PATH on port $FILE_SERVER_PORT..."
  python3 -c "
import http.server, os, socket

class CORSHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()
    def log_message(self, format, *args):
        pass  # suppress request logs

os.chdir('$ZARR_PATH')
server = http.server.HTTPServer(('', $FILE_SERVER_PORT), CORSHandler)
server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.serve_forever()
" &
  FILE_SERVER_PID=$!
  DATA_URL="http://localhost:${FILE_SERVER_PORT}/${GROUP}"
else
  DATA_URL="${ZARR_PATH}/${GROUP}"
fi

echo "Data URL: $DATA_URL"
ENCODED_URL=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$DATA_URL', safe=''))")

cd "$SCRIPT_DIR"
if [[ ! -d node_modules ]]; then
  echo "Installing dependencies..."
  npm install
fi

echo ""
echo "Open in browser:"
echo "  http://localhost:5173/?data=${DATA_URL}"
echo ""

npx vite --open "/?data=${ENCODED_URL}"
