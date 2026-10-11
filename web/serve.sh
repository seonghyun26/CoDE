#!/usr/bin/env bash
# Serve web/site.
#
#   bash web/serve.sh                 # http  on 0.0.0.0:8741
#   PORT=9000 bash web/serve.sh       # another port
#   bash web/serve.sh --https         # https with a self-signed cert (see below)
#
# ABOUT "Not secure". Nothing on the page needs a secure context -- no clipboard API,
# no service worker, no getUserMedia -- so plain http works completely. The label is
# the browser telling you the hop is unencrypted, which on a lab LAN is a judgement
# call, not a bug.
#
# The clean fix is an SSH tunnel, because browsers treat localhost as a secure origin:
#
#     ssh -N -L 8741:localhost:8741 <server>  # on your own machine
#     open http://localhost:8741
#
# That gives a real encrypted hop (ssh), no certificate warning, and a secure context.
# --https below is the alternative when other machines must reach the server directly: it is
# genuinely encrypted, but the certificate is self-signed, so every browser shows a
# full "Your connection is not private" interstitial you have to click through once.
# That is louder than the label it replaces, which is why it is not the default.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE="$HERE/site"
# shellcheck source=../voxbind/scripts/lib/common.sh
source "$HERE/../voxbind/scripts/lib/common.sh"
PY="${PY:-$(voxbind_conda_bin voxbind)/python}"
PORT="${PORT:-8741}"
# Localhost only: binding 0.0.0.0 would publish an unauthenticated page to every network the
# server is on. Reach it with:
#     ssh -N -L 8741:localhost:8741 <server>
HOST="${HOST:-127.0.0.1}"
CERT="$HERE/.dev-cert.pem"

[ -d "$SITE" ] || { echo "missing $SITE -- run: $PY $HERE/build.py" >&2; exit 1; }

if [ "${1:-}" = "--https" ]; then
    if [ ! -f "$CERT" ]; then
        echo "[serve] generating a self-signed certificate for $(hostname)"
        openssl req -x509 -newkey rsa:2048 -nodes -days 825 \
            -keyout "$CERT" -out "$CERT" \
            -subj "/CN=$(hostname)" \
            -addext "subjectAltName=DNS:$(hostname),DNS:localhost,IP:127.0.0.1" \
            2>/dev/null
        chmod 600 "$CERT"
    fi
    echo "[serve] https://localhost:$PORT  (bound to $HOST; self-signed)"
    exec "$PY" -c "
import http.server, ssl, functools, os
os.chdir('$SITE')
ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain('$CERT')
h = functools.partial(http.server.SimpleHTTPRequestHandler, directory='$SITE')
srv = http.server.ThreadingHTTPServer(('$HOST', $PORT), h)
srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
srv.serve_forever()
"
fi

echo "[serve] http://localhost:$PORT  (bound to $HOST)"
echo "[serve] from another machine: ssh -N -L $PORT:localhost:$PORT $(hostname)"
# http.server ignores Range and answers 200 with the whole file. The viewer asks for
# one SDF record out of a ~220 KB file, so without this it pays the full price every
# time (the page copes, but a pocket costs 1.3 MB instead of ~13 KB). A CDN such as
# Hugging Face already does this; locally we have to add it.
exec "$PY" - "$SITE" "$HOST" "$PORT" <<'PYEOF'
import os, re, sys, http.server, functools

SITE, HOST, PORT = sys.argv[1], sys.argv[2], int(sys.argv[3])


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    def send_head(self):
        rng = self.headers.get("Range")
        if not rng:
            return super().send_head()
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())
        path = self.translate_path(self.path)
        if not m or os.path.isdir(path):
            return super().send_head()
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404)
            return None
        size = os.fstat(f.fileno()).st_size
        first, last = m.group(1), m.group(2)
        if first == "":                      # suffix range: last N bytes
            if last == "":
                f.close(); self.send_error(400); return None
            start, end = max(0, size - int(last)), size - 1
        else:
            start = int(first)
            end = int(last) if last else size - 1
        end = min(end, size - 1)
        if start > end or start >= size:
            f.close()
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()        # end_headers() adds Accept-Ranges for every reply
        return _Slice(f, end - start + 1)

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()


class _Slice:
    """Hands copyfile() exactly the requested bytes and nothing after them."""

    def __init__(self, f, remaining):
        self.f, self.remaining = f, remaining

    def read(self, n=-1):
        if self.remaining <= 0:
            return b""
        if n is None or n < 0:
            n = self.remaining
        data = self.f.read(min(n, self.remaining))
        self.remaining -= len(data)
        return data

    def close(self):
        self.f.close()


h = functools.partial(RangeHandler, directory=SITE)
http.server.ThreadingHTTPServer((HOST, PORT), h).serve_forever()
PYEOF
