"""Serve a stream of lightsail tracking records as Server-Sent Events (standard library only).

By default the stream is SIMULATED: a hidden 'true' sail is drawn from the uncertainty ranges
around Kulkarni, Lubin & Zhang (2018), Table 1, and its Doppler speed measurements are sent in
the order the reflected light would reach Earth. With --csv, a file of real (or other) records is
replayed instead. The browser simulation's "Live data" tab connects to /stream.

    python scripts/stream_server.py                      # simulated, whole stream in ~45 s
    python scripts/stream_server.py --csv records.csv    # replay measurements from a file
    python scripts/stream_server.py --speed 1            # real time (the acceleration takes minutes)

Endpoints:  /stream (text/event-stream)  ·  /records.csv  ·  /  (help)
Each record is sent as `data: {"seq", "t", "received_at", "beta", "sigma"}`; an `event: meta`
message comes first and an `event: end` message last.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lightsail import PRESETS  # noqa: E402
from lightsail.telemetry import from_csv, sample_truth, simulated_stream, to_csv  # noqa: E402

DESIGNS = {"kulkarni": "Kulkarni et al. 2018, Table 1 (1 g)", "ucsb": "UCSB calculator baseline"}


def build_records(args):
    if args.csv:
        return from_csv(Path(args.csv).read_text()), {"simulated": False, "source": Path(args.csv).name}
    design = PRESETS[DESIGNS[args.design]]
    truth = sample_truth(design, seed=args.seed)
    recs = simulated_stream(truth, noise=args.noise, seed=args.seed)
    return recs, {"simulated": True, "design": DESIGNS[args.design], "noise": args.noise, "seed": args.seed}


def wall_times(recs, args):
    """Wall-clock send time for each record: linear speed-up, or log-time over --duration seconds."""
    t = [r.received_at for r in recs]
    if args.speed:
        return [(x - t[0]) / args.speed for x in t]
    t0, t1 = max(t[0] * 0.5, 1e-3), t[-1]
    return [args.duration * math.log(max(x, t0) / t0) / math.log(t1 / t0) for x in t]


def make_handler(recs, meta, args):
    class Handler(BaseHTTPRequestHandler):
        def _headers(self, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

        def do_GET(self):
            if self.path.startswith("/stream"):
                self._headers("text/event-stream")
                try:
                    self.wfile.write(f"event: meta\ndata: {json.dumps({**meta, 'n': len(recs)})}\n\n".encode())
                    self.wfile.flush()
                    start = time.monotonic()
                    for r, w in zip(recs, wall_times(recs, args)):
                        delay = w - (time.monotonic() - start)
                        if delay > 0:
                            time.sleep(delay)
                        msg = {"seq": r.seq, "t": r.t, "received_at": r.received_at, "beta": r.beta, "sigma": r.sigma}
                        self.wfile.write(f"data: {json.dumps(msg)}\n\n".encode())
                        self.wfile.flush()
                    self.wfile.write(b"event: end\ndata: {}\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            elif self.path.startswith("/records.csv"):
                self._headers("text/csv")
                head = "# SIMULATED lightsail tracking records (no real flight data exist)\n" if meta["simulated"] else ""
                self.wfile.write((head + to_csv(recs)).encode())
            else:
                self._headers("text/plain; charset=utf-8")
                self.wfile.write((__doc__ or "").encode())

        def log_message(self, fmt, *a):
            sys.stderr.write("stream_server: " + fmt % a + "\n")

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--design", choices=sorted(DESIGNS), default="kulkarni")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--noise", type=float, default=0.002, help="1-sigma noise on beta")
    ap.add_argument("--csv", help="replay records from this CSV instead of simulating")
    ap.add_argument("--duration", type=float, default=45.0, help="log-time playback length [s]")
    ap.add_argument("--speed", type=float, default=0.0, help="linear speed-up instead (1 = real time)")
    args = ap.parse_args()
    recs, meta = build_records(args)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(recs, meta, args))
    kind = "SIMULATED" if meta["simulated"] else "replayed"
    print(f"Serving {len(recs)} {kind} records at http://localhost:{args.port}/stream  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
