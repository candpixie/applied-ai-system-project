#!/usr/bin/env python3
"""
Local web UI for MoodLens.

    python serve.py            # http://localhost:8000
    python serve.py --port 8080

Standard library only, no Flask, no new dependencies. It wraps the same
MoodAgent the CLI uses, so what you see in the browser is the same decision
object, including the per-component signals and the execution trace.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from moodlens import MoodAgent

AGENT = MoodAgent()

SAMPLES = [
    ("Sarcasm", "oh perfect, my laptop died right before the deadline"),
    ("Negation", "not bad at all, I actually had fun"),
    ("Mixed", "grateful for the trip but the flights were exhausting"),
    ("Neutral", "the shipment arrives on Tuesday"),
    ("Out of domain", "quarterly synergy realignment scheduled for Q3"),
    ("Safety hold", "honestly I want to die, nothing is helping"),
]

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MoodLens</title>
<style>
  :root {
    --bg:#0f1115; --panel:#171a21; --line:#262b36; --text:#e8eaf0;
    --muted:#9aa3b5; --accent:#7aa2f7;
    --pos:#5ec27a; --neg:#e5687a; --neu:#8b93a7; --mix:#d9a441; --hold:#e5484d;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
    font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif; }
  .wrap { max-width:860px; margin:0 auto; padding:40px 20px 80px; }
  h1 { font-size:28px; margin:0 0 6px; letter-spacing:-.02em; }
  .sub { color:var(--muted); margin:0 0 28px; }
  form { display:flex; gap:10px; margin-bottom:14px; }
  input[type=text] { flex:1; background:var(--panel); border:1px solid var(--line);
    color:var(--text); border-radius:10px; padding:13px 15px; font-size:15px; }
  input[type=text]:focus { outline:none; border-color:var(--accent); }
  button { background:var(--accent); color:#0f1115; border:0; border-radius:10px;
    padding:13px 22px; font-size:15px; font-weight:650; cursor:pointer; }
  button:disabled { opacity:.5; cursor:default; }
  .chips { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:30px; }
  .chip { background:var(--panel); border:1px solid var(--line); color:var(--muted);
    border-radius:999px; padding:6px 13px; font-size:13px; cursor:pointer; }
  .chip:hover { border-color:var(--accent); color:var(--text); }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:14px;
    padding:24px; margin-bottom:16px; }
  .verdict { display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; }
  .label { font-size:32px; font-weight:700; letter-spacing:-.02em; }
  .positive{color:var(--pos)} .negative{color:var(--neg)}
  .neutral{color:var(--neu)} .mixed{color:var(--mix)} .uncertain{color:var(--hold)}
  .status { font-size:12px; text-transform:uppercase; letter-spacing:.09em;
    padding:4px 10px; border-radius:6px; border:1px solid var(--line); color:var(--muted); }
  .status.needs_review { color:var(--mix); border-color:var(--mix); }
  .status.safety_hold, .status.blocked { color:var(--hold); border-color:var(--hold); }
  .bar { height:6px; background:var(--line); border-radius:99px; margin:16px 0 6px; }
  .bar > i { display:block; height:100%; border-radius:99px; background:var(--accent); }
  .conf { color:var(--muted); font-size:13px; }
  .why { margin-top:16px; color:var(--muted); }
  .sig { display:grid; grid-template-columns:96px 82px 62px 1fr; gap:10px;
    padding:11px 0; border-top:1px solid var(--line); font-size:13px; align-items:start; }
  .src { color:var(--text); font-weight:600; }
  .mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px;
    color:var(--muted); word-break:break-word; }
  .abstain { opacity:.45; }
  h3 { font-size:12px; text-transform:uppercase; letter-spacing:.09em;
    color:var(--muted); margin:0 0 6px; font-weight:600; }
  .hold-note { border-left:3px solid var(--hold); padding-left:14px; }
</style></head><body><div class="wrap">
<h1>MoodLens</h1>
<p class="sub">A mood classifier that knows when it does not know.</p>

<form id="f">
  <input type="text" id="q" placeholder="Type a short post..." autocomplete="off" autofocus>
  <button id="go">Analyze</button>
</form>
<div class="chips" id="chips"></div>
<div id="out"></div>

<script>
const SAMPLES = __SAMPLES__;
const chips = document.getElementById('chips');
SAMPLES.forEach(([name, text]) => {
  const b = document.createElement('button');
  b.type = 'button'; b.className = 'chip'; b.textContent = name;
  b.onclick = () => { document.getElementById('q').value = text; run(); };
  chips.appendChild(b);
});

const esc = s => String(s).replace(/[&<>"]/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

async function run() {
  const text = document.getElementById('q').value;
  if (!text.trim()) return;
  const btn = document.getElementById('go');
  btn.disabled = true;
  try {
    const r = await fetch('/api/analyze', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text})
    });
    render(await r.json());
  } catch (e) {
    document.getElementById('out').innerHTML =
      '<div class="card">Request failed: ' + esc(e.message) + '</div>';
  } finally { btn.disabled = false; }
}

function render(d) {
  const pct = Math.round(d.confidence * 100);
  let signals = '';
  if (d.signals.length) {
    signals = '<div class="card"><h3>What each component said</h3>' +
      d.signals.map(s => `
        <div class="sig ${s.label ? '' : 'abstain'}">
          <div class="src">${esc(s.source)}</div>
          <div class="${esc(s.label || 'uncertain')}">${esc(s.label || 'abstained')}</div>
          <div class="mono">${s.confidence.toFixed(2)}</div>
          <div class="mono">${esc(s.rationale)}</div>
        </div>`).join('') + '</div>';
  }
  const trace = d.trace.length
    ? `<div class="card"><h3>Trace</h3><div class="mono">${esc(d.trace.join('  ->  '))}</div></div>`
    : '';
  const hold = (d.status === 'safety_hold' || d.status === 'blocked');
  document.getElementById('out').innerHTML = `
    <div class="card">
      <div class="verdict">
        <span class="label ${esc(d.label)}">${esc(d.label)}</span>
        <span class="status ${esc(d.status)}">${esc(d.status.replace('_',' '))}</span>
      </div>
      ${hold ? '' : `<div class="bar"><i style="width:${pct}%"></i></div>
        <div class="conf">confidence ${d.confidence.toFixed(2)}</div>`}
      <div class="why ${hold ? 'hold-note' : ''}">${esc(d.rationale)}</div>
    </div>${signals}${trace}`;
}

document.getElementById('f').onsubmit = e => { e.preventDefault(); run(); };
</script></div></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, content_type):
        payload = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            page = PAGE.replace("__SAMPLES__", json.dumps(SAMPLES))
            self._send(200, page, "text/html; charset=utf-8")
        else:
            self._send(404, "not found", "text/plain; charset=utf-8")

    def do_POST(self):
        if self.path != "/api/analyze":
            self._send(404, '{"error":"not found"}', "application/json")
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            decision = AGENT.analyze(payload.get("text", ""))
            self._send(200, json.dumps(decision.to_dict()), "application/json")
        except Exception as exc:  # noqa: BLE001 - a demo server must not die
            self._send(500, json.dumps({"error": str(exc)}), "application/json")

    def log_message(self, *args):
        pass  # keep the terminal clean while presenting


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"\n  MoodLens running at  http://localhost:{args.port}\n")
    print("  Ctrl+C to stop.\n")
    try:
        HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
