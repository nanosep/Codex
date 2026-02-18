import argparse
import html
import json
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from nanin_runner import NaninRunner


JOBS = {}
JOBS_LOCK = threading.Lock()


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NANIN Runner</title>
  <style>
    body { font-family: Segoe UI, Arial, sans-serif; margin: 24px; background: #f7f5ef; color: #1f2937; }
    .card { max-width: 980px; margin: 0 auto; background: #fff; border: 1px solid #ddd; border-radius: 12px; padding: 16px; }
    textarea, input[type=text], input[type=number], input[type=password] { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px; }
    textarea { min-height: 120px; resize: vertical; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .checks { margin-top: 10px; display: flex; gap: 14px; flex-wrap: wrap; }
    .catbox { margin-top: 12px; border: 1px solid #ddd; border-radius: 10px; padding: 10px; background: #fcfcfc; }
    .catrow { margin: 8px 0; }
    .catrow small { color: #4b5563; display: block; margin-left: 22px; }
    .artifactbox { margin-top: 12px; border: 1px solid #ddd; border-radius: 10px; padding: 10px; background: #fffef8; max-height: 280px; overflow: auto; }
    .artifactrow { margin: 8px 0; }
    .artifactrow small { color: #4b5563; display: block; margin-left: 22px; }
    button { margin-top: 12px; background: #0f766e; color: #fff; border: 0; border-radius: 8px; padding: 10px 14px; cursor: pointer; }
    .status { margin-top: 16px; padding: 10px; background: #ecfeff; border: 1px solid #bae6fd; border-radius: 8px; }
    .error { margin-top: 16px; padding: 10px; background: #fee2e2; border: 1px solid #fecaca; color: #7f1d1d; border-radius: 8px; white-space: pre-wrap; }
    .item { margin-top: 16px; border: 1px solid #ddd; border-radius: 10px; padding: 10px; background: #fafafa; }
    .item h3 { margin: 0 0 8px 0; font-size: 16px; }
    pre { white-space: pre-wrap; overflow: auto; background: #111827; color: #e5e7eb; border-radius: 8px; padding: 10px; }
    @media (max-width: 720px) { .row { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="card">
    <h1>NANIN 6-Prompt Runner</h1>
    <p>Generates prompts sequentially with live progress.</p>

    <label>OpenAI API Key (or leave empty if OPENAI_API_KEY is set)</label>
    <input id="apiKey" type="password" placeholder="sk-..." autocomplete="off" />

    <label style="margin-top:10px;">Input</label>
    <textarea id="inputText" placeholder="Describe your case..."></textarea>

    <div class="row" style="margin-top:10px;">
      <div>
        <label>Model</label>
        <input id="model" type="text" value="gpt-5.2-codex" />
      </div>
      <div>
        <label>Seed (optional)</label>
        <input id="seed" type="number" />
      </div>
    </div>

    <div class="checks">
      <label><input id="fastMode" type="checkbox" checked /> Fast mode (recommended)</label>
      <label><input id="noEmbeddings" type="checkbox" /> No embeddings</label>
      <label><input id="includeWhy" type="checkbox" /> Include why_selected</label>
      <label><input id="showMetadata" type="checkbox" /> Show internal metadata</label>
    </div>

    <div class="catbox">
      <strong>Select what to generate (not mandatory all 6)</strong>
      <div class="catrow">
        <label><input type="checkbox" name="cat" value="Artifacts" checked /> Artifacts</label>
        <small>Best for: framing leverage and analytical structure when input is ambiguous.</small>
      </div>
      <div class="catrow">
        <label><input type="checkbox" name="cat" value="Bonds & Beasts" checked /> Bonds & Beasts</label>
        <small>Best for: social dynamics, trust/fear patterns, stakeholder behavior under pressure.</small>
      </div>
      <div class="catrow">
        <label><input type="checkbox" name="cat" value="Cards" checked /> Cards</label>
        <small>Best for: lightweight reasoning moves and tactical lenses for concrete decisions.</small>
      </div>
      <div class="catrow">
        <label><input type="checkbox" name="cat" value="Glamour" checked /> Glamour</label>
        <small>Best for: narrative packaging, persuasion layer, and message salience.</small>
      </div>
      <div class="catrow">
        <label><input type="checkbox" name="cat" value="Rituals" checked /> Rituals</label>
        <small>Best for: repeatable operating routines (used as-defined, no reinterpretation).</small>
      </div>
      <div class="catrow">
        <label><input type="checkbox" name="cat" value="Weapons" checked /> Weapons</label>
        <small>Best for: decisive interventions, adversarial framing, and pressure testing.</small>
      </div>
    </div>
    <div class="artifactbox">
      <strong>Artifacts filter (optional)</strong>
      <p style="margin:6px 0 0 0; color:#4b5563;">If Artifacts is selected, you can force specific artifacts here.</p>
      {artifact_options_html}
    </div>

    <button id="runBtn">Run</button>

    <div id="status" class="status" style="display:none;"></div>
    <div id="error" class="error" style="display:none;"></div>
    <div id="results"></div>
  </div>

  <script>
    const runBtn = document.getElementById("runBtn");
    const statusEl = document.getElementById("status");
    const errorEl = document.getElementById("error");
    const resultsEl = document.getElementById("results");
    let timer = null;

    function esc(s) {
      return String(s || "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }

    function renderState(data) {
      statusEl.style.display = "block";
      statusEl.textContent = `Status: ${data.status} | Step: ${data.step || "-"} | Generated: ${data.generated_count || 0}/6`;

      if (data.error) {
        errorEl.style.display = "block";
        errorEl.textContent = data.error;
      } else {
        errorEl.style.display = "none";
      }

      resultsEl.innerHTML = "";
      (data.partial_items || []).forEach(item => {
        const div = document.createElement("div");
        div.className = "item";
        div.innerHTML = `<h3>${esc(item.category)}</h3><pre>${esc(item.main_prompt)}</pre>`;
        resultsEl.appendChild(div);
      });

      if (data.status === "completed" && data.result) {
        statusEl.textContent = `Status: completed | Generated: ${data.generated_count || 0}/${data.total_count || 0}`;
      }
    }

    async function poll(jobId) {
      try {
        const r = await fetch(`/status?job_id=${encodeURIComponent(jobId)}`);
        const data = await r.json();
        renderState(data);
        if (data.status === "completed" || data.status === "error") {
          clearInterval(timer);
          timer = null;
          runBtn.disabled = false;
        }
      } catch (e) {
        clearInterval(timer);
        timer = null;
        runBtn.disabled = false;
        errorEl.style.display = "block";
        errorEl.textContent = String(e);
      }
    }

    runBtn.addEventListener("click", async () => {
      errorEl.style.display = "none";
      resultsEl.innerHTML = "";
      statusEl.style.display = "block";
      statusEl.textContent = "Starting...";
      runBtn.disabled = true;

      const inputText = document.getElementById("inputText").value.trim();
      if (!inputText) {
        runBtn.disabled = false;
        errorEl.style.display = "block";
        errorEl.textContent = "Input is required.";
        return;
      }

      const form = new URLSearchParams();
      form.set("api_key", document.getElementById("apiKey").value.trim());
      form.set("input_text", inputText);
      form.set("model", document.getElementById("model").value.trim() || "gpt-5.2-codex");
      form.set("seed", document.getElementById("seed").value.trim());
      form.set("fast_mode", document.getElementById("fastMode").checked ? "1" : "0");
      form.set("no_embeddings", document.getElementById("noEmbeddings").checked ? "1" : "0");
      form.set("include_why", document.getElementById("includeWhy").checked ? "1" : "0");
      form.set("show_metadata", document.getElementById("showMetadata").checked ? "1" : "0");
      const selectedCats = Array.from(document.querySelectorAll("input[name='cat']:checked")).map(el => el.value);
      form.set("selected_categories", selectedCats.join(","));
      const selectedArtifacts = Array.from(document.querySelectorAll("input[name='artifact']:checked")).map(el => el.value);
      form.set("selected_artifacts", selectedArtifacts.join(","));

      try {
        const r = await fetch("/start", {
          method: "POST",
          headers: { "Content-Type": "application/x-www-form-urlencoded" },
          body: form.toString()
        });
        const data = await r.json();
        if (!data.job_id) {
          throw new Error(data.error || "Could not start job.");
        }
        if (timer) clearInterval(timer);
        timer = setInterval(() => poll(data.job_id), 1000);
        poll(data.job_id);
      } catch (e) {
        runBtn.disabled = false;
        errorEl.style.display = "block";
        errorEl.textContent = String(e);
      }
    });
  </script>
</body>
</html>
"""


def load_artifact_choices() -> list:
    try:
        path = Path("Nanin Artifacts.json")
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(payload, list):
            return []
        out = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            slug = str(item.get("slug_id", "")).strip()
            name = str(item.get("object", slug or "artifact")).strip()
            when = item.get("when_to_use")
            brief = ""
            if isinstance(when, list) and when:
                brief = str(when[0])
            elif isinstance(item.get("council_objective"), str):
                brief = item.get("council_objective", "")
            if slug:
                out.append({"slug": slug, "name": name, "brief": brief})
        return out
    except Exception:
        return []


def render_artifact_options_html() -> str:
    rows = []
    for art in load_artifact_choices():
        rows.append(
            "<div class=\"artifactrow\">"
            f"<label><input type=\"checkbox\" name=\"artifact\" value=\"{html.escape(art['slug'], quote=True)}\" /> "
            f"{html.escape(art['name'])}</label>"
            f"<small>{html.escape(art['brief'])}</small>"
            "</div>"
        )
    if not rows:
        return "<small>No artifacts loaded from `Nanin Artifacts.json`.</small>"
    return "".join(rows)


def render_html_page() -> str:
    return HTML_PAGE.replace("{artifact_options_html}", render_artifact_options_html())


def _now() -> float:
    return time.time()


def create_job(data: dict) -> str:
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "step": "queued",
            "created_at": _now(),
            "updated_at": _now(),
            "generated_count": 0,
            "total_count": len(data.get("selected_categories", [])),
            "partial_items": [],
            "result": None,
            "error": None,
            "params": data,
        }
    return job_id


def update_job(job_id: str, **kwargs) -> None:
    with JOBS_LOCK:
        if job_id not in JOBS:
            return
        JOBS[job_id].update(kwargs)
        JOBS[job_id]["updated_at"] = _now()


def get_job(job_id: str) -> dict:
    with JOBS_LOCK:
        return dict(JOBS.get(job_id, {"status": "not_found", "error": "Job not found"}))


def run_job(job_id: str) -> None:
    params = get_job(job_id).get("params", {})
    try:
        update_job(job_id, status="running", step="initializing")
        runner = NaninRunner(
            data_dir=Path("."),
            ethos_path=None,
            model=params.get("model", "gpt-5.2-codex"),
            embedding_model="text-embedding-3-large",
            seed=params.get("seed"),
            use_embeddings=not params.get("no_embeddings", False),
            cache_path=Path(".nanin_cache/embeddings.json"),
            include_why=params.get("include_why", False),
            api_key=params.get("api_key") or None,
            fast_mode=params.get("fast_mode", True),
            include_metadata=params.get("show_metadata", False),
        )
        selected_categories = params.get("selected_categories") or []

        def on_progress(event: dict) -> None:
            stage = event.get("stage", "")
            if stage == "ethos_selected":
                update_job(job_id, step=f"ETHOS selected ({event.get('ethos_name', 'unknown')})")
            elif stage == "selection_done":
                update_job(job_id, step=f"Selection done: {event.get('category')}")
            elif stage == "prompt_done":
                state = get_job(job_id)
                partial = list(state.get("partial_items", []))
                item = event.get("item")
                if item:
                    partial.append(item)
                update_job(
                    job_id,
                    step=f"Generated: {event.get('category')}",
                    partial_items=partial,
                    generated_count=len(partial),
                    total_count=len(selected_categories),
                )
            elif stage == "completed":
                update_job(job_id, step="completed")

        result = runner.run(
            params.get("input_text", ""),
            progress_callback=on_progress,
            selected_categories=selected_categories,
            selected_artifact_slugs=params.get("selected_artifacts", []),
        )
        update_job(job_id, status="completed", step="completed", result=result)
    except Exception as exc:
        update_job(job_id, status="error", error=f"{exc}\n\n{traceback.format_exc(limit=5)}")


class NaninWebHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(render_html_page())
            return
        if parsed.path == "/status":
            qs = parse_qs(parsed.query or "")
            job_id = (qs.get("job_id", [""])[0] or "").strip()
            self._send_json(get_job(job_id))
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/start":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(length).decode("utf-8", errors="replace")
        form = parse_qs(payload)

        input_text = (form.get("input_text", [""])[0] or "").strip()
        if not input_text:
            self._send_json({"error": "Input is required."}, status=400)
            return

        seed_text = (form.get("seed", [""])[0] or "").strip()
        seed = None
        if seed_text:
            try:
                seed = int(seed_text)
            except ValueError:
                self._send_json({"error": "Seed must be a valid integer."}, status=400)
                return

        params = {
            "api_key": (form.get("api_key", [""])[0] or "").strip(),
            "input_text": input_text,
            "model": (form.get("model", ["gpt-5.2-codex"])[0] or "gpt-5.2-codex").strip(),
            "seed": seed,
            "fast_mode": (form.get("fast_mode", ["1"])[0] == "1"),
            "no_embeddings": (form.get("no_embeddings", ["0"])[0] == "1"),
            "include_why": (form.get("include_why", ["0"])[0] == "1"),
            "show_metadata": (form.get("show_metadata", ["0"])[0] == "1"),
            "selected_categories": [
                c.strip()
                for c in (form.get("selected_categories", [""])[0] or "").split(",")
                if c.strip()
            ],
            "selected_artifacts": [
                a.strip()
                for a in (form.get("selected_artifacts", [""])[0] or "").split(",")
                if a.strip()
            ],
        }
        if not params["selected_categories"]:
            self._send_json({"error": "Select at least one category."}, status=400)
            return

        job_id = create_job(params)
        thread = threading.Thread(target=run_job, args=(job_id,), daemon=True)
        thread.start()
        self._send_json({"job_id": job_id})

    def _send_html(self, body: str) -> None:
        raw = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_json(self, data: dict, status: int = 200) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format: str, *args) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NANIN web UI with live progress")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), NaninWebHandler)
    print(f"NANIN web UI running on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
