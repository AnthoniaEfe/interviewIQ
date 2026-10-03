"""Offline single-file gold-set annotator (no external scripts)."""

from __future__ import annotations

import json
from typing import Any


def render_annotator_html(items: list[dict[str, Any]], orders: dict[str, list[str]]) -> str:
    payload = json.dumps({"items": items, "orders": orders}, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>InterviewIQ gold-set annotator</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 0 auto; padding: 16px; }}
.bar {{ display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-bottom:12px; }}
button, select {{ padding: 6px 10px; font-size: 14px; }}
.req {{ font-size: 1.15rem; font-weight: 600; margin: 12px 0; }}
.search {{ width: 100%; padding: 8px; margin: 8px 0; }}
.cv {{ white-space: pre-wrap; border: 1px solid #ccc; padding: 12px; max-height: 420px; overflow: auto; line-height: 1.4; }}
mark {{ background: #ffe08a; }}
details {{ margin: 12px 0; }}
textarea {{ width: 100%; min-height: 64px; }}
.progress {{ margin-left: auto; }}
.cal {{ color: #8a4b08; font-weight: 600; }}
kbd {{ border: 1px solid #888; padding: 0 4px; border-radius: 3px; }}
</style>
</head>
<body>
<h1>Gold-set annotation</h1>
<p>Judge the requirement against the <strong>whole CV</strong>. Keys: <kbd>0</kbd> not evidenced, <kbd>1</kbd> partial, <kbd>2</kbd> clearly met, <kbd>C</kbd> cannot judge. Never infer a system score — none are shown.</p>
<div class="bar">
  <label>Annotator
    <select id="who"><option value="annotator_A">A</option><option value="annotator_B">B</option></select>
  </label>
  <button type="button" id="prev">Prev</button>
  <button type="button" id="next">Next</button>
  <span class="progress" id="prog"></span>
  <button type="button" id="exportCsv">Export CSV</button>
  <button type="button" id="exportJson">Export JSON</button>
  <label>Import <input type="file" id="imp" accept=".json,.csv,application/json,text/csv"/></label>
</div>
<div id="calflag" class="cal"></div>
<div class="req" id="req"></div>
<input class="search" id="q" placeholder="Search CV (highlights matches)"/>
<div class="cv" id="cv"></div>
<details><summary>Job description (click to expand)</summary><pre id="jd" class="cv"></pre></details>
<div class="bar">
  <button type="button" data-lab="0">0 not evidenced</button>
  <button type="button" data-lab="1">1 partial</button>
  <button type="button" data-lab="2">2 clearly met</button>
  <button type="button" data-lab="C">cannot judge</button>
</div>
<p>Optional evidence snippet (paste from CV):</p>
<textarea id="snip"></textarea>
<p>Notes:</p>
<textarea id="notes"></textarea>
<script>
const DATA = {payload};
const ITEMS = Object.fromEntries(DATA.items.map(x => [x.item_id, x]));
const storeKey = () => "goldset_" + document.getElementById("who").value;
function loadState() {{
  try {{ return JSON.parse(localStorage.getItem(storeKey()) || "{{}}"); }}
  catch (e) {{ return {{}}; }}
}}
function saveState(st) {{ localStorage.setItem(storeKey(), JSON.stringify(st)); }}
let idx = 0;
function order() {{
  const who = document.getElementById("who").value;
  return DATA.orders[who] || DATA.items.map(x => x.item_id);
}}
function currentId() {{ return order()[idx]; }}
function render() {{
  const ids = order();
  if (idx >= ids.length) idx = ids.length - 1;
  if (idx < 0) idx = 0;
  const it = ITEMS[ids[idx]];
  document.getElementById("prog").textContent = (idx+1) + " / " + ids.length;
  document.getElementById("calflag").textContent = it.is_calibration ? "CALIBRATION ITEM" : "";
  document.getElementById("req").textContent = it.requirement_text || "";
  document.getElementById("jd").textContent = it.jd_text || "";
  const st = loadState();
  const rec = st[it.item_id] || {{}};
  document.getElementById("snip").value = rec.snippet || "";
  document.getElementById("notes").value = rec.notes || "";
  highlight(it.cv_text || "", document.getElementById("q").value);
}}
function highlight(text, q) {{
  const el = document.getElementById("cv");
  if (!q) {{ el.textContent = text; return; }}
  const esc = s => s.replace(/[&<>]/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;"}}[c]));
  const re = new RegExp(q.replace(/[.*+?^${{}}()|[\\]\\\\]/g, "\\\\$&"), "gi");
  el.innerHTML = esc(text).replace(re, m => "<mark>"+m+"</mark>");
}}
function setLabel(lab) {{
  const st = loadState();
  const id = currentId();
  st[id] = st[id] || {{item_id: id}};
  if (lab === "C") {{ st[id].cannot_judge = true; st[id].label = ""; }}
  else {{ st[id].cannot_judge = false; st[id].label = lab; }}
  st[id].snippet = document.getElementById("snip").value;
  st[id].notes = document.getElementById("notes").value;
  saveState(st);
  idx = Math.min(idx+1, order().length-1);
  render();
}}
function persistText() {{
  const st = loadState();
  const id = currentId();
  st[id] = st[id] || {{item_id: id}};
  st[id].snippet = document.getElementById("snip").value;
  st[id].notes = document.getElementById("notes").value;
  saveState(st);
}}
function rows() {{
  persistText();
  const st = loadState();
  return order().map(id => {{
    const r = st[id] || {{item_id: id}};
    return {{
      item_id: id,
      label: r.label ?? "",
      cannot_judge: r.cannot_judge ? "true" : "false",
      snippet: r.snippet || "",
      notes: r.notes || ""
    }};
  }});
}}
function exportCsv() {{
  const rs = rows();
  const header = "item_id,label,cannot_judge,snippet,notes";
  const body = rs.map(r => [r.item_id, r.label, r.cannot_judge,
    JSON.stringify(r.snippet||""), JSON.stringify(r.notes||"")].join(",")).join("\\n");
  download(header+"\\n"+body, "annotations.csv", "text/csv");
}}
function exportJson() {{
  download(JSON.stringify(rows(), null, 2), "annotations.json", "application/json");
}}
function download(text, name, type) {{
  const blob = new Blob([text], {{type}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
}}
document.getElementById("prev").onclick = () => {{ persistText(); idx--; render(); }};
document.getElementById("next").onclick = () => {{ persistText(); idx++; render(); }};
document.getElementById("who").onchange = () => {{ idx = 0; render(); }};
document.getElementById("q").oninput = () => {{
  const it = ITEMS[currentId()];
  highlight(it.cv_text || "", document.getElementById("q").value);
}};
document.getElementById("exportCsv").onclick = exportCsv;
document.getElementById("exportJson").onclick = exportJson;
document.querySelectorAll("button[data-lab]").forEach(b => b.onclick = () => setLabel(b.dataset.lab));
document.getElementById("imp").onchange = (ev) => {{
  const f = ev.target.files[0];
  if (!f) return;
  const reader = new FileReader();
  reader.onload = () => {{
    let recs = [];
    try {{ recs = JSON.parse(reader.result); }}
    catch (e) {{
      recs = reader.result.trim().split(/\\n/).slice(1).map(line => {{
        const p = line.split(",");
        return {{item_id: p[0], label: p[1], cannot_judge: p[2], snippet: p.slice(3).join(",")}};
      }});
    }}
    const st = loadState();
    (Array.isArray(recs) ? recs : []).forEach(r => {{ if (r.item_id) st[r.item_id] = r; }});
    saveState(st);
    render();
  }};
  reader.readAsText(f);
}};
document.addEventListener("keydown", (e) => {{
  if (["INPUT","TEXTAREA"].includes(e.target.tagName)) return;
  if (e.key === "0" || e.key === "1" || e.key === "2") setLabel(e.key);
  if (e.key === "c" || e.key === "C") setLabel("C");
}});
document.getElementById("snip").onblur = persistText;
document.getElementById("notes").onblur = persistText;
render();
</script>
</body>
</html>
"""
