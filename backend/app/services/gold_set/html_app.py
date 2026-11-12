"""Offline single-file gold-set annotator (no external scripts)."""

from __future__ import annotations

import json
from typing import Any

_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>InterviewIQ gold-set annotator</title>
<style>
:root {
  --bg: #efe8dc;
  --card: #fffdf8;
  --ink: #1f1b16;
  --muted: #5c564c;
  --line: #d9d0c3;
  --accent: #8a4b08;
  --chip: #f3ebe0;
}
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; background: var(--bg); color: var(--ink); overflow: hidden; }
body {
  font-family: "Iowan Old Style", "Palatino Linotype", Palatino, "Times New Roman", serif;
  line-height: 1.45;
}
.shell {
  height: 100vh; height: 100dvh;
  max-width: 1440px; margin: 0 auto;
  padding: 10px 12px;
  display: grid;
  grid-template-rows: auto auto minmax(0, 1fr) auto;
  gap: 8px;
}
.topbar, .item-head, .panel, .label-dock {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 12px;
}
.topbar {
  display: flex; flex-wrap: wrap; gap: 8px 10px; align-items: center;
  padding: 8px 12px;
}
.brand { font-weight: 700; margin-right: 4px; }
.topbar label { font-family: system-ui, sans-serif; font-size: 13px; color: var(--muted); }
button, select, input, textarea { font-family: system-ui, sans-serif; font-size: 14px; }
button, select { padding: 6px 10px; border-radius: 8px; border: 1px solid var(--line); background: #fff; cursor: pointer; }
button:hover { border-color: var(--ink); }
.progress { margin-left: auto; font-family: system-ui, sans-serif; font-variant-numeric: tabular-nums; font-size: 13px; color: var(--muted); }
.item-head { padding: 10px 14px 12px; min-height: 0; }
.badge {
  display: none; font-family: system-ui, sans-serif; font-size: 11px;
  letter-spacing: 0.04em; text-transform: uppercase; font-weight: 700;
  color: var(--accent); margin-bottom: 4px;
}
.badge.is-on { display: inline-block; }
.meta { font-family: system-ui, sans-serif; font-size: 12px; color: var(--muted); margin: 0 0 4px; }
.req-kicker {
  font-family: system-ui, sans-serif; font-size: 11px; font-weight: 700;
  letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); margin: 0;
}
.req { font-size: 1.28rem; font-weight: 700; margin: 2px 0 0; line-height: 1.25; }
.workspace {
  display: grid; grid-template-columns: minmax(0, 1.45fr) minmax(240px, 0.95fr);
  gap: 8px; min-height: 0;
}
@media (max-width: 900px) {
  .workspace { grid-template-columns: 1fr; grid-template-rows: minmax(0, 1.25fr) minmax(0, 0.85fr); }
}
.panel { display: flex; flex-direction: column; min-height: 0; overflow: hidden; }
.panel-bar {
  display: flex; align-items: center; gap: 8px;
  padding: 8px 12px 0; flex-shrink: 0;
}
.panel-title {
  font-family: system-ui, sans-serif; font-size: 11px; font-weight: 700;
  letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted);
}
.search { flex: 1; margin: 0; padding: 6px 8px; border: 1px solid var(--line); border-radius: 8px; min-width: 0; }
.cv {
  flex: 1; min-height: 0; overflow: auto;
  padding: 8px 14px 16px;
  font-size: 15.5px; line-height: 1.55;
}
.cv p { margin: 0 0 0.65em; }
.cv h3 {
  font-family: system-ui, sans-serif; font-size: 12px; letter-spacing: 0.05em;
  text-transform: uppercase; color: var(--accent); margin: 1.15em 0 0.4em;
  border-bottom: 1px solid var(--line); padding-bottom: 3px;
}
.cv h3:first-child { margin-top: 0.2em; }
.cv .job { margin: 0.9em 0 0.15em; font-weight: 700; font-size: 1.02em; }
.cv .when { margin: 0 0 0.45em; font-family: system-ui, sans-serif; font-size: 12.5px; color: var(--muted); }
.cv .field-k { font-family: system-ui, sans-serif; font-size: 11px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted); margin: 0.85em 0 0.15em; }
.cv .field-v { margin: 0 0 0.55em; }
mark { background: #ffe08a; padding: 0 0.12em; }
.label-dock {
  padding: 8px 12px 10px;
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: 8px 14px;
  align-items: start;
}
.labels { display: flex; flex-wrap: wrap; gap: 6px; margin: 0; }
.label-btn { min-width: 0; text-align: left; }
.label-btn.is-on { background: #1f1b16; color: #fff; border-color: #1f1b16; }
.label-btn[data-lab="C"].is-on { background: var(--accent); border-color: var(--accent); }
.notes-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; min-width: 0; }
@media (max-width: 780px) {
  .label-dock { grid-template-columns: 1fr; }
  .notes-grid { grid-template-columns: 1fr; }
}
.notes-grid label { display: block; font-family: system-ui, sans-serif; font-size: 11px; color: var(--muted); margin-bottom: 3px; }
textarea {
  width: 100%; height: 40px; min-height: 40px; max-height: 120px;
  padding: 6px 8px; border: 1px solid var(--line); border-radius: 8px; resize: vertical;
}
kbd { border: 1px solid #888; padding: 0 4px; border-radius: 3px; font-size: 11px; }
</style>
</head>
<body>
<div class="shell">
  <div class="topbar">
    <span class="brand">Gold-set annotation</span>
    <label>Annotator
      <select id="who">__OPTIONS__</select>
    </label>
    <button type="button" id="prev">Previous</button>
    <button type="button" id="next">Next</button>
    <span class="progress" id="prog"></span>
    <button type="button" id="exportCsv">Export CSV</button>
    <button type="button" id="exportJson">Export JSON</button>
    <label>Import <input type="file" id="imp" accept=".json,.csv,application/json,text/csv"/></label>
  </div>
  <header class="item-head">
    <div class="badge" id="calflag"></div>
    <p class="meta" id="meta"></p>
    <p class="req-kicker">Requirement to judge against the whole CV</p>
    <h2 class="req" id="req"></h2>
  </header>
  <div class="workspace">
    <section class="panel">
      <div class="panel-bar">
        <div class="panel-title">Candidate CV</div>
        <input class="search" id="q" placeholder="Find in this CV" autocomplete="off"/>
      </div>
      <div class="cv" id="cv"></div>
    </section>
    <section class="panel">
      <div class="panel-bar">
        <div class="panel-title">Job description</div>
      </div>
      <div class="cv" id="jd"></div>
    </section>
  </div>
  <div class="label-dock">
    <div>
      <div class="panel-title">Your label</div>
      <div class="labels">
        <button type="button" class="label-btn" data-lab="0">0 — not evidenced</button>
        <button type="button" class="label-btn" data-lab="1">1 — partially met</button>
        <button type="button" class="label-btn" data-lab="2">2 — clearly met</button>
        <button type="button" class="label-btn" data-lab="C">C — cannot judge</button>
      </div>
    </div>
    <div class="notes-grid">
      <div>
        <label for="snip">Evidence snippet (optional)</label>
        <textarea id="snip" placeholder="Paste a short CV span"></textarea>
      </div>
      <div>
        <label for="notes">Notes (optional)</label>
        <textarea id="notes"></textarea>
      </div>
    </div>
  </div>
</div>
<script>
const DATA = """

_TAIL = r"""
;
const ITEMS = Object.fromEntries(DATA.items.map(x => [x.item_id, x]));
const HEADERS = [
  "Professional Summary","Career Overview","Career Objective","Objective","Summary","Highlights",
  "Core Qualifications","Qualifications","Work History","Work Experience",
  "Professional Experience","Experience","Education","Skills","Technical Skills",
  "Accomplishments","Achievements","Certifications","Projects","Publications",
  "Awards","Languages","Interests","References"
].sort((a, b) => b.length - a.length);
const JD_LABELS = [
  "Position Title","Organization","Department","Position Type","Salary","Total Compensation",
  "Time Away from Work","Must Haves","Must Have","Requirements","Qualifications",
  "Responsibilities","Preferred Qualifications","Location","Employment Type",
  "Job Requisition ID","Position Overview","About","Benefits","Immediate need"
].sort((a, b) => b.length - a.length);
function storeKey() { return "goldset_" + document.getElementById("who").value; }
function loadState() {
  try { return JSON.parse(localStorage.getItem(storeKey()) || "{}"); }
  catch (e) { return {}; }
}
function saveState(st) { localStorage.setItem(storeKey(), JSON.stringify(st)); }
let idx = 0;
function order() {
  const who = document.getElementById("who").value;
  return DATA.orders[who] || DATA.items.map(x => x.item_id);
}
function currentId() { return order()[idx]; }
function esc(s) {
  return s.replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
}
function escapeRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }
function tidySpaces(text) {
  return (text || "").replace(/\r\n/g, "\n").replace(/\t/g, " ").replace(/ +/g, " ").trim();
}
function formatResume(text) {
  let t = tidySpaces(text);
  const names = HEADERS.map(escapeRe).join("|");
  t = t.replace(new RegExp("(^|\\n)(" + names + ")(?=[A-Z(])", "g"), "$1$2\n\n");
  t = t.replace(new RegExp("([a-z0-9,.)'\"])(" + names + ")(?=[A-Z(])", "g"), "$1\n\n$2\n\n");
  t = t.replace(new RegExp("(^|\\n)\\s*(" + names + ")\\s*(?=\\n|$)", "gi"), "$1\n\n$2\n\n");
  t = t.replace(/,?\s*((?:0[1-9]|1[0-2])\/\d{4})-(Present|\d{2}\/\d{4}|\d{4})/g, "\n$1 – $2\n");
  t = t.replace(/(Present|\b(?:19|20)\d{2}\b)(?=[A-Z])/g, "$1\n");
  t = t.replace(/(\b(?:19|20)\d{2})-([A-Z][a-z])/g, "$1\n$2");
  t = t.replace(/([A-Z]{2}),(?=[A-Z])/g, "$1\n");
  t = t.replace(/,([A-Z]{2})(\n|$)/g, ", $1$2");
  t = t.replace(/([A-Za-z.])[–—]([A-Z])/g, "$1\n$2");
  t = t.replace(/:([A-Z])/g, ": $1");
  t = t.replace(/([A-Z]{2})(GPA:)/g, "$1\n$2");
  t = t.replace(/\bin((?:19|20)\d{2}|January|February|March|April|May|June|July|August|September|October|November|December)/g, "in $1");
  t = t.replace(/\b(Sr|Jr|Mr|Ms|Mrs|Dr|Inc|Ltd|Co|St)\./g, "$1\u00B7");
  t = t.replace(/\.([A-Z])/g, ".\n$1");
  t = t.replace(/\. +([A-Z])/g, ".\n$1");
  t = t.replace(/\u00B7/g, ".");
  t = t.replace(/\n{3,}/g, "\n\n");
  return t.trim();
}
function formatJob(text) {
  let t = tidySpaces(text);
  t = t.replace(/hours\s*week/gi, "hours/week");
  t = t.replace(/\$([\d,]+)(year|month|week)\b/gi, "$$$1/$2");
  const names = JD_LABELS.map(escapeRe).join("|");
  t = t.replace(new RegExp("\\s*#?\\s*(" + names + ")\\s*:", "gi"), "\n\n$1:\n");
  t = t.replace(/\.([A-Z])/g, ".\n$1");
  t = t.replace(/\. +([A-Z])/g, ".\n$1");
  t = t.replace(/\n{3,}/g, "\n\n");
  return t.trim();
}
function isHeaderLine(line) {
  return HEADERS.some(h => h.toLowerCase() === line.toLowerCase());
}
function isDateLine(line) {
  return /^(0[1-9]|1[0-2])\/\d{4}\s+[–-]\s+(Present|\d{2}\/\d{4}|\d{4})/.test(line);
}
function renderBodyLines(lines) {
  if (!lines.length) return "";
  const chunks = [];
  let i = 0;
  while (i < lines.length) {
    if (isDateLine(lines[i])) {
      const bits = [lines[i]];
      i += 1;
      if (lines[i] && lines[i].length < 90 && !lines[i].endsWith(".")) {
        bits.push(lines[i]);
        i += 1;
        if (lines[i] && /^[A-Za-z .'-]+,\s*[A-Z]{2}$/.test(lines[i])) {
          bits.push(lines[i]);
          i += 1;
        }
      }
      chunks.push("<p class='when'>" + esc(bits.join(" · ")) + "</p>");
      continue;
    }
    const para = [];
    while (i < lines.length && !isDateLine(lines[i])) {
      para.push(lines[i]);
      i += 1;
    }
    chunks.push("<p>" + esc(para.join("\n")).replace(/\n/g, "<br>") + "</p>");
  }
  return chunks.join("");
}
function renderResume(text, query) {
  const formatted = formatResume(text);
  const parts = formatted.split(/\n{2,}/);
  const html = parts.map(block => {
    const lines = block.split("\n").map(s => s.trim()).filter(Boolean);
    if (!lines.length) return "";
    if (isHeaderLine(lines[0])) {
      return "<h3>" + esc(lines[0]) + "</h3>" + renderBodyLines(lines.slice(1));
    }
    if (isDateLine(lines[0]) || (lines.length > 1 && isDateLine(lines[1]))) {
      const title = isDateLine(lines[0]) ? "" : lines[0];
      const rest = isDateLine(lines[0]) ? lines : lines.slice(1);
      return (title ? "<p class='job'>" + esc(title) + "</p>" : "") + renderBodyLines(rest);
    }
    return renderBodyLines(lines);
  }).join("");
  return applyMarks(html, query);
}
function renderJob(text) {
  const formatted = formatJob(text);
  const parts = formatted.split(/\n{2,}/);
  return parts.map(block => {
    const lines = block.split("\n").map(s => s.trim()).filter(Boolean);
    if (!lines.length) return "";
    const m = lines[0].match(/^([^:]{2,40}):\s*(.*)$/);
    if (m && JD_LABELS.some(l => l.toLowerCase() === m[1].toLowerCase())) {
      const rest = [m[2]].concat(lines.slice(1)).filter(Boolean).join("\n");
      return "<div class='field-k'>" + esc(m[1]) + "</div>" +
        (rest ? "<p class='field-v'>" + esc(rest).replace(/\n/g, "<br>") + "</p>" : "");
    }
    if (/:$/.test(lines[0]) && lines[0].length < 48) {
      const rest = lines.slice(1).join("\n");
      return "<div class='field-k'>" + esc(lines[0].replace(/:$/, "")) + "</div>" +
        (rest ? "<p class='field-v'>" + esc(rest).replace(/\n/g, "<br>") + "</p>" : "");
    }
    return "<p>" + esc(lines.join("\n")).replace(/\n/g, "<br>") + "</p>";
  }).join("");
}
function applyMarks(html, query) {
  if (!query) return html;
  const re = new RegExp(escapeRe(query), "gi");
  return html.replace(re, m => "<mark>" + m + "</mark>");
}
function currentLabel(rec) {
  if (rec.cannot_judge || rec.label === "C") return "C";
  return rec.label || "";
}
function paintLabels(lab) {
  document.querySelectorAll(".label-btn").forEach(b => {
    b.classList.toggle("is-on", b.dataset.lab === lab);
  });
}
function labeledCount() {
  const st = loadState();
  return order().filter(id => {
    const r = st[id] || {};
    return r.cannot_judge || r.label === "0" || r.label === "1" || r.label === "2" || r.label === "C";
  }).length;
}
function render() {
  const ids = order();
  if (idx >= ids.length) idx = ids.length - 1;
  if (idx < 0) idx = 0;
  const it = ITEMS[ids[idx]];
  const st = loadState();
  const rec = st[it.item_id] || {};
  const kind = it.is_calibration ? "Calibration" : "Main item";
  document.getElementById("prog").textContent =
    (idx + 1) + " / " + ids.length + " · " + labeledCount() + " labeled";
  const cal = document.getElementById("calflag");
  cal.textContent = it.is_calibration
    ? "Calibration item — discuss after all four annotators finish these ten"
    : "";
  cal.classList.toggle("is-on", !!it.is_calibration);
  document.getElementById("meta").textContent = kind + " · " + it.item_id;
  document.getElementById("req").textContent = it.requirement_text || "";
  document.getElementById("jd").innerHTML = renderJob(it.jd_text || "");
  document.getElementById("snip").value = rec.snippet || "";
  document.getElementById("notes").value = rec.notes || "";
  paintLabels(currentLabel(rec));
  highlight();
  document.getElementById("cv").scrollTop = 0;
  document.getElementById("jd").scrollTop = 0;
}
function highlight() {
  const it = ITEMS[currentId()];
  const q = document.getElementById("q").value.trim();
  const req = (it.requirement_text || "").trim();
  let html = renderResume(it.cv_text || "", req);
  if (q && q.toLowerCase() !== req.toLowerCase()) html = applyMarks(html, q);
  document.getElementById("cv").innerHTML = html;
}
function setLabel(lab) {
  const st = loadState();
  const id = currentId();
  st[id] = st[id] || {item_id: id};
  if (lab === "C") { st[id].cannot_judge = true; st[id].label = ""; }
  else { st[id].cannot_judge = false; st[id].label = lab; }
  st[id].snippet = document.getElementById("snip").value;
  st[id].notes = document.getElementById("notes").value;
  saveState(st);
  idx = Math.min(idx + 1, order().length - 1);
  render();
}
function persistText() {
  const st = loadState();
  const id = currentId();
  st[id] = st[id] || {item_id: id};
  st[id].snippet = document.getElementById("snip").value;
  st[id].notes = document.getElementById("notes").value;
  saveState(st);
}
function rows() {
  persistText();
  const st = loadState();
  return order().map(id => {
    const r = st[id] || {item_id: id};
    return {
      item_id: id,
      label: r.label ?? "",
      cannot_judge: r.cannot_judge ? "true" : "false",
      snippet: r.snippet || "",
      notes: r.notes || ""
    };
  });
}
function exportCsv() {
  const rs = rows();
  const header = "item_id,label,cannot_judge,snippet,notes";
  const body = rs.map(r => [r.item_id, r.label, r.cannot_judge,
    JSON.stringify(r.snippet||""), JSON.stringify(r.notes||"")].join(",")).join("\n");
  download(header+"\n"+body, document.getElementById("who").value + ".csv", "text/csv");
}
function exportJson() {
  download(JSON.stringify(rows(), null, 2), document.getElementById("who").value + ".json", "application/json");
}
function download(text, name, type) {
  const blob = new Blob([text], {type});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
}
document.getElementById("prev").onclick = () => { persistText(); idx--; render(); };
document.getElementById("next").onclick = () => { persistText(); idx++; render(); };
document.getElementById("who").onchange = () => { idx = 0; render(); };
document.getElementById("q").oninput = highlight;
document.getElementById("exportCsv").onclick = exportCsv;
document.getElementById("exportJson").onclick = exportJson;
document.querySelectorAll("button[data-lab]").forEach(b => b.onclick = () => setLabel(b.dataset.lab));
document.getElementById("imp").onchange = (ev) => {
  const f = ev.target.files[0];
  if (!f) return;
  const reader = new FileReader();
  reader.onload = () => {
    let recs = [];
    try { recs = JSON.parse(reader.result); }
    catch (e) {
      recs = reader.result.trim().split(/\n/).slice(1).map(line => {
        const p = line.split(",");
        return {item_id: p[0], label: p[1], cannot_judge: p[2], snippet: p.slice(3).join(",")};
      });
    }
    const st = loadState();
    (Array.isArray(recs) ? recs : []).forEach(r => { if (r.item_id) st[r.item_id] = r; });
    saveState(st);
    render();
  };
  reader.readAsText(f);
};
document.addEventListener("keydown", (e) => {
  if (["INPUT","TEXTAREA","SELECT"].includes(e.target.tagName)) return;
  if (e.key === "0" || e.key === "1" || e.key === "2") setLabel(e.key);
  if (e.key === "c" || e.key === "C") setLabel("C");
  if (e.key === "ArrowLeft") { persistText(); idx--; render(); }
  if (e.key === "ArrowRight") { persistText(); idx++; render(); }
});
document.getElementById("snip").onblur = persistText;
document.getElementById("notes").onblur = persistText;
render();
</script>
</body>
</html>
"""


def render_annotator_html(items: list[dict[str, Any]], orders: dict[str, list[str]]) -> str:
    payload = json.dumps({"items": items, "orders": orders}, ensure_ascii=False)
    option_html = "".join(
        f'<option value="{name}">{name.replace("annotator_", "")}</option>'
        for name in orders
    )
    return _HEAD.replace("__OPTIONS__", option_html) + payload + _TAIL
