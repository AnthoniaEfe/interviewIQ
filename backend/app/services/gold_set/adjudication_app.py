"""Offline page for two adjudicators to agree the main-set disagreements.

The page embeds blind CV and JD text only. It does not embed system labels,
scores, split names, or fit labels.
"""

from __future__ import annotations

import json
from typing import Any


def render_adjudication_html(cases: list[dict[str, Any]], calibration: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        {"cases": [_public_case(row) for row in cases], "calibration": [_public_example(row) for row in calibration]},
        ensure_ascii=False,
    ).replace("<", "\\u003c")
    return _PAGE.replace("__DATA__", payload)


def _public_case(row: dict[str, Any]) -> dict[str, Any]:
    phrase = row.get("requirement_phrase_in_cv")
    return {
        "item_id": str(row["item_id"]),
        "requirement_text": str(row.get("requirement_text") or ""),
        "cv_text": str(row.get("cv_text") or ""),
        "jd_text": str(row.get("jd_text") or ""),
        "annotator_1": _short_name(row.get("annotator_1")),
        "label_1": _label_text(row.get("label_1")),
        "notes_1": str(row.get("notes_1") or ""),
        "annotator_2": _short_name(row.get("annotator_2")),
        "label_2": _label_text(row.get("label_2")),
        "notes_2": str(row.get("notes_2") or ""),
        "phrase_in_cv": bool(phrase) if isinstance(phrase, bool) else str(phrase).strip().lower() == "true",
    }


def _public_example(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "item_id": str(row["item_id"]),
        "requirement_text": str(row.get("requirement_text") or ""),
        "label": _label_text(row.get("label")),
    }


def _short_name(value: Any) -> str:
    text = str(value or "").strip()
    return text.removeprefix("annotator_") or text


def _label_text(value: Any) -> str:
    text = str(value or "").strip()
    if text.upper() == "C":
        return "C"
    if text in {"0", "1", "2"}:
        return text
    return ""


_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Gold-set adjudication</title>
<style>
:root {
  --bg: #efe8dc;
  --card: #fffdf8;
  --ink: #1f1b16;
  --muted: #5c564c;
  --line: #d9d0c3;
  --accent: #8a4b08;
}
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; background: var(--bg); color: var(--ink); overflow: hidden; }
body { font-family: "Iowan Old Style", Palatino, "Times New Roman", serif; line-height: 1.45; }
.shell {
  height: 100vh; height: 100dvh;
  max-width: 1440px; margin: 0 auto; padding: 8px 12px 10px;
  display: grid; grid-template-rows: auto auto minmax(0, 1fr) auto; gap: 8px;
}
.topbar, .item-head, .panel, .label-dock, .overlay-card {
  background: var(--card); border: 1px solid var(--line); border-radius: 12px;
}
.topbar, .item-head { padding: 8px 12px; }
.topbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.brand { font-weight: 700; }
button, select, input, textarea { font-family: system-ui, sans-serif; font-size: 14px; }
button, select { padding: 6px 10px; border-radius: 8px; border: 1px solid var(--line); background: #fff; cursor: pointer; }
button:hover { border-color: var(--ink); }
.name { width: 9.5rem; padding: 6px 8px; border: 1px solid var(--line); border-radius: 8px; }
.progress { margin-left: auto; font-family: system-ui, sans-serif; font-size: 13px; color: var(--muted); font-variant-numeric: tabular-nums; }
.rule { margin: 0; font-family: system-ui, sans-serif; font-size: 13px; color: var(--muted); }
.meta { margin: 0; font-family: system-ui, sans-serif; font-size: 12px; color: var(--muted); }
.req-kicker { margin: 4px 0 0; font-family: system-ui, sans-serif; font-size: 11px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }
.req { margin: 2px 0 6px; font-size: 1.25rem; line-height: 1.25; }
.priors { display: flex; flex-wrap: wrap; gap: 8px; margin: 0; }
.chip { font-family: system-ui, sans-serif; font-size: 13px; background: #f3ebe0; border-radius: 999px; padding: 4px 10px; }
.chip b { font-variant-numeric: tabular-nums; }
.prior-note { margin: 6px 0 0; font-family: system-ui, sans-serif; font-size: 13px; }
.phrase { margin: 6px 0 0; font-family: system-ui, sans-serif; font-size: 12px; color: var(--muted); }
.workspace { display: grid; grid-template-columns: minmax(0, 1.45fr) minmax(240px, 0.95fr); gap: 8px; min-height: 0; }
@media (max-width: 900px) {
  .workspace { grid-template-columns: 1fr; grid-template-rows: minmax(0, 1.2fr) minmax(140px, 0.7fr); }
}
.panel { display: flex; flex-direction: column; min-height: 0; overflow: hidden; }
.panel-bar { display: flex; align-items: center; gap: 8px; padding: 8px 12px 0; }
.panel-title { font-family: system-ui, sans-serif; font-size: 11px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }
.search { flex: 1; margin: 0; padding: 6px 8px; border: 1px solid var(--line); border-radius: 8px; min-width: 0; }
.cv { flex: 1; min-height: 0; overflow: auto; padding: 8px 14px 16px; font-size: 15.5px; }
.cv p { margin: 0 0 0.65em; }
.cv h3 { font-family: system-ui, sans-serif; font-size: 12px; letter-spacing: 0.05em; text-transform: uppercase; color: var(--accent); margin: 1.1em 0 0.35em; border-bottom: 1px solid var(--line); }
.cv .job { margin: 0.85em 0 0.15em; font-weight: 700; }
.cv .when { margin: 0 0 0.4em; font-family: system-ui, sans-serif; font-size: 12.5px; color: var(--muted); }
.cv .field-k { font-family: system-ui, sans-serif; font-size: 11px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted); margin: 0.8em 0 0.15em; }
mark { background: #ffe08a; }
.label-dock { padding: 8px 12px 10px; display: grid; grid-template-columns: 1fr 1fr minmax(180px, 0.8fr) minmax(180px, 0.8fr); gap: 10px; }
@media (max-width: 980px) { .label-dock { grid-template-columns: 1fr 1fr; } }
.seat-title { font-family: system-ui, sans-serif; font-size: 12px; font-weight: 700; margin-bottom: 6px; }
.labels { display: flex; flex-wrap: wrap; gap: 6px; }
.label-btn.is-on { background: #1f1b16; color: #fff; border-color: #1f1b16; }
.label-btn[data-lab="C"].is-on { background: var(--accent); border-color: var(--accent); }
.agree { min-height: 1.3em; margin: 6px 0 0; font-family: system-ui, sans-serif; font-size: 13px; }
.agree.is-yes { color: #1d6b3a; font-weight: 700; }
.agree.is-no { color: var(--accent); }
textarea { width: 100%; height: 52px; padding: 6px 8px; border: 1px solid var(--line); border-radius: 8px; resize: vertical; }
.notes-grid label { display: block; font-family: system-ui, sans-serif; font-size: 11px; color: var(--muted); margin-bottom: 3px; }
.overlay { position: fixed; inset: 0; background: rgba(31, 27, 22, 0.35); display: none; align-items: center; justify-content: center; padding: 20px; }
.overlay.is-on { display: flex; }
.overlay-card { max-width: 720px; width: 100%; max-height: 85vh; overflow: auto; padding: 16px 18px 18px; }
.overlay-card h2 { margin: 0 0 8px; font-size: 1.25rem; }
.overlay-card p, .overlay-card li { font-family: system-ui, sans-serif; font-size: 14px; }
.overlay-card table { width: 100%; border-collapse: collapse; font-family: system-ui, sans-serif; font-size: 13px; }
.overlay-card th, .overlay-card td { text-align: left; border-bottom: 1px solid var(--line); padding: 4px 6px; vertical-align: top; }
</style>
</head>
<body>
<div class="shell">
  <div class="topbar">
    <span class="brand">Main-set adjudication</span>
    <input class="name" id="name1" aria-label="First adjudicator" placeholder="Adjudicator 1"/>
    <input class="name" id="name2" aria-label="Second adjudicator" placeholder="Adjudicator 2"/>
    <button type="button" id="prev">Previous</button>
    <button type="button" id="next">Next</button>
    <select id="jump" aria-label="Jump to item"></select>
    <label style="font-family:system-ui,sans-serif;font-size:13px;color:var(--muted)"><input type="checkbox" id="unresolved"/> Unresolved only</label>
    <button type="button" id="ruleBtn">Rule and calibration key</button>
    <span class="progress" id="prog"></span>
    <button type="button" id="exportCsv">Export adjudicated.csv</button>
    <label>Import <input type="file" id="imp" accept=".csv,text/csv"/></label>
  </div>
  <header class="item-head">
    <p class="rule">2 only when the CV shows this requirement. Related experience is 1 at most. Neither is 0. The job description defines the requirement. It is not evidence in the CV.</p>
    <p class="meta" id="meta"></p>
    <p class="req-kicker">Requirement to judge against the whole CV</p>
    <h2 class="req" id="req"></h2>
    <div class="priors" id="priors"></div>
    <p class="prior-note" id="priorNotes"></p>
    <p class="phrase" id="phrase"></p>
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
      <div class="panel-bar"><div class="panel-title">Job description</div></div>
      <div class="cv" id="jd"></div>
    </section>
  </div>
  <div class="label-dock">
    <div>
      <div class="seat-title" id="seat1">Adjudicator 1</div>
      <div class="labels" data-seat="a"></div>
    </div>
    <div>
      <div class="seat-title" id="seat2">Adjudicator 2</div>
      <div class="labels" data-seat="b"></div>
    </div>
    <div class="notes-grid">
      <label for="snip">Evidence snippet from the CV</label>
      <textarea id="snip" placeholder="Paste a short span if the agreed label is 2 or 1"></textarea>
    </div>
    <div class="notes-grid">
      <label for="notes">Adjudication notes</label>
      <textarea id="notes"></textarea>
      <p class="agree" id="agree"></p>
    </div>
  </div>
</div>
<div class="overlay" id="overlay">
  <div class="overlay-card">
    <h2>Worked rule</h2>
    <p>A requirement is met (2) only when the CV shows that requirement: the skill, qualification, or experience itself, at about the level the requirement names. Related or adjacent experience is partial (1) at most. If the CV shows neither, the label is not evidenced (0). Use cannot judge only when the requirement is too vague to interpret or the CV text is unreadable.</p>
    <p>The ten calibration decisions below are the discussion key. They are not part of this file. This page records one agreed label per main-set disagreement. Export writes <code>adjudicated.csv</code> with columns item_id, label, cannot_judge, snippet, notes. A row is included only when both adjudicators have chosen the same label.</p>
    <table id="calTable"></table>
    <p><button type="button" id="closeRule">Close</button></p>
  </div>
</div>
<script>
const DATA = __DATA__;
const CASES = Object.fromEntries(DATA.cases.map(row => [row.item_id, row]));
const ORDER = DATA.cases.map(row => row.item_id);
const LABELS = [
  ["0", "0 not evidenced"],
  ["1", "1 partial"],
  ["2", "2 met"],
  ["C", "C cannot judge"]
];
const HEADERS = ["Professional Summary","Career Overview","Career Objective","Objective","Summary","Highlights","Core Qualifications","Qualifications","Work History","Work Experience","Professional Experience","Experience","Education","Skills","Technical Skills","Accomplishments","Achievements","Certifications","Projects","Publications","Awards","Languages","Interests","References"].sort((a, b) => b.length - a.length);
const JD_LABELS = ["Position Title","Organization","Department","Position Type","Salary","Total Compensation","Time Away from Work","Must Haves","Must Have","Requirements","Qualifications","Responsibilities","Preferred Qualifications","Location","Employment Type","Job Requisition ID","Position Overview","About","Benefits","Immediate need"].sort((a, b) => b.length - a.length);
const STORE = "goldset_adjudication_v1";
let idx = 0;
function loadState() {
  try { return JSON.parse(localStorage.getItem(STORE) || "{}"); }
  catch (e) { return {}; }
}
function saveState(st) { localStorage.setItem(STORE, JSON.stringify(st)); }
function blank() { return {names: ["", ""], items: {}}; }
function state() {
  const st = Object.assign(blank(), loadState());
  st.items = st.items || {};
  st.names = Array.isArray(st.names) ? st.names : ["", ""];
  return st;
}
function rec(st, id) {
  st.items[id] = Object.assign({a: "", b: "", snippet: "", notes: ""}, st.items[id] || {});
  return st.items[id];
}
function isAgreed(row) { return !!(row && row.a && row.a === row.b); }
function visibleIds() {
  if (!document.getElementById("unresolved").checked) return ORDER.slice();
  const st = state();
  const pending = ORDER.filter(id => !isAgreed(rec(st, id)));
  return pending.length ? pending : ORDER.slice();
}
function currentId() { return visibleIds()[idx]; }
function esc(s) { return String(s || "").replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }
function escapeRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }
function tidySpaces(text) { return (text || "").replace(/\r\n/g, "\n").replace(/\t/g, " ").replace(/ +/g, " ").trim(); }
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
function isHeaderLine(line) { return HEADERS.some(h => h.toLowerCase() === line.toLowerCase()); }
function isDateLine(line) { return /^(0[1-9]|1[0-2])\/\d{4}\s+[–-]\s+(Present|\d{2}\/\d{4}|\d{4})/.test(line); }
function renderBodyLines(lines) {
  if (!lines.length) return "";
  const chunks = [];
  let i = 0;
  while (i < lines.length) {
    if (isDateLine(lines[i])) {
      const bits = [lines[i]];
      i += 1;
      if (lines[i] && lines[i].length < 90 && !lines[i].endsWith(".")) {
        bits.push(lines[i]); i += 1;
        if (lines[i] && /^[A-Za-z .'-]+,\s*[A-Z]{2}$/.test(lines[i])) { bits.push(lines[i]); i += 1; }
      }
      chunks.push("<p class='when'>" + esc(bits.join(" · ")) + "</p>");
      continue;
    }
    const para = [];
    while (i < lines.length && !isDateLine(lines[i])) { para.push(lines[i]); i += 1; }
    chunks.push("<p>" + esc(para.join("\n")).replace(/\n/g, "<br>") + "</p>");
  }
  return chunks.join("");
}
function renderResume(text, query) {
  const parts = formatResume(text).split(/\n{2,}/);
  const html = parts.map(block => {
    const lines = block.split("\n").map(s => s.trim()).filter(Boolean);
    if (!lines.length) return "";
    if (isHeaderLine(lines[0])) return "<h3>" + esc(lines[0]) + "</h3>" + renderBodyLines(lines.slice(1));
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
  return formatJob(text).split(/\n{2,}/).map(block => {
    const lines = block.split("\n").map(s => s.trim()).filter(Boolean);
    if (!lines.length) return "";
    const m = lines[0].match(/^([^:]{2,40}):\s*(.*)$/);
    if (m && JD_LABELS.some(l => l.toLowerCase() === m[1].toLowerCase())) {
      const rest = [m[2]].concat(lines.slice(1)).filter(Boolean).join("\n");
      return "<div class='field-k'>" + esc(m[1]) + "</div>" + (rest ? "<p>" + esc(rest).replace(/\n/g, "<br>") + "</p>" : "");
    }
    return "<p>" + esc(lines.join("\n")).replace(/\n/g, "<br>") + "</p>";
  }).join("");
}
function applyMarks(html, query) {
  if (!query) return html;
  return html.replace(new RegExp(escapeRe(query), "gi"), m => "<mark>" + m + "</mark>");
}
function labelWord(lab) {
  return {0: "not evidenced", 1: "partial", 2: "met", C: "cannot judge"}[lab] || lab;
}
function seatName(which) {
  const typed = document.getElementById(which === "a" ? "name1" : "name2").value.trim();
  return typed || (which === "a" ? "Adjudicator 1" : "Adjudicator 2");
}
function persistNames() {
  const st = state();
  st.names = [document.getElementById("name1").value, document.getElementById("name2").value];
  saveState(st);
}
function persistText() {
  const st = state();
  const row = rec(st, currentId());
  row.snippet = document.getElementById("snip").value;
  row.notes = document.getElementById("notes").value;
  saveState(st);
}
function agreedCount() {
  const st = state();
  return ORDER.filter(id => isAgreed(rec(st, id))).length;
}
function paintSeats(row) {
  document.querySelectorAll(".label-btn").forEach(btn => {
    btn.classList.toggle("is-on", row[btn.dataset.seat] === btn.dataset.lab);
  });
  const agree = document.getElementById("agree");
  if (isAgreed(row)) {
    agree.textContent = "Agreed label " + row.a + " — " + labelWord(row.a);
    agree.className = "agree is-yes";
  } else if (row.a || row.b) {
    agree.textContent = "The two choices differ. This item stays out of adjudicated.csv until they match.";
    agree.className = "agree is-no";
  } else {
    agree.textContent = "Each adjudicator chooses a label. The row is saved when the two choices match.";
    agree.className = "agree";
  }
}
function fillJump() {
  const ids = visibleIds();
  const st = state();
  const jump = document.getElementById("jump");
  const current = ids[idx] || ids[0];
  jump.innerHTML = ids.map(id => {
    const mark = isAgreed(rec(st, id)) ? "agreed" : "open";
    return "<option value='" + id + "'" + (id === current ? " selected" : "") + ">" + id + " · " + mark + "</option>";
  }).join("");
}
function render() {
  const ids = visibleIds();
  if (idx >= ids.length) idx = ids.length - 1;
  if (idx < 0) idx = 0;
  const id = ids[idx];
  const item = CASES[id];
  const st = state();
  const row = rec(st, id);
  document.getElementById("prog").textContent = (idx + 1) + " / " + ids.length + " · " + agreedCount() + " / " + ORDER.length + " agreed";
  document.getElementById("meta").textContent = item.item_id + " · original pair " + item.annotator_1 + " and " + item.annotator_2;
  document.getElementById("req").textContent = item.requirement_text;
  document.getElementById("priors").innerHTML =
    "<span class='chip'>" + esc(item.annotator_1) + " labeled <b>" + esc(item.label_1) + "</b> " + esc(labelWord(item.label_1)) + "</span>" +
    "<span class='chip'>" + esc(item.annotator_2) + " labeled <b>" + esc(item.label_2) + "</b> " + esc(labelWord(item.label_2)) + "</span>";
  const notes = [];
  if (item.notes_1) notes.push(item.annotator_1 + ": " + item.notes_1);
  if (item.notes_2) notes.push(item.annotator_2 + ": " + item.notes_2);
  document.getElementById("priorNotes").textContent = notes.join(" ");
  document.getElementById("phrase").textContent = item.phrase_in_cv
    ? "The exact requirement string occurs in the CV. That check is a prompt, not the label."
    : "The exact requirement string does not occur in the CV. That check is a prompt, not the label.";
  document.getElementById("seat1").textContent = seatName("a");
  document.getElementById("seat2").textContent = seatName("b");
  document.getElementById("snip").value = row.snippet || "";
  document.getElementById("notes").value = row.notes || "";
  document.getElementById("jd").innerHTML = renderJob(item.jd_text || "");
  paintSeats(row);
  highlight();
  fillJump();
  document.getElementById("cv").scrollTop = 0;
  document.getElementById("jd").scrollTop = 0;
}
function highlight() {
  const item = CASES[currentId()];
  const q = document.getElementById("q").value.trim();
  let html = renderResume(item.cv_text || "", item.requirement_text || "");
  if (q && q.toLowerCase() !== (item.requirement_text || "").toLowerCase()) html = applyMarks(html, q);
  document.getElementById("cv").innerHTML = html;
}
function setLabel(seat, lab) {
  const st = state();
  const row = rec(st, currentId());
  row[seat] = lab;
  row.snippet = document.getElementById("snip").value;
  row.notes = document.getElementById("notes").value;
  saveState(st);
  render();
}
function csvField(value) {
  const text = String(value ?? "");
  if (/[",\n\r]/.test(text)) return '"' + text.replace(/"/g, '""') + '"';
  return text;
}
function exportCsv() {
  persistText();
  const st = state();
  const lines = ["item_id,label,cannot_judge,snippet,notes"];
  ORDER.forEach(id => {
    const row = rec(st, id);
    if (!isAgreed(row)) return;
    const cannot = row.a === "C";
    lines.push([id, cannot ? "" : row.a, cannot ? "true" : "false", row.snippet || "", row.notes || ""].map(csvField).join(","));
  });
  const n = lines.length - 1;
  if (!n) { document.getElementById("agree").textContent = "No agreed rows yet."; return; }
  const blob = new Blob([lines.join("\n") + "\n"], {type: "text/csv"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "adjudicated.csv";
  a.click();
}
function parseCsv(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i += 1; }
      else if (c === '"') quoted = false;
      else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n") { row.push(field); rows.push(row); row = []; field = ""; }
    else if (c !== "\r") field += c;
  }
  if (field.length || row.length) { row.push(field); rows.push(row); }
  return rows;
}
function buildLabelButtons() {
  document.querySelectorAll(".labels").forEach(box => {
    box.innerHTML = LABELS.map(([lab, text]) =>
      "<button type='button' class='label-btn' data-seat='" + box.dataset.seat + "' data-lab='" + lab + "'>" + text + "</button>"
    ).join("");
  });
}
function fillCalibration() {
  const rows = DATA.calibration.map(row =>
    "<tr><td>" + esc(row.item_id) + "</td><td>" + esc(row.requirement_text) + "</td><td>" + esc(row.label) + "</td></tr>"
  ).join("");
  document.getElementById("calTable").innerHTML = "<thead><tr><th>Item</th><th>Requirement</th><th>Agreed label</th></tr></thead><tbody>" + rows + "</tbody>";
}
buildLabelButtons();
fillCalibration();
const stored = state();
document.getElementById("name1").value = stored.names[0] || "";
document.getElementById("name2").value = stored.names[1] || "";
document.getElementById("prev").onclick = () => { persistText(); idx -= 1; render(); };
document.getElementById("next").onclick = () => { persistText(); idx += 1; render(); };
document.getElementById("jump").onchange = (ev) => {
  persistText();
  idx = Math.max(0, visibleIds().indexOf(ev.target.value));
  render();
};
document.getElementById("unresolved").onchange = () => { idx = 0; render(); };
document.getElementById("q").oninput = highlight;
document.getElementById("exportCsv").onclick = exportCsv;
document.getElementById("name1").onchange = () => { persistNames(); render(); };
document.getElementById("name2").onchange = () => { persistNames(); render(); };
document.getElementById("snip").onblur = persistText;
document.getElementById("notes").onblur = persistText;
document.getElementById("ruleBtn").onclick = () => document.getElementById("overlay").classList.add("is-on");
document.getElementById("closeRule").onclick = () => document.getElementById("overlay").classList.remove("is-on");
document.querySelectorAll(".label-btn").forEach(btn => {
  btn.onclick = () => setLabel(btn.dataset.seat, btn.dataset.lab);
});
document.getElementById("imp").onchange = (ev) => {
  const file = ev.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    const table = parseCsv(String(reader.result || ""));
    const header = (table.shift() || []).map(cell => cell.trim());
    const col = name => header.indexOf(name);
    const st = state();
    table.forEach(cells => {
      if (!cells.length || !cells[0]) return;
      const id = cells[col("item_id")] || cells[0];
      if (!CASES[id]) return;
      const label = (cells[col("label")] || "").trim();
      const cannot = String(cells[col("cannot_judge")] || "").trim().toLowerCase() === "true";
      const lab = cannot || label.toUpperCase() === "C" ? "C" : label;
      const row = rec(st, id);
      if (lab === "0" || lab === "1" || lab === "2" || lab === "C") { row.a = lab; row.b = lab; }
      row.snippet = cells[col("snippet")] || "";
      row.notes = cells[col("notes")] || "";
    });
    saveState(st);
    render();
  };
  reader.readAsText(file);
};
render();
</script>
</body>
</html>
"""
