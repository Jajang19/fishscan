"""
history_manager.py
HistoryManager keeps a list of past scans in the USER'S OWN BROWSER
(localStorage), never on the Streamlit server:
  - the app is public, so scans from different people must not be mixed, and
  - files written on Streamlit Community Cloud are lost when the app restarts.

How it works (Streamlit cannot read browser storage back into Python, so the
browser does the work):
  1. After a scan, Python builds a small entry (thumbnail + results) and sends
     a tiny script that ADDS it to localStorage on this device (save_html).
  2. The History page is a small HTML/JavaScript view (page_html) that READS
     localStorage and draws the list, the detail view and the delete buttons.

Only the newest MAX_SCANS (20) scans are kept, so the data stays far below the
browser's storage limit (usually about 5 MB per site).

This file only DEFINES things. Nothing runs when it is imported.
"""

import base64
import io
import json

from PIL import Image, ImageOps

MAX_SCANS = 20
STORAGE_KEY = "fishscan_history_v1"   # name of the list in localStorage
THUMB_SIZE = 160                      # thumbnail size in pixels
THUMB_QUALITY = 60                    # JPEG quality (small file, still clear)


class HistoryManager:
    """Builds the browser-side code that saves, lists and deletes past scans."""

    def __init__(self, css_text):
        # The History page runs in its own small frame, so it needs our stylesheet too
        self._css = css_text

    # ------------------------------------------------------------------
    # 1. Making an entry
    # ------------------------------------------------------------------
    @staticmethod
    def make_thumbnail(image):
        """Small square JPEG of the photo as text (data URI), about 5-10 KB."""
        image = ImageOps.exif_transpose(image).convert("RGB")
        side = min(image.size)
        left, top = (image.width - side) // 2, (image.height - side) // 2
        image = image.crop((left, top, left + side, top + side)).resize((THUMB_SIZE, THUMB_SIZE))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=THUMB_QUALITY)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()

    def make_entry(self, scan_id, image, result, card_html):
        """Everything we remember about one scan (results only, no model data).

        status: "unrecognized" (fish-eye check failed), "low" (a top probability
        below the threshold) or "ok". card_html is the same result card the
        Scan page showed, so the detail view looks identical.
        """
        species, freshness = result["species"], result["freshness"]
        if not result["eye_check"]["recognized"]:
            status = "unrecognized"
        elif species["low_confidence"] or freshness["low_confidence"]:
            status = "low"
        else:
            status = "ok"
        return {
            "id": scan_id,
            "thumb": self.make_thumbnail(image),
            "species": species["top_class"],
            "species_p": round(species["top_probability"], 4),
            "freshness": freshness["top_class"],
            "freshness_p": round(freshness["top_probability"], 4),
            "status": status,
            "card": card_html,
        }

    @staticmethod
    def _as_js(value):
        """Python value -> JavaScript literal (safe to put inside a <script>)."""
        return json.dumps(value).replace("</", "<\\/")

    # ------------------------------------------------------------------
    # 2. Saving (runs in the browser, invisible)
    # ------------------------------------------------------------------
    def save_html(self, entry):
        """A tiny invisible script that adds `entry` to this device's history.

        Streamlit re-runs the page often, so the script may run more than once;
        it checks the scan id first, so the same scan is never saved twice.
        The date/time is taken from the device clock (the user's own time zone).
        """
        return f"""<script>
        (function () {{
          const KEY = {self._as_js(STORAGE_KEY)}, MAX = {MAX_SCANS};
          const entry = {self._as_js(entry)};
          let list = [];
          try {{ list = JSON.parse(localStorage.getItem(KEY)) || []; }} catch (e) {{ list = []; }}
          if (list.some(x => x.id === entry.id)) return;      // already saved
          entry.savedAt = new Date().toISOString();
          list.unshift(entry);                                  // newest first
          list = list.slice(0, MAX);                            // keep only 20
          while (list.length) {{                                // if storage is full,
            try {{ localStorage.setItem(KEY, JSON.stringify(list)); break; }}
            catch (e) {{ list.pop(); }}                         // drop the oldest and retry
          }}
        }})();
        </script>"""

    # ------------------------------------------------------------------
    # 3. The History page (list, detail, delete, clear all)
    # ------------------------------------------------------------------
    def page_html(self):
        """A complete small web page that shows this device's history."""
        return f"""<!doctype html><html><head><meta name="viewport" content="width=device-width">
<style>
{self._css}
/* ---- extra rules for the History view ---- */
:root {{ --fs-text: #0b2e2e; }}
@media (prefers-color-scheme: dark) {{ :root {{ --fs-text: #e6f4f1; }} }}
body {{ margin: 0; padding: 2px 8px 2px 2px; font-family: "Source Sans", system-ui, sans-serif;
        color: var(--fs-text); background: transparent; }}
.hist-card {{ background: var(--fs-card); border: 1px solid var(--fs-border); border-radius: 16px;
             box-shadow: var(--fs-shadow); padding: 0.9rem 1rem; margin-bottom: 0.75rem; }}
.hist-row {{ display: flex; align-items: center; gap: 0.75rem; }}
.hist-row img {{ width: 64px; height: 64px; border-radius: 12px; object-fit: cover; flex: none; }}
.hist-open {{ flex: 1; min-width: 0; background: none; border: none; text-align: left; padding: 0;
             color: inherit; font: inherit; cursor: pointer; min-height: 48px; }}
.hist-title {{ font-weight: 800; font-size: 1.05rem; }}
.hist-sub, .hist-date {{ color: var(--fs-muted); font-size: 0.85rem; }}
.chip {{ display: inline-block; font-size: 0.75rem; font-weight: 700; padding: 1px 8px;
         border-radius: 999px; margin-left: 4px; }}
.chip.low {{ background: var(--fs-amber-soft); color: var(--fs-amber); }}
.chip.unrecognized {{ background: var(--fs-red-soft); color: var(--fs-red); }}
.btn {{ min-height: 44px; min-width: 44px; padding: 0 0.9rem; border-radius: 12px; font: inherit;
        font-weight: 600; border: 1px solid var(--fs-border); background: var(--fs-card);
        color: inherit; cursor: pointer; }}
.btn.danger {{ color: var(--fs-red); border-color: var(--fs-red); }}
.btn.primary {{ background: var(--fs-primary); color: var(--fs-on-primary); border-color: var(--fs-primary); }}
.hist-top {{ display: flex; justify-content: space-between; align-items: center; gap: 0.5rem;
             flex-wrap: wrap; margin-bottom: 0.6rem; }}
.hist-empty {{ text-align: center; padding: 2rem 1rem; color: var(--fs-muted); }}
.hist-detail img.big {{ width: 100%; max-width: 260px; border-radius: 12px; display: block; margin: 0 auto 0.8rem; }}
</style></head><body>
<div id="app"></div>
<script>
// Use the same font as the rest of the app: copy Streamlit's "Source Sans"
// font definition from the main page (this frame is on the same site).
try {{
  for (const sheet of parent.document.styleSheets) {{
    for (const rule of sheet.cssRules) {{
      if (rule instanceof parent.CSSFontFaceRule && /Source Sans/.test(rule.style.fontFamily)) {{
        const src = rule.style.getPropertyValue("src").replace(/url\\("?([^")]+)"?\\)/g,
                    (m, url) => 'url("' + new URL(url, sheet.href || parent.location.href) + '")');
        const css = "@font-face{{font-family:" + rule.style.fontFamily + ";font-weight:" +
                    rule.style.fontWeight + ";font-style:" + rule.style.fontStyle + ";src:" + src + "}}";
        document.head.insertAdjacentHTML("beforeend", "<style>" + css + "</style>");
      }}
    }}
  }}
}} catch (e) {{ /* if this fails, the system font is used instead */ }}

const KEY = {self._as_js(STORAGE_KEY)};
let confirmClear = false;   // "Clear all" needs a second tap to confirm

// --- read / write this device's history ---
function load() {{ try {{ return JSON.parse(localStorage.getItem(KEY)) || []; }} catch (e) {{ return []; }} }}
function store(list) {{ localStorage.setItem(KEY, JSON.stringify(list)); }}
function remove(id) {{ store(load().filter(x => x.id !== id)); showList(); }}
function clearAll() {{
  if (!confirmClear) {{ confirmClear = true; showList(); return; }}
  localStorage.removeItem(KEY); confirmClear = false; showList();
}}
function when(iso) {{ return iso ? new Date(iso).toLocaleString() : ""; }}
function pct(p) {{ return Math.round(p * 100) + "%"; }}
function esc(s) {{ return String(s).replace(/[&<>"]/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}}[c])); }}

const NOTE = '<div class="fs-note">🔒 History is saved only on this device. ' +
             'It is not uploaded, and other people cannot see it.</div>';

// --- list view: newest first ---
function showList() {{
  const list = load();
  let html = '<div class="hist-top"><div class="fs-label">' + list.length + ' of 20 saved scans</div>';
  if (list.length) html += '<button class="btn danger" onclick="clearAll()">' +
      (confirmClear ? 'Tap again to clear all' : 'Clear all history') + '</button>';
  html += '</div>' + NOTE;
  if (!list.length) {{
    html += '<div class="hist-card hist-empty">No scans yet.<br>Your results will appear here ' +
            'after you tap <b>Analyze</b> on the Scan page.</div>';
  }}
  for (const x of list) {{
    const title = x.status === "unrecognized" ? "🚫 Not recognized" : esc(x.species) + " · " + pct(x.species_p);
    const sub = x.status === "unrecognized" ? "Not a fish-eye close-up" :
                esc(x.freshness) + " · " + pct(x.freshness_p);
    const chip = x.status === "low" ? '<span class="chip low">Low confidence</span>' :
                 x.status === "unrecognized" ? '<span class="chip unrecognized">Not recognized</span>' : "";
    html += '<div class="hist-card hist-row"><img src="' + x.thumb + '" alt="">' +
            '<button class="hist-open" onclick="showDetail(\\'' + x.id + '\\')">' +
            '<div class="hist-title">' + title + '</div><div class="hist-sub">' + sub + chip + '</div>' +
            '<div class="hist-date">' + when(x.savedAt) + '</div></button>' +
            '<button class="btn danger" aria-label="Delete this scan" onclick="remove(\\'' + x.id + '\\')">Delete</button></div>';
  }}
  document.getElementById("app").innerHTML = html;
}}

// --- detail view: the same result card the Scan page showed ---
function showDetail(id) {{
  const x = load().find(e => e.id === id);
  if (!x) return showList();
  document.getElementById("app").innerHTML =
    '<div class="hist-top"><button class="btn" onclick="showList()">← Back to history</button>' +
    '<button class="btn danger" onclick="remove(\\'' + x.id + '\\')">Delete</button></div>' +
    '<div class="hist-card hist-detail"><div class="hist-date" style="margin-bottom:.5rem">' +
    when(x.savedAt) + '</div><img class="big" src="' + x.thumb + '" alt="">' + x.card + '</div>';
  window.scrollTo(0, 0);
}}

showList();
</script></body></html>"""
