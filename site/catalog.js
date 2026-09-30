const grid = document.getElementById("experiment-grid");
const manifestUrl = document.body.dataset.manifest;
const routePrefix = document.body.dataset.routePrefix || "./";

function esc(v) {
  return String(v).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
}

async function main() {
  try {
    const response = await fetch(manifestUrl);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const items = await response.json();
    grid.innerHTML = items.map((item, index) => {
      const href = routePrefix + item.route;
      return `
        <a class="experiment-card" href="${esc(href)}">
          <div class="card-top">
            <span class="index">EXP ${String(index + 1).padStart(2, "0")}</span>
            <span class="status">${esc(item.status)}</span>
          </div>
          <h3>${esc(item.title)}</h3>
          <p>${esc(item.summary)}</p>
          <div class="facts">${item.facts.map(x => `<span>${esc(x)}</span>`).join("")}</div>
          <div class="tags">${item.tags.map(x => `<span>#${esc(x)}</span>`).join("")}</div>
          <div class="card-footer"><span>${esc(item.date)}</span><strong>Open experiment →</strong></div>
        </a>`;
    }).join("");
  } catch (err) {
    grid.innerHTML = `<div class="error">Could not load experiment manifest: ${esc(err.message)}</div>`;
  }
}
main();
