const $ = id => document.getElementById(id);
const EVENT_LABELS = {
  no_event: "no event",
  vehicle_cut_in: "vehicle cut-in",
  lead_vehicle_hard_brake: "lead hard brake",
  lead_vehicle_accelerate: "lead accelerate",
  lead_vehicle_exit: "lead exits lane",
};
const ACTION_LABELS = {
  hard_brake: "hard brake",
  brake: "brake",
  keep_speed: "keep speed",
  accelerate: "accelerate",
  hard_accelerate: "hard accelerate",
};
const MODEL = {
  jev: { prefix: "jev" },
  dashscope: { prefix: "dash" },
};

let data;
let pairIndex = 0;
let step = 0;
let timer = null;
let playing = false;

async function init() {
  try {
    const res = await fetch("./data/probability-world.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    data = await res.json();

    $("run-link").href = data.meta.run_url;
    $("step-max").textContent = data.meta.steps_per_worldline - 1;
    $("step-range").max = data.meta.steps_per_worldline - 1;

    populateWorldSelect();
    renderOverview();
    bindControls();

    const requested = new URLSearchParams(location.search).get("world");
    if (requested !== null) {
      const idx = data.pairs.findIndex(p => String(p.id) === String(requested));
      if (idx >= 0) pairIndex = idx;
    }

    $("world-select").value = String(data.pairs[pairIndex].id);
    $("embed-note").textContent =
      `${data.meta.worldlines_embedded} representative pairs embedded · aggregate cards use all ${data.meta.worldlines_total_per_model.toLocaleString()} per model`;

    $("loading").hidden = true;
    $("app").hidden = false;
    renderAll();
  } catch (err) {
    $("loading").textContent = `Could not load demo data: ${err.message}`;
  }
}

function populateWorldSelect() {
  for (const p of data.pairs) {
    const opt = document.createElement("option");
    opt.value = p.id;
    opt.textContent = `#${p.id} — ${p.tags.slice(0, 2).join(" · ")}`;
    $("world-select").append(opt);
  }
}

function renderOverview() {
  const j = data.summary.jev;
  const d = data.summary.dashscope;
  const metrics = [
    [
      "Mean external events / world",
      j.event_process.mean_events_per_worldline,
      d.event_process.mean_events_per_worldline,
      v => v.toFixed(2),
    ],
    ["Near-miss rate", j.outcomes.near_miss_rate, d.outcomes.near_miss_rate, pct],
    [
      "Mean minimum gap",
      j.outcomes.min_gap_m.mean,
      d.outcomes.min_gap_m.mean,
      v => `${v.toFixed(1)} m`,
    ],
    [
      "Mean final speed",
      j.outcomes.final_speed_mph.mean,
      d.outcomes.final_speed_mph.mean,
      v => `${v.toFixed(1)} mph`,
    ],
  ];

  $("overview-grid").innerHTML = metrics
    .map(
      ([name, jv, dv, fmt]) => `
      <article class="metric-card panel">
        <div class="metric-name">${name}</div>
        <div class="metric-values">
          <div class="metric-value jev"><span>JEV</span><strong>${fmt(jv)}</strong></div>
          <div class="metric-value dash"><span>DASHSCOPE</span><strong>${fmt(dv)}</strong></div>
        </div>
      </article>`,
    )
    .join("");
}

function bindControls() {
  $("world-select").addEventListener("change", e => selectWorldById(e.target.value));
  $("random-world").addEventListener("click", () => {
    let next = pairIndex;
    while (next === pairIndex && data.pairs.length > 1) {
      next = Math.floor(Math.random() * data.pairs.length);
    }
    pairIndex = next;
    step = 0;
    stop();
    syncWorldUrl();
    renderAll();
  });

  $("play-toggle").addEventListener("click", () => (playing ? stop() : play()));
  $("prev-step").addEventListener("click", () => {
    stop();
    seek(step - 1);
  });
  $("next-step").addEventListener("click", () => {
    stop();
    seek(step + 1);
  });
  $("step-range").addEventListener("input", e => {
    stop();
    seek(Number(e.target.value));
  });
  $("speed-select").addEventListener("change", () => {
    if (playing) {
      stop();
      play();
    }
  });

  addEventListener("keydown", e => {
    if (["INPUT", "SELECT"].includes(document.activeElement?.tagName)) return;
    if (e.code === "Space") {
      e.preventDefault();
      playing ? stop() : play();
    } else if (e.code === "ArrowRight") {
      stop();
      seek(step + 1);
    } else if (e.code === "ArrowLeft") {
      stop();
      seek(step - 1);
    }
  });
}

function selectWorldById(id) {
  const idx = data.pairs.findIndex(p => String(p.id) === String(id));
  if (idx < 0) return;
  pairIndex = idx;
  step = 0;
  stop();
  syncWorldUrl();
  renderAll();
}

function currentPair() {
  return data.pairs[pairIndex];
}

function syncWorldUrl() {
  const pair = currentPair();
  const u = new URL(location.href);
  u.searchParams.set("world", pair.id);
  history.replaceState(null, "", u);
  $("world-select").value = String(pair.id);
}

function seek(n) {
  const max = data.meta.steps_per_worldline - 1;
  step = Math.max(0, Math.min(max, n));
  renderStep();
}

function play() {
  if (step >= data.meta.steps_per_worldline - 1) step = 0;
  playing = true;
  $("play-toggle").textContent = "Pause";
  setRoadPlaying(true);
  renderStep();

  timer = setInterval(() => {
    if (step >= data.meta.steps_per_worldline - 1) {
      stop();
      return;
    }
    step += 1;
    renderStep();
  }, Number($("speed-select").value));
}

function stop() {
  playing = false;
  $("play-toggle").textContent = "Play";
  setRoadPlaying(false);
  if (timer) clearInterval(timer);
  timer = null;
}

function setRoadPlaying(on) {
  for (const id of ["jev-road", "dash-road"]) $(id).classList.toggle("playing", on);
}

function renderAll() {
  syncWorldUrl();
  const pair = currentPair();
  $("world-id-label").textContent = `#${pair.id}`;
  $("world-tags").innerHTML = pair.tags
    .map(t => `<span class="tag">${escapeHtml(t)}</span>`)
    .join("");
  $("jev-model-id").textContent = pair.jev.reported_model || "";
  $("dash-model-id").textContent = pair.dashscope.reported_model || "";

  renderTimeline();
  renderStep();
}

function renderStep() {
  $("step-number").textContent = step;
  $("step-range").value = step;

  const pair = currentPair();
  const jev = pair.jev.steps[step];
  const dash = pair.dashscope.steps[step];

  renderModel("jev", jev);
  renderModel("dashscope", dash);

  document
    .querySelectorAll(".timeline-step")
    .forEach((el, i) => el.classList.toggle("active", i === step));

  renderCharts();

  const delta = Math.abs(jev.speed_after_mph - dash.speed_after_mph);
  $("divergence-now").textContent =
    `Δspeed ${delta.toFixed(1)} mph · event ${jev.sampled_event === dash.sampled_event ? "same" : "split"} · action ${jev.sampled_action === dash.sampled_action ? "same" : "split"}`;
}

function renderModel(key, row) {
  const pre = MODEL[key].prefix;
  $(`${pre}-speed`).textContent = `${row.speed_after_mph.toFixed(1)} mph`;
  $(`${pre}-gap`).textContent =
    row.front_gap_after_m == null ? "clear" : `${row.front_gap_after_m.toFixed(1)} m`;
  $(`${pre}-event`).textContent = labelEvent(row.sampled_event);
  $(`${pre}-action`).textContent = labelAction(row.sampled_action);
  $(`${pre}-event-draw`).textContent =
    `u=${row.event_draw.toFixed(3)} · ${row.sampled_event === row.event_argmax ? "argmax" : "sampled tail"}`;
  $(`${pre}-action-draw`).textContent =
    `u=${row.actor_draw.toFixed(3)} · ${row.sampled_action === row.actor_argmax ? "argmax" : "sampled tail"}`;

  renderBars(`${pre}-event-bars`, row.event_probabilities, row.sampled_event, "event");
  renderBars(`${pre}-action-bars`, row.actor_probabilities, row.sampled_action, "action");
  renderRoad(pre, row);
  renderBadges(`${pre}-risk-badges`, row);
}

function renderBars(id, probs, sampled, kind) {
  const label = kind === "event" ? labelEvent : labelAction;
  $(id).innerHTML = Object.entries(probs)
    .map(
      ([name, val]) => `
      <div class="prob-row ${name === sampled ? "sampled" : ""}">
        <div class="name" title="${name}">${label(name)}${name === sampled ? " ←" : ""}</div>
        <div class="prob-track"><div class="prob-fill" style="width:${Math.max(0.5, val * 100)}%"></div></div>
        <div class="prob-value">${pct(val)}</div>
      </div>`,
    )
    .join("");
}

function renderRoad(pre, row) {
  const lead = $(`${pre}-lead`);
  const clear = $(`${pre}-clear`);

  if (row.front_gap_after_m == null) {
    lead.style.opacity = "0";
    clear.style.opacity = "1";
    return;
  }

  lead.style.opacity = "1";
  clear.style.opacity = "0";
  const left = 27 + Math.min(55, Math.max(0, (row.front_gap_after_m / 50) * 55));
  lead.style.left = `${left}%`;
}

function renderBadges(id, row) {
  let html;
  if (row.collision_floor_hit) html = '<span class="badge danger">1m FLOOR</span>';
  else if (row.critical_gap) html = '<span class="badge danger">CRITICAL &lt;3m</span>';
  else if (row.near_miss) html = '<span class="badge warn">NEAR MISS &lt;7m</span>';
  else html = '<span class="badge safe">NORMAL</span>';
  $(id).innerHTML = html;
}

function renderCharts() {
  const pair = currentPair();
  renderLineChart(
    $("speed-chart"),
    [
      { cls: "jev", values: pair.jev.steps.map(s => s.speed_after_mph) },
      { cls: "dash", values: pair.dashscope.steps.map(s => s.speed_after_mph) },
    ],
    { min: 0, label: "mph" },
  );

  const gaps = [
    ...pair.jev.steps.map(s => s.front_gap_after_m),
    ...pair.dashscope.steps.map(s => s.front_gap_after_m),
  ].filter(v => v != null);

  renderLineChart(
    $("gap-chart"),
    [
      { cls: "jev", values: pair.jev.steps.map(s => s.front_gap_after_m) },
      { cls: "dash", values: pair.dashscope.steps.map(s => s.front_gap_after_m) },
    ],
    { min: 0, max: Math.max(30, ...gaps), label: "m", risk: 7 },
  );
}

function renderLineChart(svg, series, opts = {}) {
  const W = 760;
  const H = 240;
  const pad = { l: 42, r: 16, t: 16, b: 28 };
  const all = series.flatMap(s => s.values.filter(v => v != null));
  const min = opts.min ?? Math.min(...all, 0);
  const max = opts.max ?? Math.max(...all, 1);
  const safeMax = max === min ? min + 1 : max;
  const n = Math.max(...series.map(s => s.values.length));

  const x = i => pad.l + (W - pad.l - pad.r) * (i / (n - 1));
  const y = v => pad.t + (H - pad.t - pad.b) * (1 - (v - min) / (safeMax - min));

  const grid = [];
  for (let i = 0; i < 5; i++) {
    const val = min + (safeMax - min) * (i / 4);
    const yy = y(val);
    grid.push(
      `<line class="chart-grid-line" x1="${pad.l}" y1="${yy}" x2="${W - pad.r}" y2="${yy}"/>`,
      `<text class="chart-axis-label" x="3" y="${yy + 3}">${val.toFixed(val < 10 ? 1 : 0)}${opts.label ? " " + opts.label : ""}</text>`,
    );
  }

  for (let i = 0; i < n; i += 3) {
    grid.push(`<text class="chart-axis-label" x="${x(i) - 3}" y="${H - 6}">${i}</text>`);
  }

  const paths = series
    .map(s => {
      let d = "";
      let open = false;
      s.values.forEach((v, i) => {
        if (v == null) {
          open = false;
          return;
        }
        d += `${open ? " L" : " M"} ${x(i).toFixed(1)} ${y(v).toFixed(1)}`;
        open = true;
      });
      const point = s.values[step];
      const fill = s.cls === "jev" ? "var(--jev)" : "var(--dash)";
      return (
        `<path class="chart-line ${s.cls}" d="${d}"/>` +
        (point == null
          ? ""
          : `<circle class="chart-point" cx="${x(step)}" cy="${y(point)}" r="5" fill="${fill}"/>`)
      );
    })
    .join("");

  const risk =
    opts.risk != null && opts.risk >= min && opts.risk <= safeMax
      ? `<line class="chart-risk" x1="${pad.l}" y1="${y(opts.risk)}" x2="${W - pad.r}" y2="${y(opts.risk)}"/>`
      : "";

  svg.innerHTML =
    grid.join("") +
    risk +
    paths +
    `<line class="chart-progress" x1="${x(step)}" y1="${pad.t}" x2="${x(step)}" y2="${H - pad.b}"/>`;
}

function renderTimeline() {
  const pair = currentPair();
  $("timeline").innerHTML = pair.jev.steps
    .map((jev, i) => {
      const dash = pair.dashscope.steps[i];
      return `
        <button class="timeline-step" data-step="${i}">
          <div class="t">t+${i}s</div>
          <div class="timeline-model">
            <span class="timeline-dot jev"></span>
            <span class="timeline-code" title="${labelEvent(jev.sampled_event)} / ${labelAction(jev.sampled_action)}">${short(jev.sampled_event)} · ${short(jev.sampled_action)}</span>
          </div>
          <div class="timeline-model">
            <span class="timeline-dot dash"></span>
            <span class="timeline-code" title="${labelEvent(dash.sampled_event)} / ${labelAction(dash.sampled_action)}">${short(dash.sampled_event)} · ${short(dash.sampled_action)}</span>
          </div>
        </button>`;
    })
    .join("");

  document.querySelectorAll(".timeline-step").forEach(el =>
    el.addEventListener("click", () => {
      stop();
      seek(Number(el.dataset.step));
    }),
  );
}

function pct(v) {
  return `${(v * 100).toFixed(v < 0.1 ? 1 : 0)}%`;
}
function labelEvent(k) {
  return EVENT_LABELS[k] || k.replaceAll("_", " ");
}
function labelAction(k) {
  return ACTION_LABELS[k] || k.replaceAll("_", " ");
}
function short(k) {
  return k
    .replace("lead_vehicle_", "lead:")
    .replace("vehicle_", "")
    .replace("hard_accelerate", "hard+")
    .replace("hard_brake", "hard-")
    .replace("keep_speed", "keep")
    .replace("accelerate", "accel")
    .replace("no_event", "none");
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, c => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
  })[c]);
}

init();
