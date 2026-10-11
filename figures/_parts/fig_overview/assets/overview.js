"use strict";

const plot = document.querySelector("#plot");
const statusNode = document.querySelector("#status");
const isoInput = document.querySelector("#iso");
const sampleInput = document.querySelector("#sample-select");
const definitions = {
  overview: [
    ["Ligand ball-and-stick", [6, 7], "#F5B27E"],
    ["Protein pocket ball-and-stick", [4, 5], "#8291E8"],
    ["Ligand occupancy", [0], "#F8D3B0"],
    ["Pocket occupancy", [1], "#B4BEF0"],
    ["Electron density", [2], "#B58FDB"],
    ["Voxel grid", [3], "#626262"],
  ],
  gradient: [["Density gradient |∇ρ|", [0], "#D9B7E4"], ["Voxel grid", [1], "#626262"]],
  masked: [
    ["Ligand occupancy", [1], "#F8D3B0"], ["Pocket occupancy", [0], "#B4BEF0"],
    ["Electron density", [2], "#B58FDB"], ["Masking blocks", [3, 4], "#aaa"],
    ["Voxel grid", [5], "#626262"],
  ],
};
const cache = {};
const states = {};
let config;
let currentView;
let busy = false;

function status(message, error = false) {
  statusNode.textContent = message;
  statusNode.classList.toggle("error", error);
}

function lock(value) {
  busy = value;
  document.querySelectorAll("button, input, select").forEach(el => { el.disabled = value; });
}

async function api(path, options) {
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `Request failed: ${response.status}`);
  return body;
}

function camera() {
  // Read the live WebGL camera, including a just-finished drag/scroll. Layout
  // camera values can lag behind the rendered scene until plotly_relayout.
  const scene = plot._fullLayout.scene;
  return structuredClone(scene._scene?.getCamera() ?? scene.camera);
}

function aspectratio() {
  const scene = plot._fullLayout.scene;
  return structuredClone(scene._scene?.glplot.getAspectratio() ?? scene.aspectratio);
}

function viewportLayout() {
  return {"scene.camera": camera(), "scene.aspectmode": "manual", "scene.aspectratio": aspectratio()};
}

async function updateLayers(changes, indices) {
  const keep = viewportLayout();
  const layout = {...keep};
  for (const axis of ["xaxis", "yaxis", "zaxis"]) {
    layout[`scene.${axis}.autorange`] = false;
    layout[`scene.${axis}.range`] = [-8, 8];
  }
  // Submit visibility/geometry and the camera in one update. Isosurface
  // changes may rebuild the WebGL scene, so explicitly restore afterwards too.
  await Plotly.update(plot, changes, layout, indices);
  await Plotly.relayout(plot, keep);
}

function remember() {
  if (!currentView) return;
  states[currentView] = {
    visible: plot.data.map(t => t.visible !== false && t.visible !== "legendonly"),
    iso: Number(isoInput.value),
  };
}

function applyPalette() {
  const p = config.palette;
  const colors = {
    overview: [p.ligand, p.pocket, p.ligand_occupancy, p.pocket_occupancy, p.density, p.grid],
    gradient: [p.gradient, p.grid],
    masked: [p.ligand_occupancy, p.pocket_occupancy, p.density, "#aaa", p.grid],
  };
  for (const view of Object.keys(definitions)) {
    definitions[view].forEach((definition, i) => { definition[2] = colors[view][i]; });
  }
}

function dataUrl(endpoint, view, extra = "") {
  return `/api/${endpoint}?sample=${config.sample}&view=${view}${extra}`;
}

function layerControls() {
  const container = document.querySelector("#layers");
  container.replaceChildren();
  const legend = document.querySelector("#element-legend");
  legend.replaceChildren();
  legend.hidden = currentView !== "overview";
  for (const [element, color] of Object.entries(config.ligand_elements)) {
    const item = document.createElement("span");
    const dot = document.createElement("i");
    dot.className = "dot"; dot.style.setProperty("--color", color);
    item.append(dot, document.createTextNode(element));
    legend.append(item);
  }
  for (const [label, indices, color] of definitions[currentView]) {
    const button = document.createElement("button");
    button.className = "layer";
    button.style.setProperty("--color", color);
    button.setAttribute("aria-label", label);
    const visible = plot.data[indices[0]].visible;
    button.setAttribute("aria-pressed", visible !== false && visible !== "legendonly");
    const dot = document.createElement("span"); dot.className = "dot";
    const title = document.createElement("span"); title.textContent = label;
    const toggle = document.createElement("span"); toggle.className = "switch";
    button.append(dot, title, toggle);
    button.addEventListener("click", async () => {
      if (busy) return;
      lock(true);
      const enabled = button.getAttribute("aria-pressed") !== "true";
      try {
        await updateLayers({visible: enabled}, indices);
        button.setAttribute("aria-pressed", String(enabled));
      } catch (error) { status(error.message, true); }
      finally { lock(false); }
    });
    container.append(button);
  }
}

async function switchView(view) {
  if (busy || currentView === view) return;
  lock(true);
  status("3D 화면을 준비하고 있어요…");
  try {
    // All three representations share the current camera and orthographic
    // zoom. Only layer visibility and thresholds belong to an individual tab.
    const sharedViewport = currentView ? viewportLayout() : null;
    remember();
    if (!cache[view]) cache[view] = await api(dataUrl("figure", view));
    const figure = structuredClone(cache[view]);
    const previous = states[view];
    if (previous) {
      figure.data.forEach((trace, index) => { trace.visible = previous.visible[index]; });
    }
    if (sharedViewport) {
      figure.layout.scene.camera = sharedViewport["scene.camera"];
      figure.layout.scene.aspectmode = "manual";
      figure.layout.scene.aspectratio = sharedViewport["scene.aspectratio"];
    }
    const iso = previous?.iso ?? (view === "gradient" ? 0.35 : 0.1);
    if (view === "gradient") {
      figure.data[0].isomin = iso; figure.data[0].isomax = iso + 0.01;
    } else if (iso !== 0.1) {
      const density = await api(dataUrl("density", view, `&iso=${iso}`));
      for (const axis of ["x", "y", "z"]) figure.data[2][axis] = density.data[0][axis];
    }
    delete figure.layout.width;
    delete figure.layout.height;
    figure.layout.autosize = true;
    figure.layout.uirevision = config.sample;
    figure.layout.scene.uirevision = config.sample;
    await Plotly.react(plot, figure.data, figure.layout, {
      responsive: true, displayModeBar: false, scrollZoom: true, displaylogo: false,
      plotGlPixelRatio: 2,
    });
    if (sharedViewport) await Plotly.relayout(plot, sharedViewport);
    currentView = view;
    isoInput.max = view === "gradient" ? "2" : String(config.density_max);
    isoInput.value = String(iso);
    document.querySelector("#iso-value").value = iso.toFixed(2);
    document.querySelector("#iso-label").textContent = view === "gradient" ? "Gradient iso" : "Density iso";
    document.querySelector("#name").value = `overview-${config.sample}-${view}`;
    document.querySelectorAll("[data-view]").forEach(button => {
      button.setAttribute("aria-pressed", String(button.dataset.view === view));
    });
    layerControls();
    status("준비됐어요. 레이어와 시점을 조절한 뒤 저장하세요.");
  } catch (error) { status(error.message, true); }
  finally { lock(false); }
}

isoInput.addEventListener("input", () => {
  document.querySelector("#iso-value").value = Number(isoInput.value).toFixed(2);
});
isoInput.addEventListener("change", async () => {
  if (busy) return;
  lock(true);
  const iso = Number(isoInput.value);
  try {
    if (currentView === "gradient") {
      await updateLayers({isomin: iso, isomax: iso + 0.01}, [0]);
    } else {
      status("Density를 업데이트하고 있어요…");
      const {data} = await api(dataUrl("density", currentView, `&iso=${iso}`));
      await updateLayers({x: [data[0].x], y: [data[0].y], z: [data[0].z]}, [2]);
    }
    status("Density threshold를 적용했어요.");
  } catch (error) { status(error.message, true); }
  finally { lock(false); }
});

document.querySelectorAll("[data-view]").forEach(button => {
  button.addEventListener("click", () => switchView(button.dataset.view));
});
sampleInput.addEventListener("change", async () => {
  if (busy) return;
  lock(true);
  status("새 protein–ligand pair를 불러오고 있어요…");
  try {
    const next = await api(`/api/config?sample=${encodeURIComponent(sampleInput.value)}`);
    config = next;
    for (const key of Object.keys(cache)) delete cache[key];
    for (const key of Object.keys(states)) delete states[key];
    currentView = null;
    applyPalette();
    document.querySelector("#sample").textContent = `${config.sample.toUpperCase()} · 16 Å / 64³`;
    lock(false);
    await switchView("overview");
  } catch (error) {
    sampleInput.value = config.sample;
    status(error.message, true);
    lock(false);
  }
});
document.querySelector("#reset").addEventListener("click", async () => {
  if (busy) return;
  lock(true);
  try { await Plotly.relayout(plot, {"scene.camera": structuredClone(config.camera),
    "scene.aspectmode": "manual", "scene.aspectratio": structuredClone(config.aspectratio)}); }
  catch (error) { status(error.message, true); }
  finally { lock(false); }
});
document.querySelector("#print-camera").addEventListener("click", () => {
  if (busy || !currentView) return;
  const current = camera();
  const delta = ["x", "y", "z"].map(axis => current.eye[axis] - current.center[axis]);
  const angles = {
    azimuth: Math.atan2(delta[1], delta[0]) * 180 / Math.PI,
    elevation: Math.atan2(delta[2], Math.hypot(delta[0], delta[1])) * 180 / Math.PI,
  };
  const snapshot = {
    sample: config.sample, view: currentView,
    angles_deg: angles,
    scene: {camera: current, aspectmode: "manual", aspectratio: aspectratio()},
  };
  document.querySelector("#camera-output").value = JSON.stringify(snapshot, null, 2);
  document.querySelector("#camera-summary").textContent =
    `출력 시점: ${config.sample.toUpperCase()} / ${currentView} · ` +
    `방위각(X→Y) ${angles.azimuth.toFixed(2)}° · 고도(XY 평면 기준) ${angles.elevation.toFixed(2)}°`;
  document.querySelector("#camera-report").hidden = false;
  status("아래에 현재 카메라 각도와 줌 설정을 출력했어요.");
});
document.querySelector("#copy-camera").addEventListener("click", async () => {
  const output = document.querySelector("#camera-output");
  try {
    await navigator.clipboard.writeText(output.value);
    status("카메라 설정을 복사했어요.");
  } catch (_) {
    output.focus(); output.select();
    status("설정을 선택했어요. Ctrl+C 또는 ⌘C로 복사하세요.");
  }
});
document.querySelector("#save").addEventListener("click", async () => {
  if (busy || !currentView) return;
  const name = document.querySelector("#name");
  if (!name.value || !name.reportValidity()) return;
  lock(true);
  status("현재 화면을 이미지로 저장하고 있어요…");
  try {
    // Freeze the current scene camera before Plotly creates the export scene.
    await Plotly.relayout(plot, viewportLayout());
    const image = await Plotly.toImage(plot, {format: "png", width: 760, height: 760, scale: 2});
    const result = await api("/api/save", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({name: name.value, format: document.querySelector("#format").value, image})});
    status("저장 완료: ");
    for (const file of result.files) {
      const link = document.createElement("a");
      link.href = file.url; link.textContent = file.path; link.target = "_blank"; link.rel = "noopener";
      statusNode.append(link, document.createElement("br"));
    }
  } catch (error) { status(`저장하지 못했어요: ${error.message}`, true); }
  finally { lock(false); }
});

new ResizeObserver(() => { if (plot._fullLayout && !busy) Plotly.Plots.resize(plot); }).observe(plot);

(async () => {
  try {
    config = await api("/api/config");
    applyPalette();
    for (const sample of config.samples) {
      const option = document.createElement("option");
      option.value = sample.id; option.textContent = sample.label;
      sampleInput.append(option);
    }
    sampleInput.value = config.sample;
    document.querySelector("#sample").textContent = `${config.sample.toUpperCase()} · 16 Å / 64³`;
    await switchView("overview");
  } catch (error) { status(`화면을 불러오지 못했어요: ${error.message}`, true); }
})();
