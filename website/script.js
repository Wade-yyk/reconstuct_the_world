"use strict";

const datasets = window.Q1_DATA;
const examples = { set1: 2, set2: 27 };
const state = { dataset: "set1", method: "normalized", index: examples.set1, allLines: false };
const datasetButtons = [...document.querySelectorAll("[data-dataset]")];
const methodButtons = [...document.querySelectorAll("[data-method]")];
const allLinesButton = document.querySelector("#all-lines");
const slider = document.querySelector("#match-slider");
const previousButton = document.querySelector("#previous-match");
const nextButton = document.querySelector("#next-match");
const matchOutput = document.querySelector("#match-output");
const resultRows = [...document.querySelectorAll("tbody tr")];
const views = [1, 2].map((number) => ({
  image: document.querySelector("#view-" + number),
  stage: document.querySelector("#stage-" + number),
  overlay: document.querySelector("#overlay-" + number),
  distance: document.querySelector("#point-distance-" + number),
}));

function lineInView(matrix, view, [x, y]) {
  if (view === 0) {
    return matrix[0].map((_, column) => (
      matrix[0][column] * x + matrix[1][column] * y + matrix[2][column]
    ));
  }
  return matrix.map(([a, b, c]) => a * x + b * y + c);
}

function projectToLine([x, y], [a, b, c]) {
  const length = Math.hypot(a, b);
  if (length === 0) return null;

  const normalX = a / length;
  const normalY = b / length;
  const signedDistance = (a * x + b * y + c) / length;
  return {
    x: x - signedDistance * normalX,
    y: y - signedDistance * normalY,
    normalX,
    normalY,
    distance: Math.abs(signedDistance),
  };
}

function shape(target, name, attributes) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  target.append(element);
}

function drawLine(target, foot, extent, style) {
  shape(target, "line", {
    x1: foot.x - foot.normalY * extent,
    y1: foot.y + foot.normalX * extent,
    x2: foot.x + foot.normalY * extent,
    y2: foot.y - foot.normalX * extent,
    class: style,
  });
}

function drawOverlay(target, dataset, matrix, view, index, allLines, allPoints) {
  const { width, height, points } = dataset.images[view];
  const otherPoints = dataset.images[1 - view].points;
  const extent = 2 * Math.hypot(width, height);
  target.replaceChildren();
  target.setAttribute("viewBox", [0, 0, width, height].join(" "));

  if (allLines) {
    points.forEach((point, match) => {
      if (match === index) return;
      const foot = projectToLine(point, lineInView(matrix, view, otherPoints[match]));
      if (foot) drawLine(target, foot, extent, "epipolar faint");
    });
  }

  const [x, y] = points[index];
  const foot = projectToLine([x, y], lineInView(matrix, view, otherPoints[index]));
  if (foot) drawLine(target, foot, extent, "epipolar");

  if (allPoints) {
    points.forEach(([cx, cy], match) => {
      if (match !== index) shape(target, "circle", { cx, cy, r: 3.3, class: "point faint" });
    });
  }

  if (foot && foot.distance > 4) {
    shape(target, "line", { x1: x, y1: y, x2: foot.x, y2: foot.y, class: "error-segment" });
    shape(target, "circle", { cx: foot.x, cy: foot.y, r: 3, class: "foot-point" });
  }
  shape(target, "circle", { cx: x, cy: y, r: 6, class: "point selected" });
  return foot?.distance;
}

function render() {
  const dataset = datasets[state.dataset];
  const { matrix } = dataset.methods[state.method];
  const count = dataset.images[0].points.length;
  const match = state.index + 1;

  datasetButtons.forEach((button) => button.setAttribute("aria-pressed", button.dataset.dataset === state.dataset));
  methodButtons.forEach((button) => button.setAttribute("aria-pressed", button.dataset.method === state.method));
  allLinesButton.setAttribute("aria-pressed", state.allLines);
  slider.max = count;
  slider.value = match;
  slider.setAttribute("aria-valuetext", "Match " + match + " of " + count);
  matchOutput.textContent = String(match).padStart(2, "0") + " / " + count;
  previousButton.disabled = match === 1;
  nextButton.disabled = match === count;

  views.forEach(({ image, overlay, distance }, view) => {
    const asset = dataset.images[view];
    if (image.getAttribute("src") !== asset.src) image.src = asset.src;
    const error = drawOverlay(overlay, dataset, matrix, view, state.index, state.allLines, true);
    distance.textContent = error === undefined ? "—" : error.toFixed(2) + " px";
  });

  const activeRow = state.dataset + "-" + state.method;
  resultRows.forEach((row) => row.classList.toggle("current", row.id === activeRow));
}

datasetButtons.forEach((button) => button.addEventListener("click", () => {
  if (state.dataset === button.dataset.dataset) return;
  state.dataset = button.dataset.dataset;
  state.index = examples[state.dataset];
  render();
}));

methodButtons.forEach((button) => button.addEventListener("click", () => {
  state.method = button.dataset.method;
  render();
}));

allLinesButton.addEventListener("click", () => {
  state.allLines = !state.allLines;
  render();
});

slider.addEventListener("input", () => {
  state.index = Number(slider.value) - 1;
  render();
});

previousButton.addEventListener("click", () => {
  state.index -= 1;
  render();
});

nextButton.addEventListener("click", () => {
  state.index += 1;
  render();
});

views.forEach(({ stage }, view) => stage.addEventListener("click", (event) => {
  const { points, width, height } = datasets[state.dataset].images[view];
  const bounds = stage.getBoundingClientRect();
  const x = (event.clientX - bounds.left) * width / bounds.width;
  const y = (event.clientY - bounds.top) * height / bounds.height;
  let minimum = Infinity;
  points.forEach(([px, py], match) => {
    const squaredDistance = (px - x) ** 2 + (py - y) ** 2;
    if (squaredDistance < minimum) {
      minimum = squaredDistance;
      state.index = match;
    }
  });
  render();
}));

[1, 2].forEach((number, view) => {
  const dataset = datasets.set1;
  const target = document.querySelector("#hero-overlay-" + number);
  drawOverlay(target, dataset, dataset.methods.normalized.matrix, view, examples.set1, false, false);
});

render();
