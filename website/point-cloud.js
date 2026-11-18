"use strict";

[
  {
    id: "q4",
    encoded: window.q4PointCloud,
    encodedColors: window.q4PointCloudColors,
    // Face the same way as the upright 4(f) plot.
    axisSigns: [-1, -1, 1],
    unavailableMessage: "The interactive point cloud is unavailable. See the static figure in 4(f).",
  },
  {
    id: "shouban",
    encoded: window.shoubanPointCloud,
    encodedColors: window.shoubanPointCloudColors,
    view: window.shoubanPointCloudView,
    mesh: window.shoubanMesh,
  },
].forEach(({
  id,
  encoded,
  encodedColors,
  axisSigns = [1, 1, 1],
  view = {},
  mesh,
  unavailableMessage = "The interactive point cloud is unavailable.",
}) => {
  const canvas = document.querySelector(`#${id}-point-cloud-canvas`);
  if (!canvas) return;

  const zoomIn = document.querySelector(`#${id}-cloud-zoom-in`);
  const zoomOut = document.querySelector(`#${id}-cloud-zoom-out`);
  const reset = document.querySelector(`#${id}-cloud-reset`);
  const status = document.querySelector(`#${id}-cloud-status`);
  const pointCount = document.querySelector(`#${id}-cloud-point-count`);
  const context = canvas.getContext("2d");

  function decodeBase64(encoded) {
    const binary = atob(encoded);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes;
  }

  let points, fitRadius, radius;
  try {
    if (!context || typeof encoded !== "string") throw new Error();

    const bytes = decodeBase64(encoded);
    if (bytes.length < 12 || bytes.length % 12 !== 0) throw new Error();
    const source = new DataView(bytes.buffer);
    points = new Float32Array(bytes.length / 4);
    const center = [0, 0, 0];
    for (let i = 0; i < points.length; i++) {
      const value = source.getFloat32(4 * i, true);
      if (!Number.isFinite(value)) throw new Error();
      points[i] = value;
      center[i % 3] += value;
    }

    const distances = new Float32Array(points.length / 3);
    for (let i = 0; i < 3; i++) center[i] /= distances.length;
    for (let i = 0; i < points.length; i += 3) {
      points[i] = axisSigns[0] * (points[i] - center[0]);
      points[i + 1] = axisSigns[1] * (points[i + 1] - center[1]);
      points[i + 2] = axisSigns[2] * (points[i + 2] - center[2]);
      distances[i / 3] = Math.hypot(points[i], points[i + 1], points[i + 2]);
    }
    distances.sort();
    radius = distances[distances.length - 1] || 1;
    // A few distant matches should not determine the initial framing.
    fitRadius = distances[Math.floor((distances.length - 1) * 0.99)] || radius;
  } catch {
    status.textContent = unavailableMessage;
    for (const button of [zoomIn, zoomOut, reset]) button.disabled = true;
    canvas.tabIndex = -1;
    return;
  }
  const pointCountLabel = `${(points.length / 3).toLocaleString("en-US")} POINTS`;
  if (pointCount) pointCount.textContent = pointCountLabel;

  let colors = null;
  try {
    if (typeof encodedColors === "string") {
      const rgb = decodeBase64(encodedColors);
      if (rgb.length === points.length) colors = rgb;
    }
  } catch {
    // A missing or invalid color asset leaves the depth shading available.
  }

  const { pointRadius = 1.05 } = view;
  const initialYaw = mesh?.view?.yaw ?? mesh?.yaw ?? view.yaw ?? 0;
  const initialPitch = mesh?.view?.pitch ?? mesh?.pitch ?? view.pitch ?? -Math.PI / 18;
  const minZoom = 0.45;
  const maxZoom = 3.5;
  let pitch = initialPitch;
  let yaw = initialYaw;
  let zoom = 1;
  let activePointer = null;
  let scheduled = false;
  let frame, depthBuffer;
  const drawMesh = mesh ? window.createFigurineMeshView?.(mesh, requestDraw) : null;

  function draw() {
    const { width: cssWidth, height: cssHeight } = canvas.getBoundingClientRect();
    if (!cssWidth || !cssHeight) return;

    const resolution = Math.min(window.devicePixelRatio || 1, 1.5, 1200 / Math.max(cssWidth, cssHeight));
    const width = Math.max(1, Math.round(cssWidth * resolution));
    const height = Math.max(1, Math.round(cssHeight * resolution));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const meshImage = drawMesh?.(width, height, yaw, pitch, zoom);
    if (meshImage) {
      if (pointCount && pointCount.textContent !== "TEXTURED SURFACE") pointCount.textContent = "TEXTURED SURFACE";
      context.clearRect(0, 0, width, height);
      context.drawImage(meshImage, 0, 0);
      return;
    }
    if (pointCount && pointCount.textContent !== pointCountLabel) pointCount.textContent = pointCountLabel;
    if (!frame || frame.width !== width || frame.height !== height) {
      frame = context.createImageData(width, height);
      depthBuffer = new Float32Array(width * height);
    }

    const pixels = frame.data;
    pixels.fill(0);
    depthBuffer.fill(Infinity);
    const cameraDistance = 3.5 * radius;
    const scale = 0.46 * Math.min(width, height) * zoom / fitRadius;
    // Give points a centered circular footprint of about two CSS pixels.
    const edgeRadius = pointRadius * resolution + 0.5;
    const extent = Math.floor(edgeRadius);
    const footprint = [];
    for (let dy = -extent; dy <= extent; dy++) {
      for (let dx = -extent; dx <= extent; dx++) {
        const coverage = Math.min(1, edgeRadius - Math.hypot(dx, dy));
        // Where nearby splats overlap, favor solid centers over faint edges.
        if (coverage > 0.08) footprint.push([dx, dy, 235 * coverage, (1 - coverage) * 0.03 * fitRadius]);
      }
    }
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    const cp = Math.cos(pitch), sp = Math.sin(pitch);

    for (let i = 0; i < points.length; i += 3) {
      const x = cy * points[i] - sy * points[i + 2];
      const z = sy * points[i] + cy * points[i + 2];
      const y = cp * points[i + 1] - sp * z;
      const depth = sp * points[i + 1] + cp * z;
      const perspective = cameraDistance / (cameraDistance + depth);
      const column = Math.round(width / 2 + x * scale * perspective);
      const row = Math.round(height / 2 + y * scale * perspective);
      if (column < -extent || column >= width + extent || row < -extent || row >= height + extent) continue;

      let red, green, blue;
      if (colors) {
        red = colors[i];
        green = colors[i + 1];
        blue = colors[i + 2];
      } else {
        const shade = Math.max(0, Math.min(1, 0.5 - depth / (2 * fitRadius)));
        red = 80 + 125 * shade;
        green = 51 + 83 * shade;
        blue = 20 + 32 * shade;
      }
      for (const [dx, dy, opacity, edgeBias] of footprint) {
        const px = column + dx;
        const py = row + dy;
        if (px < 0 || px >= width || py < 0 || py >= height) continue;
        const pixel = py * width + px;
        const splatDepth = depth + edgeBias;
        if (splatDepth >= depthBuffer[pixel]) continue;

        depthBuffer[pixel] = splatDepth;
        pixels[4 * pixel] = red;
        pixels[4 * pixel + 1] = green;
        pixels[4 * pixel + 2] = blue;
        pixels[4 * pixel + 3] = opacity;
      }
    }
    context.putImageData(frame, 0, 0);
  }

  function requestDraw() {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => {
      scheduled = false;
      draw();
    });
  }

  function orbit(horizontal, vertical) {
    yaw = (yaw + horizontal) % (2 * Math.PI);
    pitch = Math.max(-1.55, Math.min(1.55, pitch + vertical));
    requestDraw();
  }

  function changeZoom(factor) {
    zoom = Math.max(minZoom, Math.min(maxZoom, zoom * factor));
    zoomIn.disabled = zoom === maxZoom;
    zoomOut.disabled = zoom === minZoom;
    requestDraw();
  }

  function resetView() {
    yaw = initialYaw;
    pitch = initialPitch;
    zoom = 1;
    changeZoom(1);
  }

  canvas.addEventListener("pointerdown", (event) => {
    if (activePointer !== null || event.button !== 0) return;
    activePointer = { id: event.pointerId, x: event.clientX, y: event.clientY };
    canvas.setPointerCapture(event.pointerId);
    canvas.focus({ preventScroll: true });
  });
  canvas.addEventListener("pointermove", (event) => {
    if (activePointer?.id !== event.pointerId) return;
    orbit(0.009 * (event.clientX - activePointer.x), -0.009 * (event.clientY - activePointer.y));
    activePointer.x = event.clientX;
    activePointer.y = event.clientY;
  });
  canvas.addEventListener("lostpointercapture", (event) => {
    if (activePointer?.id === event.pointerId) activePointer = null;
  });
  canvas.addEventListener("wheel", (event) => {
    if (document.activeElement !== canvas || event.ctrlKey || event.metaKey || !event.deltaY) return;
    event.preventDefault();
    changeZoom(event.deltaY < 0 ? 1.1 : 1 / 1.1);
  }, { passive: false });
  canvas.addEventListener("keydown", (event) => {
    if (event.ctrlKey || event.altKey || event.metaKey) return;
    switch (event.key) {
      case "ArrowLeft": orbit(-0.12, 0); break;
      case "ArrowRight": orbit(0.12, 0); break;
      case "ArrowUp": orbit(0, 0.12); break;
      case "ArrowDown": orbit(0, -0.12); break;
      case "+":
      case "=": changeZoom(1.2); break;
      case "-": changeZoom(1 / 1.2); break;
      case "Home": resetView(); break;
      default: return;
    }
    event.preventDefault();
  });

  zoomIn.addEventListener("click", () => changeZoom(1.2));
  zoomOut.addEventListener("click", () => changeZoom(1 / 1.2));
  reset.addEventListener("click", resetView);
  if ("ResizeObserver" in window) {
    new ResizeObserver(requestDraw).observe(canvas);
  } else {
    window.addEventListener("resize", requestDraw);
  }
  requestDraw();
});
