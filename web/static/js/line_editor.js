(function () {
  const detail = document.querySelector("[data-camera-id]");
  const canvas = document.getElementById("lineCanvas");
  const image = document.getElementById("editorStream");
  if (!detail || !canvas || !image) return;

  const cameraId = detail.dataset.cameraId;
  const ctx = canvas.getContext("2d");
  let lines = [];
  let current = [];
  let selectedLineId = null;
  let draggingIndex = -1;

  function resize() {
    const rect = image.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  function toCanvas(point) {
    return [point[0] * canvas.clientWidth, point[1] * canvas.clientHeight];
  }

  function pointer(event) {
    const rect = canvas.getBoundingClientRect();
    return [
      clamp((event.clientX - rect.left) / rect.width),
      clamp((event.clientY - rect.top) / rect.height),
    ];
  }

  function clamp(value) {
    return Math.max(0, Math.min(1, value));
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    for (const line of lines) {
      drawLine(line.point1, line.point2, line.id === selectedLineId ? "#72d79b" : "rgba(160,175,185,.65)", line.direction);
    }
    if (current.length === 1) drawPoint(current[0], "#72d79b");
    if (current.length === 2) drawLine(current[0], current[1], "#72d79b", document.getElementById("lineDirection").value);
  }

  function drawLine(point1, point2, color, direction) {
    const [x1, y1] = toCanvas(point1);
    const [x2, y2] = toCanvas(point2);
    ctx.save();
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    drawPoint(point1, color);
    drawPoint(point2, color);

    const mx = (x1 + x2) / 2;
    const my = (y1 + y2) / 2;
    const dx = x2 - x1;
    const dy = y2 - y1;
    const length = Math.max(1, Math.hypot(dx, dy));
    let nx = -dy / length;
    let ny = dx / length;
    if (direction === "reverse") {
      nx *= -1;
      ny *= -1;
    }
    drawArrow(mx, my, mx + nx * 44, my + ny * 44, color);
    ctx.restore();
  }

  function drawPoint(point, color) {
    const [x, y] = toCanvas(point);
    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#20272a";
    ctx.stroke();
    ctx.restore();
  }

  function drawArrow(x1, y1, x2, y2, color) {
    const angle = Math.atan2(y2 - y1, x2 - x1);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - 11 * Math.cos(angle - Math.PI / 6), y2 - 11 * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(x2 - 11 * Math.cos(angle + Math.PI / 6), y2 - 11 * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
  }

  function nearestEndpoint(point) {
    let best = -1;
    let bestDistance = 16;
    current.forEach((endpoint, index) => {
      const [x, y] = toCanvas(endpoint);
      const [px, py] = toCanvas(point);
      const distance = Math.hypot(x - px, y - py);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = index;
      }
    });
    return best;
  }

  canvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    const point = pointer(event);
    const endpoint = nearestEndpoint(point);
    if (endpoint >= 0) {
      draggingIndex = endpoint;
      canvas.setPointerCapture(event.pointerId);
      return;
    }
    selectedLineId = null;
    if (current.length < 2) current.push(point);
    else current = [point];
    draw();
  });

  canvas.addEventListener("pointermove", (event) => {
    if (draggingIndex < 0) return;
    current[draggingIndex] = pointer(event);
    draw();
  });

  canvas.addEventListener("pointerup", () => {
    draggingIndex = -1;
  });

  document.getElementById("lineDirection")?.addEventListener("change", draw);

  document.getElementById("clearLineButton")?.addEventListener("click", () => {
    current = [];
    selectedLineId = null;
    draw();
  });

  document.getElementById("saveLineButton")?.addEventListener("click", async () => {
    if (current.length !== 2) {
      window.SCT.toast("Line incomplete", "Line needs 2 endpoints");
      return;
    }
    const payload = {
      id: selectedLineId || undefined,
      name: document.getElementById("lineName").value || "Line",
      point1: current[0].map(round),
      point2: current[1].map(round),
      direction: document.getElementById("lineDirection").value,
    };
    const saved = await window.SCT.request(`/api/cameras/${encodeURIComponent(cameraId)}/lines`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    selectedLineId = saved.id;
    await loadLines();
    window.SCT.toast("Line saved", saved.name);
  });

  document.getElementById("lineList")?.addEventListener("click", async (event) => {
    const loadButton = event.target.closest("[data-load-line]");
    const deleteButton = event.target.closest("[data-delete-line]");
    if (loadButton) {
      const line = lines.find((item) => item.id === loadButton.dataset.loadLine);
      if (!line) return;
      selectedLineId = line.id;
      current = [
        [Number(line.point1[0]), Number(line.point1[1])],
        [Number(line.point2[0]), Number(line.point2[1])],
      ];
      document.getElementById("lineName").value = line.name || "Line";
      document.getElementById("lineDirection").value = line.direction || "forward";
      draw();
    }
    if (deleteButton) {
      await window.SCT.request(
        `/api/cameras/${encodeURIComponent(cameraId)}/lines/${encodeURIComponent(deleteButton.dataset.deleteLine)}`,
        { method: "DELETE" }
      );
      selectedLineId = null;
      current = [];
      await loadLines();
      window.SCT.toast("Line deleted", deleteButton.dataset.deleteLine);
    }
  });

  async function loadLines() {
    lines = await window.SCT.request(`/api/cameras/${encodeURIComponent(cameraId)}/lines`);
    renderLines();
    draw();
  }

  function renderLines() {
    const list = document.getElementById("lineList");
    if (!list) return;
    list.innerHTML = lines
      .map(
        (line) => `
        <div class="compact-item">
          <div>
            <strong>${escapeHtml(line.name)}</strong>
            <span>${escapeHtml(line.direction || "forward")}</span>
          </div>
          <div>
            <button class="button small" type="button" data-load-line="${escapeHtml(line.id)}">Edit</button>
            <button class="button danger small" type="button" data-delete-line="${escapeHtml(line.id)}">Delete</button>
          </div>
        </div>`
      )
      .join("");
  }

  function round(value) {
    return Number(value.toFixed(5));
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  image.addEventListener("load", resize);
  window.addEventListener("resize", resize);
  new ResizeObserver(resize).observe(image);
  loadLines().catch((error) => window.SCT.toast("Line API error", error.message));
})();
