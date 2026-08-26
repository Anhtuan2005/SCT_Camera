(function () {
  function directionForPoint(point1, point2, point, fallback = "forward") {
    const dx = point2[0] - point1[0];
    const dy = point2[1] - point1[1];
    const length = Math.hypot(dx, dy);
    if (length < 1) return fallback;
    const distance = (dx * (point[1] - point1[1]) - dy * (point[0] - point1[0])) / length;
    if (Math.abs(distance) < 6) return fallback;
    return distance > 0 ? "forward" : "reverse";
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { directionForPoint };
  }
  if (typeof document === "undefined") return;

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
  let draggingDirection = false;
  let draftDirty = false;
  const saveButton = document.getElementById("saveLineButton");
  const draftStatus = document.getElementById("lineDraftStatus");
  const directionSelect = document.getElementById("lineDirection");

  canvas.style.cursor = "crosshair";
  canvas.style.touchAction = "none";
  canvas.title = "Drag an arrow across its line to change the IN side";

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
      if (line.id === selectedLineId && current.length === 2) continue;
      drawLine(line.point1, line.point2, line.id === selectedLineId ? "#72d79b" : "rgba(160,175,185,.65)", line.direction);
    }
    if (current.length === 1) drawPoint(current[0], "#72d79b");
    if (current.length === 2) drawLine(current[0], current[1], "#72d79b", directionSelect.value);
  }

  function setDraftDirty(dirty) {
    draftDirty = dirty;
    if (draftStatus) {
      draftStatus.textContent = dirty ? "Unsaved changes" : "Saved";
      draftStatus.classList.toggle("dirty", dirty);
    }
    if (saveButton) {
      saveButton.disabled = !dirty || current.length !== 2;
    }
  }

  function resetDraft() {
    current = [];
    selectedLineId = null;
    document.getElementById("lineName").value = "New Line";
    directionSelect.value = "forward";
    setDraftDirty(false);
    draw();
  }

  function loadLine(line) {
    selectedLineId = line.id;
    current = [
      [Number(line.point1[0]), Number(line.point1[1])],
      [Number(line.point2[0]), Number(line.point2[1])],
    ];
    document.getElementById("lineName").value = line.name || "Line";
    directionSelect.value = line.direction || "forward";
    setDraftDirty(false);
    draw();
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

    const arrow = arrowGeometry(point1, point2, direction);
    drawArrow(arrow.start[0], arrow.start[1], arrow.tip[0], arrow.tip[1], color);
    ctx.restore();
  }

  function arrowGeometry(point1, point2, direction) {
    const [x1, y1] = toCanvas(point1);
    const [x2, y2] = toCanvas(point2);
    const mx = (x1 + x2) / 2;
    const my = (y1 + y2) / 2;
    const dx = x2 - x1;
    const dy = y2 - y1;
    const length = Math.max(1, Math.hypot(dx, dy));
    const sign = direction === "reverse" ? -1 : 1;
    return {
      start: [mx, my],
      tip: [mx + (-dy / length) * 44 * sign, my + (dx / length) * 44 * sign],
    };
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

  function distanceToSegment(point, start, end) {
    const dx = end[0] - start[0];
    const dy = end[1] - start[1];
    const lengthSquared = dx * dx + dy * dy;
    const t = lengthSquared
      ? Math.max(0, Math.min(1, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / lengthSquared))
      : 0;
    return Math.hypot(point[0] - (start[0] + t * dx), point[1] - (start[1] + t * dy));
  }

  function directionTargetAt(point) {
    const candidates = [];
    if (current.length === 2) {
      candidates.push({
        isCurrent: true,
        line: { point1: current[0], point2: current[1], direction: directionSelect.value },
      });
    }
    for (const line of lines) {
      if (line.id !== selectedLineId) candidates.push({ isCurrent: false, line });
    }
    const canvasPoint = toCanvas(point);
    return candidates.find(({ line }) => {
      const arrow = arrowGeometry(line.point1, line.point2, line.direction || "forward");
      return distanceToSegment(canvasPoint, arrow.start, arrow.tip) <= 16;
    });
  }

  canvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    const point = pointer(event);
    const endpoint = nearestEndpoint(point);
    if (endpoint >= 0) {
      draggingIndex = endpoint;
      canvas.style.cursor = "grabbing";
      canvas.setPointerCapture(event.pointerId);
      return;
    }
    const directionTarget = directionTargetAt(point);
    if (directionTarget) {
      if (!directionTarget.isCurrent) loadLine(directionTarget.line);
      draggingDirection = true;
      canvas.style.cursor = "grabbing";
      canvas.setPointerCapture(event.pointerId);
      return;
    }
    selectedLineId = null;
    if (current.length < 2) current.push(point);
    else current = [point];
    setDraftDirty(true);
    draw();
  });

  canvas.addEventListener("pointermove", (event) => {
    const point = pointer(event);
    if (draggingDirection) {
      const nextDirection = directionForPoint(
        toCanvas(current[0]),
        toCanvas(current[1]),
        toCanvas(point),
        directionSelect.value
      );
      if (nextDirection !== directionSelect.value) {
        directionSelect.value = nextDirection;
        setDraftDirty(true);
        draw();
      }
      return;
    }
    if (draggingIndex >= 0) {
      current[draggingIndex] = point;
      setDraftDirty(true);
      draw();
      return;
    }
    canvas.style.cursor = nearestEndpoint(point) >= 0 || directionTargetAt(point) ? "grab" : "crosshair";
  });

  function stopDragging() {
    draggingIndex = -1;
    draggingDirection = false;
    canvas.style.cursor = "crosshair";
  }

  canvas.addEventListener("pointerup", stopDragging);
  canvas.addEventListener("pointercancel", stopDragging);

  directionSelect?.addEventListener("change", () => {
    setDraftDirty(true);
    draw();
  });

  document.getElementById("clearLineButton")?.addEventListener("click", () => {
    resetDraft();
    window.SCT.toast("Draft discarded", "Saved counting lines were not changed");
  });

  saveButton?.addEventListener("click", async () => {
    if (current.length !== 2) {
      window.SCT.toast("Line incomplete", "Line needs 2 endpoints");
      return;
    }
    const payload = {
      id: selectedLineId || undefined,
      name: document.getElementById("lineName").value || "Line",
      point1: current[0].map(round),
      point2: current[1].map(round),
      direction: directionSelect.value,
    };
    const saved = await window.SCT.request(`/api/cameras/${encodeURIComponent(cameraId)}/lines`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    selectedLineId = saved.id;
    await loadLines();
    setDraftDirty(false);
    window.SCT.toast("Line saved", saved.name);
  });

  document.getElementById("lineList")?.addEventListener("click", async (event) => {
    const loadButton = event.target.closest("[data-load-line]");
    const deleteButton = event.target.closest("[data-delete-line]");
    if (loadButton) {
      const line = lines.find((item) => item.id === loadButton.dataset.loadLine);
      if (!line) return;
      loadLine(line);
    }
    if (deleteButton) {
      await window.SCT.request(
        `/api/cameras/${encodeURIComponent(cameraId)}/lines/${encodeURIComponent(deleteButton.dataset.deleteLine)}`,
        { method: "DELETE" }
      );
      resetDraft();
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
  document.getElementById("lineName")?.addEventListener("input", () => setDraftDirty(true));
  window.addEventListener("beforeunload", (event) => {
    if (!draftDirty) return;
    event.preventDefault();
    event.returnValue = "";
  });
  new ResizeObserver(resize).observe(image);
  setDraftDirty(false);
  loadLines().catch((error) => window.SCT.toast("Line API error", error.message));
})();
