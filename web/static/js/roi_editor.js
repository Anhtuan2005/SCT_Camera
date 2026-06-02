(function () {
  const detail = document.querySelector("[data-camera-id]");
  const canvas = document.getElementById("roiCanvas");
  const image = document.getElementById("editorStream");
  if (!detail || !canvas || !image) return;

  const cameraId = detail.dataset.cameraId;
  const ctx = canvas.getContext("2d");
  let zones = [];
  let current = [];
  let selectedZoneId = null;
  let draggingIndex = -1;
  const DEFAULT_ZONE_TYPE = "all";
  const DEFAULT_THRESHOLD_SECONDS = 15;
  const TIMED_ZONE_TYPES = new Set(["all", "loitering", "stranger_watch", "asset_watch"]);
  const ZONE_TYPE_LABELS = {
    all: "All Behaviors",
    intrusion: "Intrusion",
    loitering: "Loitering",
    stranger_watch: "Stranger Watch",
    asset_watch: "Asset Watch",
    counting: "Counting",
  };

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
    for (const zone of zones) {
      drawPolygon(zone.polygon || [], zone.id === selectedZoneId ? "#72d79b" : "rgba(160,175,185,.55)", false);
    }
    drawPolygon(current, "#72d79b", true);
  }

  function drawPolygon(points, color, active) {
    if (!points.length) return;
    ctx.save();
    ctx.lineWidth = active ? 2.5 : 1.5;
    ctx.strokeStyle = color;
    ctx.fillStyle = active ? "rgba(84, 195, 145, .18)" : "rgba(140, 160, 170, .10)";
    ctx.beginPath();
    points.forEach((point, index) => {
      const [x, y] = toCanvas(point);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    if (points.length >= 3) ctx.closePath();
    ctx.fill();
    ctx.stroke();
    for (const point of points) {
      const [x, y] = toCanvas(point);
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fillStyle = active ? "#72d79b" : "#9db0b8";
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#20272a";
      ctx.stroke();
    }
    ctx.restore();
  }

  function nearestVertex(point) {
    let best = -1;
    let bestDistance = 14;
    current.forEach((vertex, index) => {
      const [x, y] = toCanvas(vertex);
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
    const vertex = nearestVertex(point);
    if (vertex >= 0) {
      draggingIndex = vertex;
      canvas.setPointerCapture(event.pointerId);
    } else {
      current.push(point);
      selectedZoneId = null;
      draw();
    }
  });

  canvas.addEventListener("pointermove", (event) => {
    if (draggingIndex < 0) return;
    current[draggingIndex] = pointer(event);
    draw();
  });

  canvas.addEventListener("pointerup", () => {
    draggingIndex = -1;
  });

  canvas.addEventListener("dblclick", (event) => {
    event.preventDefault();
    if (current.length >= 3) {
      window.SCT.toast("ROI ready", `${current.length} vertices selected`);
      draw();
    }
  });

  canvas.addEventListener("contextmenu", (event) => {
    event.preventDefault();
    const vertex = nearestVertex(pointer(event));
    if (vertex >= 0) {
      current.splice(vertex, 1);
      draw();
    }
  });

  document.getElementById("clearRoiButton")?.addEventListener("click", async () => {
    if (selectedZoneId) {
      await deleteZone(selectedZoneId);
      window.SCT.toast("Zone deleted", selectedZoneId);
      return;
    }

    if (zones.length && !current.length) {
      await Promise.all(zones.map((zone) => deleteZone(zone.id, false)));
      await loadZones();
      window.SCT.toast("Zones cleared", "All saved ROI zones were deleted");
      return;
    }

    current = [];
    selectedZoneId = null;
    draw();
  });

  document.getElementById("saveZoneButton")?.addEventListener("click", async () => {
    if (current.length < 3) {
      window.SCT.toast("ROI incomplete", "Polygon needs at least 3 points");
      return;
    }
    const zoneType = DEFAULT_ZONE_TYPE;
    const payload = {
      id: selectedZoneId || undefined,
      name: document.getElementById("zoneName").value || "Zone",
      type: zoneType,
      polygon: current.map(([x, y]) => [round(x), round(y)]),
    };
    if (TIMED_ZONE_TYPES.has(zoneType)) {
      payload.threshold_seconds = Number(document.getElementById("zoneThreshold").value || DEFAULT_THRESHOLD_SECONDS);
    }
    const saved = await window.SCT.request(`/api/cameras/${encodeURIComponent(cameraId)}/zones`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    selectedZoneId = saved.id;
    await loadZones();
    window.SCT.toast("Zone saved", saved.name);
  });

  document.getElementById("zoneList")?.addEventListener("click", async (event) => {
    const loadButton = event.target.closest("[data-load-zone]");
    const deleteButton = event.target.closest("[data-delete-zone]");
    if (loadButton) {
      const zone = zones.find((item) => item.id === loadButton.dataset.loadZone);
      if (!zone) return;
      selectedZoneId = zone.id;
      current = (zone.polygon || []).map((point) => [Number(point[0]), Number(point[1])]);
      document.getElementById("zoneName").value = zone.name || "Zone";
      const zoneTypeInput = document.getElementById("zoneType");
      if (zoneTypeInput && zoneTypeInput.type !== "hidden") {
        zoneTypeInput.value = zone.behaviorType || zone.type || DEFAULT_ZONE_TYPE;
      }
      const zoneThreshold = document.getElementById("zoneThreshold");
      if (zoneThreshold) {
        zoneThreshold.value = zone.threshold_seconds || DEFAULT_THRESHOLD_SECONDS;
      }
      updateThresholdVisibility();
      draw();
    }
    if (deleteButton) {
      await deleteZone(deleteButton.dataset.deleteZone);
      window.SCT.toast("Zone deleted", deleteButton.dataset.deleteZone);
    }
  });

  async function deleteZone(zoneId, reload = true) {
    await window.SCT.request(
      `/api/cameras/${encodeURIComponent(cameraId)}/zones/${encodeURIComponent(zoneId)}`,
      { method: "DELETE" }
    );
    if (selectedZoneId === zoneId) {
      selectedZoneId = null;
      current = [];
    }
    if (reload) {
      await loadZones();
    }
  }

  async function loadZones() {
    const loadedZones = await window.SCT.request(`/api/cameras/${encodeURIComponent(cameraId)}/zones`);
    zones = loadedZones.map((zone) => ({
      ...zone,
      behaviorType: zone.type || DEFAULT_ZONE_TYPE,
      type: zoneTypeLabel(zone.type),
    }));
    renderZones();
    draw();
  }

  function renderZones() {
    const list = document.getElementById("zoneList");
    if (!list) return;
    list.innerHTML = zones
      .map(
        (zone) => `
        <div class="compact-item">
          <div>
            <strong>${escapeHtml(zone.name)}</strong>
            <span>${escapeHtml(zone.type)} · ${(zone.polygon || []).length} points</span>
          </div>
          <div>
            <button class="button small" type="button" data-load-zone="${escapeHtml(zone.id)}">Edit</button>
            <button class="button danger small" type="button" data-delete-zone="${escapeHtml(zone.id)}">Delete</button>
          </div>
        </div>`
      )
      .join("");
  }

  function round(value) {
    return Number(value.toFixed(5));
  }

  function zoneTypeLabel(type) {
    return ZONE_TYPE_LABELS[type] || type || DEFAULT_ZONE_TYPE;
  }

  function updateThresholdVisibility() {
    const zoneType = document.getElementById("zoneType");
    const thresholdLabel = document.getElementById("zoneThresholdLabel");
    if (!zoneType || !thresholdLabel) return;
    thresholdLabel.hidden = !TIMED_ZONE_TYPES.has(zoneType.value);
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
  document.getElementById("zoneType")?.addEventListener("change", updateThresholdVisibility);
  new ResizeObserver(resize).observe(image);
  updateThresholdVisibility();
  loadZones().catch((error) => window.SCT.toast("Zone API error", error.message));
})();
