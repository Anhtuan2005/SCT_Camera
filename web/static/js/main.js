(function () {
  const ALERT_TYPE_LABELS = {
    intrusion: "Intrusion",
    loitering: "Loitering",
    stranger_detected: "Stranger detected",
    suspicious_stranger: "Suspicious stranger",
    asset_missing: "Possible theft",
    asset_removed: "Possible theft",
    suspicious_theft_behavior: "Possible theft",
    line_crossing: "Line crossing",
  };

  function alertTypeLabel(value) {
    const type = String(value || "");
    return ALERT_TYPE_LABELS[type] || type.replaceAll("_", " ");
  }

  const toastStack = document.getElementById("toastStack");
  const seenAlerts = new Set();
  let firstAlertPoll = true;
  let editingCameraId = "";
  let cameraEditPrimed = false;

  async function request(url, options = {}) {
    const response = await fetch(url, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || `${response.status} ${response.statusText}`);
    }
    return response.json();
  }

  function toast(title, message) {
    if (!toastStack) return;
    const node = document.createElement("div");
    node.className = "toast";
    node.innerHTML = `<strong>${escapeHtml(title)}</strong><span>${escapeHtml(message)}</span>`;
    toastStack.appendChild(node);
    window.setTimeout(() => node.remove(), 5200);
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function setStatusBadge(node, status) {
    if (!node) return;
    node.className = `status-badge ${status || "offline"}`;
    node.textContent = status || "offline";
  }

  function setSwitchState(node, active) {
    if (!node) return;
    node.setAttribute("aria-checked", active ? "true" : "false");
    node.classList.toggle("active", !!active);
  }

  function nextSwitchState(node) {
    return node?.getAttribute("aria-checked") !== "true";
  }

  function selectedNotificationChannels(root) {
    return Array.from(root.querySelectorAll('input[name="notification_channels"]:checked, [data-camera-channel]:checked'))
      .map((input) => input.value)
      .filter((value, index, items) => value && items.indexOf(value) === index);
  }

  function formatAiLatency(camera) {
    const latency = Number(camera.ai_latency_ms || 0);
    return `${latency.toFixed(0)} ms`;
  }

  function cameraHasChannel(camera, channel) {
    const channels = Array.isArray(camera.notification_channels) && camera.notification_channels.length
      ? camera.notification_channels
      : ["telegram"];
    return channels.includes(channel);
  }

  function setCameraFormMode(camera) {
    const form = document.getElementById("cameraForm");
    if (!form) return;
    const submit = document.getElementById("cameraSubmitButton");
    const cancel = document.getElementById("cameraEditCancel");
    const cameraId = camera ? String(camera.camera_id || "") : "";
    editingCameraId = cameraId;
    form.querySelector('[name="camera_id"]').value = cameraId;
    form.querySelector('[name="name"]').value = camera ? camera.name || "" : "";
    form.querySelector('[name="source"]').value = camera ? camera.source ?? "" : "";
    form.querySelector('[name="enabled"]').checked = camera ? camera.enabled !== false : true;
    for (const input of form.querySelectorAll('input[name="notification_channels"]')) {
      input.checked = camera ? cameraHasChannel(camera, input.value) : input.value === "telegram";
    }
    if (submit) submit.textContent = camera ? "Save" : "Add";
    if (cancel) cancel.hidden = !camera;
    updateEditingCameraRow();
  }

  function updateEditingCameraRow() {
    document.querySelectorAll("[data-settings-camera]").forEach((row) => {
      row.classList.toggle("editing", row.dataset.settingsCamera === editingCameraId);
    });
  }

  function primeCameraEditFromQuery(cameras) {
    if (cameraEditPrimed) return;
    cameraEditPrimed = true;
    const cameraId = new URLSearchParams(window.location.search).get("camera");
    if (!cameraId) return;
    const camera = cameras.find((item) => item.camera_id === cameraId);
    if (!camera) return;
    setCameraFormMode(camera);
    document.getElementById("cameras")?.scrollIntoView({ block: "start" });
  }

  async function refreshCameras() {
    const cameras = await request("/api/cameras");
    updateDashboard(cameras);
    updateDetailHeader(cameras);
    updateSettingsCameraList(cameras);
    primeCameraEditFromQuery(cameras);
    await pollAlerts(cameras);
  }

  function updateDashboard(cameras) {
    const total = document.getElementById("summaryTotal");
    const online = document.getElementById("summaryOnline");
    const objects = document.getElementById("summaryObjects");
    const alerts = document.getElementById("summaryAlerts");
    if (total) total.textContent = cameras.length;
    if (online) online.textContent = cameras.filter((camera) => camera.status === "online").length;
    if (objects) objects.textContent = cameras.reduce((sum, camera) => sum + (camera.object_count || 0), 0);
    if (alerts) alerts.textContent = cameras.reduce((sum, camera) => sum + (camera.alert_count || 0), 0);

    let anyEnabled = false;
    for (const camera of cameras) {
      const enabled = camera.enabled !== false;
      if (enabled) anyEnabled = true;
      const card = document.querySelector(`[data-camera-card="${camera.camera_id}"]`);
      if (!card) continue;
      setStatusBadge(card.querySelector("[data-status]"), camera.status);
      card.classList.toggle("disabled", !enabled);
      const objectNode = card.querySelector("[data-objects]");
      const alertNode = card.querySelector("[data-alerts]");
      const fpsNode = card.querySelector("[data-fps]");
      const aiLatencyNode = card.querySelector("[data-ai-latency]");
      if (objectNode) objectNode.textContent = camera.object_count || 0;
      if (alertNode) alertNode.textContent = camera.alert_count || 0;
      if (fpsNode) fpsNode.textContent = Number(camera.fps || 0).toFixed(1);
      if (aiLatencyNode) aiLatencyNode.textContent = formatAiLatency(camera);

      const enabledToggle = card.querySelector(`[data-camera-enabled-toggle="${camera.camera_id}"]`);
      if (enabledToggle && !enabledToggle._userChanging) {
        setSwitchState(enabledToggle, enabled);
      }
      const enabledText = card.querySelector("[data-camera-enabled-text]");
      if (enabledText) enabledText.textContent = enabled ? "Online" : "Offline";
      const enabledLabel = card.querySelector(`[data-toggle-camera-enabled="${camera.camera_id}"]`);
      if (enabledLabel) enabledLabel.classList.toggle("active", enabled);

      card.classList.toggle("paused", camera.status === "paused");
    }

    // Sync toggle-all cameras switch
    const toggleAll = document.getElementById("toggleAllCameras");
    if (toggleAll && !toggleAll._userChanging) {
      setSwitchState(toggleAll, anyEnabled);
    }
    const toggleAllText = document.getElementById("toggleAllText");
    if (toggleAllText) toggleAllText.textContent = anyEnabled ? "On" : "Off";
  }

  function updateDetailHeader(cameras) {
    const detail = document.querySelector("[data-camera-id]");
    if (!detail) return;
    const camId = detail.dataset.cameraId;
    const camera = cameras.find((item) => item.camera_id === camId);
    if (!camera) return;
    const enabled = camera.enabled !== false;
    setStatusBadge(document.querySelector("[data-detail-status]"), camera.status);
    const objectNode = document.querySelector("[data-detail-objects]");
    const alertNode = document.querySelector("[data-detail-alerts]");
    const fpsNode = document.querySelector("[data-detail-fps]");
    const aiLatencyNode = document.querySelector("[data-detail-ai-latency]");
    if (objectNode) objectNode.textContent = camera.object_count || 0;
    if (alertNode) alertNode.textContent = camera.alert_count || 0;
    if (fpsNode) fpsNode.textContent = Number(camera.fps || 0).toFixed(1);
    if (aiLatencyNode) aiLatencyNode.textContent = formatAiLatency(camera);

    const detailEnabledToggle = document.getElementById("detailEnabledToggleLabel");
    if (detailEnabledToggle && !detailEnabledToggle._userChanging) {
      setSwitchState(detailEnabledToggle, enabled);
    }
    const detailEnabledToggleText = document.getElementById("detailEnabledToggleText");
    if (detailEnabledToggleText) detailEnabledToggleText.textContent = enabled ? "Online" : "Offline";
    const detailEnabledToggleLabel = document.getElementById("detailEnabledToggleLabel");
    if (detailEnabledToggleLabel) detailEnabledToggleLabel.classList.toggle("active", enabled);

  }

  function updateSettingsCameraList(cameras) {
    const list = document.getElementById("settingsCameraList");
    if (!list) return;
    list.innerHTML = cameras
      .map(
        (camera) => `
        <div class="camera-list-row${editingCameraId === camera.camera_id ? " editing" : ""}" data-settings-camera="${escapeHtml(camera.camera_id)}"
             data-camera-name="${escapeHtml(camera.name)}"
             data-camera-source="${escapeHtml(camera.source)}"
             data-camera-enabled="${camera.enabled !== false ? "true" : "false"}">
          <div>
            <strong>${escapeHtml(camera.name)}</strong>
            <span class="mono">${escapeHtml(camera.source)}</span>
          </div>
          <div class="camera-channel-options" data-camera-channels="${escapeHtml(camera.camera_id)}">
            <label class="checkbox-row"><input data-camera-channel="telegram" type="checkbox" value="telegram" ${
              cameraHasChannel(camera, "telegram") ? "checked" : ""
            }>Telegram</label>
            <label class="checkbox-row"><input data-camera-channel="discord" type="checkbox" value="discord" ${
              cameraHasChannel(camera, "discord") ? "checked" : ""
            }>Discord</label>
          </div>
          <span class="status-badge ${escapeHtml(camera.status)}">${escapeHtml(camera.status)}</span>
          <a class="button small" href="/camera/${encodeURIComponent(camera.camera_id)}">Open</a>
          <button class="button small" type="button" data-edit-camera="${escapeHtml(camera.camera_id)}">Edit</button>
          <button class="button danger small" type="button" data-delete-camera="${escapeHtml(camera.camera_id)}">Delete</button>
        </div>`
      )
      .join("");
  }

  async function pollAlerts(cameras) {
    const ids = cameras.map((camera) => camera.camera_id);
    const detail = document.querySelector("[data-camera-id]");
    if (detail && !ids.includes(detail.dataset.cameraId)) ids.push(detail.dataset.cameraId);

    for (const cameraId of ids) {
      const alerts = await request(`/api/alerts/${encodeURIComponent(cameraId)}?limit=10`).catch(() => []);
      if (detail && detail.dataset.cameraId === cameraId) renderAlertTable(alerts);
      for (const alert of alerts) {
        const key = alertKey(cameraId, alert);
        if (seenAlerts.has(key)) continue;
        seenAlerts.add(key);
        if (!firstAlertPoll && !alert.suppressed) {
          const target = alert.zone_name || alert.line_name || "-";
          toast(`${alertTypeLabel(alert.type)} · ${alert.camera_name}`, `${target} · track #${alert.track_id}`);
        }
      }
    }
    firstAlertPoll = false;
  }

  function alertKey(cameraId, alert) {
    return [
      cameraId,
      alert.timestamp || alert.received_at,
      alert.type,
      alert.track_id,
      alert.zone_name || alert.line_name || "",
    ].join("|");
  }

  function renderAlertTable(alerts) {
    const body = document.getElementById("alertTableBody");
    if (!body) return;
    if (!alerts.length) {
      body.innerHTML = `<tr><td colspan="4">No alerts yet</td></tr>`;
      return;
    }
    body.innerHTML = alerts
      .map((alert) => {
        const target = alert.zone_name || alert.line_name || "-";
        const status = alert.suppressed ? "Cooldown" : alert.sent ? "Sent" : "Queued";
        return `
          <tr>
            <td>${escapeHtml(alert.timestamp || alert.received_at || "")}</td>
            <td>${escapeHtml(alertTypeLabel(alert.type))}</td>
            <td>${escapeHtml(target)}</td>
            <td>${escapeHtml(status)}</td>
          </tr>`;
      })
      .join("");
  }

  function setupEditorTabs() {
    const buttons = document.querySelectorAll("[data-editor-mode]");
    if (!buttons.length) return;
    buttons.forEach((button) => {
      button.addEventListener("click", () => {
        const mode = button.dataset.editorMode;
        buttons.forEach((item) => item.classList.toggle("active", item === button));
        document.getElementById("roiCanvas")?.classList.toggle("active", mode === "roi");
        document.getElementById("lineCanvas")?.classList.toggle("active", mode === "line");
        document.getElementById("roiPanel")?.classList.toggle("hidden", mode !== "roi");
        document.getElementById("linePanel")?.classList.toggle("hidden", mode !== "line");
        window.dispatchEvent(new Event("resize"));
      });
    });
  }

  function setupSettingsForms() {
    const telegramForm = document.getElementById("telegramForm");
    telegramForm?.addEventListener("submit", async (event) => {
      event.preventDefault();
      const form = new FormData(telegramForm);
      await request("/api/settings", {
        method: "PUT",
        body: JSON.stringify({
          telegram: {
            bot_token: form.get("bot_token"),
            chat_id: form.get("chat_id"),
            enabled: form.get("enabled") === "on",
            cooldown_seconds: Number(form.get("cooldown_seconds") || 10),
          },
        }),
      });
      toast("Settings saved", "Telegram configuration updated");
    });

    const discordForm = document.getElementById("discordForm");
    discordForm?.addEventListener("submit", async (event) => {
      event.preventDefault();
      const form = new FormData(discordForm);
      await request("/api/settings", {
        method: "PUT",
        body: JSON.stringify({
          discord: {
            webhook_url: form.get("webhook_url"),
            username: form.get("username") || "SCT Camera",
            enabled: form.get("enabled") === "on",
            max_retries: Number(form.get("max_retries") || 3),
          },
        }),
      });
      toast("Settings saved", "Discord configuration updated");
    });

    const thresholdForm = document.getElementById("thresholdForm");
    thresholdForm?.addEventListener("submit", async (event) => {
      event.preventDefault();
      const form = new FormData(thresholdForm);
      const classes = Array.from(thresholdForm.querySelectorAll('input[name="classes"]:checked')).map((input) =>
        Number(input.value)
      );
      if (!classes.length) {
        toast("Detection classes", "Select at least one class");
        return;
      }
      await request("/api/settings", {
        method: "PUT",
        body: JSON.stringify({
          detection: {
            model: form.get("model") || "yolo11s.pt",
            confidence: Number(form.get("confidence") || 0.25),
            iou: Number(form.get("iou") || 0.55),
            imgsz: Number(form.get("imgsz") || 800),
            classes,
            device: form.get("device"),
            half: true,
          },
          pose: {
            enabled: true,
            model: form.get("pose_model") || "yolo11n-pose.pt",
            allow_download: true,
            confidence: 0.2,
            imgsz: Number(form.get("pose_imgsz") || 640),
            match_iou: 0.2,
          },
          behavior: {
            loitering_threshold_seconds: Number(form.get("loitering_threshold_seconds") || 30),
            stranger_watch_seconds: Number(form.get("stranger_watch_seconds") || 180),
            asset_missing_seconds: Number(form.get("asset_missing_seconds") || 6),
            theft: {
              enabled: form.get("theft_enabled") === "on",
              proximity_seconds: Number(form.get("theft_proximity_seconds") || 10),
              proximity_distance_meters: Number(form.get("theft_proximity_distance_meters") || 1.5),
              meters_per_frame_diagonal: Number(form.get("theft_meters_per_frame_diagonal") || 12),
              pacing_min_passes: Number(form.get("theft_pacing_min_passes") || 2),
              score_threshold: Number(form.get("theft_score_threshold") || 2),
              require_vehicle_signal: form.get("theft_require_vehicle_signal") === "on",
            },
          },
          behavior_learning: {
            enabled: form.get("learning_enabled") === "on",
            log_candidates: form.get("learning_log_candidates") === "on",
            event_log_path: form.get("learning_event_log_path") || "data/behavior_events.jsonl",
            model_path: form.get("learning_model_path") || "models/behavior_classifier.npz",
            gate_alerts: form.get("learning_gate_alerts") === "on",
            min_risk_score: Number(form.get("learning_min_risk_score") || 0.65),
          },
          pipeline: {
            frame_skip: Number(form.get("frame_skip") || 1),
            ai_max_fps: Number(form.get("ai_max_fps") || 10),
            processing_max_height: Number(form.get("processing_max_height") || 720),
          },
          tracking: {
            track_grace_frames: Number(form.get("track_grace_frames") || 15),
            duplicate_iou_threshold: Number(form.get("duplicate_iou_threshold") || 0.85),
          },
          siren: {
            enabled: form.get("siren_enabled") === "on",
          },
        }),
      });
      toast("Settings saved", "Runtime thresholds updated");
    });

    const cameraForm = document.getElementById("cameraForm");
    cameraForm?.addEventListener("submit", async (event) => {
      event.preventDefault();
      const form = new FormData(cameraForm);
      const sourceValue = String(form.get("source") || "").trim();
      const cameraId = String(form.get("camera_id") || "").trim();
      const notificationChannels = selectedNotificationChannels(cameraForm);
      if (!notificationChannels.length) {
        toast("Alert channels", "Select Telegram or Discord");
        return;
      }
      await request("/api/cameras", {
        method: "POST",
        body: JSON.stringify({
          ...(cameraId ? { camera_id: cameraId } : {}),
          name: form.get("name"),
          source: /^\d+$/.test(sourceValue) ? Number(sourceValue) : sourceValue,
          enabled: form.get("enabled") === "on",
          notification_channels: notificationChannels,
        }),
      });
      setCameraFormMode(null);
      await refreshCameras();
      toast("Camera saved", "Camera list updated");
    });

    document.getElementById("settingsCameraList")?.addEventListener("click", async (event) => {
      const editButton = event.target.closest("[data-edit-camera]");
      if (editButton) {
        const row = editButton.closest("[data-settings-camera]");
        if (!row) return;
        setCameraFormMode({
          camera_id: row.dataset.settingsCamera,
          name: row.dataset.cameraName || "",
          source: row.dataset.cameraSource || "",
          enabled: row.dataset.cameraEnabled !== "false",
          notification_channels: selectedNotificationChannels(row),
        });
        document.getElementById("cameras")?.scrollIntoView({ block: "start" });
        return;
      }
      const button = event.target.closest("[data-delete-camera]");
      if (!button) return;
      await request(`/api/cameras/${encodeURIComponent(button.dataset.deleteCamera)}`, { method: "DELETE" });
      if (editingCameraId === button.dataset.deleteCamera) setCameraFormMode(null);
      await refreshCameras();
      toast("Camera deleted", button.dataset.deleteCamera);
    });

    document.getElementById("cameraEditCancel")?.addEventListener("click", () => {
      setCameraFormMode(null);
    });

    document.getElementById("settingsCameraList")?.addEventListener("change", async (event) => {
      const input = event.target.closest("[data-camera-channel]");
      if (!input) return;
      const row = input.closest("[data-settings-camera]");
      if (!row) return;
      const notificationChannels = selectedNotificationChannels(row);
      if (!notificationChannels.length) {
        input.checked = true;
        toast("Alert channels", "Each camera needs at least one channel");
        return;
      }
      const sourceValue = String(row.dataset.cameraSource || "").trim();
      await request("/api/cameras", {
        method: "POST",
        body: JSON.stringify({
          camera_id: row.dataset.settingsCamera,
          name: row.dataset.cameraName || row.dataset.settingsCamera,
          source: /^\d+$/.test(sourceValue) ? Number(sourceValue) : sourceValue,
          enabled: row.dataset.cameraEnabled !== "false",
          notification_channels: notificationChannels,
        }),
      });
      await refreshCameras();
      toast("Camera saved", "Alert channels updated");
    });

    document.getElementById("testTelegramButton")?.addEventListener("click", async () => {
      const result = await request("/api/settings/telegram/test", { method: "POST", body: "{}" });
      toast("Telegram test", result.sent ? "Message sent" : "Send skipped or failed");
    });

    document.getElementById("testDiscordButton")?.addEventListener("click", async () => {
      const result = await request("/api/settings/discord/test", { method: "POST", body: "{}" });
      toast("Discord test", result.sent ? "Message sent" : "Send skipped or failed");
    });
  }

  function setupCameraToggles() {
    async function setCameraEnabled(toggle, camId, enabled) {
      toggle._userChanging = true;
      try {
        await request(`/api/cameras/${encodeURIComponent(camId)}/enabled`, {
          method: "POST",
          body: JSON.stringify({ enabled }),
        });
        toast("Camera", `${camId}: ${enabled ? "online" : "offline"}`);
        await refreshCameras();
      } catch (err) {
        toast("Error", err.message);
        setSwitchState(toggle, !enabled);
      } finally {
        toggle._userChanging = false;
      }
    }

    async function setAllCamerasEnabled(toggle, enabled) {
      toggle._userChanging = true;
      try {
        const cameras = await request("/api/cameras");
        await Promise.all(
          cameras.map((camera) =>
            request(`/api/cameras/${encodeURIComponent(camera.camera_id)}/enabled`, {
              method: "POST",
              body: JSON.stringify({ enabled }),
            })
          )
        );
        toast("Camera", enabled ? "All cameras online" : "All cameras offline");
        await refreshCameras();
      } catch (err) {
        toast("Error", err.message);
        setSwitchState(toggle, !enabled);
      } finally {
        toggle._userChanging = false;
      }
    }

    // Toggle all cameras
    const toggleAll = document.getElementById("toggleAllCameras");
    if (toggleAll) {
      toggleAll.addEventListener("click", async () => {
        const enabled = nextSwitchState(toggleAll);
        setSwitchState(toggleAll, enabled);
        await setAllCamerasEnabled(toggleAll, enabled);
      });
    }

    // Per-camera toggles (event delegation on camera grid)
    const grid = document.getElementById("cameraGrid");
    if (grid) {
      grid.addEventListener("click", async (event) => {
        const enabledToggle = event.target.closest("[data-camera-enabled-toggle]");
        if (enabledToggle) {
          const enabled = nextSwitchState(enabledToggle);
          setSwitchState(enabledToggle, enabled);
          await setCameraEnabled(enabledToggle, enabledToggle.dataset.cameraEnabledToggle, enabled);
        }
      });
    }

    // Detail page camera online/offline toggle
    const detailEnabledToggle = document.getElementById("detailEnabledToggleLabel");
    if (detailEnabledToggle) {
      detailEnabledToggle.addEventListener("click", async () => {
        const enabled = nextSwitchState(detailEnabledToggle);
        setSwitchState(detailEnabledToggle, enabled);
        await setCameraEnabled(detailEnabledToggle, detailEnabledToggle.dataset.cameraId, enabled);
      });
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    setupEditorTabs();
    setupSettingsForms();
    setupCameraToggles();
    refreshCameras().catch((error) => toast("API error", error.message));
    window.setInterval(() => refreshCameras().catch(() => {}), 2500);
  });

  window.SCT = { request, toast };
})();
