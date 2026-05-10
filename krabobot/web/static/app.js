/**
 * Local web chat: OpenAI-compatible API + session list + attachments.
 */
(function () {
  "use strict";

  const SESSION_KEY = "krabobot_web_session_id";
  const logEl = document.getElementById("kb-log");
  const formEl = document.getElementById("kb-form");
  const inputEl = document.getElementById("kb-input");
  const sendBtn = document.getElementById("kb-send");
  const statusEl = document.getElementById("kb-status");
  const newSessionBtn = document.getElementById("kb-new-session");
  const refreshSessionsBtn = document.getElementById("kb-refresh-sessions");
  const sessionListEl = document.getElementById("kb-session-list");
  const attachBtn = document.getElementById("kb-attach");
  const fileInputEl = document.getElementById("kb-files");
  const tabChat = document.getElementById("kb-tab-chat");
  const tabSettings = document.getElementById("kb-tab-settings");
  const viewChat = document.getElementById("kb-view-chat");
  const viewSettings = document.getElementById("kb-view-settings");
  const copySessionBtn = document.getElementById("kb-copy-session");

  let modelId = null;
  /** @type {File[]} */
  let pendingFiles = [];

  function getSessionId() {
    let id = localStorage.getItem(SESSION_KEY);
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem(SESSION_KEY, id);
    }
    return id;
  }

  function setSessionId(id) {
    localStorage.setItem(SESSION_KEY, id);
  }

  function setStatus(text) {
    statusEl.textContent = text || "";
  }

  function switchTab(which) {
    const isChat = which === "chat";
    if (viewChat && viewSettings) {
      viewChat.classList.toggle("kb-view--hidden", !isChat);
      viewSettings.classList.toggle("kb-view--hidden", isChat);
      viewSettings.hidden = isChat;
    }
    if (tabChat && tabSettings) {
      tabChat.classList.toggle("kb-tab--active", isChat);
      tabSettings.classList.toggle("kb-tab--active", !isChat);
      tabChat.setAttribute("aria-selected", String(isChat));
      tabSettings.setAttribute("aria-selected", String(!isChat));
    }
    if (!isChat) {
      refreshSettingsPanel();
    }
  }

  async function refreshSettingsPanel() {
    const elSession = document.getElementById("kb-st-session");
    const elModel = document.getElementById("kb-st-model");
    const elHealth = document.getElementById("kb-st-health");
    if (elSession) {
      elSession.textContent = getSessionId();
    }
    if (elModel) {
      elModel.textContent = modelId || "—";
    }
    if (elHealth) {
      elHealth.textContent = "…";
      try {
        const r = await fetch("/health");
        const j = await r.json().catch(() => ({}));
        elHealth.textContent =
          r.ok && j.status === "ok" ? "ok" : "HTTP " + r.status;
      } catch {
        elHealth.textContent = "недоступно";
      }
    }
  }

  function appendMessage(role, text, kind) {
    const wrap = document.createElement("div");
    wrap.className =
      "kb-msg " +
      (kind === "error"
        ? "kb-msg--error"
        : role === "user"
          ? "kb-msg--user"
          : "kb-msg--assistant");
    const label = document.createElement("div");
    label.className = "kb-msg-label";
    label.textContent =
      kind === "error" ? "ошибка" : role === "user" ? "вы" : "ассистент";
    const body = document.createElement("div");
    body.className = "kb-msg-body";
    body.textContent = text;
    wrap.appendChild(label);
    wrap.appendChild(body);
    logEl.appendChild(wrap);
    logEl.scrollTop = logEl.scrollHeight;
  }

  function readAsBase64(file) {
    return new Promise((resolve, reject) => {
      const fr = new FileReader();
      fr.onload = () => {
        const s = String(fr.result || "");
        const i = s.indexOf(",");
        resolve(i >= 0 ? s.slice(i + 1) : s);
      };
      fr.onerror = () => reject(fr.error);
      fr.readAsDataURL(file);
    });
  }

  function audioFormat(mime, name) {
    const m = (mime || "").toLowerCase();
    if (m.includes("wav")) return "wav";
    if (m.includes("mpeg") || m.includes("mp3")) return "mp3";
    if (m.includes("webm")) return "webm";
    if (m.includes("ogg")) return "ogg";
    if (m.includes("mp4") || m.includes("m4a")) return "mp4";
    if (/\.(wav)$/i.test(name)) return "wav";
    if (/\.(mp3)$/i.test(name)) return "mp3";
    if (/\.(webm|ogg)$/i.test(name)) return "webm";
    return "wav";
  }

  /**
   * OpenAI-style content parts from text + files.
   */
  async function buildContentPayload(text, files) {
    const parts = [];
    const trim = (text || "").trim();

    for (const file of files) {
      const mt = file.type || "";
      const name = file.name || "file";

      if (mt.startsWith("image/")) {
        const b64 = await readAsBase64(file);
        parts.push({
          type: "image_url",
          image_url: { url: `data:${mt};base64,${b64}` },
        });
        continue;
      }

      if (mt.startsWith("audio/") || /\.(mp3|wav|ogg|webm|m4a|flac)$/i.test(name)) {
        const b64 = await readAsBase64(file);
        parts.push({
          type: "input_audio",
          input_audio: { data: b64, format: audioFormat(mt, name) },
        });
        continue;
      }

      if (
        mt.startsWith("text/") ||
        /\.(txt|md|csv|json|xml|yaml|yml|log|ini|env)$/i.test(name)
      ) {
        let t = await file.text();
        if (t.length > 120000) {
          t = t.slice(0, 120000) + "\n…(обрезано)";
        }
        parts.push({ type: "text", text: `--- ${name} ---\n${t}` });
        continue;
      }

      // PDF, Office, архивы и остальной бинарник — на сервер как base64, сохраняется в workspace
      const b64 = await readAsBase64(file);
      parts.push({
        type: "kb_file",
        kb_file: {
          filename: name,
          mime: mt || "application/octet-stream",
          data: b64,
        },
      });
    }

    if (trim && parts.length === 0) {
      return trim;
    }

    const out = [];
    if (trim) {
      out.push({ type: "text", text: trim });
    }
    out.push(...parts);

    if (out.length === 0) {
      return "…";
    }
    if (out.length === 1 && out[0].type === "text") {
      return out[0].text;
    }
    return out;
  }

  async function fetchModel() {
    const r = await fetch("/v1/models");
    if (!r.ok) {
      throw new Error("GET /v1/models: " + r.status);
    }
    const data = await r.json();
    const id = data?.data?.[0]?.id;
    if (!id) {
      throw new Error("Нет модели в ответе /v1/models");
    }
    return id;
  }

  async function fetchSessionList() {
    const r = await fetch("/v1/web/sessions");
    if (!r.ok) {
      throw new Error("Список диалогов: " + r.status);
    }
    const data = await r.json();
    return Array.isArray(data.data) ? data.data : [];
  }

  async function deleteSessionOnServer(id) {
    const r = await fetch("/v1/web/sessions/" + encodeURIComponent(id), {
      method: "DELETE",
    });
    if (r.status === 404) {
      return false;
    }
    if (!r.ok) {
      throw new Error("Удаление: " + r.status);
    }
    return true;
  }

  async function fetchSessionMessages(id) {
    const r = await fetch(
      "/v1/web/sessions/" + encodeURIComponent(id) + "/messages"
    );
    if (!r.ok) {
      throw new Error("История: " + r.status);
    }
    const data = await r.json();
    return Array.isArray(data.data) ? data.data : [];
  }

  function renderSessionList(rows, currentId) {
    sessionListEl.innerHTML = "";
    for (const row of rows) {
      const id = row.id;
      const li = document.createElement("li");
      li.className = "kb-sess" + (id === currentId ? " kb-sess--active" : "");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "kb-sess-btn";
      const prev = (row.preview || "").trim() || "—";
      const when = row.updated_at
        ? String(row.updated_at).slice(0, 16).replace("T", " ")
        : "";
      btn.textContent = prev.slice(0, 52) + (prev.length > 52 ? "…" : "");
      btn.title = when + "\n" + id;
      btn.addEventListener("click", async () => {
        setSessionId(id);
        await refreshSessions();
        await loadHistoryForSession(id);
      });
      const del = document.createElement("button");
      del.type = "button";
      del.className = "kb-sess-del";
      del.textContent = "×";
      del.title = "Удалить диалог";
      del.addEventListener("click", async (e) => {
        e.stopPropagation();
        if (!window.confirm("Удалить этот диалог с сервера?")) {
          return;
        }
        try {
          await deleteSessionOnServer(id);
          if (getSessionId() === id) {
            const nid = crypto.randomUUID();
            setSessionId(nid);
            logEl.innerHTML = "";
          }
          await refreshSessions();
          setStatus("Удалено");
        } catch (err) {
          appendMessage("assistant", String(err.message || err), "error");
        }
        setTimeout(() => setStatus(""), 2000);
      });
      li.appendChild(btn);
      li.appendChild(del);
      sessionListEl.appendChild(li);
    }
  }

  async function refreshSessions() {
    const current = getSessionId();
    try {
      const rows = await fetchSessionList();
      renderSessionList(rows, current);
    } catch (err) {
      setStatus("");
      console.warn(err);
    }
  }

  async function loadHistoryForSession(id) {
    logEl.innerHTML = "";
    setStatus("Загрузка истории…");
    try {
      const msgs = await fetchSessionMessages(id);
      for (const m of msgs) {
        const role = m.role === "user" ? "user" : "assistant";
        appendMessage(role, String(m.content || ""));
      }
      setStatus("");
    } catch (err) {
      setStatus("");
      appendMessage("assistant", String(err.message || err), "error");
    }
  }

  async function sendMessage(userText, files) {
    const body = {
      model: modelId,
      messages: [{ role: "user", content: await buildContentPayload(userText, files) }],
      stream: false,
      session_id: getSessionId(),
    };
    const r = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      const msg =
        data?.error?.message ||
        (typeof data === "object" ? JSON.stringify(data) : String(data));
      throw new Error(msg || "HTTP " + r.status);
    }
    const content = data?.choices?.[0]?.message?.content;
    if (typeof content !== "string") {
      throw new Error("Неверный ответ API");
    }
    return content;
  }

  formEl.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = inputEl.value.trim();
    if ((!text && pendingFiles.length === 0) || !modelId) {
      return;
    }

    const filesSnapshot = pendingFiles.slice();
    pendingFiles = [];
    renderAttachments();

    const userVisible =
      text +
      (filesSnapshot.length
        ? "\n" + filesSnapshot.map((f) => "📎 " + f.name).join("\n")
        : "");
    appendMessage("user", userVisible || "(вложения)");
    inputEl.value = "";
    sendBtn.disabled = true;
    setStatus("Запрос…");

    try {
      const reply = await sendMessage(text, filesSnapshot);
      appendMessage("assistant", reply);
      setStatus("");
      await refreshSessions();
    } catch (err) {
      appendMessage("assistant", String(err.message || err), "error");
      setStatus("");
    } finally {
      sendBtn.disabled = false;
      inputEl.focus();
    }
  });

  newSessionBtn.addEventListener("click", async () => {
    const nid = crypto.randomUUID();
    setSessionId(nid);
    logEl.innerHTML = "";
    setStatus("Новый диалог");
    await refreshSessions();
    setTimeout(() => setStatus(""), 2000);
  });

  refreshSessionsBtn.addEventListener("click", () => refreshSessions());

  if (tabChat) {
    tabChat.addEventListener("click", () => switchTab("chat"));
  }
  if (tabSettings) {
    tabSettings.addEventListener("click", () => switchTab("settings"));
  }

  if (copySessionBtn) {
    copySessionBtn.addEventListener("click", async () => {
      const id = getSessionId();
      try {
        await navigator.clipboard.writeText(id);
        setStatus("ID сессии скопирован");
        setTimeout(() => setStatus(""), 2200);
      } catch {
        setStatus("Не удалось скопировать");
        setTimeout(() => setStatus(""), 2200);
      }
    });
  }

  attachBtn.addEventListener("click", () => fileInputEl.click());

  fileInputEl.addEventListener("change", () => {
    const files = Array.from(fileInputEl.files || []);
    fileInputEl.value = "";
    pendingFiles.push(...files);
    renderAttachments();
  });

  function renderAttachments() {
    const bar = document.getElementById("kb-attachments");
    if (!bar) {
      return;
    }
    bar.innerHTML = "";
    pendingFiles.forEach((f) => {
      const tag = document.createElement("span");
      tag.className = "kb-attach-tag";
      tag.textContent = f.name;
      const x = document.createElement("button");
      x.type = "button";
      x.textContent = "×";
      x.addEventListener("click", () => {
        pendingFiles = pendingFiles.filter((p) => p !== f);
        renderAttachments();
      });
      tag.appendChild(x);
      bar.appendChild(tag);
    });
  }

  (async function init() {
    try {
      modelId = await fetchModel();
      setStatus("Модель: " + modelId);
      await refreshSettingsPanel();
      await refreshSessions();
      await loadHistoryForSession(getSessionId());
    } catch (err) {
      appendMessage(
        "assistant",
        "Не удалось инициализировать: " + (err.message || err),
        "error"
      );
      sendBtn.disabled = true;
    }
  })();
})();
