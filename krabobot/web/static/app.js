/**
 * Local web chat: OpenAI-compatible API + session list + attachments.
 */
(function () {
  "use strict";

  const SESSION_KEY = "krabobot_web_session_id";
  const AUTH_TOKEN_KEY = "krabobot_web_auth_token";
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
  const menuBtn = document.getElementById("kb-menu-btn");
  const menuDropdown = document.getElementById("kb-menu-dropdown");
  const navChat = document.getElementById("kb-nav-chat");
  const navSettings = document.getElementById("kb-nav-settings");
  const navAdmin = document.getElementById("kb-nav-admin");
  const navLogout = document.getElementById("kb-nav-logout");
  const viewChat = document.getElementById("kb-view-chat");
  const viewSettings = document.getElementById("kb-view-settings");
  const viewAdmin = document.getElementById("kb-view-admin");
  const authGate = document.getElementById("kb-auth-gate");
  const layoutElRoot = document.getElementById("kb-layout");
  const copySessionBtn = document.getElementById("kb-copy-session");
  const cmdMenuBtn = document.getElementById("kb-cmd-menu-btn");
  const cmdMenuDropdown = document.getElementById("kb-cmd-dropdown");
  const layoutEl = document.getElementById("kb-layout");
  const sidebarEl = document.getElementById("kb-sidebar");
  const sidebarToggle = document.getElementById("kb-sidebar-toggle");
  const sidebarExpand = document.getElementById("kb-sidebar-expand");

  const SIDEBAR_COLLAPSED_KEY = "krabobot_web_sidebar_collapsed";

  function getAuthToken() {
    try {
      return sessionStorage.getItem(AUTH_TOKEN_KEY) || "";
    } catch (_e) {
      return "";
    }
  }

  function setAuthToken(token) {
    try {
      if (token) {
        sessionStorage.setItem(AUTH_TOKEN_KEY, token);
      } else {
        sessionStorage.removeItem(AUTH_TOKEN_KEY);
      }
    } catch (_e) {
      /* ignore */
    }
  }

  /**
   * Authenticated fetch for /v1/* (cookie + optional Bearer from login).
   * @param {string} url
   * @param {RequestInit} [opts]
   */
  function kbApiFetch(url, opts) {
    const o = Object.assign({ credentials: "include" }, opts || {});
    const headers = new Headers(o.headers || {});
    const tok = getAuthToken();
    if (tok && !headers.has("Authorization")) {
      headers.set("Authorization", "Bearer " + tok);
    }
    o.headers = headers;
    return fetch(url, o);
  }

  /** Prefer server Russian message; map bare 401 to a clear RU string. */
  function kbHttpErrorMessage(status, data, fallback) {
    const fromApi =
      data && data.error && data.error.message ? String(data.error.message) : "";
    if (fromApi) {
      return fromApi;
    }
    if (status === 401) {
      return "Требуется аутентификация. Войдите снова.";
    }
    return fallback || "Ошибка HTTP " + status;
  }

  async function kbHandleUnauthorized() {
    setAuthToken("");
    showAuthGate(false);
  }


  /** UUID v4; works over HTTP where crypto.randomUUID is unavailable. */
  function kbRandomUuid() {
    if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
      return crypto.randomUUID();
    }
    if (typeof crypto !== "undefined" && typeof crypto.getRandomValues === "function") {
      const bytes = new Uint8Array(16);
      crypto.getRandomValues(bytes);
      bytes[6] = (bytes[6] & 0x0f) | 0x40;
      bytes[8] = (bytes[8] & 0x3f) | 0x80;
      const hex = Array.from(bytes, function (b) {
        return b.toString(16).padStart(2, "0");
      }).join("");
      return (
        hex.slice(0, 8) +
        "-" +
        hex.slice(8, 12) +
        "-" +
        hex.slice(12, 16) +
        "-" +
        hex.slice(16, 20) +
        "-" +
        hex.slice(20)
      );
    }
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
      const r = (Math.random() * 16) | 0;
      const v = c === "x" ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });
  }

  /** @type {{ cmd: string, label: string, hint: string }[]} */
  const KB_CMD_MENU_ITEMS = [
    { cmd: "/help", label: "/help", hint: "Список команд" },
    { cmd: "/start", label: "/start", hint: "Начало работы и доступ" },
    { cmd: "/id", label: "/id", hint: "Ваши ID и привязки (без веб-сессий)" },
    { cmd: "/link", label: "/link", hint: "Код привязки другого канала" },
    { cmd: "/new", label: "/new", hint: "Новый диалог на сервере" },
    { cmd: "/clear_memory", label: "/clear_memory", hint: "Очистить память (с архивом)" },
    { cmd: "/tts status", label: "/tts status", hint: "Статус голосовых ответов (VK/TG)" },
    { cmd: "/tts on", label: "/tts on", hint: "Включить TTS" },
    { cmd: "/tts off", label: "/tts off", hint: "Выключить TTS" },
    { cmd: "/reg", label: "/reg", hint: "Регистрация" },
    { cmd: "/stop", label: "/stop", hint: "Остановить текущую задачу" },
    { cmd: "/status", label: "/status", hint: "Статус бота (владелец)" },
    { cmd: "/restart", label: "/restart", hint: "Перезапуск (владелец)" },
  ];

  let modelId = null;
  /** @type {object|null} */
  let kbCfgLastLoaded = null;
  let kbMarkdownHooksInstalled = false;

  function kbInitMarkdownLibs() {
    if (kbMarkdownHooksInstalled) {
      return;
    }
    if (typeof marked === "undefined" || typeof DOMPurify === "undefined") {
      return;
    }
    marked.setOptions({ gfm: true, breaks: true });
    DOMPurify.addHook("afterSanitizeAttributes", function (node) {
      if (node.tagName === "A" && node.hasAttribute("href")) {
        node.setAttribute("target", "_blank");
        node.setAttribute("rel", "noopener noreferrer");
      }
    });
    kbMarkdownHooksInstalled = true;
  }

  /**
   * @param {string} text
   * @returns {string|null} sanitized HTML or null if libs missing / parse error
   */
  function kbAssistantMarkdownToHtml(text) {
    kbInitMarkdownLibs();
    if (typeof marked === "undefined" || typeof DOMPurify === "undefined") {
      return null;
    }
    try {
      const raw = marked.parse(String(text || ""), { async: false });
      return DOMPurify.sanitize(raw);
    } catch (_e) {
      return null;
    }
  }
  /**
   * Pending attachments: either a browser File, or an already-uploaded multipart ref
   * (video/docs/large) so we never FileReader them later at send time.
   * @type {Array<File|{__kbUploaded:true,name:string,rows:object[]}>}
   */
  let pendingFiles = [];

  function getSessionId() {
    let id = localStorage.getItem(SESSION_KEY);
    if (!id) {
      id = kbRandomUuid();
      localStorage.setItem(SESSION_KEY, id);
    }
    return id;
  }

  function setSessionId(id) {
    localStorage.setItem(SESSION_KEY, id);
  }

  function setStatus(text) {
    statusEl.classList.remove("kb-status--busy");
    statusEl.textContent = text || "";
  }

  /** Status with spinner + optional percent (upload progress). */
  function setUploadProgressStatus(label, percent) {
    const pct =
      typeof percent === "number" && Number.isFinite(percent)
        ? Math.max(0, Math.min(100, Math.round(percent)))
        : null;
    const line = pct != null ? label + " " + pct + "%" : label;
    statusEl.classList.add("kb-status--busy");
    statusEl.replaceChildren();
    const spin = document.createElement("span");
    spin.className = "kb-status-spinner";
    spin.setAttribute("aria-hidden", "true");
    const text = document.createElement("span");
    text.className = "kb-status-text";
    text.textContent = line;
    statusEl.appendChild(spin);
    statusEl.appendChild(text);
  }

  const KB_CFG_AGENT_ORDER = ["model", "provider", "workspace", "anonymize", "shortMemory"];

  /** @type {Record<string,string>} */
  const KB_CFG_HINTS = {
    sendProgress: "Стриминг текста ответа в канал",
    sendToolHints: "Показывать вызовы инструментов",
    sendMaxRetries: "Повторы доставки в канал",
    anonymize: "Анонимизация PII перед отправкой в LLM",
    shortMemory: "Краткая память (только отмеченные факты)",
  };

  /** @type {Record<string,string>} */
  const KB_CFG_OTHER_SECTIONS = {
    api: "API-сервер (krabobot serve)",
    gateway: "Шлюз",
    tools: "Инструменты агента",
    tts: "Синтез речи (TTS)",
    stt: "Распознавание речи (STT)",
  };

  /** @type {Record<string,string>} */
  const KB_CHANNEL_TITLES = {
    telegram: "Telegram",
    vk: "ВКонтакте",
    email: "Почта",
  };

  /** Admin UI: channel keys → Russian labels (account links). api hidden — auto-linked. */
  /** @type {Record<string,string>} */
  const KB_ADMIN_CHANNEL_LABELS = {
    telegram: "Telegram",
    vk: "VK",
    email: "Email",
    cli: "Терминал (krabobot agent)",
  };

  /** Manual add-link / create-user channels (api sessions auto-link to owner). */
  const KB_ADMIN_MANUAL_CHANNELS = ["telegram", "vk", "email", "cli"];

  /** @type {Record<string,string>} */
  const KB_ADMIN_SENDER_HINTS = {
    telegram: "Telegram user ID (число)",
    vk: "VK user ID",
    email: "Адрес email",
    cli: "Локальный id (например user@host)",
  };

  function kbAdminChannelLabel(channel) {
    const ch = String(channel || "").trim();
    return KB_ADMIN_CHANNEL_LABELS[ch] || ch || "—";
  }

  function kbAdminSenderHint(channel) {
    const ch = String(channel || "").trim();
    return KB_ADMIN_SENDER_HINTS[ch] || "Идентификатор в канале";
  }

  function kbAdminParseAccount(acc) {
    const s = String(acc || "");
    const i = s.indexOf(":");
    if (i < 0) {
      return { channel: s, sender_id: "" };
    }
    return { channel: s.slice(0, i), sender_id: s.slice(i + 1) };
  }

  function kbAdminFillChannelSelect(selectEl, opts) {
    const includeEmpty = !!(opts && opts.includeEmpty);
    const emptyLabel = (opts && opts.emptyLabel) || "— выберите канал —";
    selectEl.replaceChildren();
    if (includeEmpty) {
      const o = document.createElement("option");
      o.value = "";
      o.textContent = emptyLabel;
      selectEl.appendChild(o);
    }
    KB_ADMIN_MANUAL_CHANNELS.forEach((ch) => {
      const o = document.createElement("option");
      o.value = ch;
      o.textContent = kbAdminChannelLabel(ch);
      selectEl.appendChild(o);
    });
  }

  function kbAdminBindSenderHint(channelSelect, senderInput, captionEl) {
    const sync = () => {
      const ch = channelSelect.value;
      const hint = ch ? kbAdminSenderHint(ch) : "Идентификатор в канале";
      senderInput.placeholder = ch ? hint : "Сначала выберите канал";
      if (captionEl) {
        captionEl.textContent = hint;
      }
    };
    channelSelect.addEventListener("change", sync);
    sync();
  }

  function kbAdminMakeRemoveLinkBtn(userId, account) {
    const rm = document.createElement("button");
    rm.type = "button";
    rm.className = "kb-btn kb-btn-secondary";
    rm.textContent = "Удалить";
    rm.addEventListener("click", async () => {
      const r = await kbApiFetch(
        "/v1/web/users/" + encodeURIComponent(userId) + "/links",
        {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ account: account }),
        }
      );
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        setAdminBanner("", (d.error && d.error.message) || "Не удалось удалить связь");
        return;
      }
      refreshAdminUsers();
    });
    return rm;
  }

  function kbAdminAppendLinkRow(listEl, userId, account) {
    const { channel, sender_id } = kbAdminParseAccount(account);
    const li = document.createElement("li");
    li.className = "kb-admin-link-row";
    const span = document.createElement("span");
    span.className = "kb-admin-link-text";
    const label = document.createElement("strong");
    label.textContent = kbAdminChannelLabel(channel);
    span.appendChild(label);
    span.appendChild(document.createTextNode(" · "));
    const code = document.createElement("code");
    code.textContent = sender_id || account;
    span.appendChild(code);
    li.appendChild(span);
    li.appendChild(kbAdminMakeRemoveLinkBtn(userId, account));
    listEl.appendChild(li);
  }

  function kbCfgFmtKey(key) {
    const h = KB_CFG_HINTS[key];
    if (h) {
      return h;
    }
    return key
      .replace(/([A-Z])/g, " $1")
      .replace(/^./, function (s) {
        return s.toUpperCase();
      })
      .trim();
  }

  function kbCfgSortedKeys(obj, priorityList) {
    const keys = Object.keys(obj || {});
    const head = [];
    for (const k of priorityList) {
      if (keys.includes(k)) {
        head.push(k);
      }
    }
    const tail = keys.filter(function (k) {
      return !priorityList.includes(k);
    });
    tail.sort();
    return head.concat(tail);
  }

  function kbCfgKeyLooksSecret(leafKey) {
    const n = leafKey.replace(/[^a-zA-Z0-9]/g, "").toLowerCase();
    if (n.length < 2) {
      return false;
    }
    if (n === "apikey" || n === "password" || n === "passwd" || n === "secret") {
      return true;
    }
    if (n.endsWith("apikey")) {
      return true;
    }
    if (n.endsWith("password") || n.endsWith("passwd")) {
      return true;
    }
    if (n.endsWith("tokens")) {
      return false;
    }
    if (n.endsWith("secret")) {
      return true;
    }
    if (n.endsWith("token")) {
      return true;
    }
    return false;
  }

  /** @param {string} path */
  function kbCfgPathLooksSecret(path) {
    const seg = path.split(".").pop() || "";
    return kbCfgKeyLooksSecret(seg);
  }

  /** @param {string} path */
  function kbCfgPathParts(path) {
    return path.split(".").filter(Boolean);
  }

  /** @param {Record<string,*>} root @param {string[]} parts */
  function kbCfgSetDeep(root, parts, val) {
    let o = root;
    for (let i = 0; i < parts.length - 1; i++) {
      const k = parts[i];
      if (
        o[k] === undefined ||
        o[k] === null ||
        typeof o[k] !== "object" ||
        Array.isArray(o[k])
      ) {
        o[k] = {};
      }
      o = o[k];
    }
    o[parts[parts.length - 1]] = val;
  }

  function kbCfgSectionsSnapshot(payload) {
    return {
      core: JSON.parse(JSON.stringify(payload.core || {})),
      channels: JSON.parse(JSON.stringify(payload.channels || {})),
      other: JSON.parse(JSON.stringify(payload.other || {})),
    };
  }

  /** @param {HTMLElement} root */
  function kbCfgCollectSectionsFromDom(root, snap) {
    root.querySelectorAll("select[data-kb-path]").forEach(function (sel) {
      const path = sel.dataset.kbPath;
      if (!path) {
        return;
      }
      kbCfgSetDeep(snap, kbCfgPathParts(path), sel.value);
    });
    root.querySelectorAll("input[data-kb-path]").forEach(function (inp) {
      const path = inp.dataset.kbPath;
      if (!path) {
        return;
      }
      if (inp.type === "checkbox") {
        kbCfgSetDeep(snap, kbCfgPathParts(path), inp.checked);
      } else if (inp.type === "number") {
        const raw = inp.value.trim();
        if (raw === "") {
          return;
        }
        const n = Number(raw);
        kbCfgSetDeep(snap, kbCfgPathParts(path), Number.isFinite(n) ? n : raw);
      } else if (inp.type === "password") {
        const v = inp.value.trim();
        if (v === "") {
          return;
        }
        kbCfgSetDeep(snap, kbCfgPathParts(path), v);
      } else {
        kbCfgSetDeep(snap, kbCfgPathParts(path), inp.value);
      }
    });
    root.querySelectorAll("textarea[data-kb-path]").forEach(function (ta) {
      const path = ta.dataset.kbPath;
      if (!path) {
        return;
      }
      kbCfgSetDeep(snap, kbCfgPathParts(path), ta.value);
    });
    return snap;
  }

  function kbCfgCollectSectionsFromForm() {
    const root = document.getElementById("kb-cfg-root");
    if (!root || !kbCfgLastLoaded) {
      return null;
    }
    const snap = kbCfgSectionsSnapshot(kbCfgLastLoaded);
    kbCfgCollectSectionsFromDom(root, snap);
    return snap;
  }

  async function kbCfgRefreshBackupSelect() {
    const sel = document.getElementById("kb-cfg-backups");
    if (!sel) {
      return;
    }
    const cur = sel.value;
    sel.innerHTML = "";
    const def = document.createElement("option");
    def.value = "";
    def.textContent = "— выберите бэкап —";
    sel.appendChild(def);
    try {
      const r = await kbApiFetch("/v1/web/config/backups");
      const data = await r.json().catch(function () {
        return {};
      });
      if (!r.ok) {
        return;
      }
      const rows = Array.isArray(data.data) ? data.data : [];
      for (const row of rows) {
        const name = row && row.name;
        if (!name) {
          continue;
        }
        const o = document.createElement("option");
        o.value = name;
        o.textContent = name;
        sel.appendChild(o);
      }
    } catch (_e) {
      /* ignore */
    }
    if (cur && Array.from(sel.options).some((o) => o.value === cur)) {
      sel.value = cur;
    }
  }

  /** @param {HTMLElement} cell @param {string} path @param {*} val */
  function kbCfgAppendEditableValue(cell, path, val) {
    const tailProp = path.split(".").pop() || path;

    if (val === null || val === undefined) {
      cell.appendChild(document.createTextNode("—"));
      return;
    }

    if (typeof val === "boolean") {
      const wrap = document.createElement("label");
      wrap.className = "kb-cfg-cb-wrap";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = !!val;
      cb.dataset.kbPath = path;
      cb.setAttribute("aria-label", kbCfgFmtKey(tailProp));
      wrap.appendChild(cb);
      wrap.appendChild(document.createTextNode("да / нет"));
      cell.appendChild(wrap);
      return;
    }

    if (typeof val === "number" && Number.isFinite(val)) {
      const inp = document.createElement("input");
      inp.type = "number";
      inp.className = "kb-cfg-input";
      inp.step = "any";
      inp.dataset.kbPath = path;
      inp.value = String(val);
      inp.setAttribute("aria-label", kbCfgFmtKey(tailProp));
      cell.appendChild(inp);
      return;
    }

    if (typeof val !== "object") {
      let sval = String(val);
      const isBulletMask = /^[\u2022\u00b7\u2027]+$/.test(sval.trim());
      const secret = kbCfgPathLooksSecret(path) || (isBulletMask && kbCfgPathLooksSecret(path));

      if (secret) {
        const inp = document.createElement("input");
        inp.type = "password";
        inp.autocomplete = "off";
        inp.className = "kb-cfg-input";
        inp.dataset.kbPath = path;
        inp.value = "";
        inp.placeholder = isBulletMask || sval.includes("•") ? "без изменений" : "новое значение";
        inp.setAttribute("aria-label", kbCfgFmtKey(tailProp));
        cell.appendChild(inp);
        const hint = document.createElement("p");
        hint.className = "kb-cfg-hint-secret";
        hint.textContent =
          "Пустое поле — не менять сохранённый ключ. Новое значение — только если хотите заменить.";
        cell.appendChild(hint);
        return;
      }

      if (sval.length > 100 || /\r|\n/.test(sval)) {
        const ta = document.createElement("textarea");
        ta.className = "kb-cfg-textarea";
        ta.rows = Math.min(10, Math.max(3, Math.ceil(sval.length / 80)));
        ta.dataset.kbPath = path;
        ta.value = sval;
        ta.setAttribute("aria-label", kbCfgFmtKey(tailProp));
        cell.appendChild(ta);
        return;
      }

      const inp = document.createElement("input");
      inp.type = "text";
      inp.className = "kb-cfg-input";
      inp.dataset.kbPath = path;
      inp.value = sval;
      inp.setAttribute("aria-label", kbCfgFmtKey(tailProp));
      cell.appendChild(inp);
      return;
    }

    if (Array.isArray(val)) {
      if (val.length === 0) {
        cell.appendChild(document.createTextNode("—"));
        return;
      }
      if (
        val.every(function (x) {
          return x === null || ["string", "number", "boolean"].includes(typeof x);
        })
      ) {
        const ul = document.createElement("ul");
        ul.className = "kb-cfg-array-simple kb-cfg-readonly-note";
        for (const item of val) {
          const li = document.createElement("li");
          li.textContent = String(item);
          ul.appendChild(li);
        }
        cell.appendChild(ul);
      } else {
        const pre = document.createElement("pre");
        pre.className = "kb-cfg-pre";
        pre.textContent = JSON.stringify(val, null, 2);
        cell.appendChild(pre);
      }
      const note = document.createElement("p");
      note.className = "kb-cfg-muted";
      note.textContent =
        "Этот массив только для просмотра — при необходимости правьте config.json.";
      cell.appendChild(note);
      return;
    }

    const inner = document.createElement("div");
    inner.className = "kb-cfg-nested";
    kbCfgRenderObjectAt(inner, val, path);
    cell.appendChild(inner);
  }

  /** @param {Record<string,*>} obj @param {string} prefix dotted path matching GET shapes */
  function kbCfgRenderObjectAt(container, obj, prefix) {
    const keys = Object.keys(obj || {});
    if (keys.length === 0) {
      const p = document.createElement("p");
      p.className = "kb-cfg-empty";
      p.textContent = "пусто";
      container.appendChild(p);
      return;
    }
    for (const k of keys.slice().sort()) {
      const row = document.createElement("div");
      row.className = "kb-cfg-kv";
      const dt = document.createElement("div");
      dt.className = "kb-cfg-k";
      dt.textContent = kbCfgFmtKey(k);
      const dd = document.createElement("div");
      dd.className = "kb-cfg-v";
      const fullPath = prefix ? `${prefix}.${k}` : k;
      kbCfgAppendEditableValue(dd, fullPath, obj[k]);
      row.appendChild(dt);
      row.appendChild(dd);
      container.appendChild(row);
    }
  }

  /** @param {Record<string,*>} obj */
  function kbCfgRenderObjectOrderedAt(container, obj, priorityList, prefix) {
    const keys = kbCfgSortedKeys(obj, priorityList || []);
    if (keys.length === 0) {
      const p = document.createElement("p");
      p.className = "kb-cfg-empty";
      p.textContent = "пусто";
      container.appendChild(p);
      return;
    }
    for (const k of keys) {
      const row = document.createElement("div");
      row.className = "kb-cfg-kv";
      const dt = document.createElement("div");
      dt.className = "kb-cfg-k";
      dt.textContent = kbCfgFmtKey(k);
      const dd = document.createElement("div");
      dd.className = "kb-cfg-v";
      const fullPath = prefix ? `${prefix}.${k}` : k;
      kbCfgAppendEditableValue(dd, fullPath, obj[k]);
      row.appendChild(dt);
      row.appendChild(dd);
      container.appendChild(row);
    }
  }

  /** @param {*} val */
  function kbCfgAppendReadonlyFallback(cell, val) {
    if (val === null || val === undefined) {
      cell.appendChild(document.createTextNode("—"));
      return;
    }
    if (typeof val === "boolean") {
      cell.appendChild(document.createTextNode(val ? "да" : "нет"));
      return;
    }
    if (typeof val !== "object") {
      cell.appendChild(document.createTextNode(String(val)));
      return;
    }
    const pre = document.createElement("pre");
    pre.className = "kb-cfg-pre";
    pre.textContent = JSON.stringify(val, null, 2);
    cell.appendChild(pre);
  }

  /** @param {HTMLElement} wrap */
  function kbCfgProviderSelect(wrap, providerChoices, currentProvider) {
    const block = document.createElement("div");
    block.className = "kb-cfg-block";
    const ht = document.createElement("h3");
    ht.className = "kb-cfg-block-h";
    ht.textContent = "Выбор провайдера LLM";
    block.appendChild(ht);
    const p = document.createElement("p");
    p.className = "kb-cfg-muted";
    p.textContent =
      "agents.defaults.provider (для вашей модели). После смены нажмите «Сохранить» вверху.";
    block.appendChild(p);
    const row = document.createElement("div");
    row.className = "kb-cfg-kv";
    const dt = document.createElement("div");
    dt.className = "kb-cfg-k";
    dt.textContent = "Текущее значение";
    const dd = document.createElement("div");
    dd.className = "kb-cfg-v";
    const sel = document.createElement("select");
    sel.className = "kb-cfg-select";
    sel.dataset.kbPath = "core.agents.defaults.provider";
    sel.setAttribute("aria-label", "Провайдер LLM");
    const seen = new Set();
    for (const opt of providerChoices || []) {
      seen.add(opt.value);
      const o = document.createElement("option");
      o.value = opt.value;
      o.textContent = opt.label || opt.value;
      if (opt.value === currentProvider) {
        o.selected = true;
      }
      sel.appendChild(o);
    }
    if (currentProvider && !seen.has(currentProvider)) {
      const o = document.createElement("option");
      o.value = currentProvider;
      o.textContent = currentProvider;
      o.selected = true;
      sel.appendChild(o);
    }
    dd.appendChild(sel);
    row.appendChild(dt);
    row.appendChild(dd);
    block.appendChild(row);
    wrap.appendChild(block);
  }

  /** @param {HTMLElement} container */
  function kbCfgRenderCore(container, payload) {
    container.innerHTML = "";
    const core = payload.core || {};
    const defaults = core.agents && core.agents.defaults ? core.agents.defaults : {};
    const defsForRows = Object.assign({}, defaults);
    delete defsForRows.provider;

    kbCfgProviderSelect(container, payload.providerChoices, defaults.provider || "");

    const agentBlock = document.createElement("div");
    agentBlock.className = "kb-cfg-block";
    const ha = document.createElement("h3");
    ha.className = "kb-cfg-block-h";
    ha.textContent = "Параметры агента (agents.defaults)";
    agentBlock.appendChild(ha);
    const innerAgent = document.createElement("div");
    innerAgent.className = "kb-cfg-rows";
    kbCfgRenderObjectOrderedAt(
      innerAgent,
      defsForRows,
      KB_CFG_AGENT_ORDER,
      "core.agents.defaults",
    );
    agentBlock.appendChild(innerAgent);
    container.appendChild(agentBlock);

    const prov = core.providers || {};
    const provWrap = document.createElement("div");
    provWrap.className = "kb-cfg-block";
    const hp = document.createElement("h3");
    hp.className = "kb-cfg-block-h";
    hp.textContent = "Настройка провайдеров (providers.*)";
    provWrap.appendChild(hp);
    const provKeys = Object.keys(prov).sort();
    for (const name of provKeys) {
      const sub = document.createElement("div");
      sub.className = "kb-cfg-subblock";
      const sh = document.createElement("h4");
      sh.className = "kb-cfg-subblock-h";
      sh.textContent = name;
      sub.appendChild(sh);
      const body = document.createElement("div");
      body.className = "kb-cfg-rows";
      const pconf = prov[name];
      if (pconf && typeof pconf === "object" && !Array.isArray(pconf)) {
        kbCfgRenderObjectOrderedAt(
          body,
          pconf,
          ["apiKey", "apiBase", "extraHeaders", "useMaxCompletionTokens"],
          `core.providers.${name}`,
        );
      } else {
        kbCfgAppendReadonlyFallback(body, pconf);
      }
      sub.appendChild(body);
      provWrap.appendChild(sub);
    }
    container.appendChild(provWrap);
  }

  /** @param {HTMLElement} container */
  function kbCfgRenderChannels(container, payload) {
    container.innerHTML = "";
    const ch = payload.channels || {};
    const common = ch.common || {};
    const named = ch.named || {};

    if (Object.keys(common).length) {
      const block = document.createElement("div");
      block.className = "kb-cfg-block";
      const h = document.createElement("h3");
      h.className = "kb-cfg-block-h";
      h.textContent = "Общие параметры каналов";
      block.appendChild(h);
      const rows = document.createElement("div");
      rows.className = "kb-cfg-rows";
      kbCfgRenderObjectOrderedAt(
        rows,
        common,
        ["sendProgress", "sendToolHints", "sendMaxRetries"],
        "channels.common",
      );
      block.appendChild(rows);
      container.appendChild(block);
    }

    const chNames = Object.keys(named).sort();
    if (chNames.length === 0 && Object.keys(common).length === 0) {
      const p = document.createElement("p");
      p.className = "kb-cfg-empty";
      p.textContent = "В конфигурации не найдено секций каналов.";
      container.appendChild(p);
      return;
    }

    for (const name of chNames) {
      const block = document.createElement("div");
      block.className = "kb-cfg-block";
      const h = document.createElement("h3");
      h.className = "kb-cfg-block-h";
      h.textContent = KB_CHANNEL_TITLES[name] || name;
      block.appendChild(h);
      const rows = document.createElement("div");
      rows.className = "kb-cfg-rows";
      const conf = named[name];
      if (conf && typeof conf === "object" && !Array.isArray(conf)) {
        kbCfgRenderObjectOrderedAt(
          rows,
          conf,
          ["enabled", "streaming", "token"],
          `channels.named.${name}`,
        );
      } else {
        kbCfgAppendReadonlyFallback(rows, conf);
      }
      block.appendChild(rows);
      container.appendChild(block);
    }
  }

  /** @param {HTMLElement} container */
  function kbCfgRenderOther(container, payload) {
    container.innerHTML = "";
    const other = payload.other || {};
    const order = ["api", "gateway", "tools", "tts", "stt"];
    let any = false;
    for (const key of order) {
      if (!(key in other)) {
        continue;
      }
      any = true;
      const block = document.createElement("div");
      block.className = "kb-cfg-block";
      const h = document.createElement("h3");
      h.className = "kb-cfg-block-h";
      h.textContent = KB_CFG_OTHER_SECTIONS[key] || key;
      block.appendChild(h);
      const rows = document.createElement("div");
      rows.className = "kb-cfg-rows";
      const obj = other[key];
      if (obj && typeof obj === "object" && !Array.isArray(obj)) {
        if (key === "tools") {
          const tk = kbCfgSortedKeys(obj, [
            "web",
            "exec",
            "restrictToWorkspace",
            "multiUser",
            "mcpServers",
          ]);
          for (const subKey of tk) {
            const sub = document.createElement("div");
            sub.className = "kb-cfg-subblock";
            const sh = document.createElement("h4");
            sh.className = "kb-cfg-subblock-h";
            sh.textContent = kbCfgFmtKey(subKey);
            sub.appendChild(sh);
            const body = document.createElement("div");
            body.className = "kb-cfg-rows";
            const v = obj[subKey];
            if (v && typeof v === "object" && !Array.isArray(v)) {
              kbCfgRenderObjectAt(body, v, `other.tools.${subKey}`);
            } else {
              kbCfgAppendReadonlyFallback(body, v);
            }
            sub.appendChild(body);
            rows.appendChild(sub);
          }
        } else if (key === "gateway") {
          kbCfgRenderObjectOrderedAt(rows, obj, ["host", "port", "heartbeat"], `other.${key}`);
        } else {
          kbCfgRenderObjectAt(rows, obj, `other.${key}`);
        }
      } else {
        kbCfgAppendReadonlyFallback(rows, obj);
      }
      block.appendChild(rows);
      container.appendChild(block);
    }
    if (!any) {
      const p = document.createElement("p");
      p.className = "kb-cfg-empty";
      p.textContent = "Нет дополнительных секций.";
      container.appendChild(p);
    }
  }

  async function loadWebConfig() {
    const errEl = document.getElementById("kb-cfg-error");
    const rootEl = document.getElementById("kb-cfg-root");
    const pathEl = document.getElementById("kb-cfg-path");
    const mainEl = document.getElementById("kb-cfg-main");
    const chEl = document.getElementById("kb-cfg-channels");
    const otEl = document.getElementById("kb-cfg-other");
    if (!errEl || !rootEl || !pathEl || !mainEl || !chEl || !otEl) {
      return;
    }
    errEl.hidden = true;
    errEl.textContent = "";
    try {
      const r = await kbApiFetch("/v1/web/config");
      const data = await r.json().catch(function () {
        return {};
      });
      if (!r.ok) {
        rootEl.hidden = true;
        kbCfgLastLoaded = null;
        const msg =
          (data.error && data.error.message) ||
          data.detail ||
          "Не удалось загрузить конфигурацию.";
        errEl.textContent =
          msg + (r.status === 404 ? " Запустите krabobot с существующим config.json." : "");
        errEl.hidden = false;
        return;
      }
      pathEl.textContent = data.path || "";
      kbCfgLastLoaded = data;
      const okBanner = document.getElementById("kb-cfg-ok");
      if (okBanner) {
        okBanner.hidden = true;
        okBanner.textContent = "";
      }
      kbCfgRenderCore(mainEl, data);
      kbCfgRenderChannels(chEl, data);
      kbCfgRenderOther(otEl, data);
      rootEl.hidden = false;
      await kbCfgRefreshBackupSelect();
    } catch (_e) {
      rootEl.hidden = true;
      errEl.textContent = "Ошибка сети при загрузке конфигурации.";
      errEl.hidden = false;
    }
  }

  function closeCmdMenu() {
    if (cmdMenuDropdown) {
      cmdMenuDropdown.hidden = true;
    }
    if (cmdMenuBtn) {
      cmdMenuBtn.setAttribute("aria-expanded", "false");
    }
  }

  function toggleCmdMenu() {
    if (!cmdMenuDropdown || !cmdMenuBtn) {
      return;
    }
    closeMenu();
    if (cmdMenuDropdown.hidden) {
      cmdMenuDropdown.hidden = false;
      cmdMenuBtn.setAttribute("aria-expanded", "true");
    } else {
      closeCmdMenu();
    }
  }

  function closeMenu() {
    if (menuDropdown) {
      menuDropdown.hidden = true;
    }
    if (menuBtn) {
      menuBtn.setAttribute("aria-expanded", "false");
    }
  }

  function openMenu() {
    if (menuDropdown) {
      menuDropdown.hidden = false;
    }
    if (menuBtn) {
      menuBtn.setAttribute("aria-expanded", "true");
    }
  }

  function toggleMenu() {
    if (!menuDropdown || !menuBtn) {
      return;
    }
    if (menuDropdown.hidden) {
      closeCmdMenu();
      openMenu();
    } else {
      closeMenu();
    }
  }

  function switchTab(which) {
    const isChat = which === "chat";
    const isSettings = which === "settings";
    const isAdmin = which === "admin";
    if (viewChat) {
      viewChat.classList.toggle("kb-view--hidden", !isChat);
    }
    if (viewSettings) {
      viewSettings.classList.toggle("kb-view--hidden", !isSettings);
      viewSettings.hidden = !isSettings;
    }
    if (viewAdmin) {
      viewAdmin.classList.toggle("kb-view--hidden", !isAdmin);
      viewAdmin.hidden = !isAdmin;
    }
    if (navChat) {
      navChat.classList.toggle("kb-dropdown-item--active", isChat);
    }
    if (navSettings) {
      navSettings.classList.toggle("kb-dropdown-item--active", isSettings);
    }
    if (navAdmin) {
      navAdmin.classList.toggle("kb-dropdown-item--active", isAdmin);
    }
    if (isSettings) {
      refreshSettingsPanel();
    }
    if (isAdmin) {
      refreshAdminUsers();
    }
    closeMenu();
    closeCmdMenu();
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
        if (r.ok && typeof j.maxUploadHardBytes === "number" && j.maxUploadHardBytes > 0) {
          maxAttachHardBytes = j.maxUploadHardBytes;
        } else if (r.ok && typeof j.maxUploadHardMb === "number" && j.maxUploadHardMb > 0) {
          maxAttachHardBytes = j.maxUploadHardMb * 1024 * 1024;
        }
      } catch {
        elHealth.textContent = "недоступно";
      }
    }
    await loadWebConfig();
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
    const s = String(text ?? "");
    if (kind === "error" || role === "user") {
      body.textContent = s;
    } else {
      const html = kbAssistantMarkdownToHtml(s);
      if (html !== null) {
        body.classList.add("kb-msg-body--md");
        body.innerHTML = html;
      } else {
        body.textContent = s;
      }
    }
    wrap.appendChild(label);
    wrap.appendChild(body);
    logEl.appendChild(wrap);
    logEl.scrollTop = logEl.scrollHeight;
  }

  /** Soft: ≤50 MiB — usual attach (multipart or inline). Above — folder stream upload. */
  const MAX_ATTACH_SOFT_BYTES = 50 * 1024 * 1024;
  /** Hard max for large folder uploads; refreshed from GET /health when available. */
  let maxAttachHardBytes = 2048 * 1024 * 1024;

  function formatSizeMb(n) {
    return (Number(n) / (1024 * 1024)).toFixed(0);
  }

  function isUploadedRef(item) {
    return !!(item && item.__kbUploaded === true);
  }

  function attachDisplayName(item) {
    if (isUploadedRef(item)) {
      return item.name || "файл";
    }
    return (item && item.name) || "файл";
  }

  /** Browser NotReadableError / locked OneDrive-Teams file → clear Russian text. */
  function friendlyAttachError(err, fileName) {
    const name = fileName || "файл";
    const raw = err && (err.message || err.name) ? String(err.message || err.name) : String(err || "");
    const isNotReadable =
      (err && err.name === "NotReadableError") ||
      /could not be read/i.test(raw) ||
      /NotReadableError/i.test(raw) ||
      /permission problems that have occurred after a reference/i.test(raw);
    if (isNotReadable) {
      return (
        "Не удалось прочитать «" +
        name +
        "». Файл, скорее всего, ещё записывается или заблокирован (Teams/OneDrive). " +
        "Дождитесь окончания записи/синхронизации и выберите файл снова. " +
        "Видео в чат уходит как обычная загрузка файла (без чтения в память браузера)."
      );
    }
    return raw || "Ошибка чтения файла «" + name + "»";
  }

  function readAsBase64(file) {
    return new Promise((resolve, reject) => {
      const fr = new FileReader();
      fr.onload = () => {
        const s = String(fr.result || "");
        const i = s.indexOf(",");
        resolve(i >= 0 ? s.slice(i + 1) : s);
      };
      fr.onerror = () => {
        const err = fr.error;
        const name = file && file.name ? file.name : "файл";
        reject(new Error(friendlyAttachError(err || new Error("read failed"), name)));
      };
      fr.readAsDataURL(file);
    });
  }

  function assertAttachHardMax(file) {
    if (!file || typeof file.size !== "number") {
      return;
    }
    if (file.size > maxAttachHardBytes) {
      throw new Error(
        "Файл «" +
          (file.name || "без имени") +
          "» слишком большой (" +
          formatSizeMb(file.size) +
          " МБ). Максимум — " +
          formatSizeMb(maxAttachHardBytes) +
          " МБ. Уменьшите файл или поднимите api.maxUploadMb в config.json."
      );
    }
  }

  function isLargeAttach(file) {
    return file && typeof file.size === "number" && file.size > MAX_ATTACH_SOFT_BYTES;
  }

  function isVideoFile(file) {
    const mt = (file.type || "").toLowerCase();
    const name = file.name || "";
    // Extension wins: Teams/OneDrive often mislabel .mp4 as audio/* or octet-stream.
    if (/\.(mp4|m4v|mov|mkv|avi|webm)$/i.test(name)) {
      // Mic audio/webm stays audio; only bare/unknown .webm counts as video above.
      if (/\.webm$/i.test(name) && mt.startsWith("audio/")) {
        return false;
      }
      return true;
    }
    return mt.startsWith("video/");
  }

  /** Large / binary attachments go via multipart — not FileReader base64 in JSON. */
  function needsMultipartUpload(file) {
    if (isUploadedRef(file)) {
      return false;
    }
    const mt = (file.type || "").toLowerCase();
    const name = file.name || "";
    if (isVideoFile(file)) {
      return true;
    }
    if (mt.startsWith("image/")) {
      return false;
    }
    if (mt.startsWith("audio/") || /\.(mp3|wav|ogg|m4a|flac)$/i.test(name)) {
      return false;
    }
    if (
      mt.startsWith("text/") ||
      /\.(txt|md|csv|json|xml|yaml|yml|log|ini|env)$/i.test(name)
    ) {
      return false;
    }
    return true;
  }

  /**
   * POST FormData with upload progress (XHR). Returns { ok, status, data }.
   * @param {(loaded: number, total: number) => void} [onProgress]
   */
  function postFormDataWithProgress(url, formData, onProgress) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", url);
      xhr.withCredentials = true;
      const tok = getAuthToken();
      if (tok) {
        xhr.setRequestHeader("Authorization", "Bearer " + tok);
      }
      xhr.upload.onprogress = (ev) => {
        if (typeof onProgress === "function") {
          onProgress(ev.loaded, ev.lengthComputable ? ev.total : 0);
        }
      };
      xhr.onload = () => {
        let data = {};
        try {
          data = xhr.responseText ? JSON.parse(xhr.responseText) : {};
        } catch (_e) {
          data = {};
        }
        resolve({
          ok: xhr.status >= 200 && xhr.status < 300,
          status: xhr.status,
          data: data,
        });
      };
      xhr.onerror = () => reject(new Error("NetworkError"));
      xhr.onabort = () => reject(new Error("Aborted"));
      xhr.send(formData);
    });
  }

  async function uploadFilesMultipart(files, opts) {
    if (!files.length) {
      return [];
    }
    for (const f of files) {
      assertAttachHardMax(f);
    }
    const large = files.some(isLargeAttach);
    const statusLabel = large
      ? "Загрузка большого файла…"
      : (opts && opts.status) || "Загрузка файла…";
    setUploadProgressStatus(statusLabel, 0);
    const fd = new FormData();
    fd.append("session_id", getSessionId());
    for (const f of files) {
      // Pass File/Blob directly — never FileReader / base64 into JS memory.
      fd.append("files", f, f.name || "file.bin");
    }
    let r;
    try {
      r = await postFormDataWithProgress("/v1/web/uploads", fd, (loaded, total) => {
        if (total > 0) {
          setUploadProgressStatus(statusLabel, (loaded / total) * 100);
        } else {
          setUploadProgressStatus(statusLabel, null);
        }
      });
    } catch (err) {
      setStatus("");
      const n = files[0] && files[0].name ? files[0].name : "файл";
      throw new Error(friendlyAttachError(err, n));
    }
    const data = r.data && typeof r.data === "object" ? r.data : {};
    if (!r.ok) {
      setStatus("");
      const msg =
        data?.error?.message ||
        (typeof data === "object" ? JSON.stringify(data) : String(data));
      throw new Error(msg || "Загрузка файла: HTTP " + r.status);
    }
    const rows = Array.isArray(data.data) ? data.data : [];
    if (!rows.length) {
      setStatus("");
      throw new Error("Сервер не вернул сохранённые файлы");
    }
    return rows;
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
   * ≤50 MiB: images/audio/text stay inline (base64); video/docs via multipart.
   * >50 MiB: stream multipart to workspace folder (no FileReader), then path notes.
   * Pre-uploaded refs (from attach-time multipart) only add path notes — no re-read.
   */
  async function buildContentPayload(text, files) {
    const parts = [];
    const trim = (text || "").trim();
    const multipartFiles = [];
    const inlineFiles = [];

    for (const item of files) {
      if (isUploadedRef(item)) {
        for (const u of item.rows || []) {
          const p = u.path || u.saved_as || u.filename;
          parts.push({
            type: "text",
            text: "[Файл сохранён в workspace: " + p + "]",
          });
        }
        continue;
      }
      const file = item;
      assertAttachHardMax(file);
      // Large / video / binary: never FileReader/base64 — FormData only.
      if (isLargeAttach(file) || needsMultipartUpload(file)) {
        multipartFiles.push(file);
      } else {
        inlineFiles.push(file);
      }
    }

    if (multipartFiles.length) {
      const uploaded = await uploadFilesMultipart(multipartFiles);
      for (const u of uploaded) {
        const p = u.path || u.saved_as || u.filename;
        parts.push({
          type: "text",
          text: "[Файл сохранён в workspace: " + p + "]",
        });
      }
    }

    for (const file of inlineFiles) {
      const mt = file.type || "";
      const name = file.name || "file";

      try {
        if (mt.startsWith("image/")) {
          const b64 = await readAsBase64(file);
          parts.push({
            type: "image_url",
            image_url: { url: `data:${mt};base64,${b64}` },
          });
          continue;
        }

        if (mt.startsWith("audio/") || /\.(mp3|wav|ogg|webm|m4a|flac)$/i.test(name)) {
          // Never inline true video containers (e.g. mislabeled .mp4).
          if (isVideoFile(file)) {
            const uploaded = await uploadFilesMultipart([file]);
            for (const u of uploaded) {
              parts.push({
                type: "text",
                text: "[Файл сохранён в workspace: " + (u.path || u.filename) + "]",
              });
            }
            continue;
          }
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

        // Fallback (should be rare — needsMultipartUpload covers the rest)
        const uploaded = await uploadFilesMultipart([file]);
        for (const u of uploaded) {
          parts.push({
            type: "text",
            text: "[Файл сохранён в workspace: " + (u.path || u.filename) + "]",
          });
        }
      } catch (err) {
        throw new Error(friendlyAttachError(err, name));
      }
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
    const r = await kbApiFetch("/v1/models");
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      if (r.status === 401) {
        await kbHandleUnauthorized();
      }
      throw new Error(kbHttpErrorMessage(r.status, data, "GET /v1/models: " + r.status));
    }
    const id = data?.data?.[0]?.id;
    if (!id) {
      throw new Error("Нет модели в ответе /v1/models");
    }
    return id;
  }

  async function fetchSessionList() {
    const r = await kbApiFetch("/v1/web/sessions");
    if (!r.ok) {
      throw new Error("Список диалогов: " + r.status);
    }
    const data = await r.json();
    return Array.isArray(data.data) ? data.data : [];
  }

  async function deleteSessionOnServer(id) {
    const r = await kbApiFetch("/v1/web/sessions/" + encodeURIComponent(id), {
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
    const r = await kbApiFetch(
      "/v1/web/sessions/" + encodeURIComponent(id) + "/messages"
    );
    if (!r.ok) {
      throw new Error("История: " + r.status);
    }
    const data = await r.json();
    return Array.isArray(data.data) ? data.data : [];
  }

  /** @param {{ id?: string, updated_at?: string, created_at?: string }} row */
  function kbFormatSessionListLabel(row) {
    const ts = row.updated_at || row.created_at;
    if (ts) {
      const d = new Date(ts);
      if (!Number.isNaN(d.getTime())) {
        return d.toLocaleString("ru-RU", {
          day: "2-digit",
          month: "2-digit",
          year: "numeric",
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        });
      }
      const s = String(ts);
      return s.length > 19 ? s.slice(0, 19).replace("T", " ") : s.replace("T", " ");
    }
    return "—";
  }

  function applySidebarCollapsed(collapsed) {
    if (!layoutEl || !sidebarEl) {
      return;
    }
    layoutEl.classList.toggle("kb-layout--sidebar-collapsed", collapsed);
    const expand = collapsed ? "Развернуть список диалогов" : "Свернуть список диалогов";
    if (sidebarToggle) {
      sidebarToggle.setAttribute("aria-expanded", String(!collapsed));
      sidebarToggle.setAttribute("aria-label", expand);
      sidebarToggle.title = expand;
      sidebarToggle.hidden = collapsed;
    }
    if (sidebarExpand) {
      sidebarExpand.hidden = !collapsed;
      sidebarExpand.setAttribute("aria-expanded", String(!collapsed));
    }
    sidebarEl.setAttribute("aria-hidden", collapsed ? "true" : "false");
    try {
      localStorage.setItem(SIDEBAR_COLLAPSED_KEY, collapsed ? "1" : "0");
    } catch (_e) {
      /* ignore */
    }
  }

  if (layoutEl && sidebarEl) {
    try {
      if (localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "1") {
        applySidebarCollapsed(true);
      }
    } catch (_e) {
      /* ignore */
    }
    if (sidebarToggle) {
      sidebarToggle.addEventListener("click", () => {
        applySidebarCollapsed(true);
      });
    }
    if (sidebarExpand) {
      sidebarExpand.addEventListener("click", () => {
        applySidebarCollapsed(false);
      });
    }
  }

  function renderSessionList(rows, currentId) {
    sessionListEl.innerHTML = "";
    const labelCounts = {};
    for (const row of rows) {
      const l = kbFormatSessionListLabel(row);
      labelCounts[l] = (labelCounts[l] || 0) + 1;
    }
    for (const row of rows) {
      const id = row.id;
      const li = document.createElement("li");
      li.className = "kb-sess" + (id === currentId ? " kb-sess--active" : "");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "kb-sess-btn";
      let label = kbFormatSessionListLabel(row);
      if (labelCounts[label] > 1) {
        label = label + " · " + String(id || "").slice(0, 8);
      }
      btn.textContent = label;
      const preview = (row.preview || "").trim();
      btn.title = String(id || "") + (preview ? "\n" + preview.slice(0, 220) : "");
      const ariaPrev = preview ? ". Последнее: " + preview.slice(0, 100) : "";
      btn.setAttribute("aria-label", "Открыть диалог " + label + ariaPrev);
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
            const nid = kbRandomUuid();
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
    const r = await kbApiFetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      if (r.status === 401) {
        await kbHandleUnauthorized();
      }
      throw new Error(
        kbHttpErrorMessage(
          r.status,
          data,
          typeof data === "object" ? JSON.stringify(data) : String(data)
        )
      );
    }
    const content = data?.choices?.[0]?.message?.content;
    if (typeof content !== "string") {
      throw new Error("Неверный ответ API");
    }
    return content;
  }

  async function postChatTurn(displayText, apiText, filesSnapshot) {
    appendMessage("user", displayText);
    sendBtn.disabled = true;
    setStatus("Запрос…");
    try {
      const reply = await sendMessage(apiText, filesSnapshot);
      appendMessage("assistant", reply);
      setStatus("");
      await refreshSessions();
    } catch (err) {
      const hint =
        filesSnapshot && filesSnapshot.length
          ? attachDisplayName(filesSnapshot[0])
          : "";
      appendMessage("assistant", friendlyAttachError(err, hint), "error");
      setStatus("");
    } finally {
      sendBtn.disabled = false;
      inputEl.focus();
    }
  }

  function populateCmdMenu() {
    if (!cmdMenuDropdown) {
      return;
    }
    cmdMenuDropdown.innerHTML = "";
    for (const item of KB_CMD_MENU_ITEMS) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "kb-cmd-dropdown-item";
      b.setAttribute("role", "menuitem");
      const lab = document.createElement("span");
      lab.className = "kb-cmd-item-label";
      lab.textContent = item.label;
      const hint = document.createElement("span");
      hint.className = "kb-cmd-item-hint";
      hint.textContent = item.hint;
      b.appendChild(lab);
      b.appendChild(hint);
      b.addEventListener("click", async function (ev) {
        ev.preventDefault();
        ev.stopPropagation();
        const cmd = item.cmd;
        closeCmdMenu();
        if (!modelId) {
          return;
        }
        await postChatTurn(cmd, cmd, []);
      });
      cmdMenuDropdown.appendChild(b);
    }
  }

  populateCmdMenu();

  formEl.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = inputEl.value.trim();
    if (!text && pendingFiles.length === 0) {
      return;
    }
    if (!modelId) {
      appendMessage(
        "assistant",
        "Модель ещё не загружена или сессия не авторизована. Обновите страницу или войдите снова.",
        "error"
      );
      return;
    }

    const filesSnapshot = pendingFiles.slice();
    pendingFiles = [];
    renderAttachments();

    const userVisible =
      text +
      (filesSnapshot.length
        ? "\n" + filesSnapshot.map((f) => "📎 " + attachDisplayName(f)).join("\n")
        : "");
    inputEl.value = "";
    await postChatTurn(userVisible || "(вложения)", text, filesSnapshot);
  });

  newSessionBtn.addEventListener("click", async () => {
    const nid = kbRandomUuid();
    setSessionId(nid);
    logEl.innerHTML = "";
    setStatus("Новый диалог");
    await refreshSessions();
    setTimeout(() => setStatus(""), 2000);
  });

  refreshSessionsBtn.addEventListener("click", () => refreshSessions());

  if (menuBtn) {
    menuBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleMenu();
    });
  }

  document.addEventListener("click", () => {
    closeMenu();
    closeCmdMenu();
  });

  if (menuDropdown) {
    menuDropdown.addEventListener("click", (e) => {
      e.stopPropagation();
    });
  }

  if (cmdMenuBtn) {
    cmdMenuBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleCmdMenu();
    });
  }
  if (cmdMenuDropdown) {
    cmdMenuDropdown.addEventListener("click", (e) => {
      e.stopPropagation();
    });
  }

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeMenu();
      closeCmdMenu();
      return;
    }
    if (e.key === "Enter" && !e.shiftKey && e.target === inputEl) {
      e.preventDefault();
      formEl.requestSubmit();
    }
  });

  if (navChat) {
    navChat.addEventListener("click", () => switchTab("chat"));
  }
  if (navSettings) {
    navSettings.addEventListener("click", () => switchTab("settings"));
  }
  if (navAdmin) {
    navAdmin.addEventListener("click", () => switchTab("admin"));
  }
  if (navLogout) {
    navLogout.addEventListener("click", () => {
      closeMenu();
      kbLogout();
    });
  }

  function showAuthGate(setupMode) {
    if (authGate) {
      authGate.hidden = false;
    }
    if (layoutElRoot) {
      layoutElRoot.hidden = true;
    }
    const lead = document.getElementById("kb-auth-lead");
    const submit = document.getElementById("kb-auth-submit");
    const tokenLabel = document.querySelector(".kb-auth-token-label");
    const tokenInput = document.getElementById("kb-auth-token");
    if (setupMode) {
      if (lead) {
        lead.textContent =
          "Первый запуск: задайте пароль администратора (минимум 6 символов). " +
          "Он сохранится в config.json как api.auth.passwordHash.";
      }
      if (submit) {
        submit.textContent = "Задать пароль";
      }
      if (tokenLabel) {
        tokenLabel.hidden = true;
      }
      if (tokenInput) {
        tokenInput.hidden = true;
      }
    } else {
      if (lead) {
        lead.textContent = "Для веб-интерфейса нужен пароль или API-ключ администратора.";
      }
      if (submit) {
        submit.textContent = "Войти";
      }
      if (tokenLabel) {
        tokenLabel.hidden = false;
      }
      if (tokenInput) {
        tokenInput.hidden = false;
      }
    }
  }

  function showAppShell() {
    if (authGate) {
      authGate.hidden = true;
    }
    if (layoutElRoot) {
      layoutElRoot.hidden = false;
    }
  }

  function setAuthError(msg) {
    const el = document.getElementById("kb-auth-error");
    if (!el) {
      return;
    }
    if (msg) {
      el.hidden = false;
      el.textContent = msg;
    } else {
      el.hidden = true;
      el.textContent = "";
    }
  }

  async function kbLogout() {
    try {
      await kbApiFetch("/v1/web/auth/logout", { method: "POST" });
    } catch (_e) {
      /* ignore */
    }
    setAuthToken("");
    showAuthGate(false);
  }

  /**
   * @returns {Promise<boolean>} true if app may proceed
   */
  async function ensureAuthenticated() {
    let status;
    try {
      const r = await fetch("/v1/web/auth/status", { credentials: "include" });
      status = await r.json();
    } catch (err) {
      setAuthError("Не удалось связаться с сервером: " + (err.message || err));
      showAuthGate(false);
      return false;
    }
    if (status.configured && status.authenticated) {
      showAppShell();
      return true;
    }
    if (!status.configured) {
      showAuthGate(true);
      return false;
    }
    // Configured but not authenticated — try Bearer from sessionStorage
    if (getAuthToken()) {
      try {
        const r2 = await kbApiFetch("/v1/web/auth/status");
        const s2 = await r2.json();
        if (s2.authenticated) {
          showAppShell();
          return true;
        }
      } catch (_e) {
        /* fall through */
      }
    }
    showAuthGate(false);
    return false;
  }

  const authForm = document.getElementById("kb-auth-form");
  if (authForm) {
    authForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      setAuthError("");
      const passwordEl = document.getElementById("kb-auth-password");
      const tokenEl = document.getElementById("kb-auth-token");
      const password = passwordEl ? passwordEl.value : "";
      const token = tokenEl && !tokenEl.hidden ? tokenEl.value.trim() : "";
      let setupMode = false;
      try {
        const st = await fetch("/v1/web/auth/status", { credentials: "include" });
        const sj = await st.json();
        setupMode = !sj.configured;
      } catch (_e) {
        /* assume login */
      }
      const url = setupMode ? "/v1/web/auth/setup" : "/v1/web/auth/login";
      const body = setupMode
        ? { password: password }
        : token
          ? { token: token }
          : { password: password };
      try {
        const r = await fetch(url, {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) {
          setAuthError(
            (data.error && data.error.message) || "Ошибка входа (HTTP " + r.status + ")"
          );
          return;
        }
        if (data.token) {
          setAuthToken(data.token);
        }
        showAppShell();
        setAuthError("");
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
        }
      } catch (err) {
        setAuthError(String(err.message || err));
      }
    });
  }

  function setAdminBanner(okMsg, errMsg) {
    const ok = document.getElementById("kb-admin-ok");
    const err = document.getElementById("kb-admin-error");
    if (ok) {
      if (okMsg) {
        ok.hidden = false;
        ok.textContent = okMsg;
      } else {
        ok.hidden = true;
        ok.textContent = "";
      }
    }
    if (err) {
      if (errMsg) {
        err.hidden = false;
        err.textContent = errMsg;
      } else {
        err.hidden = true;
        err.textContent = "";
      }
    }
  }

  async function refreshAdminUsers() {
    setAdminBanner("", "");
    const listEl = document.getElementById("kb-admin-user-list");
    const pendingEl = document.getElementById("kb-admin-pending");
    const pendingEmpty = document.getElementById("kb-admin-pending-empty");
    if (!listEl) {
      return;
    }
    try {
      const [usersR, regsR] = await Promise.all([
        kbApiFetch("/v1/web/users"),
        kbApiFetch("/v1/web/registrations"),
      ]);
      if (!usersR.ok) {
        const d = await usersR.json().catch(() => ({}));
        throw new Error((d.error && d.error.message) || "HTTP " + usersR.status);
      }
      const usersData = await usersR.json();
      const regsData = regsR.ok ? await regsR.json() : { data: [] };
      const users = Array.isArray(usersData.data) ? usersData.data : [];
      const regs = Array.isArray(regsData.data) ? regsData.data : [];

      if (pendingEl) {
        pendingEl.innerHTML = "";
        regs.forEach((reg) => {
          const li = document.createElement("li");
          li.className = "kb-admin-pending-item";
          const meta = document.createElement("div");
          meta.className = "kb-admin-pending-meta";
          meta.textContent =
            kbAdminChannelLabel(reg.channel) +
            " · " +
            reg.sender_id +
            (reg.note ? " — " + reg.note : "") +
            (reg.created_at ? " (" + reg.created_at + ")" : "");
          const actions = document.createElement("div");
          actions.className = "kb-admin-pending-actions";
          const approve = document.createElement("button");
          approve.type = "button";
          approve.className = "kb-btn kb-btn-primary";
          approve.textContent = "Одобрить";
          approve.addEventListener("click", async () => {
            const r = await kbApiFetch(
              "/v1/web/registrations/" + encodeURIComponent(reg.request_id) + "/approve",
              { method: "POST" }
            );
            if (!r.ok) {
              const d = await r.json().catch(() => ({}));
              setAdminBanner("", (d.error && d.error.message) || "Ошибка одобрения");
              return;
            }
            setAdminBanner("Заявка одобрена", "");
            refreshAdminUsers();
          });
          const reject = document.createElement("button");
          reject.type = "button";
          reject.className = "kb-btn kb-btn-secondary";
          reject.textContent = "Отклонить";
          reject.addEventListener("click", async () => {
            const r = await kbApiFetch(
              "/v1/web/registrations/" + encodeURIComponent(reg.request_id) + "/reject",
              { method: "POST" }
            );
            if (!r.ok) {
              const d = await r.json().catch(() => ({}));
              setAdminBanner("", (d.error && d.error.message) || "Ошибка отклонения");
              return;
            }
            setAdminBanner("Заявка отклонена", "");
            refreshAdminUsers();
          });
          actions.appendChild(approve);
          actions.appendChild(reject);
          li.appendChild(meta);
          li.appendChild(actions);
          pendingEl.appendChild(li);
        });
      }
      if (pendingEmpty) {
        pendingEmpty.hidden = regs.length > 0;
      }

      listEl.innerHTML = "";
      users.forEach((u) => {
        listEl.appendChild(renderUserCard(u));
      });
      if (!users.length) {
        const p = document.createElement("p");
        p.className = "kb-settings-p";
        p.textContent = "Пользователей пока нет.";
        listEl.appendChild(p);
      }
    } catch (err) {
      setAdminBanner("", String(err.message || err));
    }
  }

  function renderUserCard(u) {
    const card = document.createElement("article");
    card.className = "kb-admin-user-card";
    card.dataset.userId = u.user_id;

    const head = document.createElement("div");
    head.className = "kb-admin-user-card-head";
    const nameInput = document.createElement("input");
    nameInput.type = "text";
    nameInput.className = "kb-input kb-admin-user-name-input";
    nameInput.value = u.display_name || "";
    nameInput.placeholder = "Имя";
    head.appendChild(nameInput);
    if (u.is_owner) {
      const badge = document.createElement("span");
      badge.className = "kb-admin-badge";
      badge.textContent = "владелец";
      head.appendChild(badge);
    }
    card.appendChild(head);

    const idRo = document.createElement("p");
    idRo.className = "kb-admin-ro";
    idRo.textContent = "user_id: " + u.user_id;
    card.appendChild(idRo);

    const wsRo = document.createElement("p");
    wsRo.className = "kb-admin-ro";
    wsRo.textContent = "workspace: " + (u.workspace || "—");
    card.appendChild(wsRo);

    const ttsLabel = document.createElement("label");
    ttsLabel.className = "kb-cfg-label";
    const ttsCb = document.createElement("input");
    ttsCb.type = "checkbox";
    ttsCb.checked = !!u.tts_enabled;
    ttsLabel.appendChild(ttsCb);
    ttsLabel.appendChild(document.createTextNode(" TTS включён"));
    card.appendChild(ttsLabel);

    const linksTitle = document.createElement("p");
    linksTitle.className = "kb-settings-p";
    linksTitle.style.marginTop = "0.5rem";
    linksTitle.textContent = "Привязки каналов:";
    card.appendChild(linksTitle);

    const accounts = (Array.isArray(u.accounts) ? u.accounts : []).filter((acc) => {
      return kbAdminParseAccount(acc).channel !== "api";
    });

    const linksUl = document.createElement("ul");
    linksUl.className = "kb-admin-links";
    accounts.forEach((acc) => kbAdminAppendLinkRow(linksUl, u.user_id, acc));
    card.appendChild(linksUl);

    const addRow = document.createElement("div");
    addRow.className = "kb-admin-link-add";
    const chIn = document.createElement("select");
    chIn.className = "kb-input";
    chIn.required = true;
    chIn.setAttribute("aria-label", "Канал");
    kbAdminFillChannelSelect(chIn, {
      includeEmpty: true,
      emptyLabel: "— канал —",
    });
    const sidIn = document.createElement("input");
    sidIn.type = "text";
    sidIn.className = "kb-input";
    sidIn.setAttribute("aria-label", "Идентификатор в канале");
    kbAdminBindSenderHint(chIn, sidIn, null);
    const addBtn = document.createElement("button");
    addBtn.type = "button";
    addBtn.className = "kb-btn kb-btn-secondary";
    addBtn.textContent = "Добавить связь";
    addBtn.addEventListener("click", async () => {
      const channel = chIn.value.trim();
      const sender_id = sidIn.value.trim();
      if (!channel) {
        setAdminBanner("", "Выберите канал");
        return;
      }
      if (!sender_id) {
        setAdminBanner("", "Укажите: " + kbAdminSenderHint(channel));
        return;
      }
      const r = await kbApiFetch(
        "/v1/web/users/" + encodeURIComponent(u.user_id) + "/links",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ channel: channel, sender_id: sender_id }),
        }
      );
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        setAdminBanner("", (d.error && d.error.message) || "Не удалось добавить связь");
        return;
      }
      refreshAdminUsers();
    });
    addRow.appendChild(chIn);
    addRow.appendChild(sidIn);
    addRow.appendChild(addBtn);
    card.appendChild(addRow);

    const actions = document.createElement("div");
    actions.className = "kb-admin-user-actions";
    const saveBtn = document.createElement("button");
    saveBtn.type = "button";
    saveBtn.className = "kb-btn kb-btn-primary";
    saveBtn.textContent = "Сохранить";
    saveBtn.addEventListener("click", async () => {
      const r = await kbApiFetch("/v1/web/users/" + encodeURIComponent(u.user_id), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          display_name: nameInput.value.trim(),
          tts_enabled: ttsCb.checked,
        }),
      });
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        setAdminBanner("", (d.error && d.error.message) || "Ошибка сохранения");
        return;
      }
      setAdminBanner("Пользователь сохранён", "");
      refreshAdminUsers();
    });
    const delBtn = document.createElement("button");
    delBtn.type = "button";
    delBtn.className = "kb-btn kb-btn-danger";
    delBtn.textContent = "Удалить";
    delBtn.disabled = !!u.is_owner;
    delBtn.title = u.is_owner ? "Нельзя удалить владельца" : "Удалить пользователя и workspace";
    delBtn.addEventListener("click", async () => {
      if (u.is_owner) {
        return;
      }
      const ok = window.confirm(
        "Удалить пользователя " +
          (u.display_name || u.user_id) +
          "?\nБудут удалены все привязки и каталог workspace."
      );
      if (!ok) {
        return;
      }
      const r = await kbApiFetch("/v1/web/users/" + encodeURIComponent(u.user_id), {
        method: "DELETE",
      });
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        setAdminBanner("", (d.error && d.error.message) || "Ошибка удаления");
        return;
      }
      setAdminBanner("Пользователь удалён", "");
      refreshAdminUsers();
    });
    actions.appendChild(saveBtn);
    actions.appendChild(delBtn);
    card.appendChild(actions);
    return card;
  }

  const adminCreateForm = document.getElementById("kb-admin-create-form");
  const adminCreateChannel = document.getElementById("kb-admin-create-channel");
  const adminCreateSender = document.getElementById("kb-admin-create-sender");
  const adminCreateSenderCaption = document.getElementById("kb-admin-create-sender-caption");
  if (adminCreateChannel && adminCreateSender) {
    kbAdminBindSenderHint(adminCreateChannel, adminCreateSender, adminCreateSenderCaption);
  }
  if (adminCreateForm) {
    adminCreateForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const name = (document.getElementById("kb-admin-create-name") || {}).value || "";
      const channel = (adminCreateChannel && adminCreateChannel.value) || "";
      const sender = (adminCreateSender && adminCreateSender.value) || "";
      const body = { display_name: String(name).trim() };
      if (String(channel).trim() && String(sender).trim()) {
        body.channel = String(channel).trim();
        body.sender_id = String(sender).trim();
      } else if (String(channel).trim() || String(sender).trim()) {
        setAdminBanner("", "Укажите и канал, и идентификатор — или оставьте оба пустыми");
        return;
      }
      const r = await kbApiFetch("/v1/web/users", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        setAdminBanner("", (d.error && d.error.message) || "Ошибка создания");
        return;
      }
      adminCreateForm.reset();
      if (adminCreateChannel) {
        adminCreateChannel.dispatchEvent(new Event("change"));
      }
      setAdminBanner("Пользователь создан", "");
      refreshAdminUsers();
    });
  }

  const adminRefreshBtn = document.getElementById("kb-admin-refresh");
  if (adminRefreshBtn) {
    adminRefreshBtn.addEventListener("click", () => refreshAdminUsers());
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

  const kbCfgSaveBtn = document.getElementById("kb-cfg-save");
  const kbCfgReloadBtn = document.getElementById("kb-cfg-reload");
  const kbCfgRestoreBtn = document.getElementById("kb-cfg-restore");
  const kbCfgDownloadBackupBtn = document.getElementById("kb-cfg-download-backup");

  function kbCfgFormatSaveError(data, status) {
    const err = data && data.error ? data.error : null;
    let msg =
      (err && err.message) || data.detail || "Не удалось сохранить (" + status + ").";
    if (err && err.detail != null && typeof err.detail !== "string") {
      msg += "\n" + JSON.stringify(err.detail).slice(0, 900);
    }
    return msg;
  }

  if (kbCfgDownloadBackupBtn) {
    kbCfgDownloadBackupBtn.addEventListener("click", () => {
      const a = document.createElement("a");
      a.href = "/v1/web/backup/download";
      a.download = "";
      a.rel = "noopener";
      document.body.appendChild(a);
      a.click();
      a.remove();
    });
  }

  if (kbCfgSaveBtn) {
    kbCfgSaveBtn.addEventListener("click", async () => {
      const errEl = document.getElementById("kb-cfg-error");
      const okEl = document.getElementById("kb-cfg-ok");
      if (errEl) {
        errEl.hidden = true;
        errEl.textContent = "";
      }
      if (okEl) {
        okEl.hidden = true;
        okEl.textContent = "";
      }
      kbCfgSaveBtn.disabled = true;
      try {
        const sections = kbCfgCollectSectionsFromForm();
        if (
          sections == null ||
          typeof sections !== "object" ||
          !kbCfgLastLoaded
        ) {
          throw new Error(
            "Загрузите настройки: откройте вкладку и дождитесь загрузки.",
          );
        }
        const r = await kbApiFetch("/v1/web/config", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(sections),
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) {
          if (errEl) {
            errEl.textContent = kbCfgFormatSaveError(data, r.status);
            errEl.hidden = false;
          }
          return;
        }
        const bak = data.backupCreated || "";
        if (okEl) {
          okEl.textContent = bak
            ? "Готово. Перед сохранением создан бэкап: " + bak
            : "Сохранено.";
          okEl.hidden = false;
        }
        await refreshSettingsPanel();
      } catch (e) {
        if (errEl) {
          errEl.textContent =
            String((e && e.message) || e) || "Ошибка при сохранении.";
          errEl.hidden = false;
        }
      } finally {
        kbCfgSaveBtn.disabled = false;
      }
    });
  }

  if (kbCfgReloadBtn) {
    kbCfgReloadBtn.addEventListener("click", async () => {
      await refreshSettingsPanel();
    });
  }

  if (kbCfgRestoreBtn) {
    kbCfgRestoreBtn.addEventListener("click", async () => {
      const sel = document.getElementById("kb-cfg-backups");
      const errEl = document.getElementById("kb-cfg-error");
      const okEl = document.getElementById("kb-cfg-ok");
      const name = (sel && sel.value && sel.value.trim()) || "";
      if (!name) {
        alert("Выберите файл резервной копии из списка.");
        return;
      }
      if (
        !window.confirm(
          "Заменить текущий config.json содержимым «" +
            name +
            "»? Текущий файл будет сохранён в отдельный бэкап перед заменой.",
        )
      ) {
        return;
      }
      if (errEl) {
        errEl.hidden = true;
        errEl.textContent = "";
      }
      if (okEl) {
        okEl.hidden = true;
        okEl.textContent = "";
      }
      kbCfgRestoreBtn.disabled = true;
      try {
        const r = await kbApiFetch("/v1/web/config/restore", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ backup: name }),
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) {
          if (errEl) {
            errEl.textContent = kbCfgFormatSaveError(data, r.status);
            errEl.hidden = false;
          }
          return;
        }
        const pre = data.previousBackedUpAs || "";
        if (okEl) {
          okEl.textContent =
            "Восстановлено из «" +
            name +
            "»." +
            (pre ? " Резервная копия прежней версии: " + pre : "");
          okEl.hidden = false;
        }
        await refreshSettingsPanel();
      } finally {
        kbCfgRestoreBtn.disabled = false;
      }
    });
  }

  attachBtn.addEventListener("click", () => fileInputEl.click());

  fileInputEl.addEventListener("change", async () => {
    const files = Array.from(fileInputEl.files || []);
    // Keep File references; clear only the input so the same path can be re-picked.
    fileInputEl.value = "";
    for (const f of files) {
      try {
        assertAttachHardMax(f);
        // Video / docs / large: upload immediately via FormData (no FileReader).
        // Avoids stale handles and never loads the file into JS memory as base64.
        if (isLargeAttach(f) || needsMultipartUpload(f)) {
          setStatus("Загрузка «" + (f.name || "файл") + "»…");
          sendBtn.disabled = true;
          const rows = await uploadFilesMultipart([f], {
            status: "Загрузка «" + (f.name || "файл") + "»…",
          });
          pendingFiles.push({
            __kbUploaded: true,
            name: f.name || "file.bin",
            rows: rows,
          });
        } else {
          pendingFiles.push(f);
        }
      } catch (err) {
        appendMessage(
          "assistant",
          friendlyAttachError(err, f && f.name ? f.name : "файл"),
          "error"
        );
      } finally {
        sendBtn.disabled = false;
      }
    }
    renderAttachments();
    setStatus("");
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
      tag.textContent = attachDisplayName(f);
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
    const ok = await ensureAuthenticated();
    if (!ok) {
      return;
    }
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
