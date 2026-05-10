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
  const menuBtn = document.getElementById("kb-menu-btn");
  const menuDropdown = document.getElementById("kb-menu-dropdown");
  const navChat = document.getElementById("kb-nav-chat");
  const navSettings = document.getElementById("kb-nav-settings");
  const viewChat = document.getElementById("kb-view-chat");
  const viewSettings = document.getElementById("kb-view-settings");
  const copySessionBtn = document.getElementById("kb-copy-session");
  const cmdMenuBtn = document.getElementById("kb-cmd-menu-btn");
  const cmdMenuDropdown = document.getElementById("kb-cmd-dropdown");
  const layoutEl = document.getElementById("kb-layout");
  const sidebarEl = document.getElementById("kb-sidebar");
  const sidebarToggle = document.getElementById("kb-sidebar-toggle");

  const SIDEBAR_COLLAPSED_KEY = "krabobot_web_sidebar_collapsed";

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

  const KB_CFG_AGENT_ORDER = ["model", "provider", "workspace"];

  /** @type {Record<string,string>} */
  const KB_CFG_HINTS = {
    sendProgress: "Стриминг текста ответа в канал",
    sendToolHints: "Показывать вызовы инструментов",
    sendMaxRetries: "Повторы доставки в канал",
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
      const r = await fetch("/v1/web/config/backups");
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
      const r = await fetch("/v1/web/config");
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
    if (viewChat && viewSettings) {
      viewChat.classList.toggle("kb-view--hidden", !isChat);
      viewSettings.classList.toggle("kb-view--hidden", isChat);
      viewSettings.hidden = isChat;
    }
    if (navChat && navSettings) {
      navChat.classList.toggle("kb-dropdown-item--active", isChat);
      navSettings.classList.toggle("kb-dropdown-item--active", !isChat);
    }
    if (!isChat) {
      refreshSettingsPanel();
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
    if (!layoutEl || !sidebarToggle || !sidebarEl) {
      return;
    }
    layoutEl.classList.toggle("kb-layout--sidebar-collapsed", collapsed);
    sidebarToggle.setAttribute("aria-expanded", String(!collapsed));
    const expand = collapsed ? "Развернуть список диалогов" : "Свернуть список диалогов";
    sidebarToggle.setAttribute("aria-label", expand);
    sidebarToggle.title = expand;
    sidebarEl.setAttribute("aria-hidden", collapsed ? "true" : "false");
    try {
      localStorage.setItem(SIDEBAR_COLLAPSED_KEY, collapsed ? "1" : "0");
    } catch (_e) {
      /* ignore */
    }
  }

  if (layoutEl && sidebarToggle && sidebarEl) {
    try {
      if (localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "1") {
        applySidebarCollapsed(true);
      }
    } catch (_e) {
      /* ignore */
    }
    sidebarToggle.addEventListener("click", () => {
      applySidebarCollapsed(
        !layoutEl.classList.contains("kb-layout--sidebar-collapsed"),
      );
    });
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
      appendMessage("assistant", String(err.message || err), "error");
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
    inputEl.value = "";
    await postChatTurn(userVisible || "(вложения)", text, filesSnapshot);
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
    }
  });

  if (navChat) {
    navChat.addEventListener("click", () => switchTab("chat"));
  }
  if (navSettings) {
    navSettings.addEventListener("click", () => switchTab("settings"));
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

  function kbCfgFormatSaveError(data, status) {
    const err = data && data.error ? data.error : null;
    let msg =
      (err && err.message) || data.detail || "Не удалось сохранить (" + status + ").";
    if (err && err.detail != null && typeof err.detail !== "string") {
      msg += "\n" + JSON.stringify(err.detail).slice(0, 900);
    }
    return msg;
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
        const r = await fetch("/v1/web/config", {
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
        const r = await fetch("/v1/web/config/restore", {
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
