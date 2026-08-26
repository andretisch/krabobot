(function () {
  "use strict";

  const FEATURES = [
    {
      icon: "📡",
      title: "Многоканальность",
      text: "Telegram, VK и Email в одном процессе — сообщения приходят в общую очередь агента.",
    },
    {
      icon: "👤",
      title: "Единый профиль",
      text: "Единый профиль одного пользователя между Telegram, VK и Email — связка через /link, общая память и история.",
    },
    {
      icon: "🔐",
      title: "Регистрация и владелец",
      text: "Первый пользователь — owner. Остальные проходят /reg и подтверждение; коды /regcode.",
    },
    {
      icon: "🧠",
      title: "Многопользовательский режим",
      text: "Многопользовательский режим: каждый собеседник узнаётся отдельно, свой workspace в users/<user_id>/; консолидация и /clear_memory с архивом.",
    },
    {
      icon: "🎙️",
      title: "Локальные STT и TTS",
      text: "Распознавание и озвучка через sherpa-onnx; голос в чатах — по /tts on у каждого.",
    },
    {
      icon: "🌐",
      title: "Веб-чат и API",
      text: "krabobot serve поднимает HTTP API и UI; настройка config.json прямо в браузере.",
    },
    {
      icon: "⏰",
      title: "Cron и heartbeat",
      text: "Отложенные задачи, напоминания и фоновые проверки по расписанию.",
    },
    {
      icon: "🛠️",
      title: "Инструменты агента",
      text: "Файлы, shell, веб-поиск, MCP, под-агенты и встроенные skills (погода, память, tmux и др.).",
    },
    {
      icon: "🤖",
      title: "Любой LLM",
      text: "OpenAI-compatible провайдеры: custom, OpenRouter, ProxyAPI, Ollama и другие из config.json.",
    },
  ];

  const CHANNELS = [
    {
      name: "Telegram",
      text: "Бот через BotFather: личные чаты и группы (mention или open), стриминг ответов, транскрипция голосовых.",
      tag: "python-telegram-bot",
    },
    {
      name: "ВКонтакте",
      text: "Сообщества с Long Poll API: личные сообщения и беседы, реакции, голосовые вложения.",
      tag: "Bots Long Poll",
    },
    {
      name: "Email",
      text: "IMAP/SMTP с автоответами, DKIM/SPF-проверкой и ответами только зарегистрированным пользователям.",
      tag: "IMAP · SMTP",
    },
  ];

  const COMMANDS = [
    ["/start", "Начало работы и проверка доступа"],
    ["/help", "Список доступных команд"],
    ["/new", "Новый разговор в сессии"],
    ["/link", "Привязать аккаунт между каналами"],
    ["/tts on|off", "Голосовые ответы для пользователя"],
    ["/reg", "Заявка на регистрацию или ввод кода"],
    ["/status", "Статус бота (только владелец)"],
  ];

  const INSTALL = `git clone https://github.com/andretisch/krabobot.git
cd krabobot
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,api]"
krabobot onboard
krabobot gateway`;

  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    if (attrs) {
      Object.entries(attrs).forEach(([k, v]) => {
        if (k === "className") node.className = v;
        else if (k === "textContent") node.textContent = v;
        else node.setAttribute(k, v);
      });
    }
    (children || []).forEach((c) => {
      if (typeof c === "string") node.appendChild(document.createTextNode(c));
      else if (c) node.appendChild(c);
    });
    return node;
  }

  function renderFeatures() {
    const root = document.getElementById("features-grid");
    if (!root) return;
    FEATURES.forEach((f) => {
      root.appendChild(
        el("li", { className: "feature-card" }, [
          el("div", { className: "feature-card__icon", textContent: f.icon }),
          el("h3", { textContent: f.title }),
          el("p", { textContent: f.text }),
        ])
      );
    });
  }

  function renderChannels() {
    const root = document.getElementById("channels-grid");
    if (!root) return;
    CHANNELS.forEach((ch) => {
      root.appendChild(
        el("li", { className: "channel-card" }, [
          el("h3", { textContent: ch.name }),
          el("p", { textContent: ch.text }),
          el("span", { className: "channel-card__tag", textContent: ch.tag }),
        ])
      );
    });
  }

  function renderCommands() {
    const root = document.getElementById("cmd-list");
    if (!root) return;
    COMMANDS.forEach(([cmd, desc]) => {
      const row = el("div");
      row.appendChild(el("dt", { textContent: cmd }));
      row.appendChild(el("dd", { textContent: desc }));
      root.appendChild(row);
    });
  }

  function initNav() {
    const toggle = document.getElementById("nav-toggle");
    const nav = document.getElementById("site-nav");
    if (!toggle || !nav) return;

    toggle.addEventListener("click", () => {
      const open = nav.classList.toggle("is-open");
      toggle.setAttribute("aria-expanded", String(open));
    });

    nav.querySelectorAll("a[href^='#']").forEach((link) => {
      link.addEventListener("click", () => {
        nav.classList.remove("is-open");
        toggle.setAttribute("aria-expanded", "false");
      });
    });
  }

  function initYear() {
    const y = document.getElementById("year");
    if (y) y.textContent = String(new Date().getFullYear());
  }

  function initInstall() {
    const block = document.getElementById("install-snippet");
    if (block) block.textContent = INSTALL;
  }

  renderFeatures();
  renderChannels();
  renderCommands();
  initNav();
  initYear();
  initInstall();
})();
