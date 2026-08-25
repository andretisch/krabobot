![](https://raw.githubusercontent.com/andretisch/krabobot/refs/heads/main/133a0f87b-9f20-4de3-8238-3b946b16f3f9.png)

# krabobot

`krabobot` — локальный многоканальный AI-бот с акцентом на:

- русскоязычную аудиторию;
- работу через Telegram, VK и Email;
- единый профиль пользователя между каналами;
- регистрацию с подтверждением владельца;
- локальные STT/TTS через `sherpa-onnx`.

---

## Что умеет

- Режимы запуска: `gateway`, `agent`, `serve`.
- Каналы: `telegram`, `vk`, `email`.
- Команды: `/start`, `/help`, `/new`, `/clear_memory`, `/id`, `/link`, `/tts`, `/reg`, `/regcode`, `/status`, `/restart`.
- Голосовые ответы в VK/Telegram включаются **только** командой **`/tts on`** у каждого пользователя (глобальных флагов в `config.json` нет).
- Встроенная модель доступа:
  - первый пользователь становится владельцем;
  - остальные проходят регистрацию (`/reg`) и подтверждение владельцем.
- Персональная память по каждому пользователю (отдельные workspace).

---

## Требования

- Linux/macOS/Windows
- Python `3.11+`
- Для аудио/STT используется `imageio-ffmpeg` (кроссплатформенно, включая Windows); при его недоступности используется системный `ffmpeg` из `PATH`

---

## Установка

```bash
git clone https://github.com/andretisch/krabobot.git
cd krabobot
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,api]"
```

**Скрипт установки** (то же самое + опционально `krabobot onboard`):

```bash
bash scripts/install.sh
# справка: bash scripts/install.sh --help
```

Windows (PowerShell из корня репозитория): `.\scripts\install.ps1`

Пакеты **`numpy`** и **`sherpa-onnx`** (локальные STT/TTS) входят в обычную зависимость `pip install -e .` — extras `[stt]` / `[tts]` не нужны.

---

## Первый запуск

По умолчанию после `krabobot onboard` используется **Ollama** на `localhost:11434` и модель **`gemma4:cloud`**. Локально облачные модели Ollama можно запускать через установленный Ollama и `ollama signin`. Альтернатива — прямой доступ к **Ollama Cloud API** без локального демона:

```json
{
  "agents": { "defaults": { "provider": "ollama", "model": "gemma4:cloud" } },
  "providers": {
    "ollama": {
      "apiKey": "YOUR_OLLAMA_API_KEY",
      "apiBase": "https://ollama.com"
    }
  }
}
```

Ключ создаётся на [ollama.com/settings/keys](https://ollama.com/settings/keys) или через переменную окружения `OLLAMA_API_KEY`. Локальный Ollama (`http://localhost:11434/v1`) по-прежнему работает без ключа.

```bash
krabobot onboard
krabobot gateway
```

Полезно:

```bash
krabobot agent
krabobot serve
krabobot --help
```

После `krabobot serve` в браузере откройте `http://127.0.0.1:<порт>/` (порт из `config.api`) — веб-чат к тому же API без CORS.

Во вкладке **Настройки** можно править тот же файл `config.json`, с которым запущен сервер (в т.ч. путь из `krabobot serve -c …`):

- **Сохранить в config.json** записывает изменения на диск и перед записью делает резервную копию рядом с файлом конфигурации (`config.backup.ГГГГММДД-ЧЧММСС-микросекунды.json`; старые имена вида без микросекунд тоже учитываются для списка и отката).
- Поля с ключами и токенами на странице скрыты от показа; если поле секрета **оставить пустым** и сохранить, в файле сохранится прежнее значение.
- Через список можно **восстановить** выбранный бэкап; текущий `config.json` перед этим тоже сохраняется в бэкап.
- Сервер API и UI по умолчанию **без отдельной авторизации** для этой страницы — не выставляйте порт наружу без прокси/ограничений.

Для автоматизации доступны HTTP-эндпоинты: `GET /v1/web/config`, `PUT /v1/web/config` (тело в форме ответа GET: `core`, `channels`, `other`), `GET /v1/web/config/backups`, `POST /v1/web/config/restore` с телом вида `{ "backup": "config.backup.YYYYMMDD-HHMMSS-ffffff.json" }` (basename из списка бэкапов).

Основной конфиг:

- `~/.krabobot/config.json`

---

## Каталог `~/.krabobot`

При `krabobot onboard` создаётся домашний каталог экземпляра. Если конфиг задан через `-c /path/to/config.json`, каталоги `logs/` и `media/` создаются **рядом с этим файлом**. История CLI (`history/cli_history`) всегда в `~/.krabobot/history/`.

### Дерево файлов и папок

```
~/.krabobot/
├── config.json                 # Главный конфиг (провайдер, каналы, workspace, tools)
├── config.backup.*.json        # Резервные копии config (веб-настройки, restore)
├── logs/                       # Логи процесса (например krabobot.log)
├── media/                      # Временные файлы вложений с каналов
│   ├── telegram/
│   ├── vk/
│   └── email/
├── models/                     # Локальные модели sherpa-onnx (STT/TTS)
│   ├── stt/
│   └── tts/
├── history/
│   └── cli_history             # История ввода в `krabobot agent` (readline)
└── workspace/                  # Рабочая область (agents.defaults.workspace)
    ├── AGENTS.md               # Инструкции агенту (шаблон при onboard)
    ├── SOUL.md                 # Стиль и «характер» ответов
    ├── USER.md                 # О пользователе и предпочтениях
    ├── TOOLS.md                # Заметки по инструментам и окружению
    ├── HEARTBEAT.md            # Периодические задачи (heartbeat-сервис)
    ├── memory/
    │   ├── MEMORY.md           # Долговременная память (в контекст LLM)
    │   └── HISTORY.md          # Журнал событий (поиск через инструменты)
    ├── skills/                 # Пользовательские skills
    │   └── <имя-скилла>/SKILL.md
    ├── sessions/               # История диалогов: <канал>_<chat_id>.jsonl
    ├── cron/
    │   └── jobs.json           # Запланированные задания (cron tool)
    ├── identity/
    │   └── user_links.json     # Аккаунты, owner, /reg, /link, TTS на пользователя
    └── users/                  # Изолированный workspace каждого user_id
        └── <user_id>/          # Та же структура (memory, sessions, skills, …)
```

Cron и сессии хранятся **только** в `workspace/` (или в `users/<user_id>/`), не в корне `~/.krabobot/`.

### Назначение ключевых частей

| Путь | Назначение |
|------|------------|
| `config.json` | Единственный источник настроек: LLM, каналы, API, MCP, heartbeat, STT/TTS. |
| `workspace/` | Файлы, которые агент читает и правит; корень для `read_file` / `write_file` (если не включён `restrictToWorkspace`). |
| `workspace/identity/` | Связка Telegram/VK/Email/web в одного пользователя, владелец, очередь `/reg`, коды `/link` и `/regcode`. |
| `workspace/users/<id>/` | Отдельная память, сессии и skills для каждого внутреннего `user_id`. |
| `workspace/sessions/*.jsonl` | Полная переписка по чату; в LLM уходит только «хвост» после консолидации. |
| `workspace/memory/MEMORY.md` | Сжатые факты; подмешиваются в system prompt каждый запрос. |
| `workspace/memory/HISTORY.md` | Подробный лог; в промпт **не** входит, только по запросу через инструменты. |
| `workspace/cron/jobs.json` | Напоминания и расписание. |
| `models/` | Кэш весов sherpa-onnx; скачиваются при `autoDownloadModels: true`. |
| `media/` | Скачанные голосовые, фото и вложения с каналов перед обработкой. |
| `history/cli_history` | Удобство CLI, к агенту не относится. |

Шаблоны `AGENTS.md`, `SOUL.md`, `USER.md`, `TOOLS.md` и пустой `memory/MEMORY.md` копируются из пакета **только если файла ещё нет** (`sync_workspace_templates`). Существующие файлы не перезаписываются.

### Что всегда в контексте LLM, а что подключается по запросу

**Постоянно в system prompt (каждый вызов модели):**

- Блок идентичности krabobot (runtime, путь workspace, правила безопасности).
- Файлы bootstrap, если есть: `AGENTS.md`, `SOUL.md`, `USER.md`, `TOOLS.md`.
- Содержимое `memory/MEMORY.md` (долговременная память).
- Skills с флагом `always: true` в frontmatter — по умолчанию встроенные `memory` и `multimodal-hints` (полный текст `SKILL.md`).
- Краткий **каталог skills** (XML: имя, описание, путь, `available=true/false`) — чтобы агент знал, что можно открыть через `read_file`.

**В каждом сообщении пользователя (не system):**

- Метаданные сессии: время, канал, `chat_id`.
- Несжатая история из `sessions/*.jsonl` (после `last_consolidated`).
- Текст и вложения текущего сообщения (в т.ч. изображения как multimodal-блоки).

**Подключается по необходимости (агент сам читает инструментами):**

- Полный текст любого skill: `workspace/skills/.../SKILL.md` или встроенный skill из пакета.
- `memory/HISTORY.md` — журнал для grep/поиска.
- Любые другие файлы в workspace пользователя.
- `HEARTBEAT.md` — отдельно читает **heartbeat-сервис** по расписанию из `gateway.heartbeat`, в обычный чат не подмешивается.

**Настраивается в `config.json`, не хранится в `.krabobot` как текст контекста:**

- Каналы (`telegram`, `vk`, `email`) — подключаются при `krabobot gateway`.
- MCP-серверы (`tools.mcpServers`) — ленивое подключение при первом сообщении, инструменты регистрируются в рантайме.
- Провайдер LLM, STT/TTS, веб-поиск — влияют на поведение, но не копируются в промпт целиком.

**Консолидация памяти:** когда история не помещается в окно контекста, старые сообщения из `sessions/*.jsonl` суммаризуются в `MEMORY.md` / `HISTORY.md`; сами строки JSONL при этом не удаляются, меняется только смещение `last_consolidated`.

Пользовательские skills кладите в `~/.krabobot/workspace/skills/<имя>/SKILL.md` (или в `users/<id>/skills/`). Имя каталога с тем же именем **перекрывает** встроенный skill из пакета.

### Обновление со старых версий

Если после апгрейда остались каталоги `~/.krabobot/cron/` или `~/.krabobot/sessions/` (старые раскладки до workspace):

```bash
mkdir -p ~/.krabobot/workspace/cron ~/.krabobot/workspace/sessions
mv ~/.krabobot/cron/jobs.json ~/.krabobot/workspace/cron/ 2>/dev/null || true
mv ~/.krabobot/sessions/*.jsonl ~/.krabobot/workspace/sessions/ 2>/dev/null || true
rmdir ~/.krabobot/cron ~/.krabobot/sessions 2>/dev/null || true
```

---

## Базовая структура конфига

Провайдер задаётся в `agents.defaults.provider`. Удобный вариант — **`custom`**: один блок с ключом и OpenAI-compatible `apiBase`, без отдельного имени вида ProxyAPI/OpenRouter в структуре.

```json
{
  "agents": {
    "defaults": {
      "workspace": "~/.krabobot/workspace",
      "model": "gpt-5.4-nano",
      "provider": "custom"
    }
  },
  "providers": {
    "custom": {
      "apiKey": "YOUR_API_KEY",
      "apiBase": "https://api.proxyapi.ru/openai/v1",
      "useMaxCompletionTokens": true
    }
  },
  "channels": {
    "telegram": {
      "enabled": false,
      "token": ""
    },
    "vk": {
      "enabled": false,
      "token": ""
    },
    "email": {
      "enabled": false,
      "consentGranted": false
    }
  }
}
```

---

## Регистрация пользователей и права

`krabobot` работает в модели owner + registration:

1. Первый пользователь, который начинает работу, становится владельцем.
2. Остальные пользователи отправляют `/reg [кто вы]` или `/reg <одноразовый_код>`.
3. Владелец подтверждает:
   - `/reg list`
   - `/reg approve <request_id>`
   - `/reg reject <request_id>`
4. Одноразовый код:
   - владелец: `/regcode create [ttl_seconds]`
   - пользователь: `/reg <код>`

Ограниченные команды владельца:

- `/status`
- `/restart`

Связка каналов одного пользователя:

- `/link` — сгенерировать код привязки
- `/link <код>` — привязать аккаунт в другом канале
- `/id` — показать IDs и список связанных каналов

---

## Настройка каналов

## 1) VK (ВКонтакте)

### Подготовка сообщества ВКонтакте

1. Создайте сообщество (группу или паблик), если его еще нет.
2. Откройте **Управление → Сообщения** и включите сообщения.
3. Откройте **Управление → Дополнительно → Работа с API → Ключи доступа**.
4. Нажмите **Создать ключ** и выберите права:
   - `Сообщения сообщества`
   - `Управление сообществом` (нужно для Bots Long Poll API)
5. Откройте **Управление → Дополнительно → Работа с API → Long Poll API**:
   - включите Long Poll API (`Включен`);
   - во вкладке **Типы событий** обязательно отметьте `Входящие сообщения`.
6. Для работы в беседах:
   - **Управление → Сообщения → Настройки для бота**
   - включите `Разрешать добавлять сообщество в чаты`.

### Конфиг VK

```json
"vk": {
  "enabled": true,
  "token": "vk1.a....",
  "reactionId": 10,
  "transcribeVoice": true,
  "transcribeAudio": false
}
```

Примечания:

- Для голосовых вложений используется `audio_message`.
- Нужны рабочие права токена на сообщения/документы.

---

## 2) Telegram

### Подготовка бота

1. Создайте бота через `@BotFather`.
2. Получите токен.
3. Если нужен доступ в группы:
   - добавьте бота в группу;
   - при необходимости отключите privacy mode в `@BotFather`.

### Конфиг Telegram

```json
"telegram": {
  "enabled": true,
  "token": "123456:ABCDEF...",
  "groupPolicy": "mention",
  "streaming": true,
  "transcribeVoice": true,
  "transcribeAudio": false,
  "welcomeMessage": ""
}
```

Опционально:

- `proxy` — прокси для Telegram API.
- `groupPolicy`:
  - `mention` — бот отвечает в группе при упоминании/reply;
  - `open` — бот отвечает на все сообщения группы.

---

## 3) Email

### Что нужно

- IMAP и SMTP одной почты/домена;
- явное согласие на отправку почты (`consentGranted: true`).

### Конфиг Email

```json
"email": {
  "enabled": true,
  "consentGranted": true,
  "imapHost": "imap.mail.ru",
  "imapPort": 993,
  "imapUsername": "bot@example.com",
  "imapPassword": "APP_PASSWORD",
  "imapMailbox": "INBOX",
  "imapUseSsl": true,
  "smtpHost": "smtp.mail.ru",
  "smtpPort": 587,
  "smtpUsername": "bot@example.com",
  "smtpPassword": "APP_PASSWORD",
  "smtpUseTls": true,
  "smtpUseSsl": false,
  "fromAddress": "bot@example.com",
  "autoReplyEnabled": true,
  "replyRegisteredOnly": true,
  "pollIntervalSeconds": 30,
  "markSeen": true,
  "maxBodyChars": 12000,
  "subjectPrefix": "Re: ",
  "verifyDkim": true,
  "verifySpf": true
}
```

Ключевые флаги:

- `autoReplyEnabled` — автоответ на входящие письма.
- `replyRegisteredOnly` — отвечать только зарегистрированным пользователям.
- `verifyDkim`/`verifySpf` — защита от spoofing.

---

## Провайдеры LLM

Поддерживаются OpenAI-compatible:

- **`custom`** — любой совместимый endpoint (ключ и `apiBase` в блоке `providers.custom`).
- **`openrouter`**, **`proxyapi`**, **`gptunnel`**, **`ollama`** — готовые слоты в `providers.<имя>`; удобны, когда не хотите класть базовый URL в `custom`.

Для моделей через ProxyAPI можно оставаться на **`provider: custom`**, указав их `apiBase`; при необходимости включите семантику **`max_completion_tokens`**:

```json
"custom": {
  "apiKey": "...",
  "apiBase": "https://api.proxyapi.ru/openai/v1",
  "useMaxCompletionTokens": true
}
```

---

## STT / TTS (sherpa-onnx)

Пример:

```json
"tts": {
  "provider": "sherpa_onnx",
  "language": "ru",
  "autoDownloadModels": true,
  "sherpaSpeed": 1.0,
  "sherpaModelsDir": "~/.krabobot/models/tts",
  "sherpaModelId": "csukuangfj/vits-piper-ru_RU-irina-medium"
},
"stt": {
  "provider": "sherpa_onnx",
  "autoDownloadModels": true,
  "sherpaModelsDir": "~/.krabobot/models/stt",
  "sherpaModelId": "csukuangfj/sherpa-onnx-nemo-transducer-punct-giga-am-v3-russian-2025-12-16",
  "sherpaNumThreads": 16,
  "sherpaProvider": "cpu"
}
```

---

## Частые проблемы

- VK не отвечает:
  - проверьте Long Poll API и тип события `Входящие сообщения`;
  - проверьте права ключа и что ключ от нужного сообщества.
- Email не отправляется:
  - проверьте `consentGranted: true`;
  - проверьте SMTP auth и TLS/SSL режим.
- `/status` недоступен:
  - команда только для владельца.
- Пользователь не может общаться:
  - он не прошел `/reg` + подтверждение владельца.

---

## Разработка

```bash
pytest
pytest tests/channels/test_email_channel.py
pytest tests/channels/test_vk_channel_helpers.py
```

---

## Лицензия

MIT
