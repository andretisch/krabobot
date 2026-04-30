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

---

## Первый запуск

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

Основной конфиг:

- `~/.krabobot/config.json`

---

## Базовая структура конфига

```json
{
  "agents": {
    "defaults": {
      "workspace": "~/.krabobot/workspace",
      "model": "gpt-5.4-nano",
      "provider": "proxyapi"
    }
  },
  "providers": {
    "proxyapi": {
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

- `custom`
- `openrouter`
- `proxyapi`
- `gptunnel`
- `ollama`

Если провайдер требует `max_completion_tokens`:

```json
"proxyapi": {
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
