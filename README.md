# krabobot

`krabobot` is a lightweight personal AI assistant framework focused on:

- simple setup and local control;
- built-in channels (`telegram`, `vk`, `email`) without third-party channel plugins;
- OpenAI-compatible providers (including custom proxy backends);
- multi-user account linking across channels;
- optional voice pipeline: TTS (`gTTS`) + STT (`gigaam` + `onnxruntime`).

---

## Features

- **CLI modes**: interactive agent, gateway worker, API server.
- **Channels**: Telegram, VK, Email.
- **Built-in commands**: `/new`, `/status`, `/help`, `/id`, `/link`.
- **Multi-user isolation**: per-user workspace, sessions, memory.
- **Message tools**: file operations, shell, web fetch/search, message send, spawn.
- **STT/TTS**:
  - STT from voice/audio with local GigaAM ONNX backend.
  - TTS voice replies per channel with `ttsEnabled`.

---

## Requirements

- Linux/macOS
- Python `3.11+`
- `ffmpeg` (recommended; required for best VK voice-note compatibility and speed transforms)

---

## Installation

### 1. Clone and create virtualenv

```bash
git clone https://github.com/andretisch/krabobot.git
cd krabobot
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install package

Base install:

```bash
pip install -e .
```

Development install:

```bash
pip install -e ".[dev,api]"
```

STT install:

```bash
pip install -e ".[stt]"
```

Optional TTS dependency (if not already present in your environment):

```bash
pip install gTTS
```

---

## First Run

Initialize config and run onboarding:

```bash
krabobot onboard
```

Start gateway (channels + agent loop):

```bash
krabobot gateway
```

Interactive local mode (without channels):

```bash
krabobot agent
```

API mode:

```bash
krabobot serve
```

---

## Configuration

Main config path:

`~/.krabobot/config.json`

### Minimal example

```json
{
  "agents": {
    "defaults": {
      "workspace": "~/.krabobot/workspace",
      "model": "openai/gpt-4o-mini",
      "provider": "custom"
    }
  },
  "providers": {
    "custom": {
      "apiKey": "YOUR_API_KEY",
      "apiBase": "https://api.openai.com/v1"
    }
  },
  "channels": {
    "telegram": {
      "enabled": true,
      "token": "YOUR_BOT_TOKEN",
      "allowFrom": ["123456789"],
      "ttsEnabled": false,
      "transcribeVoice": true,
      "transcribeAudio": false
    }
  }
}
```

---

## Providers

`krabobot` uses OpenAI-compatible providers via shared transport.

Common configured sections:

- `custom`
- `openrouter`
- `proxyapi`
- `gptunnel`
- `ollama`

For providers/proxies that require `max_completion_tokens` instead of `max_tokens`, use:

```json
{
  "providers": {
    "proxyapi": {
      "useMaxCompletionTokens": true
    }
  }
}
```

---

## Channels

### Telegram

Required:

- `channels.telegram.enabled = true`
- `channels.telegram.token`
- `channels.telegram.allowFrom`

Optional voice:

- `ttsEnabled` - send voice reply in addition to text.
- `transcribeVoice` - transcribe voice notes to text context.
- `transcribeAudio` - transcribe generic audio attachments.

### VK

Required:

- `channels.vk.enabled = true`
- `channels.vk.token`
- `channels.vk.allowFrom`

Optional voice:

- `ttsEnabled` - TTS voice replies.
- `transcribeVoice` / `transcribeAudio`.

Notes:

- Voice notes are uploaded as `audio_message`.
- `ffmpeg` is used to convert voice to OGG/Opus when needed.

### Email

Required:

- IMAP + SMTP credentials in `channels.email`.
- Explicit consent:
  - `channels.email.consentGranted = true`

Behavior:

- inbound email is polled via IMAP;
- outbound replies are sent via SMTP;
- slash commands in email body (e.g. `/new`, `/link CODE`) are supported.

---

## Multi-user and Account Linking

When multi-user mode is enabled, accounts from different channels can be linked into one internal user.

User commands:

- `/id` - show current `channel`, `sender_id`, `chat_id`, `user_id`.
- `/link` - generate one-time code.
- `/link CODE` - link current account to existing user.

CLI helpers:

- `krabobot users link ...`
- `krabobot users list`

Identity storage:

- `~/.krabobot/workspace/identity/user_links.json`

---

## Voice Pipeline

### TTS (`gTTS`)

- Enabled per channel with `ttsEnabled`.
- Slash-command responses skip TTS.
- Telegram/VK send both text and (if enabled) voice attachment.

### STT (`gigaam` + ONNX Runtime)

Recommended environment:

```bash
export STT_PROVIDER=gigaam_onnx
export GIGAAM_MODEL_VERSION=v2_ctc
# optional:
export GIGAAM_ONNX_DIR="$HOME/.krabobot/models/gigaam/onnx"
```

If `GIGAAM_ONNX_DIR` is omitted, default directory is used under `~/.krabobot/models/...`.

---

## Useful Commands

```bash
krabobot --help
krabobot onboard
krabobot gateway
krabobot agent
krabobot serve
krabobot users --help
```

---

## Troubleshooting

- **Email does not send**
  - check `channels.email.consentGranted = true`;
  - verify SMTP credentials and host/port/TLS settings.

- **VK voice attachments fail**
  - verify community token permissions for docs/messages;
  - ensure `ffmpeg` is installed;
  - inspect gateway logs for `docs.getMessagesUploadServer` / `docs.save`.

- **Provider rejects `max_tokens`**
  - enable `useMaxCompletionTokens` for that provider in config.

- **STT returns empty/error**
  - verify `gigaam` and `onnxruntime` installed;
  - check model version (`v2_ctc`) and ONNX model directory.

---

## Development

Run tests:

```bash
pytest
```

Run selected tests:

```bash
pytest tests/channels/test_email_channel.py
pytest tests/channels/test_vk_channel_helpers.py
```

---

## License

MIT.
