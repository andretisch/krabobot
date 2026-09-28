# krabobot-voice

Локальный голосовой клиент для канала **voice** в [krabobot](https://github.com/andretisch/krabobot).

Цикл:

1. Ждёт wake-фразу **«Эй, Арнольд»** (лёгкий **KWS** — MFCC-эмбеддинги + cosine) **или** **PTT** hotkey
2. Короткий **beep** (`winsound.Beep` на Windows)
3. Запись реплики:
   - после wake → energy VAD до тишины (до ~15 с)
   - после PTT → hold-to-talk (пока зажата комбинация)
4. `POST /v1/voice/turn` (STT → агент → TTS на сервере)
5. Проигрывает ответный WAV
6. Снова ждёт wake / PTT

## Почему не sherpa на клиенте

Раньше wake шёл через полный sherpa-onnx ASR (тяжёлая русская модель). Теперь по умолчанию:

- **KWS** — energy-gated sliding window + MFCC mean/std embedding + cosine similarity к коротким reference WAV («Эй, Арнольд»).
- Это ближе к OK-Google / openWakeWord-style custom verifier: без непрерывного ASR и без гигабайтной модели на клиенте.
- STT/TTS по-прежнему только на сервере.

Для кастомной русской фразы полноценный openWakeWord ONNX обычно нужно отдельно обучать; вместо этого клиент сразу работает с эмбеддингами по reference samples. Legacy sherpa wake оставлен как `wake.mode: asr`.

## Требования

- Windows / Python 3.11+
- Работающий **`krabobot serve`** на `http://127.0.0.1:8900`
- Микрофон с разрешением для Python/терминала  
  (Параметры Windows → Конфиденциальность → Микрофон → разрешить классическим приложениям)
- Для PTT: пакет `pynput` (ставится с клиентом)

## Установка

Из корня репозитория (активный `.venv`):

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e ".\clients\krabobot-voice"
```

Опционально (только для `wake.mode: asr`):

```powershell
pip install sherpa-onnx
```

## Запуск

```powershell
# serve должен уже слушать :8900
python -m krabobot_voice
```

Статус: `waiting for wake / PTT…` → `listening…` → `thinking…` → `playing…`.

### Wake (KWS)

Фраза: **«Эй, Арнольд»**.

Reference WAV ищутся в `%LOCALAPPDATA%\krabobot-voice\wake_refs\`.  
Если папка пуста и `wake.auto_enroll_tts: true`, на Windows клиент попытается сгенерировать несколько SAPI-семплов.

Лучшее качество — положить 2–5 своих записей фразы (16 kHz mono WAV) в `wake_refs` и при необходимости подкрутить `wake.threshold` (по умолчанию `0.82`).

### PTT

По умолчанию: удерживайте **Ctrl+Alt+Space**, говорите, отпустите — клип уходит на сервер.

## Конфиг

Приоритет:

1. Путь аргументом: `python -m krabobot_voice C:\path\config.yaml`
2. `%LOCALAPPDATA%\krabobot-voice\config.yaml`
3. Env: `KRABOBOT_URL`, `KRABOBOT_TOKEN`, `KRABOBOT_DEVICE_ID` (также `KRABOBOT_VOICE_*`)
4. Пустой `token` → `~/.krabobot/config.json` → `api.auth.adminToken`

Пример: [`config.example.yaml`](config.example.yaml).

| Ключ / переменная | Смысл |
|-------------------|--------|
| `wake.mode` / `KRABOBOT_VOICE_WAKE_MODE` | `kws` (default) \| `asr` \| `off` |
| `wake.threshold` | Cosine threshold для KWS |
| `wake.refs_dir` | Папка reference WAV |
| `ptt.hotkey` / `KRABOBOT_VOICE_PTT_HOTKEY` | Например `ctrl+alt+space` |
| `KRABOBOT_URL` | Base URL (по умолчанию `http://127.0.0.1:8900`) |
| `KRABOBOT_DEVICE_ID` | Стабильный device id |
| `KRABOBOT_TOKEN` | Bearer |

При старте клиент:

- берёт `device_id` (по умолчанию `local-<hostname>`)
- читает Bearer-токен из конфига / env / `api.auth.adminToken`
- **автоматически привязывает** `voice:<device_id>` к owner через `GET /v1/web/users` + `POST .../links`

## Layout

```
krabobot_voice/
  app.py         # главный цикл
  kws.py         # MFCC embedding KWS
  ptt.py         # hold-to-talk hotkey
  wake.py        # текстовый match (legacy ASR mode)
  local_asr.py   # sherpa-onnx offline (опционально, wake.mode=asr)
  vad.py         # energy VAD
  audio_io.py    # mic / beep / play
  link.py        # auto-link device → owner
  http_client.py # POST /v1/voice/turn
  config.py
```
