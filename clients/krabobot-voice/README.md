# krabobot-voice

Локальный голосовой клиент для канала **voice** в [krabobot](https://github.com/andretisch/krabobot).

Цикл:

1. Ждёт wake-фразу **«Эй, Арнольд»** / **«Привет, Арнольд»** (локальный **sherpa-onnx ASR** по умолчанию), **PTT** hotkey **или** toggle **Meeting**
2. Короткий **beep** (`winsound.Beep` на Windows) — только на первом входе в реплику (не на follow-up)
3. Запись:
   - после wake / follow-up → energy VAD до тишины (до ~15 с), **только микрофон**
   - после PTT → hold-to-talk (пока зажата комбинация), **только микрофон**
   - Meeting → фон.поток до повторного hotkey или голосовой «стоп запись»; `meeting.capture`: `mic` | `loopback` | `mix` (по умолчанию **`mix`**)
4. Локальный sherpa на клипе: если фраза — команда (совещание / выход) → обработать **без** сервера
5. Иначе клиент **выравнивает** WAV → `POST /v1/voice/turn` (STT → агент → TTS; для meeting — ещё `instruct`)
6. Проигрывает ответный WAV
7. **Dialog follow-up:** снова слушает ~`talk.follow_up_s` секунд без wake; при тишине — обратно к шагу 1

## Wake: sherpa ASR (default)

По умолчанию `wake.mode: asr`:

1. Energy-gate на скользящем окне (~2 с)
2. **Preprocess** (DC → 16 kHz mono → trim silence → peak/RMS normalize)
3. Локальный **sherpa-onnx** decode (та же русская модель, что у сервера в `~/.krabobot/models/stt/…`)
4. Fuzzy-match текста на «эй» + «арнольд»

Опционально лёгкий **KWS** (`wake.mode: kws`) — MFCC embedding + cosine к reference WAV, без локального sherpa.

## Выравнивание аудио

Перед **локальным wake ASR** и перед **upload Talk/Meeting** клиент:

- приводит к **16 kHz mono int16** и переписывает WAV-заголовок
- убирает DC offset
- аккуратно обрезает тишину по краям
- поднимает тихий микрофон (RMS/peak normalize без жёсткого клиппинга)

Сервер дополнительно прогоняет тот же alignment в `voice_io` / sherpa STT — защита от «байты есть, transcription empty».

## Требования

- Windows / Python 3.11+
- Работающий **`krabobot serve`** на `http://127.0.0.1:8900`
- Микрофон с разрешением для Python/терминала  
  (Параметры Windows → Конфиденциальность → Микрофон → разрешить классическим приложениям)
- Для PTT: пакет `pynput` (ставится с клиентом)
- Для `wake.mode: asr` (default): `sherpa-onnx` + скачанная STT-модель (`krabobot serve` один раз)

## Установка

Из корня репозитория (активный `.venv`):

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e ".\clients\krabobot-voice[asr]"
```

Без локального wake (только PTT / KWS):

```powershell
pip install -e ".\clients\krabobot-voice"
```

## Запуск

```powershell
# serve должен уже слушать :8900
python -m krabobot_voice
```

Статус: `waiting for wake…` → `listening…` → (`local command` \| `thinking…` → `playing…` → `follow-up listening…`) → снова wake.

### Dialog follow-up

После успешного Talk-turn (wake/PTT → ответ проигран) клиент **не** ждёт «Эй, Арнольд» снова: окно VAD на `talk.follow_up_s` секунд (default `8`). Beep на follow-up по умолчанию выключен (`talk.follow_up_beep: false`). `talk.follow_up_s: 0` — старое поведение (сразу wake). Meeting не затрагивается.

### Локальные команды

При `wake.mode: asr` каждое Talk-высказывание сначала распознаётся локальным sherpa. Совпавшие фразы **не** уходят на `/v1/voice/turn`:

| Команда | Примеры фраз (настраиваются в `talk.*`) |
|---------|----------------------------------------|
| Старт совещания | «начать совещание», «начать запись», «запиши совещание» |
| Стоп совещания | «закончить совещание», «завершить запись», «стоп запись» |
| Выход из диалога | «хватит», «выход», «спокойной ночи», «отмена» |

Hotkey **Ctrl+Alt+M** по-прежнему стартует/останавливает meeting.

### Wake (ASR)

Фраза: **«Эй, Арнольд»**. Нужна модель в `~/.krabobot/models/stt/…` (как у gateway).

### Wake (KWS, optional)

`wake.mode: kws`. Reference WAV в `%LOCALAPPDATA%\krabobot-voice\wake_refs\`.  
Лучше 2–5 своих записей; SAPI auto-enroll — только запасной вариант.

### PTT

По умолчанию: удерживайте **Ctrl+Alt+Space**, говорите, отпустите — клип уходит на сервер.

### Meeting (Teams / звонок)

Wake и PTT **всегда** пишут только микрофон. Для записи встречи (удалённые участники + ваш голос) используйте Meeting:

| `meeting.capture` | Что пишется |
|-------------------|-------------|
| `mix` (**default**) | Микрофон + WASAPI loopback (то, что играет в колонки/наушники) |
| `loopback` | Только системный звук (удалённые в Teams) |
| `mic` | Только микрофон |

По умолчанию hotkey **Ctrl+Alt+M** — старт/стоп. На стопе клиент шлёт выровненный WAV (16 kHz mono int16) + `meeting.instruct` на `/v1/voice/turn`.

**Teams:**

1. В Windows выберите нужное устройство воспроизведения (тот же output, куда играет Teams).
2. В конфиге оставьте `meeting.capture: mix` (или задайте `meeting.loopback_device` / `audio.output_device` — подстрока имени).
3. Лучше **наушники**: при громких колонках `mix` может задвоить ваш голос (echo: mic + loopback).
4. Запустите клиент, в встрече нажмите `Ctrl+Alt+M` → говорите → снова `Ctrl+Alt+M` → резюме уйдёт на сервер.

Loopback идёт через **PyAudioWPatch** (ставится с клиентом на Windows). Штатный `sounddevice` без loopback-сборки PortAudio.

## Конфиг

Приоритет:

1. Путь аргументом: `python -m krabobot_voice C:\path\config.yaml`
2. `%LOCALAPPDATA%\krabobot-voice\config.yaml`
3. Env: `KRABOBOT_URL`, `KRABOBOT_TOKEN`, `KRABOBOT_DEVICE_ID` (также `KRABOBOT_VOICE_*`)
4. Пустой `token` → `~/.krabobot/config.json` → `api.auth.adminToken`

Пример: [`config.example.yaml`](config.example.yaml).

| Ключ / переменная | Смысл |
|-------------------|--------|
| `wake.mode` / `KRABOBOT_VOICE_WAKE_MODE` | `asr` (default) \| `kws` \| `off` |
| `wake.energy_threshold` | Energy-gate перед wake decode |
| `wake.window_s` / `wake.hop_s` | Окно / шаг локального wake |
| `wake.phrases` / `wake.greetings` | Фразы wake и приветствия рядом с «Арнольд» |
| `wake.threshold` | Cosine threshold для KWS |
| `wake.refs_dir` | Папка reference WAV (KWS) |
| `ptt.hotkey` / `KRABOBOT_VOICE_PTT_HOTKEY` | Например `ctrl+alt+space` |
| `talk.follow_up_s` | Окно диалога без wake после ответа (default `8`; `0` = off) |
| `talk.follow_up_beep` | Beep на follow-up (default `false`) |
| `talk.meeting_start` / `talk.meeting_stop` / `talk.exit` | Списки локальных фраз-команд |
| `meeting.capture` | `mix` (default) \| `loopback` \| `mic` |
| `meeting.hotkey` | Toggle старт/стоп (default `ctrl+alt+m`) |
| `meeting.loopback_device` | Имя/индекс loopback; пусто → default output |
| `meeting.instruct` | Текст к upload на стопе |
| `audio.input_device` / `audio.output_device` | Подсказка устройств (подстрока имени) |
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
  app.py         # главный цикл (wake / PTT / meeting / follow-up)
  dialog.py      # follow-up state machine (unit-tested)
  commands.py    # локальные фразы: meeting / exit
  preprocess.py  # 16 kHz / trim / normalize перед ASR и upload
  kws.py         # MFCC embedding KWS (optional)
  ptt.py         # hold-to-talk hotkey
  meeting.py     # meeting start/stop thread + upload
  wake.py        # текстовый fuzzy-match «Эй, Арнольд»
  local_asr.py   # sherpa-onnx offline wake + commands
  vad.py         # energy VAD
  audio_io.py    # mic / WASAPI loopback / mix / beep / play
  link.py        # auto-link device → owner
  http_client.py # POST /v1/voice/turn
  config.py
```
