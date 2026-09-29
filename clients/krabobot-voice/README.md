# krabobot-voice

Локальный голосовой клиент для канала **voice** в [krabobot](https://github.com/andretisch/krabobot).

Цикл:

1. Ждёт **вашу** wake-фразу из конфига (локальный **sherpa-onnx ASR** по умолчанию), **PTT** hotkey **или** toggle **Meeting**
2. **2 beep** при wake (фраза или wake+команда в одном сегменте)
3. Запись одним persistent VAD-сегментером (preroll/state не сбрасываются между фазами):
   - wake only → окно `talk.listen_timeout_s` (default 10 с) до следующей реплики
   - wake + текст после фразы → команда сразу (без второго listen)
   - PTT → hold-to-talk; тот же `audio.listen_source` (`mic` \| `loopback`)
   - Meeting → отдельный процесс пишет WAV в `meetings/`; mic-tap для стоп-фразы; `meeting.capture`: `mic` | `loopback` | `mix`
4. Локальный sherpa: команда (совещание / выход) → локально; иначе `POST /v1/voice/turn` (+ `client_state`)
5. Проигрывает ответный WAV; сервер может вернуть actions (`meeting_stop` / `end_dialog` / …)
6. **Dialog follow-up:** сегменты без wake до `talk.follow_up_s` тишины → **1 beep** → idle

## Wake: sherpa ASR (default)

По умолчанию `wake.mode: asr`:

1. Energy/Silero VAD: `speech_start` → буфер до `speech_end` (+ preroll), cap `wake.max_s`
   - Default **`vad.backend: silero`** (ONNX, `onnxruntime`) — отличает речь от музыки
   - Fallback **`vad.backend: energy`** — только RMS (для тестов без onnxruntime)
   - Early ASR/KWS после `speech_start`: первый probe ≤ `wake.early_asr_s` (0.8 с),
     далее ~каждые 0.6 с до silence/`max_s`; match → сразу (idle wake + meeting mic-tap)
2. **Preprocess** (DC → 16 kHz mono → trim silence → peak/RMS normalize) — внутри `LocalWakeAsr.transcribe_pcm16`
3. Локальный **sherpa-onnx** decode на закрытый сегмент **или** early-match (не sliding window)
4. Fuzzy-match текста на вашу `wake.phrase` / `wake.phrases` (части фразы + edit-distance)
5. Miss → discard (тихо, кроме `KRABOBOT_VOICE_DEBUG`); сразу готов к следующему VAD-сегменту

Опционально лёгкий **KWS** (`wake.mode: kws`) — тот же VAD, затем один MFCC cosine score к reference WAV (без spam по hop).

### Своя wake-фраза

Конфиг: `config.yaml` рядом с приложением (см. [`config.example.yaml`](config.example.yaml)).

```yaml
wake:
  mode: asr
  phrase: "Эй, Арнольд"   # ← задайте свою: "Ок, Бот", "Привет, Краб" …
  # phrases:               # опциональные алиасы
  #   - hey arnold
  silence_end_s: 0.55      # короче talk — snappy wake
  max_s: 2.0               # cap wake/команды (≤2 с by design)
  min_speech_s: 0.45
  energy_threshold: 0.006  # ниже = чувствительнее к тихому микрофону
```

`wake.phrase` — **основной** ключ (одна строка, запятая внутри — пунктуация).  
`wake.phrases` — список алиасов; если оба заданы, `phrase` идёт первым.  
`wake.greetings` можно не писать: клиент выведет приветствия из первых слов `phrase`.

## Выравнивание аудио

Перед **локальным wake ASR** и перед **upload Talk/Meeting** клиент:

- приводит к **16 kHz mono int16** и переписывает WAV-заголовок
- убирает DC offset
- аккуратно обрезает тишину по краям
- поднимает тихий микрофон (RMS/peak normalize без жёсткого клиппинга)

Точки вызова:

| Путь | Функция |
|------|---------|
| Wake ASR | `local_asr.LocalWakeAsr.transcribe_pcm16` → `preprocess.preprocess_pcm16` |
| Talk upload | `app._pcm_to_upload_wav` → `preprocess.preprocess_pcm16` |
| Meeting upload | `app._handle_meeting_upload` → multipart `files` (не `audio`) |

Сервер дополнительно прогоняет тот же alignment в `voice_io` / sherpa STT — защита от «байты есть, transcription empty».

## Требования

- Windows / Python 3.11+
- Работающий **`krabobot serve`** на `http://127.0.0.1:8900`
- Микрофон с разрешением для Python/терминала  
  (Параметры Windows → Конфиденциальность → Микрофон → разрешить классическим приложениям)
- Для PTT: пакет `pynput` (ставится с клиентом)
- Для `wake.mode: asr` (default): `sherpa-onnx` + скачанная STT-модель (`krabobot serve` один раз)
- Для Silero VAD (default): `onnxruntime` (ставится с `[asr]` / `[vad]`); модель скачивается один раз в `%LOCALAPPDATA%\krabobot-voice\models\`

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

Статус (один раз при входе в idle): `waiting for wake / PTT / meeting…` → на каждом закрытом VAD-сегменте одна строка `[wake-asr] …` → при match `wake: …` → `listening…` → (`local command` \| `thinking…` → `playing…` → `follow-up listening…`) → снова wake. Строка `waiting…` не печатается каждые N секунд / hop.

### Dialog follow-up

После успешного Talk-turn клиент слушает без wake окно `talk.follow_up_s` (default `10`). Тишина → **1 beep** → idle. `talk.follow_up_s: 0` — сразу idle после ответа. Post-wake окно: `talk.listen_timeout_s` (default `10`; fallback `no_speech_timeout_s`). Cues: `talk.beeps` (default `true`) — 2 beep на wake, 1 на возврат в idle.

### Локальные команды

При `wake.mode: asr` каждое Talk-высказывание сначала распознаётся локальным sherpa. Совпавшие фразы **не** уходят на `/v1/voice/turn`:

| Команда | Примеры фраз (настраиваются в `talk.*`) |
|---------|----------------------------------------|
| Старт совещания | «начать совещание», «начать запись», «запиши совещание» |
| Стоп совещания | «закончить запись совещания», «закончить совещание», «стоп запись» |
| Локальный тест | «выполни тест», «выполнить тест», «сделай тест», … |
| Выход из диалога | «хватит», «выход», «спокойной ночи», «отмена» |

Hotkey **Ctrl+Alt+M** по-прежнему стартует/останавливает meeting.

### Wake (ASR)

Фраза задаётся в `wake.phrase` / `wake.phrases`. Нужна модель в `~/.krabobot/models/stt/…` (как у gateway).

Чувствительность (defaults):

| Параметр | Default | Зачем |
|----------|---------|--------|
| `wake.energy_threshold` | `0.006` | Тихий mic всё ещё попадает в VAD (loopback: автониже) |
| `wake.silence_end_s` | `0.55` | Короче talk — быстрый закрытие wake-фразы |
| `wake.max_s` | `2.0` | Cap wake/команды (и meeting mic-tap); ≤2 с by design |
| `wake.early_asr_s` | `0.8` | Первый early ASR/KWS после speech_start (далее ~0.6 с) |
| `wake.min_speech_s` | `0.45` | Короткие «эй арнольд» не отбрасываются |

### VAD (Silero)

По умолчанию `vad.backend: silero` — ONNX-модель [snakers4/silero-vad](https://github.com/snakers4/silero-vad) через `onnxruntime` (без PyTorch). Отличает **речь** от музыки/шума, поэтому wordless music на loopback не держит сегмент 7–15 с.

```yaml
vad:
  backend: silero       # или energy (только RMS)
  threshold: 0.5
  energy_pregate: 0.0008
```

Первый запуск скачивает `silero_vad.onnx` в `%LOCALAPPDATA%\krabobot-voice\models\`. Без `onnxruntime` клиент пишет WARNING и падает на `energy`.

### Wake (KWS, optional)

`wake.mode: kws`. Reference WAV в `%LOCALAPPDATA%\krabobot-voice\wake_refs\`.  
Лучше 2–5 своих записей; SAPI auto-enroll — только запасной вариант.

### Wake без микрофона (loopback + TTS)

Чтобы не говорить wake каждый раз при отладке ASR:

1. В `config.yaml` рядом с приложением:

```yaml
audio:
  listen_source: loopback   # mic | loopback (default: mic)
  # output_device: ""       # подстрока имени playback, если нужно
```

Алиас: `wake.input: loopback` (то же поле).

2. **Проверьте, что loopback слышит звук** (видео/TTS должно играть на том же выходе):

```powershell
python -m krabobot_voice --test-loopback
```

Ожидание: `RMS max` заметно больше `0` (примерно `> 0.01` при нормальной громкости). Если `RMS≈0` — звук играет на другом устройстве: задайте `audio.output_device` / `meeting.loopback_device` (подстрока имени) или смените default playback в Windows.

3. Перезапустите клиент. В логе старта: `audio.listen_source=loopback`, имя/index loopback-устройства и `loopback probe 1s: rms_…`.
4. Воспроизведите речь с колонками/наушников (TTS / зацикленное видео), например:
   «Давай скажем: Эй, Арнольд» — wake ASR должен услышать фразу через **WASAPI loopback** и сработать даже с filler в начале.
5. Для повседневной работы верните `listen_source: mic`.

**Наушники vs колонки:** при loopback-тесте наушники удобнее (меньше шума комнаты).  
Если позже смешиваете mic+loopback (meeting `mix`), **рекомендуются наушники** — иначе echo: ваш голос попадает и в mic, и в loopback с колонок.

В режиме `loopback` тот же источник используется и для Talk/follow-up/PTT (удобно для теста). Meeting по-прежнему задаётся отдельно через `meeting.capture`.

### PTT

По умолчанию: удерживайте **Ctrl+Alt+Space**, говорите, отпустите — клип уходит на сервер.

### Meeting (Teams / звонок)

Для записи встречи (удалённые участники + ваш голос) используйте Meeting (независимо от `audio.listen_source`):

| `meeting.capture` | Что пишется |
|-------------------|-------------|
| `mix` (**default**) | Микрофон + WASAPI loopback (то, что играет в колонки/наушники) |
| `loopback` | Только системный звук (удалённые в Teams) |
| `mic` | Только микрофон |

По умолчанию hotkey **Ctrl+Alt+M** — старт/стоп. Запись идёт в **отдельном процессе** и сразу пишется на диск:

`%LOCALAPPDATA%\krabobot-voice\meetings\YYYYMMDD-HHMMSS.wav`

(или `meeting.save_dir`). На стопе («закончить запись совещания» / hotkey) клиент шлёт WAV как multipart **`files`** + `async=1` — **не ждёт** ответ бота/TTS; в логе: `meeting queued — результат придёт на почту`. Сервер обрабатывает запись в фоне и шлёт письмо **owner** на первый привязанный аккаунт `email:…` (веб-UI → Users → owner → Links). Нужны `channels.email` (SMTP) и `consentGranted: true`; должен работать **gateway** (или spool outbound при `krabobot serve`). Вложение — Markdown с протоколом; тело письма — краткое резюме.

`meeting.upload_as: audio` — устаревший путь (короткие клипы через STT); для длинных совещаний не используйте.

**Teams:**

1. В Windows выберите нужное устройство воспроизведения (тот же output, куда играет Teams).
2. В конфиге оставьте `meeting.capture: mix` (или задайте `meeting.loopback_device` / `audio.output_device` — подстрока имени).
3. Лучше **наушники**: при громких колонках `mix` может задвоить ваш голос (echo: mic + loopback).
4. Запустите клиент, в встрече нажмите `Ctrl+Alt+M` → говорите → снова `Ctrl+Alt+M` → резюме уйдёт на сервер.

Loopback идёт через **PyAudioWPatch** (ставится с клиентом на Windows). Штатный `sounddevice` без loopback-сборки PortAudio.

## Конфиг

Конфиг всегда рядом с приложением (не `%LOCALAPPDATA%`). Приоритет:

1. Путь аргументом: `python -m krabobot_voice C:\path\config.yaml` / `krabobot-voice.exe C:\path\config.yaml`
2. `KRABOBOT_VOICE_CONFIG` — явный путь к файлу (env)
3. `config.yaml`, затем `config.yml` в каталоге приложения:
   - frozen: родитель `krabobot-voice.exe`
   - `python -m` / editable: корень пакета (`clients/krabobot-voice/`)
   - override каталога: `KRABOBOT_VOICE_CONFIG_DIR`
   - если файла нет — копия `config.example.yaml` → `config.yaml` в том же каталоге
4. Env (часть переменных **перекрывает** YAML, если заданы — см. таблицу)
5. Пустой `token` → `~/.krabobot/config.json` → `api.auth.adminToken`

Пример: [`config.example.yaml`](config.example.yaml).

| Ключ / переменная | Смысл |
|-------------------|--------|
| `wake.mode` / `KRABOBOT_VOICE_WAKE_MODE` | `asr` (default) \| `kws` \| `off` |
| `wake.phrase` / `KRABOBOT_VOICE_WAKE_PHRASE` | **Основная** wake-фраза (строка); env перекрывает YAML |
| `wake.phrases` | Алиасы (список); мержится с `phrase` |
| `wake.greetings` | Приветствия для fuzzy name; иначе из `phrase` |
| `wake.energy_threshold` | Energy-gate для VAD speech_start (loopback: автониже) |
| `wake.silence_end_s` / `wake.max_s` / `wake.min_speech_s` | VAD wake: тишина / cap / min voiced |
| `silence_end_s` | После речи в Talk/LISTEN: закрыть сегмент (~`2.0` с тишины) |
| `talk.listen_timeout_s` / `talk.follow_up_s` | Пустое окно без речи (~`10` с) → idle |
| `wake.threshold` | Cosine threshold для KWS |
| `wake.refs_dir` | Папка reference WAV (KWS) |
| `ptt.hotkey` / `KRABOBOT_VOICE_PTT_HOTKEY` | Например `ctrl+alt+space` |
| `talk.follow_up_s` | Окно диалога без wake после ответа (default `8`; `0` = off) |
| `talk.follow_up_beep` | Beep на follow-up (default `false`) |
| `talk.meeting_start` / `talk.meeting_stop` / `talk.run_test` / `talk.exit` | Списки локальных фраз-команд |
| `meeting.capture` | `mix` (default) \| `loopback` \| `mic` |
| `meeting.hotkey` | Toggle старт/стоп (default `ctrl+alt+m`) |
| `meeting.loopback_device` | Имя/индекс loopback; пусто → default output |
| `meeting.save_dir` | Локальный каталог WAV; пусто → `%LOCALAPPDATA%\krabobot-voice\meetings` |
| `meeting.upload_as` | `file` (default, multipart files) \| `audio` (legacy STT) |
| `meeting.instruct` | Текст к upload на стопе |
| `audio.listen_source` / `KRABOBOT_VOICE_LISTEN_SOURCE` | `mic` (default) \| `loopback` — wake + Talk/PTT; env перекрывает YAML |
| `audio.input_device` / `audio.output_device` | Подсказка устройств (подстрока имени) |
| `KRABOBOT_URL` | Base URL (по умолчанию `http://127.0.0.1:8900`) |
| `KRABOBOT_DEVICE_ID` | Стабильный device id |
| `KRABOBOT_TOKEN` | Bearer |

Быстрый override без правки файла:

```powershell
$env:KRABOBOT_VOICE_LISTEN_SOURCE = "loopback"   # или VOICE_LISTEN_SOURCE
$env:KRABOBOT_VOICE_WAKE_PHRASE = "Эй, Арнольд"  # или VOICE_WAKE_PHRASE
python -m krabobot_voice
```

При старте клиент:

- берёт `device_id` (по умолчанию `local-<hostname>`)
- читает Bearer-токен из конфига / env / `api.auth.adminToken`
- **автоматически привязывает** `voice:<device_id>` к owner через `GET /v1/web/users` + `POST .../links`

## Если процесс завис (Ctrl+C не помогает)

Старые версии могли зависнуть в `PyAudio.stream.read` (C-вызов) после wake ASR —
Python не доставляет `KeyboardInterrupt`, пока PortAudio не вернётся.

1. Закройте окно терминала **или** убейте процесс:
   ```powershell
   # найти PID
   Get-Process python* | Format-Table Id, ProcessName, Path
   # или по командной строке:
   Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
     Where-Object { $_.CommandLine -match 'krabobot_voice' } |
     Select-Object ProcessId, CommandLine
   taskkill /PID <pid> /F
   ```
2. Task Manager → завершить `python.exe` с `krabobot_voice`.
3. Обновите клиент на эту ветку: чтение loopback/mic теперь с таймаутом + drain во время ASR, Ctrl+C снова работает.

## Portable build (Windows / PyInstaller)

Onedir-сборка без установленного Python на целевой машине:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e ".\clients\krabobot-voice[asr,packaging]"
.\clients\krabobot-voice\scripts\build_portable.ps1
```

Результат: `clients/krabobot-voice/build/krabobot-voice/krabobot-voice.exe`
(+ `_internal/`, `config.example.yaml`, `README.md`).

Инструкция для пользователя сборки: [`packaging/README.md`](packaging/README.md)
(копируется рядом с exe). Конфиг всегда `config.yaml` рядом с приложением
(exe или корень пакета; не LocalAppData). Модели STT/VAD не бандлятся —
runtime-пути как у обычного клиента
(`~/.krabobot/models/stt/…`, `%LOCALAPPDATA%\krabobot-voice\models\`).

Каталоги `build/` и `dist/` в `.gitignore`. Spec: `packaging/krabobot-voice.spec`.

## Layout

```
krabobot_voice/
  app.py         # главный цикл (wake / PTT / meeting / follow-up)
  dialog.py      # follow-up state machine (unit-tested)
  commands.py    # локальные фразы: meeting / exit
  preprocess.py  # 16 kHz / trim / normalize перед ASR и upload
  kws.py         # MFCC embedding KWS (optional)
  ptt.py         # hold-to-talk hotkey
  meeting.py     # meeting subprocess capture + local WAV
  meeting_worker.py  # CLI entry: python -m krabobot_voice.meeting_worker
  wake.py        # текстовый fuzzy-match по wake.phrase
  local_asr.py   # sherpa-onnx offline wake + commands
  vad.py         # energy VAD
  audio_io.py    # mic / WASAPI loopback / mix / beep / play
  link.py        # auto-link device → owner
  http_client.py # POST /v1/voice/turn
  config.py
packaging/
  krabobot-voice.spec   # PyInstaller onedir
  README.md             # инструкция рядом с portable-сборкой
scripts/
  build_portable.ps1    # сборка → build/krabobot-voice/
```
