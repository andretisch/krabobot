# Portable-сборка krabobot-voice (Windows)

Этот файл лежит в репозитории (`clients/krabobot-voice/packaging/`) и **копируется**
рядом с exe при сборке (`build/krabobot-voice-portable/README.md`).

## Что в папке сборки

| Файл / каталог | Назначение |
|----------------|------------|
| `krabobot-voice.exe` | Запуск клиента |
| `_internal/` | Python runtime + зависимости (onedir PyInstaller) |
| `models/silero_vad.onnx` | Silero VAD (бандл, без скачивания) |
| `models/stt/<folder>/` | Sherpa STT для wake ASR (бандл) |
| `config.example.yaml` | Пример конфига |
| `config.yaml` | Рабочий конфиг (**рядом с exe**; создаётся при первом запуске) |
| `README.md` | Эта инструкция |

Сборка **полностью самодостаточна** для VAD и локального wake-STT: первый запуск
работает **офлайн** по моделям из `models/` (скачиваний нет). Нужен только
доступный API-сервер `krabobot serve` (сеть к нему — отдельно).

Конфиг **не** лежит в `%LOCALAPPDATA%` — только рядом с приложением
(или путь через argv / `KRABOBOT_VOICE_CONFIG`).

## Перед первым запуском

1. **Сервер krabobot** должен слушать API (обычно `:8900`):

   ```powershell
   krabobot serve
   ```

2. **Конфиг клиента** — всегда рядом с приложением:

   ```powershell
   # Из папки сборки (рядом с krabobot-voice.exe):
   Copy-Item .\config.example.yaml .\config.yaml   # если ещё нет
   notepad .\config.yaml
   ```

   При первом запуске, если `config.yaml` / `config.yml` нет, клиент сам
   скопирует `config.example.yaml` → `config.yaml` рядом с exe.

   То же правило для `python -m krabobot_voice`: `config.yaml` в корне пакета
   (`clients/krabobot-voice/`). Overrides: argv, `KRABOBOT_VOICE_CONFIG`,
   `KRABOBOT_VOICE_CONFIG_DIR`.

3. В `config.yaml` проверьте как минимум:

   | Ключ | Что задать |
   |------|------------|
   | `wake.phrase` | Ваша wake-фраза (например `"Ок, Бот"`) |
   | `base_url` | URL API, обычно `http://127.0.0.1:8900` |
   | `device_id` | Идентификатор устройства (уникальный на ПК) |
   | `token` | Пусто → берётся `api.auth.adminToken` из `~/.krabobot/config.json` |
   | `audio.listen_source` | `mic` (по умолчанию) или `loopback` (WASAPI, для тестов) |

4. **Привязка устройства** — при первом запуске клиент линкует `device_id` к серверу
   (нужен валидный token / adminToken).

5. **Микрофон** — при первом запуске разрешите микрофон в системном окне
   Windows (клиент запрашивает доступ сам через WinRT). Portable exe и
   `python -m` — разные записи в Privacy. Если доступа нет — одно нажатие OK
   откроет Параметры. При сбое открытия:
   `.\krabobot-voice.exe --list-devices`.

## Запуск

Из папки сборки (или с полным путём):

```powershell
.\krabobot-voice.exe
```

Список входных устройств (если mic не открывается):

```powershell
.\krabobot-voice.exe --list-devices
```

Опционально — свой конфиг:

```powershell
.\krabobot-voice.exe D:\path\to\config.yaml
```

Ожидаемый статус в UI / логе: `waiting for wake / PTT / meeting…`

## UI (portable)

Сборка windowed (`console=False`): tray + окно в том же процессе, что и voice loop.
Конфиг и папка `meetings/` — рядом с exe. Кнопка Apply в окне сохраняет YAML и
перезапускает loop (не hot-reload устройств mid-stream).

## Модели (в комплекте)

| Компонент | Где лежит | Примечание |
|-----------|-----------|------------|
| Silero VAD | `models/silero_vad.onnx` | Бандл; portable **не** скачивает |
| Sherpa STT (wake ASR) | `models/stt/<folder>/` | Бандл; `stt_model_dir` пустой → этот путь |
| Записи совещаний | `<app>/meetings/` | рядом с exe; legacy LocalAppData не мигрируется |

Если модели удалили — положите файлы обратно в `models/` (см. таблицу выше).
Клиент выдаст явную ошибку «положите модель в models/…», а не тихое скачивание.

Переопределение STT (опционально):

```yaml
stt_model_dir: "D:/path/to/sherpa-onnx-…"
```

## Пересборка из исходников

Нужны Python 3.11+, venv репозитория. Скрипт копирует Silero и STT с этой машины
в `models/` рядом с exe:

| Источник (первый найденный) | Куда в сборке |
|-----------------------------|----------------|
| `clients/krabobot-voice/models/silero_vad.onnx` или `%LOCALAPPDATA%\krabobot-voice\models\silero_vad.onnx` | `models/silero_vad.onnx` |
| `clients/krabobot-voice/models/stt/<preferred>/` или `~/.krabobot/models/stt/…` | `models/stt/<имя>/` |

Preferred STT: `sherpa-onnx-nemo-transducer-punct-giga-am-v3-russian-2025-12-16`
(если нет — любой каталог с `tokens.txt` под `~/.krabobot/models/stt/`).

```powershell
# из корня репозитория krabobot
.\.venv\Scripts\Activate.ps1
pip install -e ".\clients\krabobot-voice[asr,ui,packaging]"
.\clients\krabobot-voice\scripts\build_portable.ps1
```

**Готовый каталог для раздачи:**  
`clients\krabobot-voice\build\krabobot-voice-portable\`

(Промежуточный PyInstaller dist: `build\krabobot-voice\` — тот же состав;
скрипт синхронизирует его в `-portable` и сохраняет ваш `config.yaml` при
пересборке.)

Каталоги `build/` и `dist/` в git не коммитятся. Spec: `packaging/krabobot-voice.spec`.

| Флаг | Смысл |
|------|--------|
| `-SkipInstall` | пропустить `pip install`, если deps уже стоят |
| `-SileroSource <path>` | явный путь к `silero_vad.onnx` |
| `-SttSource <dir>` | явный каталог STT (нужен `tokens.txt`) |

Подробнее: [`../README.md`](../README.md) → раздел **Portable build**.

## Устранение неполадок

- **Нет связи с API** — убедитесь, что `krabobot serve` запущен и `base_url` совпадает.
- **Нет STT/VAD в models/** — восстановите файлы в `models/` или пересоберите portable.
- **Микрофон / PaErrorCode -9996** — клиент сам запрашивает доступ (WinRT) при
  старте; при отказе — диалог → Параметры. Проверьте `--list-devices` и
  `audio.input_device`.
- **Тихий выход после `mic access: allowed`** — на Windows нельзя грузить
  `winrt` раньше `onnxruntime` (ACCESS_VIOLATION). Клиент прелоадит onnx
  до запроса микрофона; если правили код — сохраните этот порядок.
- **Wake не срабатывает** — проверьте `wake.phrase`, микрофон, `audio.listen_source`.
- **Meeting / loopback** — нужен WASAPI (Windows); в сборке должен быть `PyAudioWPatch`.
- **Отладка** — `$env:KRABOBOT_VOICE_DEBUG = "1"` перед запуском exe.
