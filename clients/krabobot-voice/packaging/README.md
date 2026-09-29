# Portable-сборка krabobot-voice (Windows)

Этот файл лежит в репозитории (`clients/krabobot-voice/packaging/`) и **копируется**
рядом с exe при сборке (`build/krabobot-voice/README.md`).

## Что в папке сборки

| Файл / каталог | Назначение |
|----------------|------------|
| `krabobot-voice.exe` | Запуск клиента |
| `_internal/` | Python runtime + зависимости (onedir PyInstaller) |
| `config.example.yaml` | Пример конфига |
| `config.yaml` | Рабочий конфиг (рядом с exe; создаётся при первом запуске) |
| `README.md` | Эта инструкция |

Модели STT/VAD **не** упакованы (сотни МБ–ГБ). Клиент использует те же пути, что и при запуске из Python.

## Перед первым запуском

1. **Сервер krabobot** должен слушать API (обычно `:8900`):

   ```powershell
   krabobot serve
   ```

   Один раз после установки gateway скачает STT-модель в `~/.krabobot/models/stt/…`
   (нужна для `wake.mode: asr`).

2. **Конфиг клиента** — всегда рядом с приложением (не `%LOCALAPPDATA%`):

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

5. Windows: разрешите микрофон классическим приложениям
   (Параметры → Конфиденциальность → Микрофон).

## Запуск

Из папки сборки (или с полным путём):

```powershell
.\krabobot-voice.exe
```

Опционально — свой конфиг:

```powershell
.\krabobot-voice.exe D:\path\to\config.yaml
```

Ожидаемый статус в консоли: `waiting for wake / PTT / meeting…`

## Модели (runtime, не в exe)

| Компонент | Где лежит | Как появляется |
|-----------|-----------|----------------|
| Sherpa STT (wake ASR) | `~/.krabobot/models/stt/…` | `krabobot serve` один раз; либо `stt_model_dir` в YAML |
| Silero VAD | `%LOCALAPPDATA%\krabobot-voice\models\silero_vad.onnx` | авто-скачивание при первом запуске |
| Записи совещаний | `%LOCALAPPDATA%\krabobot-voice\meetings\` | при meeting-режиме |

Переопределение STT:

```yaml
stt_model_dir: "C:/Users/YOU/.krabobot/models/stt/sherpa-onnx-…"
```

## Пересборка из исходников

Нужны Python 3.11+, venv репозитория и сеть (для pip).

```powershell
# из корня репозитория krabobot
.\.venv\Scripts\Activate.ps1
pip install -e ".\clients\krabobot-voice[asr,packaging]"
.\clients\krabobot-voice\scripts\build_portable.ps1
```

Готовый onedir: `clients\krabobot-voice\build\krabobot-voice\`

Каталоги `build/` и `dist/` в git не коммитятся. Spec: `packaging/krabobot-voice.spec`.

Флаг `-SkipInstall` у скрипта пропускает `pip install`, если зависимости уже стоят.

## Устранение неполадок

- **Нет связи с API** — убедитесь, что `krabobot serve` запущен и `base_url` совпадает.
- **Нет STT-модели** — запустите сервер один раз или укажите `stt_model_dir`.
- **Wake не срабатывает** — проверьте `wake.phrase`, микрофон, `audio.listen_source`.
- **Meeting / loopback** — нужен WASAPI (Windows); в сборке должен быть `PyAudioWPatch`.
- **Отладка** — `$env:KRABOBOT_VOICE_DEBUG = "1"` перед запуском exe.
