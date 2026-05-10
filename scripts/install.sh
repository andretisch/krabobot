#!/usr/bin/env bash
# Install krabobot from a git checkout: venv + editable install + optional onboard.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

VENV_DIR="${VENV_DIR:-.venv}"
EXTRAS="${EXTRAS:-dev,api}"
SKIP_VENV=0
SKIP_ONBOARD=0
YES=0

_usage() {
    cat <<'EOF'
Установка krabobot из каталога репозитория.

Использование:
  scripts/install.sh [опции]

Опции:
  --venv PATH      Каталог venv (по умолчанию: .venv в корне репозитория)
  --extras LIST    Extras для pip, через запятую (по умолчанию: dev,api)
  --skip-venv      Не создавать venv; pip install в текущий python
  --skip-onboard   Не предлагать krabobot onboard в конце
  -y, --yes        Не задавать вопросов (onboard не запускается)
  -h, --help       Эта справка

Переменные окружения:
  VENV_DIR, EXTRAS — то же, что флаги --venv и --extras

Требования: Python 3.11+
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)
            VENV_DIR="${2:?}"
            shift 2
            ;;
        --extras)
            EXTRAS="${2:?}"
            shift 2
            ;;
        --skip-venv)
            SKIP_VENV=1
            shift
            ;;
        --skip-onboard)
            SKIP_ONBOARD=1
            shift
            ;;
        -y|--yes)
            YES=1
            shift
            ;;
        -h|--help)
            _usage
            exit 0
            ;;
        *)
            echo "Неизвестный аргумент: $1" >&2
            _usage >&2
            exit 1
            ;;
    esac
done

if ! command -v python3 >/dev/null 2>&1; then
    echo "Ошибка: нужен python3 в PATH." >&2
    exit 1
fi

if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
    echo "Ошибка: нужен Python 3.11 или новее (текущий: $(python3 -V))." >&2
    exit 1
fi

PIP=(python3 -m pip)
ACTIVATE=""

if [[ "${SKIP_VENV}" -eq 0 ]]; then
    VENV_ABS="${ROOT}/${VENV_DIR}"
    echo "[install] venv: ${VENV_ABS}"
    if [[ ! -d "${VENV_ABS}" ]]; then
        python3 -m venv "${VENV_ABS}"
    fi
    # shellcheck source=/dev/null
    source "${VENV_ABS}/bin/activate"
    PIP=(python -m pip)
    ACTIVATE="source \"${VENV_ABS}/bin/activate\""
    echo "[install] активируйте окружение: ${ACTIVATE}"
fi

echo "[install] обновление pip…"
"${PIP[@]}" install --upgrade pip

EXTRA_SPEC=""
if [[ -n "${EXTRAS//,/}" ]]; then
    EXTRA_SPEC="[${EXTRAS}]"
fi

echo "[install] pip install -e .${EXTRA_SPEC}"
"${PIP[@]}" install -e ".${EXTRA_SPEC}"

echo "[install] готово: команда krabobot должна быть в PATH этого окружения."

if command -v krabobot >/dev/null 2>&1; then
    echo "[install] krabobot: $(command -v krabobot)"
    krabobot --version || true
fi

RUN_ONBOARD=0
if [[ "${SKIP_ONBOARD}" -eq 0 && "${YES}" -eq 0 ]]; then
    if [[ -t 0 ]]; then
        read -r -p "Запустить «krabobot onboard» сейчас? [y/N] " ans || true
        case "${ans:-}" in
            y|Y|yes|YES|да|Да) RUN_ONBOARD=1 ;;
            *) ;;
        esac
    fi
fi

if [[ "${RUN_ONBOARD}" -eq 1 ]]; then
    echo "[install] запуск krabobot onboard…"
    krabobot onboard
fi
