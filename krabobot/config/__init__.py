"""Configuration module for krabobot."""

from krabobot.config.loader import get_config_path, load_config
from krabobot.config.paths import (
    get_cli_history_path,
    get_data_dir,
    is_default_workspace,
    get_logs_dir,
    get_media_dir,
    get_runtime_subdir,
    get_workspace_path,
)
from krabobot.config.schema import Config

__all__ = [
    "Config",
    "load_config",
    "get_config_path",
    "get_data_dir",
    "get_runtime_subdir",
    "get_media_dir",
    "get_logs_dir",
    "get_workspace_path",
    "is_default_workspace",
    "get_cli_history_path",
]
