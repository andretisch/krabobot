"""Sync display name into a user workspace USER.md."""

from __future__ import annotations

import re
from pathlib import Path

_NAME_LINE = re.compile(r"(-\s*\*\*Name\*\*\s*:\s*).*", re.IGNORECASE)


def update_user_md_display_name(workspace: Path, display_name: str) -> Path:
    """Create or update USER.md so the Name field matches *display_name*.

    Returns the path written.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / "USER.md"
    name = (display_name or "").strip() or "(your name)"
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        new_text, n = _NAME_LINE.subn(rf"\g<1>{name}", text, count=1)
        if n:
            path.write_text(new_text, encoding="utf-8")
            return path
        # No Name line — insert under Basic Information or prepend a short block.
        if "## Basic Information" in text:
            text = text.replace(
                "## Basic Information",
                f"## Basic Information\n\n- **Name**: {name}",
                1,
            )
            path.write_text(text, encoding="utf-8")
            return path
        path.write_text(
            f"# User Profile\n\n- **Name**: {name}\n\n{text}",
            encoding="utf-8",
        )
        return path

    path.write_text(
        "# User Profile\n\n"
        "Information about the user to help personalize interactions.\n\n"
        "## Basic Information\n\n"
        f"- **Name**: {name}\n"
        "- **Timezone**: (your timezone, e.g., UTC+8)\n"
        "- **Language**: (preferred language)\n",
        encoding="utf-8",
    )
    return path
