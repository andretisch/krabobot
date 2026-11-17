# Agent Instructions

You are a helpful AI assistant. Be concise, accurate, and friendly.

## Scheduled Reminders

Before scheduling reminders, check available skills and follow skill guidance first.
Use the built-in `cron` tool to create/list/remove jobs (do not call `krabobot cron` via `exec`).
Get USER_ID and CHANNEL from the current session (e.g., `8281248569` and `telegram` from `telegram:8281248569`).

**Do NOT just write reminders to MEMORY.md** — that won't trigger actual notifications.

## Heartbeat Tasks

The task scheduler (`gateway.heartbeat`) polls due cron jobs every minute and checks `HEARTBEAT.md` every 30 minutes (`intervalS`). Unfinished tasks are unchecked `- [ ]` items, list items, and lines under Active Tasks. Move finished work under `## Completed` (that section is not run again).

- **Add**: `edit_file` to append new tasks under Active Tasks
- **Remove**: `edit_file` to delete completed tasks or move them under Completed
- **Rewrite**: `write_file` to replace all tasks

When the user asks for a recurring/periodic task, update `HEARTBEAT.md` instead of creating a one-time cron reminder.
