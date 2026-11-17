# Heartbeat Tasks

Built-in scheduler task (default every 30 min via `gateway.heartbeat.intervalS`). User cron jobs in `cron/jobs.json` are checked every minute (`tickIntervalS`).
Add tasks below that you want the agent to work on periodically.

If this file has no unfinished tasks (only headers and comments), the heartbeat skips the turn.

Unfinished work is always picked up: unchecked `- [ ]` items, list items, and lines under Active Tasks. The Completed section is ignored.

## Active Tasks

<!-- Add your periodic tasks below this line -->


## Completed

<!-- Move completed tasks here or delete them -->

