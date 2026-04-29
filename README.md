# krabobot (Migration Baseline)

This repository is now a **simplified baseline** intended for continuation in another project.

The original upstream scope was broader (many providers/channels/docs). This branch intentionally reduces complexity to make migration and refactoring easier.

## Why This Version Exists

- Reduce moving parts before extraction to a new codebase.
- Keep core agent/runtime/channel flow intact.
- Remove provider-specific complexity that blocks iteration.
- Document what remains so the next project can continue quickly.

## Current State (Important)

- Provider layer is now focused on `OpenAI-compatible` backend flow.
- Specialized provider implementations were removed from this codebase.
- Multi-user/session isolation groundwork is present (user-scoped runtime paths).
- Configuration and docs were trimmed to reduce noise.

## Removed in This Branch

The following provider implementations were intentionally removed:

- `krabobot/providers/anthropic_provider.py`
- `krabobot/providers/azure_openai_provider.py`
- `krabobot/providers/openai_codex_provider.py`
- `krabobot/providers/qwen_oauth_provider.py`

If you need any of these in the new project, reintroduce them as standalone modules with clear ownership and tests.

## What Still Works

- CLI entrypoints (`krabobot agent`, `krabobot gateway`, `krabobot serve`) 
- Channel routing and command dispatch
- Core agent loop and tool execution flow
- OpenAI-compatible provider path via `OpenAICompatProvider`
- Session/memory persistence and consolidation flow

## Migration Guide for New Project

### 1) Copy Core Runtime First

Recommended minimal set to migrate first:

- `krabobot/agent/*`
- `krabobot/bus/*`
- `krabobot/channels/*` (only channels you actively use)
- `krabobot/command/*`
- `krabobot/session/*`
- `krabobot/providers/base.py`
- `krabobot/providers/openai_compat_provider.py`
- `krabobot/providers/registry.py`
- `krabobot/config/*`

### 2) Keep One Provider Strategy Initially

Start with a single OpenAI-compatible endpoint and validate end-to-end behavior before adding new providers.

### 3) Add Providers as Isolated Plug-ins

For each new provider in the next project:

- one module per provider
- one focused test file
- explicit config schema
- explicit fallback behavior

### 4) Keep Config Lean

Avoid carrying full upstream provider matrix unless needed. Add config sections only when the feature is live.

## Minimal Config Example

```json
{
  "agents": {
    "defaults": {
      "workspace": "~/.krabobot/workspace",
      "model": "openai/gpt-4o-mini",
      "provider": "auto"
    }
  },
  "providers": {
    "openai": {
      "apiKey": "<your-key>"
    }
  },
  "channels": {
    "telegram": {
      "enabled": true,
      "token": "<bot-token>",
      "allowFrom": ["<your-user-id>"]
    }
  }
}
```

## Run

```bash
krabobot onboard
krabobot gateway
```

or for direct local chat:

```bash
krabobot agent
```

## Notes for Maintainers

- This README intentionally replaces previous long-form product documentation.
- Treat this repository as a transition snapshot, not a full-feature upstream mirror.
- Keep commits migration-oriented: smaller surface, fewer abstractions, stronger tests.

## License

MIT (same as repository).
