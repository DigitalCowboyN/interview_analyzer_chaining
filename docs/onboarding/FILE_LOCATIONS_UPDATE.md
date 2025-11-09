# Configuration File Locations - Important Changes

**Date:** November 9, 2025

## File Location Changes

The following configuration template files have been **moved from the project root** to the `docs/onboarding/` folder for better organization:

### New Locations

| Old Location | New Location | Purpose |
|--------------|--------------|---------|
| `.env.example` | `docs/onboarding/env.example` | Environment variables template |
| `.cursorrules` | `docs/onboarding/cursorrules.example` | Cursor AI configuration template |
| N/A | `docs/onboarding/devcontainer.env.example` | DevContainer environment template (NEW) |
| N/A | `docs/onboarding/devcontainer.json.example` | DevContainer configuration reference (NEW) |
| N/A | `docs/onboarding/coveragerc.example` | Code coverage configuration reference (NEW) |
| N/A | `docs/onboarding/flake8.example` | Flake8 linter configuration reference (NEW) |
| N/A | `docs/onboarding/gitignore.example` | Git exclusion patterns reference (NEW) |

## How to Use These Templates

### 1. Creating Your `.env` File

```bash
# From project root
cp docs/onboarding/env.example .env

# Edit with your actual keys
cursor .env
```

**Important:** The `.env` file in your project root is gitignored. Never commit it!

### 2. Creating Your `.cursorrules` File (Optional)

```bash
# From project root
cp docs/onboarding/cursorrules.example .cursorrules
```

This configures Cursor AI to understand your project's architecture and patterns.

### 3. Creating DevContainer Environment (If Using DevContainers)

```bash
# From project root
cp docs/onboarding/devcontainer.env.example .devcontainer/devcontainer.env
```

## Why This Change?

**Before:**
- Configuration files with real secrets were in the project root
- Risk of accidental commits
- No clear separation between templates and actual config

**After:**
- ✅ Templates are in `docs/onboarding/` (safe to commit)
- ✅ Actual config files (`.env`, `.cursorrules`) stay in root (gitignored)
- ✅ Clear separation between templates and real configuration
- ✅ All templates use `[PLACEHOLDER]` format instead of example values

## Security Note

All templates now use bracketed placeholders like `[YOUR_OPENAI_API_KEY]` instead of example values. This makes it clear what needs to be replaced and prevents confusion about whether values are real or examples.

## Quick Reference

**Template Files (Safe to Commit):**
- `docs/onboarding/env.example`
- `docs/onboarding/devcontainer.env.example`
- `docs/onboarding/devcontainer.json.example`
- `docs/onboarding/coveragerc.example`
- `docs/onboarding/flake8.example`
- `docs/onboarding/gitignore.example`
- `docs/onboarding/cursorrules.example`

**Actual Config Files:**
- `.env` (in project root) - **NEVER commit** (has secrets)
- `.devcontainer/devcontainer.env` (in .devcontainer/) - **NEVER commit** (has secrets)
- `.devcontainer/devcontainer.json` (in .devcontainer/) - Can commit (no secrets)
- `.gitignore` (in project root) - Can commit (no secrets, critical for security)
- `.coveragerc` (in project root) - Can commit (no secrets)
- `.flake8` (in project root) - Can commit (no secrets)
- `.cursorrules` (in project root, optional) - Can commit (no secrets)

**How to Check:**
```bash
# Verify actual config files are gitignored
git status .env .cursorrules .devcontainer/devcontainer.env

# Should show: "nothing to commit" or not list these files
```

---

For complete setup instructions, see [02-initial-setup.md](./02-initial-setup.md).

