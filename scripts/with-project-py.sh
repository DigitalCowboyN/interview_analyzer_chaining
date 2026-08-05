#!/usr/bin/env bash
# Resolve a Python interpreter that has the project deps (yaml), then run a module.
# Usage: with-project-py.sh <module> [args...]
# Non-blocking: if none is found, exit 0 silently.
set -u
mod="${1:-}"
[ -n "$mod" ] || exit 0
shift || true
_pyenv_python="$(command -v pyenv >/dev/null 2>&1 && pyenv which python 2>/dev/null || true)"
for py in python python3 "$HOME/.pyenv/shims/python" "$_pyenv_python"; do
  [ -n "$py" ] || continue
  if "$py" -c "import yaml" >/dev/null 2>&1; then
    exec "$py" -m "$mod" "$@"
  fi
done
exit 0
