#!/usr/bin/env bash
# Resolve a Python interpreter that has the project deps (yaml), then run tools.adr.
# Non-blocking: if none is found, exit 0 silently.
set -u
_pyenv_python="$(command -v pyenv >/dev/null 2>&1 && pyenv which python 2>/dev/null || true)"
for py in python python3 "$HOME/.pyenv/shims/python" "$_pyenv_python"; do
  [ -n "$py" ] || continue
  if "$py" -c "import yaml" >/dev/null 2>&1; then
    exec "$py" -m tools.adr "$@"
  fi
done
exit 0
