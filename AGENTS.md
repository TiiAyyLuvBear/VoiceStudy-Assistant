# Shared-code rules

- Shared utilities belong in `src/utils/`.
- Before adding a helper, search `src/utils/` and reuse an existing utility.
- Import shared utilities with `from src.utils...`; do not copy or redefine them
  inside feature modules.
- Keep utilities dependency-light and domain-neutral. Speaker-, ASR-, or
  database-specific logic remains in its own domain package.
- Add reusable helper exports to `src/utils/__init__.py` and tests when behavior
  changes.
