"""Utilities for avoiding SpeechBrain optional lazy-module import traps."""

from __future__ import annotations

import sys


def remove_speechbrain_lazy_modules() -> dict[str, object]:
    """Remove SpeechBrain lazy module placeholders that break ``inspect`` callers."""
    removed: dict[str, object] = {}
    for name, module in list(sys.modules.items()):
        module_type = type(module)
        if (
            module_type.__module__ == "speechbrain.utils.importutils"
            and module_type.__name__ in {"LazyModule", "DeprecatedModuleRedirect"}
        ):
            removed[name] = module
            sys.modules.pop(name, None)
    return removed


def restore_modules(modules: dict[str, object]) -> None:
    for name, module in modules.items():
        sys.modules.setdefault(name, module)
