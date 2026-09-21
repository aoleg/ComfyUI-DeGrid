"""Load the repo-root degrid_core.py as a shared module for the Forge Neo script.

degrid_core.py has no ComfyUI imports (torch only), so the exact same file is
the maths behind the ComfyUI node and the Forge Neo extension. It is loaded via
``importlib.util.spec_from_file_location`` under a private ``sys.modules`` key so
it never collides with another extension's module of the same bare filename.

The cache is keyed on the file's mtime and size rather than being permanent:
Forge's "Reload UI" re-executes ``scripts/*.py`` but leaves this module in
``sys.modules``, so a permanent cache would keep serving the core that was on
disk when the process started and a ``git pull`` would appear to do nothing.
"""

from __future__ import annotations

import importlib.util
import os
import sys

_MODULE_NAME = "comfyui_degrid_core"
_cached_module = None
_cached_stamp = None


def core_stamp(extension_root: str):
    """Identity of the degrid_core.py currently on disk: (path, mtime_ns, size)."""
    path = os.path.join(extension_root, "degrid_core.py")
    try:
        info = os.stat(path)
        return (path, info.st_mtime_ns, info.st_size)
    except OSError:
        return (path, None, None)


def load_core(extension_root: str):
    """Import degrid_core.py from ``extension_root``, re-importing it when it changes.

    ``extension_root`` should be captured via ``scripts.basedir()`` at the calling
    script's own import time; ``scripts.basedir()`` reflects whichever script Forge
    is currently loading and is not reliable later, e.g. inside a UI callback.
    """
    global _cached_module, _cached_stamp

    stamp = core_stamp(extension_root)
    if _cached_module is not None and _cached_stamp == stamp:
        return _cached_module

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, stamp[0])
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)

    _cached_module = module
    _cached_stamp = stamp
    return module
