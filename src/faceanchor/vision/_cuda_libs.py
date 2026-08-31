"""Preloads pip-installed CUDA/cuDNN shared libraries before onnxruntime needs them.

`onnxruntime-gpu` expects libcudnn/libcublas to be discoverable via the loader
search path, but the pip wheels (`nvidia-cudnn-cu12`, `nvidia-cublas-cu12`)
install them under site-packages, off the default path. Rather than requiring
every invocation to `export LD_LIBRARY_PATH=...`, dlopen them explicitly with
RTLD_GLOBAL so the CUDA execution provider can resolve its symbols at session
creation time.

Import this module before `onnxruntime` anywhere the CUDA EP may be selected.
Safe to import when CUDA is unavailable — failures to load are swallowed and
onnxruntime falls back to CPUExecutionProvider on its own.
"""

from __future__ import annotations

import ctypes
import glob
import sysconfig

_loaded = False


def preload() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True

    site_packages = sysconfig.get_paths()["purelib"]
    for lib in sorted(glob.glob(f"{site_packages}/nvidia/*/lib/*.so*")):
        try:
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


preload()
