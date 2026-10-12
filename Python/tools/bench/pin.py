# -*- coding: utf-8 -*-
"""
CPU placement helpers for the benchmark scripts (review A-7). Windows only (ctypes); on other platforms every function
reports "unsupported" instead of guessing.

Why: on a hybrid CPU (this machine: i7-13650HX, 6 P-cores = logical processors 0-11, 8 E-cores = 12-19) Windows places a
long single-thread run on a P- or an E-core non-deterministically. The same Python solve took 10.1-18.0 s unpinned and
23.4 s on an E-core, and .NET is about 1.4-1.5x slower on E-cores, so an interleaved Python/C# run can report a ratio
that is off by 1.5-2x for reasons that have nothing to do with the code. Both sides must run on the SAME core class.

Rules this module follows
  * The mask is applied to the child process BEFORE it executes a single instruction (the process is created suspended,
    the affinity mask is set, then it is resumed), so the .NET runtime sees the restricted CPU set at startup
    (Environment.ProcessorCount, GC and JIT worker threads) and children of the process inherit it.
  * A mask is a CPU SET, not a single CPU: pinning .NET to one logical CPU makes its JIT/GC threads compete with the
    benchmark thread and distorts the result by 2-4x.
  * The P-core set is detected from the OS (GetLogicalProcessorInformationEx efficiency classes), never hard-coded. On this
    machine the detected value is 0xFFF (P-cores) and 0xFF000 (E-cores).
"""
import ctypes
import os
import struct
import subprocess
import sys

CREATE_SUSPENDED = 0x00000004

_IS_WINDOWS = sys.platform == 'win32'


def supported():
    return _IS_WINDOWS


def _kernel32():
    from ctypes import wintypes
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.GetLogicalProcessorInformationEx.argtypes = [wintypes.DWORD, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
    k.GetLogicalProcessorInformationEx.restype = wintypes.BOOL
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    k.GetProcessAffinityMask.restype = wintypes.BOOL
    k.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    k.SetProcessAffinityMask.restype = wintypes.BOOL
    return k


def current_affinity():
    """The affinity mask this process actually runs with, read back from the OS (None if unsupported)."""
    if not _IS_WINDOWS:
        return None
    k = _kernel32()
    process, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not k.GetProcessAffinityMask(k.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(system)):
        raise OSError(ctypes.get_last_error(), 'GetProcessAffinityMask failed')
    return process.value


def core_classes():
    """{efficiency_class: mask} over the physical cores of processor group 0; None if unsupported.

    A higher efficiency class is a more performant core (P-core); class 0 is the lowest (E-core). A CPU that is not
    hybrid has a single class.
    """
    if not _IS_WINDOWS:
        return None
    from ctypes import wintypes
    k = _kernel32()
    length = wintypes.DWORD(0)
    k.GetLogicalProcessorInformationEx(0, None, ctypes.byref(length))  # RelationProcessorCore: asks for the size
    buf = ctypes.create_string_buffer(length.value)
    if not k.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(length)):
        raise OSError(ctypes.get_last_error(), 'GetLogicalProcessorInformationEx failed')
    raw = buf.raw[:length.value]
    classes = {}
    offset = 0
    while offset < len(raw):
        relationship, size = struct.unpack_from('<II', raw, offset)
        efficiency, = struct.unpack_from('<B', raw, offset + 9)  # PROCESSOR_RELATIONSHIP.EfficiencyClass
        groups, = struct.unpack_from('<H', raw, offset + 30)
        mask, group = struct.unpack_from('<QH', raw, offset + 32)  # first GROUP_AFFINITY
        if relationship == 0:
            if groups != 1 or group != 0:
                raise OSError('more than one processor group: set --affinity explicitly')
            classes[efficiency] = classes.get(efficiency, 0) | mask
        offset += size
    return classes


def is_hybrid():
    classes = core_classes()
    return bool(classes) and len(classes) > 1


def performance_core_mask():
    """Mask of the highest efficiency class (the P-cores), or None if unsupported."""
    classes = core_classes()
    return None if not classes else classes[max(classes)]


def efficiency_core_mask():
    """Mask of the lowest efficiency class (the E-cores) of a hybrid CPU; None if not hybrid or unsupported."""
    classes = core_classes()
    if not classes or len(classes) < 2:
        return None
    return classes[min(classes)]


def parse_mask(text):
    """'FFF', '0xFFF' -> int. 'auto' -> the detected P-core set."""
    text = text.strip()
    if text.lower() == 'auto':
        mask = performance_core_mask()
        if mask is None:
            raise SystemExit('--affinity auto is only supported on Windows; pass a hex mask')
        return mask
    return int(text, 16)


def describe(mask):
    return 'none' if mask is None else '0x%X' % mask


def run_pinned(cmd, mask, cwd=None, timeout=None):
    """Runs cmd with the process affinity set to `mask` before the process executes anything.

    Returns subprocess.CompletedProcess-like (returncode, stdout, stderr as text). `mask` None runs unpinned.
    """
    if mask is None:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout)
    if not _IS_WINDOWS:
        raise SystemExit('pinned runs are only supported on Windows')
    k = _kernel32()
    ntdll = ctypes.WinDLL('ntdll')
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8',
                            errors='replace', creationflags=CREATE_SUSPENDED)
    try:
        handle = int(proc._handle)  # noqa: SLF001  (the Windows process handle of the suspended child)
        if not k.SetProcessAffinityMask(handle, mask):
            raise OSError(ctypes.get_last_error(), 'SetProcessAffinityMask(0x%X) failed' % mask)
        ntdll.NtResumeProcess(ctypes.c_void_p(handle))
        out, err = proc.communicate(timeout=timeout)
    except BaseException:
        proc.kill()
        raise
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


def start_pinned(cmd, mask, cwd=None, stdout=None, stderr=None, env=None):
    """Starts cmd in the background with the process affinity set to `mask` before it executes anything (Phase 7: servers).

    Returns the subprocess.Popen (already running). `mask` None starts it unpinned. The caller stops it."""
    if mask is None:
        return subprocess.Popen(cmd, cwd=cwd, stdout=stdout, stderr=stderr, env=env)
    if not _IS_WINDOWS:
        raise SystemExit('pinned runs are only supported on Windows')
    k = _kernel32()
    ntdll = ctypes.WinDLL('ntdll')
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=stdout, stderr=stderr, env=env, creationflags=CREATE_SUSPENDED)
    try:
        handle = int(proc._handle)  # noqa: SLF001
        if not k.SetProcessAffinityMask(handle, mask):
            raise OSError(ctypes.get_last_error(), 'SetProcessAffinityMask(0x%X) failed' % mask)
        ntdll.NtResumeProcess(ctypes.c_void_p(handle))
    except BaseException:
        proc.kill()
        raise
    return proc


if __name__ == '__main__':
    classes = core_classes()
    print('process affinity now: %s' % describe(current_affinity()))
    if classes:
        for cls in sorted(classes, reverse=True):
            print('efficiency class %d: mask 0x%X (%s)' % (cls, classes[cls], 'P-cores' if cls == max(classes) and len(classes) > 1 else
                                                         ('E-cores' if len(classes) > 1 else 'all cores')))
        print('hybrid CPU: %s' % is_hybrid())
    else:
        print('core classes: unsupported on this platform')
