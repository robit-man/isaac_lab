#!/usr/bin/env python3
"""
bootstrap_isaaclab.py

One-shot installer + runner for:
  1) Python 3.11 venv
  2) Isaac Sim via pip (NVIDIA index)
  3) CUDA-enabled PyTorch wheel (cu128)
  4) Isaac Lab from source (clone + ./isaaclab.sh --install)
  5) Run an Isaac Lab script using the venv

Typical usage:
  python3 bootstrap_isaaclab.py
  python3 bootstrap_isaaclab.py --run train_ant --headless
  python3 bootstrap_isaaclab.py --base ~/work --env env_isaaclab --repo IsaacLab

Notes:
- Linux pip install of Isaac Sim requires GLIBC >= 2.35 and (typically) x86_64.
- Isaac Sim 5.x requires Python 3.11 in the environment.
- First simulator run may prompt for NVIDIA Omniverse EULA and pull extensions.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_ISAACSIM_VERSION = "5.1.0"
DEFAULT_TORCH_VERSION = "2.7.0"
DEFAULT_TORCHVISION_VERSION = "0.22.0"
DEFAULT_TORCHAUDIO_VERSION = "2.7.0"
DEFAULT_NVIDIA_PYPI = "https://pypi.nvidia.com"
DEFAULT_ISAACLAB_GIT = "https://github.com/isaac-sim/IsaacLab.git"


# -----------------------------
# helpers
# -----------------------------
def eprint(*a: object) -> None:
    print(*a, file=sys.stderr, flush=True)


def run_cmd(
    cmd: List[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    cmd_str = " ".join([shlex_quote(x) for x in cmd])
    print(f"+ {cmd_str}", flush=True)
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=check)


def shlex_quote(s: str) -> str:
    # minimal, good-enough quoting for printing
    if re.fullmatch(r"[A-Za-z0-9_./:@%+=-]+", s):
        return s
    return "'" + s.replace("'", "'\"'\"'") + "'"


def is_venv() -> bool:
    return getattr(sys, "base_prefix", sys.prefix) != sys.prefix


def venv_bin_dir(env_dir: Path) -> Path:
    return env_dir / ("Scripts" if platform.system() == "Windows" else "bin")


def venv_python(env_dir: Path) -> Path:
    b = venv_bin_dir(env_dir)
    return b / ("python.exe" if platform.system() == "Windows" else "python")


def venv_pip(env_dir: Path) -> Path:
    b = venv_bin_dir(env_dir)
    return b / ("pip.exe" if platform.system() == "Windows" else "pip")


def which(exe: str) -> Optional[str]:
    return shutil.which(exe)


def parse_version_tuple(v: str) -> Tuple[int, int, int]:
    m = re.match(r"^\s*(\d+)\.(\d+)(?:\.(\d+))?", v.strip())
    if not m:
        return (0, 0, 0)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))


def get_glibc_version_linux() -> Optional[str]:
    # Works on glibc systems (most Ubuntu/Debian). Returns None on musl or failure.
    try:
        libc = ctypes.CDLL("libc.so.6")
        gnu_get_libc_version = libc.gnu_get_libc_version
        gnu_get_libc_version.restype = ctypes.c_char_p
        v = gnu_get_libc_version().decode("ascii", errors="ignore")
        return v
    except Exception:
        return None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_marker(env_dir: Path, name: str, content: str = "ok\n") -> None:
    ensure_dir(env_dir)
    (env_dir / name).write_text(content, encoding="utf-8")


def has_marker(env_dir: Path, name: str) -> bool:
    return (env_dir / name).exists()


def prepend_path(env: Dict[str, str], p: Path) -> Dict[str, str]:
    env2 = dict(env)
    env2["PATH"] = str(p) + os.pathsep + env2.get("PATH", "")
    return env2


def git_clone_or_download(repo_url: str, dest: Path, branch: Optional[str] = None) -> None:
    if dest.exists() and (dest / ".git").exists():
        print(f"[i] IsaacLab repo already exists at: {dest}")
        return

    if dest.exists() and any(dest.iterdir()):
        raise RuntimeError(f"Destination exists and is not empty: {dest}")

    git = which("git")
    if git:
        cmd = [git, "clone", "--depth", "1"]
        if branch:
            cmd += ["--branch", branch]
        cmd += [repo_url, str(dest)]
        run_cmd(cmd)
        return

    # fallback: download zipball via curl/wget/python if git is missing
    # We keep it simple: require git unless user installs it via apt.
    raise RuntimeError(
        "git is not available and auto-download fallback is disabled for safety. "
        "Install git (e.g., `sudo apt-get install -y git`) and re-run."
    )


def apt_install_if_requested(pkgs: List[str], mode: str) -> None:
    """
    mode: 'off' | 'auto' | 'on'
    """
    if platform.system() != "Linux":
        return
    if mode == "off":
        return
    if not which("apt-get"):
        if mode == "on":
            raise RuntimeError("Requested --system-deps on, but apt-get not found.")
        return

    # only attempt on Debian/Ubuntu-like
    # this may prompt for sudo password; that's expected.
    sudo = which("sudo")
    is_root = (hasattr(os, "geteuid") and os.geteuid() == 0)
    prefix: List[str] = []
    if not is_root:
        if not sudo:
            if mode == "on":
                raise RuntimeError("Need root to install system deps, but sudo not found.")
            return
        prefix = [sudo]

    env = dict(os.environ)
    env["DEBIAN_FRONTEND"] = "noninteractive"

    run_cmd(prefix + ["apt-get", "update"], env=env)
    run_cmd(prefix + ["apt-get", "install", "-y"] + pkgs, env=env)


def find_python_311() -> Optional[str]:
    # If we're already 3.11, use this interpreter
    if sys.version_info[:2] == (3, 11):
        return sys.executable

    # Common names on Linux/macOS
    for cand in ("python3.11", "python311", "python3"):
        path = which(cand)
        if not path:
            continue
        try:
            out = subprocess.check_output([path, "-c", "import sys; print(sys.version.split()[0])"])
            ver = out.decode().strip()
            if ver.startswith("3.11."):
                return path
        except Exception:
            continue

    # Windows launcher
    if platform.system() == "Windows" and which("py"):
        try:
            out = subprocess.check_output(["py", "-3.11", "-c", "import sys; print(sys.version.split()[0])"])
            ver = out.decode().strip()
            if ver.startswith("3.11."):
                return "py -3.11"  # special token handled later
        except Exception:
            pass

    return None


def create_venv_with_python(py311: str, env_dir: Path) -> None:
    if env_dir.exists() and venv_python(env_dir).exists():
        print(f"[i] venv already exists at: {env_dir}")
        return

    ensure_dir(env_dir.parent)

    if py311 == sys.executable and sys.version_info[:2] == (3, 11):
        # Use stdlib venv if current python is 3.11
        run_cmd([py311, "-m", "venv", str(env_dir)])
        return

    if py311.startswith("py -3.11"):
        # Windows py launcher
        run_cmd(["py", "-3.11", "-m", "venv", str(env_dir)])
        return

    # External python path
    run_cmd([py311, "-m", "venv", str(env_dir)])


def ensure_in_venv(env_dir: Path, passthrough_args: List[str]) -> None:
    """
    If not currently running inside the target venv, re-exec into it.
    """
    if is_venv():
        return
    py = venv_python(env_dir)
    if not py.exists():
        raise RuntimeError(f"venv python not found at: {py}")

    # Re-exec: run this same script inside venv with internal flag
    args = [str(py), str(Path(__file__).resolve()), "--_inside-venv"] + passthrough_args
    print(f"[i] re-exec into venv: {py}")
    os.execv(str(py), args)


def pip_install(pip_path: Path, args: List[str]) -> None:
    run_cmd([str(pip_path), "install"] + args)


def pip_check_import(py_path: Path, module: str) -> bool:
    try:
        subprocess.check_call([str(py_path), "-c", f"import {module}; print({module}.__name__)"])
        return True
    except Exception:
        return False


# -----------------------------
# main flow
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Self-contained Isaac Sim (pip) + Isaac Lab (source) venv bootstrapper.")
    ap.add_argument("--base", type=str, default=str(Path.cwd()), help="Workspace directory (default: current dir).")
    ap.add_argument("--env", type=str, default="env_isaaclab", help="Venv directory name (default: env_isaaclab).")
    ap.add_argument("--repo", type=str, default="IsaacLab", help="IsaacLab checkout directory (default: IsaacLab).")
    ap.add_argument("--repo-url", type=str, default=DEFAULT_ISAACLAB_GIT, help="IsaacLab git URL.")
    ap.add_argument("--branch", type=str, default=None, help="Optional git branch/tag to checkout.")
    ap.add_argument("--isaacsim-version", type=str, default=DEFAULT_ISAACSIM_VERSION, help="isaacsim pip version.")
    ap.add_argument("--torch-version", type=str, default=DEFAULT_TORCH_VERSION, help="torch version.")
    ap.add_argument("--torchvision-version", type=str, default=DEFAULT_TORCHVISION_VERSION, help="torchvision version.")
    ap.add_argument("--torchaudio-version", type=str, default=DEFAULT_TORCHAUDIO_VERSION, help="torchaudio version.")
    ap.add_argument("--system-deps", choices=["auto", "on", "off"], default="auto",
                    help="Install system deps via apt-get (cmake, build-essential, git).")
    ap.add_argument("--force", action="store_true", help="Force re-install steps even if markers exist.")
    ap.add_argument("--run", choices=["create_empty", "train_ant", "train_anymal", "none"], default="create_empty",
                    help="What to run after install.")
    ap.add_argument("--headless", action="store_true", help="Add --headless to training run (and some scripts).")
    ap.add_argument("--_inside-venv", action="store_true", help=argparse.SUPPRESS)

    args, unknown = ap.parse_known_args()

    base = Path(args.base).expanduser().resolve()
    env_dir = base / args.env
    repo_dir = base / args.repo

    # ---- platform checks (esp. Linux glibc / arch) ----
    sysname = platform.system()
    machine = platform.machine().lower()

    if sysname == "Linux":
        glibc = get_glibc_version_linux()
        if glibc is None:
            raise RuntimeError("Could not detect GLIBC version (are you on musl/Alpine?). "
                               "Isaac Sim pip installs generally require GLIBC >= 2.35.")
        if parse_version_tuple(glibc) < (2, 35, 0):
            raise RuntimeError(f"GLIBC {glibc} detected, but Isaac Sim pip requires GLIBC >= 2.35. "
                               "Use the Isaac Sim binaries installation method on older distros.")

        # Isaac Sim pip wheels are typically manylinux_2_35_x86_64; enforce x86_64 for Linux.
        if machine not in ("x86_64", "amd64"):
            raise RuntimeError(f"Linux architecture '{machine}' detected. Isaac Sim pip install is typically x86_64-only. "
                               "Use the Isaac Sim binaries (or supported platform method) instead.")

    # ---- ensure python 3.11 for venv ----
    py311 = find_python_311()
    if not py311:
        # try to install python3.11 via apt if available and allowed
        if sysname == "Linux" and which("apt-get") and args.system_deps in ("auto", "on"):
            eprint("[i] Python 3.11 not found. Attempting to install via apt-get (may prompt for sudo)...")
            apt_install_if_requested(["python3.11", "python3.11-venv", "python3.11-distutils"], mode="on")
            py311 = find_python_311()

        if not py311:
            raise RuntimeError(
                "Python 3.11 interpreter not found. Install Python 3.11 first, then re-run.\n"
                "Examples:\n"
                "  Ubuntu/Debian: sudo apt-get install -y python3.11 python3.11-venv\n"
                "  Or use your distro's recommended Python 3.11 install method."
            )

    # ---- create venv, then re-exec into it ----
    if not args._inside_venv:
        create_venv_with_python(py311, env_dir)
        # Re-exec into that venv, preserving user args except internal flag
        passthrough = [x for x in sys.argv[1:] if x != "--_inside-venv"]
        ensure_in_venv(env_dir, passthrough)

    # From here on: running inside venv
    py = Path(sys.executable)
    pip = venv_pip(env_dir)
    if not pip.exists():
        # fallback to python -m pip
        pip = Path(sys.executable)
    print(f"[i] using venv python: {py}")
    print(f"[i] workspace base: {base}")
    print(f"[i] venv dir: {env_dir}")
    print(f"[i] IsaacLab dir: {repo_dir}")

    # ---- optional system deps ----
    # Needed by Isaac Lab optional deps (e.g., robomimic); also ensure git exists for clone.
    if sysname == "Linux":
        want = args.system_deps
        pkgs = ["cmake", "build-essential", "git"]
        apt_install_if_requested(pkgs, mode=want)

    # ---- pip bootstrap ----
    marker_bootstrap = ".isaaclab_bootstrap_ok"
    if args.force or not has_marker(env_dir, marker_bootstrap):
        # upgrade pip tooling
        if pip.name.startswith("python"):
            run_cmd([str(py), "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"])
        else:
            run_cmd([str(pip), "install", "-U", "pip", "setuptools", "wheel"])
        write_marker(env_dir, marker_bootstrap)

    # ---- install PyTorch (CUDA 12.8 wheels) ----
    marker_torch = ".isaaclab_torch_ok"
    if args.force or not has_marker(env_dir, marker_torch):
        torch_index = None
        if sysname in ("Linux", "Windows") and machine in ("x86_64", "amd64"):
            torch_index = "https://download.pytorch.org/whl/cu128"
        else:
            raise RuntimeError(f"Unsupported platform for scripted torch install: {sysname} / {machine}")

        torch_pkgs = [
            f"torch=={args.torch_version}",
            f"torchvision=={args.torchvision_version}",
            f"torchaudio=={args.torchaudio_version}",
            "--index-url", torch_index,
        ]
        if pip.name.startswith("python"):
            run_cmd([str(py), "-m", "pip", "install", "-U"] + torch_pkgs)
        else:
            run_cmd([str(pip), "install", "-U"] + torch_pkgs)

        # sanity import
        subprocess.check_call([str(py), "-c", "import torch; import torchvision; print(torch.__version__, torchvision.__version__)"])
        write_marker(env_dir, marker_torch)

    # ---- install Isaac Sim via pip ----
    marker_isaacsim = ".isaaclab_isaacsim_ok"
    if args.force or not has_marker(env_dir, marker_isaacsim):
        isaacsim_spec = f"isaacsim[all,extscache]=={args.isaacsim_version}"
        cmd = [str(py), "-m", "pip", "install", isaacsim_spec, "--extra-index-url", DEFAULT_NVIDIA_PYPI]
        run_cmd(cmd)

        # sanity import (no sim launch yet)
        subprocess.check_call([str(py), "-c", "import isaacsim; print('isaacsim import OK')"])
        write_marker(env_dir, marker_isaacsim)

    # ---- clone IsaacLab ----
    marker_repo = ".isaaclab_repo_ok"
    if args.force or not has_marker(env_dir, marker_repo):
        if not repo_dir.exists():
            git_clone_or_download(args.repo_url, repo_dir, branch=args.branch)
        else:
            print(f"[i] repo dir exists: {repo_dir}")
        write_marker(env_dir, marker_repo)

    # ---- run ./isaaclab.sh --install ----
    marker_install = ".isaaclab_install_ok"
    if args.force or not has_marker(env_dir, marker_install):
        if sysname == "Windows":
            bat = repo_dir / "isaaclab.bat"
            if not bat.exists():
                raise RuntimeError(f"Expected {bat} not found.")
            # Ensure venv python first in PATH
            env = prepend_path(dict(os.environ), venv_bin_dir(env_dir))
            run_cmd(["cmd", "/c", str(bat), "--install"], cwd=repo_dir, env=env)
        else:
            sh = repo_dir / "isaaclab.sh"
            if not sh.exists():
                raise RuntimeError(f"Expected {sh} not found.")
            # make executable just in case
            run_cmd(["chmod", "+x", str(sh)], cwd=repo_dir, check=False)
            env = prepend_path(dict(os.environ), venv_bin_dir(env_dir))
            run_cmd(["bash", str(sh), "--install"], cwd=repo_dir, env=env)
        write_marker(env_dir, marker_install)

    # ---- run something in Isaac Lab ----
    if args.run != "none":
        env = prepend_path(dict(os.environ), venv_bin_dir(env_dir))

        if sysname == "Windows":
            runner = repo_dir / "isaaclab.bat"
            if not runner.exists():
                raise RuntimeError(f"Expected {runner} not found.")
            base_cmd = ["cmd", "/c", str(runner), "-p"]
        else:
            runner = repo_dir / "isaaclab.sh"
            base_cmd = ["bash", str(runner), "-p"]

        if args.run == "create_empty":
            script_path = "scripts/tutorials/00_sim/create_empty.py"
            cmd = base_cmd + [script_path] + unknown
            # if headless requested and user didn't already pass it, append
            if args.headless and "--headless" not in cmd:
                cmd.append("--headless")
            run_cmd(cmd, cwd=repo_dir, env=env)

        elif args.run == "train_ant":
            script_path = "scripts/reinforcement_learning/rsl_rl/train.py"
            cmd = base_cmd + [script_path, "--task=Isaac-Ant-v0"] + unknown
            if args.headless and "--headless" not in cmd:
                cmd.append("--headless")
            run_cmd(cmd, cwd=repo_dir, env=env)

        elif args.run == "train_anymal":
            script_path = "scripts/reinforcement_learning/rsl_rl/train.py"
            cmd = base_cmd + [script_path, "--task=Isaac-Velocity-Rough-Anymal-C-v0"] + unknown
            if args.headless and "--headless" not in cmd:
                cmd.append("--headless")
            run_cmd(cmd, cwd=repo_dir, env=env)

    print("\n[i] Done.")
    print("[i] To use the venv later:")
    if platform.system() == "Windows":
        print(f"    {env_dir}\\Scripts\\activate")
    else:
        print(f"    source {env_dir}/bin/activate")
    print(f"[i] IsaacLab checkout: {repo_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as ex:
        eprint(f"\n[!] Command failed with exit code {ex.returncode}")
        raise
    except Exception as ex:
        eprint(f"\n[!] {ex}")
        raise
