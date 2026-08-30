import os
import platform
import shutil
from pathlib import Path


def find_blender():
    env_path = os.environ.get("BLENDER_PATH")

    if env_path:
        blender_path = Path(env_path)

        if blender_path.exists():
            return blender_path

        raise FileNotFoundError(
            f"BLENDER_PATH points to a missing file: {blender_path}"
        )

    blender_from_path = shutil.which("blender")

    if blender_from_path:
        return Path(blender_from_path)

    system = platform.system()

    candidates = []

    if system == "Darwin":
        candidates = [
            Path("/Applications/Blender.app/Contents/MacOS/Blender"),
            Path("~/Applications/Blender.app/Contents/MacOS/Blender").expanduser(),
        ]

    elif system == "Windows":
        program_files = os.environ.get(
            "PROGRAMFILES",
            r"C:\Program Files",
        )

        blender_root = Path(program_files) / "Blender Foundation"

        if blender_root.exists():
            candidates.extend(
                sorted(
                    blender_root.glob("Blender*/blender.exe"),
                    reverse=True,
                )
            )

    elif system == "Linux":
        candidates = [
            Path("/usr/bin/blender"),
            Path("/usr/local/bin/blender"),
            Path("/snap/bin/blender"),
        ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Blender executable was not found.\n"
        "Install Blender, add it to PATH, or set "
        "the BLENDER_PATH environment variable."
    )
