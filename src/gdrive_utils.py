"""Google Drive integration for saving intermediate results in Colab."""

import os
import shutil
from pathlib import Path


# Default project folder inside MyDrive
GDRIVE_PROJECT = "sae_probing"


def mount_drive(mount_point: str = "/content/drive") -> Path | None:
    """Mount Google Drive (Colab only) and return the MyDrive root.

    Returns
    -------
    Path to ``/content/drive/MyDrive`` on success, ``None`` outside Colab.
    """
    try:
        from google.colab import drive  # type: ignore
        drive.mount(mount_point, force_remount=False)
        my_drive = Path(mount_point) / "MyDrive"
        print(f"Google Drive mounted at {my_drive}")
        return my_drive
    except ImportError:
        print("Not running in Colab — Google Drive not mounted.")
        return None
    except Exception as e:
        print(f"Drive mount failed: {e}")
        return None


def get_drive_dirs(
    my_drive: Path,
    project: str = GDRIVE_PROJECT,
) -> dict[str, Path]:
    """Return and create the Drive cache/results directories.

    Parameters
    ----------
    my_drive : path returned by :func:`mount_drive`
    project  : subdirectory name inside MyDrive

    Returns
    -------
    {"cache_dir": Path, "results_dir": Path}
    """
    base = my_drive / project
    dirs = {
        "cache_dir":   base / "cache",
        "results_dir": base / "result",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    print(f"Drive dirs ready: {base}")
    return dirs


def sync_to_drive(local_dir: Path, drive_dir: Path, verbose: bool = True) -> None:
    """Copy all files from *local_dir* into *drive_dir* (non-recursive).

    Already-present files with identical size are skipped.
    """
    drive_dir.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    for src in local_dir.iterdir():
        if not src.is_file():
            continue
        dst = drive_dir / src.name
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1
    if verbose:
        print(f"sync_to_drive: {copied} copied, {skipped} skipped → {drive_dir}")


def sync_all(config: dict, drive_dirs: dict[str, Path]) -> None:
    """Convenience wrapper: sync both cache and results dirs to Drive."""
    sync_to_drive(Path(config["cache_dir"]),   drive_dirs["cache_dir"])
    sync_to_drive(Path(config["results_dir"]), drive_dirs["results_dir"])
