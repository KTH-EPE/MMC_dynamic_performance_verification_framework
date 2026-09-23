from pathlib import Path
import shutil


def move_result_file(src: Path, dest_dir: Path, new_name: str):
    """
    Move simulation result file to a target directory and rename it.

    Ensures destination directory exists before moving.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / new_name
    shutil.move(str(src), dest_path)


def format_rlc_filename(R, L, C, file_name):
    """
    Generate a standardized filename for LCR simulations.

    Inductance is converted to mH for readability.
    """
    return f"{file_name}_L{str(round(L, 3))}_C{str(round(C, 3))}_R{str(round(R, 3))}_.csv"


def format_rl_filename(R, L, file_name):
    """
    Generate a standardized filename for RL simulations.
    """
    return f"{file_name}_L{str(round(L, 3))}_R{str(round(R, 3))}_.csv"
