
import os
import shutil
from pathlib import Path


def cleanup_debug_files(workdir):
    workdir = Path(workdir)
    if not workdir.exists():
        return

    for scoring_dir in [
        workdir / "adaptive_candidate_scoring",
        workdir / "adapt_grad" / "adaptive_candidate_scoring",
        workdir / "adapt_ikkt" / "adaptive_candidate_scoring",
    ]:
        try:
            if scoring_dir.exists():
                shutil.rmtree(scoring_dir)
        except PermissionError:
            print(f"Avviso: Impossibile cancellare {scoring_dir}, file ancora in uso da XFOIL.")
        except Exception as e:
            print(f"Errore durante la rimozione di {scoring_dir}: {e}")

    for sub in ["static", "adaptive", "adapt_grad", "adapt_ikkt"]:
        base = workdir / sub
        if not base.exists():
            continue

        for root, dirs, files in os.walk(base):
            root_path = Path(root)

            for f in files:
                if not (
                    "optimized_airfoil.dat" in f
                    or f == "cp.txt"
                    or f == "polar.txt"
                ):
                    try:
                        file_to_del = root_path / f
                        if file_to_del.exists():
                            os.remove(file_to_del)
                    except Exception:
                        pass

            for d in dirs:
                if d.startswith("candidate") or d.startswith("eval"):
                    folder_to_delete = root_path / d
                    try:
                        if folder_to_delete.exists():
                            shutil.rmtree(folder_to_delete)
                    except PermissionError:
                        print(f"Avviso: Impossibile cancellare {folder_to_delete}, file in uso.")
                    except Exception as e:
                        print(f"Errore durante la rimozione di {folder_to_delete}: {e}")