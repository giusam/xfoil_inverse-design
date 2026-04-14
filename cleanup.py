import os
import shutil
from pathlib import Path


def cleanup_debug_files(workdir):
    """
    Pulisce i file temporanei prodotti dalle run di XFOIL, preservando solo
    i risultati ottimizzati necessari per il report finale.
    """
    workdir = Path(workdir)
    if not workdir.exists():
        return

    # 1. Cancella completamente la cartella di scoring dei candidati
    # Spesso contiene migliaia di piccoli file che appesantiscono il disco.
    scoring_dir = workdir / "adaptive_candidate_scoring"
    try:
        if scoring_dir.exists():
            shutil.rmtree(scoring_dir)
    except PermissionError:
        # Questo succede se XFOIL non ha ancora chiuso i file cp.txt o polar.txt
        print(f"Avviso: Impossibile cancellare {scoring_dir}, file ancora in uso da XFOIL.")
    except Exception as e:
        print(f"Errore durante la rimozione di {scoring_dir}: {e}")

    # 2. Rimuovi file e cartelle inutili dentro static/ e adaptive/
    for sub in ["static", "adaptive"]:
        base = workdir / sub
        if not base.exists():
            continue

        for root, dirs, files in os.walk(base):
            root_path = Path(root)

            # A. Tieni solo i file finali "optimized"
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
                    except:
                        pass

            # B. Elimina le cartelle di valutazione intermedia (candidate o eval)
            for d in dirs:
                if d.startswith("candidate") or d.startswith("eval"):
                    folder_to_delete = root_path / d
                    try:
                        if folder_to_delete.exists():
                            shutil.rmtree(folder_to_delete)
                    except PermissionError:
                        # Gestione sicura per processi XFOIL ancora attivi
                        print(f"Avviso: Impossibile cancellare {folder_to_delete}, file in uso.")
                    except Exception as e:
                        print(f"Errore durante la rimozione di {folder_to_delete}: {e}")
