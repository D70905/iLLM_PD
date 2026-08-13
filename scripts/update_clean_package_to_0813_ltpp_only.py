from __future__ import annotations

from pathlib import Path
import shutil
from datetime import datetime


CLEAN = Path(r"D:\iLLM_PD_new_clean_NC_resubmit_0812")
SRC = Path(r"C:\Users\Ivy\Documents\iLLM_PD\output\fig3_revised")


def ensure_inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if not str(resolved).lower().startswith(str(root_resolved).lower()):
        raise RuntimeError(f"Refusing to modify outside {root_resolved}: {resolved}")
    return resolved


def copy(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    ensure_inside(dst, CLEAN)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def backup_existing(path: Path, backup_root: Path) -> None:
    if path.exists():
        rel = path.relative_to(CLEAN)
        dst = backup_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = CLEAN / "_backup_before_0813_ltpp_only" / ts
    backup.mkdir(parents=True, exist_ok=True)

    fig_final = CLEAN / "figures" / "final"
    fig_source = CLEAN / "figures" / "source"

    old_fig3 = [
        fig_final / "Fig3_design_response_revised_editable.svg",
        fig_final / "Fig3_design_response_revised_pillow_largefont.pdf",
        fig_final / "Fig3_design_response_revised_pillow_largefont.png",
        fig_source / "Fig3_revised_source_data.csv",
        fig_source / "plot_fig3_design_response_revised_pillow.py",
        fig_source / "plot_fig3_design_response_revised_editable_svg.py",
    ]
    for p in old_fig3:
        backup_existing(p, backup)
        if p.exists() and p.name.startswith("Fig3_design_response_revised"):
            p.unlink()

    copy(SRC / "Fig3_LTPP_only_true_delivered_redraw.png", fig_final / "Fig3_LTPP_only_true_delivered_redraw.png")
    copy(SRC / "Fig3_LTPP_only_true_delivered_redraw_wrapped.svg", fig_final / "Fig3_LTPP_only_true_delivered_redraw_wrapped.svg")
    copy(SRC / "Fig3_LTPP_only_true_delivered.svg", fig_final / "Fig3_LTPP_only_true_delivered.svg")
    copy(SRC / "Fig3_LTPP_only_true_delivered_source_data.csv", fig_source / "Fig3_LTPP_only_true_delivered_source_data.csv")
    copy(Path(r"C:\Users\Ivy\Documents\iLLM_PD\scripts\plot_fig3_ltpp_only_true_delivered_redraw.py"), fig_source / "plot_fig3_ltpp_only_true_delivered_redraw.py")
    copy(Path(r"C:\Users\Ivy\Documents\iLLM_PD\scripts\plot_fig3_ltpp_only_true_delivered_svg.py"), fig_source / "plot_fig3_ltpp_only_true_delivered_svg.py")

    notes = CLEAN / "DATA_AND_CODE_PACKAGE_NOTES.md"
    text = notes.read_text(encoding="utf-8")
    if "NCAT" in text:
        text = text.replace(
            "- Whether final Source Data should be uploaded as the root `SourceData_iLLM-PD_resubmit_0730.xlsx`\n"
            "  or as separate per-figure files under `Source_Data/`\n"
            "- Whether trained PPO checkpoints should be deposited separately if required by the editor\n",
            "- Use `Source_Data/SourceData_iLLM-PD_NC_final.xlsx` as the final Source Data upload candidate.\n"
            "- Whether trained PPO checkpoints should be deposited separately if required by the editor.\n",
        )
        text = text.replace(
            "- Whether final Source Data should be uploaded as the root `SourceData_iLLM-PD_resubmit_0730.xlsx`",
            "- Use the final Source Data workbook generated for the 0813 LTPP-only manuscript",
        )
        text = text.replace(
            "NCAT material/field-data tables used in the boundary analysis, ",
            "",
        )
    notes.write_text(text, encoding="utf-8")

    readme = CLEAN / "REPRODUCIBILITY.md"
    text = readme.read_text(encoding="utf-8")
    text = text.replace(
        "| Fig. 3 | `figures/final/Fig3_design_response_revised_pillow_largefont.*`; `figures/final/Fig3_design_response_revised_editable.svg`; `figures/source/Fig3_revised_source_data.csv` |",
        "| Fig. 3 | `figures/final/Fig3_LTPP_only_true_delivered_redraw.png`; `figures/final/Fig3_LTPP_only_true_delivered_redraw_wrapped.svg`; `figures/source/Fig3_LTPP_only_true_delivered_source_data.csv` |",
    )
    readme.write_text(text, encoding="utf-8")

    print(f"Updated clean package to 0813 LTPP-only Fig. 3.")
    print(f"Backup: {backup}")


if __name__ == "__main__":
    main()
