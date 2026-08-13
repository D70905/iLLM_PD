from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Source_Data" / "_final_tables_0813"
OLD_WB = ROOT / "Source_Data" / "SourceData_iLLM-PD_resubmit_0730.xlsx"
FIG2_0813 = Path(r"C:\Users\Ivy\Documents\iLLM_PD\output\fig2_04_1034_cost_heatmap_0813")


SECTION_ORDER = [
    "16_1010", "27_1085", "04_1034", "48_1076", "12_1060", "48_0001",
    "30_7076", "04_1065", "48_1109", "06_2004", "27_2023", "12_4097",
]


def write_csv(name: str, df: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")


def section_label_map():
    return {sid: f"Section {i + 1}" for i, sid in enumerate(SECTION_ORDER)}


def main() -> None:
    label_map = section_label_map()

    fig3 = pd.read_csv(ROOT / "figures" / "source" / "Fig3_LTPP_only_true_delivered_source_data.csv")
    fig3["study_section"] = fig3["section"].map(label_map)
    table1 = fig3[
        [
            "study_section", "section", "type", "state", "climate_zone", "E_sub",
            "h1", "h2", "h3", "h4", "h5", "AC_cm", "total_thickness_cm",
            "cost_USD", "LCC_USD", "DSR", "SCR",
        ]
    ].rename(columns={
        "section": "LTPP_section_id",
        "type": "pavement_family",
        "state": "US_state",
        "E_sub": "subgrade_modulus_MPa",
        "h1": "AC1_upper_thickness_cm",
        "h2": "AC2_middle_thickness_cm",
        "h3": "AC3_lower_thickness_cm",
        "h4": "base_thickness_cm",
        "h5": "subbase_thickness_cm",
        "AC_cm": "total_asphalt_thickness_cm",
        "total_thickness_cm": "total_pavement_thickness_cm",
        "cost_USD": "construction_cost_USD_m2",
        "LCC_USD": "twenty_year_LCC_USD_m2",
        "DSR": "final_DSR",
        "SCR": "episode_SCR",
    })
    write_csv("Table1_LTPP_sections", table1)
    write_csv("Fig3_LTPP_design_response", table1)

    fig4 = pd.read_csv(ROOT / "figures" / "source" / "Fig4_comparison_practice_v2_source_data_DSR.csv")
    fig4["study_section"] = fig4["section"].map(label_map)
    fig4 = fig4[
        ["study_section", "section", "type", "method", "DSR", "pass_JTG", "B1", "B2", "B3", "B4"]
    ].rename(columns={
        "section": "LTPP_section_id",
        "type": "pavement_family",
        "DSR": "final_DSR",
        "pass_JTG": "JTG_compliant",
    })
    write_csv("Fig4_method_comparison", fig4)

    mepdg = pd.read_csv(ROOT / "figures" / "source" / "Fig4_comparison_practice_v2_source_data_MEPDG.csv")
    mepdg["study_section"] = mepdg["section_id"].map(label_map)
    mepdg = mepdg[
        [
            "study_section", "section_id", "seed", "pavement_type", "E_subgrade",
            "h_ac_cm", "RD_total_mm", "RD_HMA_mm", "RD_base_mm", "RD_subgrade_mm",
            "margin_RD", "pass_RD", "MEPDG_all_pass",
        ]
    ].rename(columns={
        "section_id": "LTPP_section_id",
        "pavement_type": "pavement_family",
        "E_subgrade": "subgrade_modulus_MPa",
        "h_ac_cm": "total_asphalt_thickness_cm",
        "RD_total_mm": "MEPDG_total_rutting_mm",
        "RD_HMA_mm": "MEPDG_HMA_rutting_mm",
        "RD_base_mm": "MEPDG_base_rutting_mm",
        "RD_subgrade_mm": "MEPDG_subgrade_rutting_mm",
        "margin_RD": "rutting_margin_to_19mm",
        "pass_RD": "MEPDG_rutting_pass",
    })
    write_csv("Fig4_MEPDG_rutting", mepdg)

    climate = pd.read_csv(ROOT / "experiments" / "batch_climate_12sections_summary.csv")
    climate["study_section"] = climate["section"].map(label_map)
    climate = climate[
        [
            "study_section", "section", "climate_zone", "MAAT_C", "E_sub_MPa",
            "eps_a_min_ue", "eps_a_max_ue", "eps_swing", "eps_a_fixed_ue",
            "Nf_fixed", "Nf_climate_eff", "fixed_over_climate", "verdict",
        ]
    ].rename(columns={
        "section": "LTPP_section_id",
        "E_sub_MPa": "subgrade_modulus_MPa",
        "eps_a_min_ue": "minimum_monthly_asphalt_tensile_strain_microstrain",
        "eps_a_max_ue": "maximum_monthly_asphalt_tensile_strain_microstrain",
        "eps_swing": "monthly_strain_swing_microstrain",
        "eps_a_fixed_ue": "fixed_20C_asphalt_tensile_strain_microstrain",
        "Nf_fixed": "fixed_20C_predicted_AC_fatigue_life",
        "Nf_climate_eff": "climate_resolved_predicted_AC_fatigue_life",
        "fixed_over_climate": "fixed_over_climate_resolved_fatigue_life_ratio",
    })
    write_csv("Fig5_temperature_fatigue", climate)

    ood = pd.read_csv(ROOT / "experiments" / "ltpp_data" / "deliverables" / "ood_stress" / "ood_aggregate_20260723_133102.csv")
    ood = ood.rename(columns={
        "case_id": "scenario_id",
        "final_dsr": "final_DSR",
        "final_scr": "single_design_SCR",
        "scr_traj": "episode_SCR",
        "B3_min": "minimum_B3_permanent_deformation_margin",
        "escalation_rate": "FEA_escalation_rate",
        "guards_total": "NumericalGuard_events_total",
    })
    write_csv("Fig6_OOD_response", ood)

    ablation = pd.read_csv(ROOT / "experiments" / "ltpp_data" / "deliverables" / "ablation_inference" / "ablation_table2.csv")
    ablation = ablation.rename(columns={
        "base_type": "pavement_family",
        "DSR_mean": "mean_DSR",
        "DSR_sd": "SD_DSR",
        "SCR_mean": "mean_episode_SCR",
        "SCR_sd": "SD_episode_SCR",
        "cost_mean": "mean_construction_cost_USD_m2",
        "cost_sd": "SD_construction_cost_USD_m2",
        "n": "number_of_runs",
    })
    write_csv("Table2_ablation", ablation)

    fig2 = pd.read_csv(FIG2_0813 / "Fig2_04_1034_panel_b_d_source_data.csv")
    fig2 = fig2.rename(columns={
        "dsr": "DSR",
        "running_scr": "running_SCR",
        "construction_cost_usd_m2": "construction_cost_USD_m2",
        "maintenance_npv_usd_m2": "maintenance_NPV_USD_m2",
        "lcc_total_npv_usd_m2": "twenty_year_LCC_USD_m2",
        "B1_asphalt_fatigue": "B1_margin_asphalt_fatigue",
        "B3_ac_permanent_deformation": "B3_margin_asphalt_permanent_deformation",
        "B4_subgrade_strain": "B4_margin_subgrade_strain",
        "AC_upper_cm": "AC1_upper_thickness_cm",
        "AC_mid_cm": "AC2_middle_thickness_cm",
        "AC_lower_cm": "AC3_lower_thickness_cm",
        "Base_cm": "base_thickness_cm",
        "Subbase_cm": "subbase_thickness_cm",
    })
    fig2["LTPP_section_id"] = "04_1034"
    fig2["study_section"] = fig2["LTPP_section_id"].map(label_map)
    fig2["seed"] = 0
    fig2 = fig2[
        [
            "study_section", "LTPP_section_id", "seed", "step", "DSR", "running_SCR",
            "reward", "compliant", "guard_rejected", "construction_cost_USD_m2",
            "maintenance_NPV_USD_m2", "twenty_year_LCC_USD_m2",
            "B1_margin_asphalt_fatigue", "B3_margin_asphalt_permanent_deformation",
            "B4_margin_subgrade_strain", "AC1_upper_thickness_cm",
            "AC2_middle_thickness_cm", "AC3_lower_thickness_cm",
            "base_thickness_cm", "subbase_thickness_cm",
        ]
    ]
    write_csv("Fig2_trajectory", fig2)

    index = pd.DataFrame([
        ["Fig2_trajectory", "Fig. 2", "Sequential design trajectory for representative flexible section", "output/fig2_04_1034_cost_heatmap_0813/Fig2_04_1034_panel_b_d_source_data.csv"],
        ["Fig3_LTPP_design_response", "Fig. 3a-c", "LTPP map/table, delivered structures and delivered-design metrics", "figures/source/Fig3_LTPP_only_true_delivered_source_data.csv"],
        ["Fig4_method_comparison", "Fig. 4a", "JTG compliance comparison across iLLM-PD, as-built and AASHTO 1993", "Fig4_comparison_practice_v2_source_data_DSR.csv"],
        ["Fig4_MEPDG_rutting", "Fig. 4b", "ME-PDG/NCHRP 1-37A rutting cross-specification check for three seeds per section", "Fig4_comparison_practice_v2_source_data_MEPDG.csv"],
        ["Fig5_temperature_fatigue", "Fig. 5", "Post-design temperature-dependent AC fatigue analysis", "experiments/batch_climate_12sections_summary.csv"],
        ["Fig6_OOD_response", "Robustness/OOD figure", "Out-of-distribution stress-test aggregate response", "ood_aggregate_20260723_133102.csv"],
        ["Table1_LTPP_sections", "Table 1 / Fig. 3", "Twelve selected LTPP sections and delivered design metrics", "figures/source/Fig3_LTPP_only_true_delivered_source_data.csv"],
        ["Table2_ablation", "Table 2", "Ablation results by pavement family", "ablation_table2.csv"],
    ], columns=["sheet_name", "paper_location", "content", "source_file"])
    write_csv("Supplementary_source_index", index)

    readme = pd.DataFrame([
        ["Workbook", "SourceData_iLLM-PD_NC_final.xlsx"],
        ["Purpose", "Final source data workbook for Nature Communications resubmission"],
        ["Structure", "Each worksheet is a flat data table with one header row and no embedded explanatory text rows"],
        ["Units", "Units are provided in column headers where applicable"],
        ["LTPP labels", "Study sections are labeled Section 1 to Section 12, with original LTPP section IDs retained in a separate column"],
        ["Generated from", "Curated clean package for Nature Communications resubmission"],
    ], columns=["item", "description"])
    write_csv("README", readme)

    print(f"Wrote 0813 final source-data tables to {OUT}")


if __name__ == "__main__":
    main()
