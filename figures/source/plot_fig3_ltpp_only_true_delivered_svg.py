from __future__ import annotations

import pandas as pd

import plot_fig3_design_response_revised_editable_svg as old
import plot_fig3_design_response_revised_pillow_largefont as src


OUT = src.OUT_DIR / "Fig3_LTPP_only_true_delivered.svg"
CSV_OUT = src.OUT_DIR / "Fig3_LTPP_only_true_delivered_source_data.csv"
W, H = 5400, 6200


def draw_map_no_ncat(svg, df: pd.DataFrame, box):
    x0, y0, x1, y1 = box
    old.header(svg, x0, y0 - 80, "a", "Field-section locations and LTPP lookup")
    map_box = (x0, y0, x0 + 3500, y1)
    tab_box = (x0 + 3560, y0, x1, y1)

    svg.rect(*map_box[:2], map_box[2] - map_box[0], map_box[3] - map_box[1], src.COL["water"], "#d7dee2", 2, 18)
    tr = src.map_transform((map_box[0] + 20, map_box[1] + 20, map_box[2] - 20, map_box[3] - 132))

    climate_by_state = {}
    for _, r in df.iterrows():
        climate_by_state.setdefault(r["state"], r["climate"])
    state_name_to_abbr = {
        "Arizona": "AZ",
        "California": "CA",
        "Florida": "FL",
        "Idaho": "ID",
        "Minnesota": "MN",
        "Montana": "MT",
        "Texas": "TX",
        "Alabama": "AL",
    }
    for name, poly in src.state_polys():
        abbr = state_name_to_abbr.get(name)
        fill = src.CLIMATE_COLORS.get(climate_by_state.get(abbr, ""), src.COL["state_fill"])
        svg.polygon([tr(lon, lat) for lon, lat in poly], fill, src.COL["state_edge"], 1.2)

    for abbr, lon, lat in src.STATE_INFO.values():
        xx, yy = tr(lon, lat)
        svg.text(xx, yy, abbr, 47, "#444444", "middle")

    for _, r in df.iterrows():
        xx, yy = tr(r["lon"], r["lat"])
        if r["type"] == "flexible":
            svg.circle(xx, yy, 19, "white", src.COL["flexible"], 6)
        else:
            svg.rect(xx - 18, yy - 18, 36, 36, "white", src.COL["semi_rigid"], 6)
        dx, dy = src.LABEL_OFFSETS.get(r["section"], (30, -12))
        lx, ly = xx + dx, yy + dy
        svg.line(xx, yy, lx - 6 if dx > 0 else lx + 112, ly + 11, "#777777", 1)
        svg.text(lx, ly, r["study_id"], 47)

    lx, ly = map_box[0] + 58, map_box[3] - 125
    for i, (name, c) in enumerate(src.CLIMATE_COLORS.items()):
        svg.rect(lx + i * 430, ly, 44, 32, c, "#555555", 1)
        svg.text(lx + i * 430 + 58, ly + 15, name.replace("NoFreeze", "No-freeze"), 40)
    tx = map_box[2] - 760
    svg.circle(tx + 21, ly + 83, 21, "white", src.COL["flexible"], 6)
    svg.text(tx + 58, ly + 83, "Flexible", 40)
    svg.rect(tx + 320, ly + 62, 42, 42, "white", src.COL["semi_rigid"], 6)
    svg.text(tx + 378, ly + 83, "Semi-rigid", 40)

    svg.rect(tab_box[0] - 34, tab_box[1] - 4, tab_box[2] - tab_box[0] + 68, tab_box[3] - tab_box[1] + 8, "white")
    svg.text((tab_box[0] + tab_box[2]) / 2, tab_box[1] + 34, "LTPP validation sections", 54, anchor="middle")

    # Wider final column keeps Esub values within the table; lines extend to the same width.
    cx = [tab_box[0] - 8, tab_box[0] + 255, tab_box[0] + 565, tab_box[0] + 900, tab_box[0] + 1275, tab_box[2] + 26]
    heads = ["Study ID", "LTPP ID", "Family", "Climate", "Esub\n(MPa)"]
    yhead = tab_box[1] + 132
    table_bottom = tab_box[3] - 14
    svg.line(cx[0], yhead - 58, cx[-1], yhead - 58, "#222222", 4)
    svg.line(cx[0], yhead + 64, cx[-1], yhead + 64, "#222222", 4)
    for i, h in enumerate(heads):
        for k, line in enumerate(h.split("\n")):
            svg.text((cx[i] + cx[i + 1]) / 2, yhead - 24 + k * 46, line, 50 if i == 4 else 54, anchor="middle")

    row_h = (table_bottom - (yhead + 84)) / len(df)
    climate_short = {"Dry-Freeze": "Dry-F", "Dry-NoFreeze": "Dry-NF", "Wet-Freeze": "Wet-F", "Wet-NoFreeze": "Wet-NF"}
    for i, r in df.iterrows():
        yy = yhead + 84 + i * row_h + row_h / 2
        vals = [
            r["study_id"],
            r["section"],
            "Flexible" if r["type"] == "flexible" else "Semi-rigid",
            climate_short.get(r["climate"], r["climate"]),
            f"{int(round(r['E_sub']))}",
        ]
        for j, val in enumerate(vals):
            svg.text((cx[j] + cx[j + 1]) / 2, yy, val, 50 if j == 4 else 54, anchor="middle")
    svg.line(cx[0], table_bottom, cx[-1], table_bottom, "#222222", 3)


def draw_matrix_white_long_bars(svg, df: pd.DataFrame, box):
    x0, y0, x1, y1 = box
    old.header(svg, x0, y0 - 125, "c", "LTPP delivered-design metrics")
    metrics = [
        ("E_sub", "Esub\n(MPa)"),
        ("total_thickness_cm", "Total h\n(cm)"),
        ("DSR", "DSR"),
        ("SCR", "Episode\nSCR"),
        ("cost_USD", "Cost\n(USD)"),
        ("LCC_USD", "20-year LCC\n(USD)"),
    ]
    cols = [x0 + 350, x0 + 1145, x0 + 1940, x0 + 2735, x0 + 3530, x0 + 4325]
    bar_w, yhead, row_h = 690, y0 + 74, 83
    svg.text(x0 + 18, yhead - 24, "Section", 54)
    for col, (_, label) in zip(cols, metrics):
        for k, line in enumerate(label.split("\n")):
            svg.text(col + bar_w / 2, yhead - 56 + k * 54, line, 54, anchor="middle")
    svg.line(x0, yhead + 55, x1, yhead + 55, "#222222", 3)

    for i, r in df.iterrows():
        yy = yhead + 110 + i * row_h
        bg = "#f7fbff" if r["type"] == "flexible" else "#fff8f2"
        c = src.COL[r["type"]]
        svg.rect(x0, yy - 37, x1 - x0, 74, bg)
        if i == 6:
            svg.line(x0, yy - 47, x1, yy - 47, "#777777", 3)
        svg.text(x0 + 18, yy, r["study_id"], 54)
        for col, (m, _) in zip(cols, metrics):
            vals = df[m].astype(float)
            lo, hi = vals.min(), vals.max()
            val = float(r[m])
            if m in ["DSR", "SCR"]:
                lab = f"{val:.3f}"
                lo, hi = (0.90, 1.0) if m == "SCR" else (0.99, 1.0)
            elif m == "E_sub":
                lab = f"{val:.0f}"
            else:
                lab = f"{val:.1f}"
            frac = old.norm(val, lo, hi)
            svg.rect(col, yy - 25, bar_w, 50, "#eeeeee")
            svg.rect(col, yy - 25, bar_w * frac, 50, c)
            fill = "white" if frac >= 0.50 else "#202020"
            svg.text(col + bar_w / 2, yy, lab, 54, fill, "middle")

    svg.text(x0, y1 - 16, "Bars are min-max scaled within each metric; costs are USD m^-2.", 40, src.COL["muted"])


def main():
    df = src.load_ltp_data()
    df.to_csv(CSV_OUT, index=False)

    svg = old.SVG(W, H)
    draw_map_no_ncat(svg, df, (230, 300, 5170, 1800))
    old.draw_structures(svg, df, (230, 2040, 5170, 4520))
    draw_matrix_white_long_bars(svg, df, (230, 4780, 5170, 6100))
    svg.save(OUT)
    print(OUT)
    print(CSV_OUT)


if __name__ == "__main__":
    main()
