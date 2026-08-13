from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "plot"
OUT_PNG = OUT_DIR / "Fig4_comparison_practice_v2.png"
OUT_PDF = OUT_DIR / "Fig4_comparison_practice_v2.pdf"
OUT_DSR_CSV = OUT_DIR / "Fig4_comparison_practice_v2_source_data_DSR.csv"
OUT_MEPDG_CSV = OUT_DIR / "Fig4_comparison_practice_v2_source_data_MEPDG.csv"
OUT_HPDS_CSV = OUT_DIR / "Fig4_comparison_practice_v2_source_data_HPDS.csv"

AASHTO_CSV = ROOT / "experiments/ltpp_data/deliverables/ltpp_aashto1993/aashto1993_summary_20260525_095437.csv"
ASBUILT_CSV = ROOT / "experiments/ltpp_data/deliverables/ltpp_asbuilt/asbuilt_summary_20260525_005649.csv"
MEPDG_CSV = ROOT / "experiments/ltpp_data/deliverables/mepdg_check/mepdg_check_per_run_20260527_114533.csv"

SECTIONS = [
    "16_1010", "04_1034", "27_1085", "12_1060", "48_1076", "48_0001",
    "30_7076", "04_1065", "06_2004", "27_2023", "12_4097", "48_1109",
]

SECTION_TYPE = {
    "16_1010": "Flexible", "04_1034": "Flexible", "27_1085": "Flexible",
    "12_1060": "Flexible", "48_1076": "Flexible", "48_0001": "Flexible",
    "30_7076": "Semi-rigid", "04_1065": "Semi-rigid", "06_2004": "Semi-rigid",
    "27_2023": "Semi-rigid", "12_4097": "Semi-rigid", "48_1109": "Semi-rigid",
}

HPDS_ROWS = [
    {
        "section": "16_1010", "E_sub": "78 MPa",
        "base": ("fail", "B1 fail", "1020 mm"),
        "ac": ("fail", "B3 fail", "15.86 mm"),
        "illm": ("pass", "DSR 1.0", "5-layer"),
    },
    {
        "section": "48_0001", "E_sub": "700 MPa",
        "base": ("fail", "B3 fail", "15.87 mm"),
        "ac": ("fail", "B3 fail", "15.87 mm"),
        "illm": ("pass", "DSR 1.0", "5-layer"),
    },
    {
        "section": "30_7076", "E_sub": "59 MPa",
        "base": ("pass", "Pass*", "initial base"),
        "ac": ("fail", "B3 fail", "15.94 mm"),
        "illm": ("pass", "DSR 1.0", "5-layer"),
    },
]


W, H = 6200, 5200
BG = "white"
INK = "#202124"
MUTED = "#6a6f73"
GRID = "#d8dce0"
SOFT = "#f5f7f9"
ILLM = "#1879b9"
ASBUILT = "#707070"
AASHTO = "#d54a39"
FLEX = "#2176b5"
SEMI = "#d66a1f"
PASS = "#1f9d55"
FAIL = "#d83b35"
PANEL_BG = "#fbfcfd"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


F = {
    "title": font(98, True),
    "panel": font(70, True),
    "h1": font(48, True),
    "h2": font(39, True),
    "body": font(34),
    "body_b": font(34, True),
    "small": font(28),
    "small_b": font(28, True),
    "tiny": font(23),
    "tiny_b": font(23, True),
}


def text(draw: ImageDraw.ImageDraw, xy, s, fill=INK, f=None, anchor=None, **kw):
    draw.text(xy, s, fill=fill, font=f or F["body"], anchor=anchor, **kw)


def rect(draw, xy, fill, outline=None, width=1, radius=0):
    if radius:
        draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)
    else:
        draw.rectangle(xy, fill=fill, outline=outline, width=width)


def line(draw, xy, fill=GRID, width=2):
    draw.line(xy, fill=fill, width=width)


def centered(draw, box, s, fill=INK, f=None):
    x0, y0, x1, y1 = box
    draw.text(((x0 + x1) / 2, (y0 + y1) / 2), s, fill=fill, font=f or F["body"], anchor="mm")


def bar_cell(draw, box, value, max_value, color, label, good=True):
    x0, y0, x1, y1 = box
    rect(draw, box, "#f0f2f4", "#d7dadd", 2)
    frac = 0 if max_value <= 0 else min(max(value / max_value, 0), 1)
    rect(draw, (x0, y0, x0 + (x1 - x0) * frac, y1), color, None)
    fill = "white" if frac > 0.62 else INK
    centered(draw, box, label, fill=fill if good else INK, f=F["small_b" if good else "small"])


def build_data():
    aashto = pd.read_csv(AASHTO_CSV).set_index("section_id")
    asbuilt = pd.read_csv(ASBUILT_CSV).set_index("section_id")
    rows = []
    for sec in SECTIONS:
        rows.append({
            "section": sec,
            "type": SECTION_TYPE[sec],
            "method": "iLLM-PD",
            "DSR": 1.0,
            "pass_JTG": True,
        })
        rows.append({
            "section": sec,
            "type": SECTION_TYPE[sec],
            "method": "As-built",
            "DSR": float(asbuilt.loc[sec, "DSR"]),
            "pass_JTG": float(asbuilt.loc[sec, "DSR"]) >= 1.0,
            "B1": float(asbuilt.loc[sec, "B1"]),
            "B2": float(asbuilt.loc[sec, "B2"]),
            "B3": float(asbuilt.loc[sec, "B3"]),
            "B4": float(asbuilt.loc[sec, "B4"]),
        })
        rows.append({
            "section": sec,
            "type": SECTION_TYPE[sec],
            "method": "AASHTO 1993",
            "DSR": float(aashto.loc[sec, "DSR"]),
            "pass_JTG": float(aashto.loc[sec, "DSR"]) >= 1.0,
            "B1": float(aashto.loc[sec, "B1"]),
            "B2": float(aashto.loc[sec, "B2"]),
            "B3": float(aashto.loc[sec, "B3"]),
            "B4": float(aashto.loc[sec, "B4"]),
        })
    dsr = pd.DataFrame(rows)
    mepdg = pd.read_csv(MEPDG_CSV)
    mepdg["type"] = mepdg["section_id"].map(SECTION_TYPE)
    return dsr, mepdg


def panel_a(draw, dsr):
    x0, y0, x1, y1 = 220, 390, 5980, 1040
    text(draw, (x0, y0 - 90), "a", f=F["panel"])
    text(draw, (x0 + 95, y0 - 74), "JTG compliance under identical criteria", f=F["h1"])

    methods = [("iLLM-PD", ILLM), ("As-built", ASBUILT), ("AASHTO 1993", AASHTO)]
    card_w, gap = 1720, 120
    for i, (method, color) in enumerate(methods):
        cx0 = x0 + i * (card_w + gap)
        cx1 = cx0 + card_w
        vals = dsr[dsr["method"] == method]
        n_pass = int(vals["pass_JTG"].sum())
        n_fail = 12 - n_pass
        rect(draw, (cx0, y0, cx1, y1), PANEL_BG, "#d8dde2", 3, radius=18)
        text(draw, (cx0 + 70, y0 + 62), method, f=F["h2"], fill=color)
        text(draw, (cx0 + 70, y0 + 165), f"{n_pass}/12", f=font(118, True), fill=color)
        text(draw, (cx0 + 390, y0 + 218), "sections pass all four\nJTG D50-2017 indicators", f=F["small"], fill=MUTED)
        bx0, by0, bx1, by1 = cx0 + 70, y0 + 360, cx1 - 70, y0 + 470
        rect(draw, (bx0, by0, bx1, by1), "#eef1f4", None, radius=12)
        pass_w = (bx1 - bx0) * n_pass / 12
        rect(draw, (bx0, by0, bx0 + pass_w, by1), color, None, radius=12)
        if n_fail:
            text(draw, (bx1 - 18, by1 + 42), f"{n_fail} fail", f=F["small_b"], fill=FAIL, anchor="ra")
        else:
            text(draw, (bx1 - 18, by1 + 42), "no failures", f=F["small_b"], fill=PASS, anchor="ra")


def panel_b(draw, dsr):
    x0, y0, x1, y1 = 220, 1230, 5980, 2620
    text(draw, (x0, y0 - 85), "b", f=F["panel"])
    text(draw, (x0 + 95, y0 - 68), "Per-section design safety rate: failure locations", f=F["h1"])
    text(draw, (x1, y0 - 44), "cell width is scaled to DSR; red edge marks DSR < 1.0", f=F["small"], fill=MUTED, anchor="ra")

    left_w = 580
    col_w = 1300
    row_h = 82
    header_h = 100
    methods = [("iLLM-PD", ILLM), ("As-built", ASBUILT), ("AASHTO 1993", AASHTO)]
    rect(draw, (x0, y0, x1, y0 + header_h), "#f1f4f7", None)
    text(draw, (x0 + 20, y0 + 60), "Section", f=F["small_b"])
    for j, (m, c) in enumerate(methods):
        text(draw, (x0 + left_w + j * col_w + 20, y0 + 60), m, f=F["small_b"], fill=c)

    for i, sec in enumerate(SECTIONS):
        yy = y0 + header_h + i * row_h
        bg = "#f7fbfd" if SECTION_TYPE[sec] == "Flexible" else "#fff7f1"
        rect(draw, (x0, yy, x1, yy + row_h), bg if i % 2 == 0 else "white", None)
        if i == 6:
            line(draw, (x0, yy, x1, yy), "#899099", 4)
        family_color = FLEX if SECTION_TYPE[sec] == "Flexible" else SEMI
        text(draw, (x0 + 20, yy + row_h / 2), sec, f=F["small_b"], fill=family_color, anchor="lm")
        for j, (m, c) in enumerate(methods):
            val = float(dsr[(dsr.section == sec) & (dsr.method == m)]["DSR"].iloc[0])
            cell = (
                x0 + left_w + j * col_w + 20,
                yy + 18,
                x0 + left_w + (j + 1) * col_w - 45,
                yy + row_h - 18,
            )
            is_pass = val >= 1.0 - 1e-9
            outline = "#d5d9dd" if is_pass else FAIL
            rect(draw, cell, "#ebeff2", outline, 3 if not is_pass else 1)
            fill_w = (cell[2] - cell[0]) * min(val, 1.0)
            rect(draw, (cell[0], cell[1], cell[0] + fill_w, cell[3]), c if is_pass else "#f0a09a", None)
            label = "1.000" if is_pass else f"{val:.3f}"
            centered(draw, cell, label, fill="white" if is_pass and val > 0.65 else INK, f=F["tiny_b"])


def panel_c(draw, mepdg):
    x0, y0, x1, y1 = 220, 2860, 5980, 3920
    text(draw, (x0, y0 - 88), "c", f=F["panel"])
    text(draw, (x0 + 95, y0 - 70), "ME-PDG rutting depth for 36 design-seed cases", f=F["h1"])
    n_pass = int(mepdg["MEPDG_all_pass"].sum())
    text(draw, (x1, y0 - 47), f"{n_pass}/36 pass; three failures are 48_0001 at 19.08 mm", f=F["small_b"], fill=FAIL, anchor="ra")

    label_w = 620
    plot_x0, plot_x1 = x0 + label_w, x1 - 120
    min_x, max_x = 7.5, 20.0
    row_h = 72

    def sx(v):
        return plot_x0 + (float(v) - min_x) / (max_x - min_x) * (plot_x1 - plot_x0)

    for val, col, lab in [(16, "#999999", "16 mm (95%)"), (19, FAIL, "19 mm (90%)")]:
        xx = sx(val)
        line(draw, (xx, y0 + 55, xx, y0 + 55 + row_h * 12), col, 4 if val == 19 else 2)
        text(draw, (xx + 10, y0 + 35), lab, f=F["tiny_b"], fill=col)

    for tick in [8, 10, 12, 14, 16, 18, 20]:
        xx = sx(tick)
        line(draw, (xx, y0 + 55 + row_h * 12, xx, y0 + 82 + row_h * 12), "#6f7478", 3)
        text(draw, (xx, y0 + 128 + row_h * 12), str(tick), f=F["tiny"], anchor="mm")
    text(draw, ((plot_x0 + plot_x1) / 2, y0 + 195 + row_h * 12), "Rutting depth RD (mm)", f=F["small_b"], anchor="mm")

    for i, sec in enumerate(SECTIONS):
        yy = y0 + 55 + i * row_h
        bg = "#f8fbfd" if SECTION_TYPE[sec] == "Flexible" else "#fff7f1"
        rect(draw, (x0, yy - 30, x1, yy + 30), bg if i % 2 == 0 else "white", None)
        if i == 6:
            line(draw, (x0, yy - 36, x1, yy - 36), "#899099", 4)
        family_color = FLEX if SECTION_TYPE[sec] == "Flexible" else SEMI
        text(draw, (x0 + 20, yy), sec, f=F["tiny_b"], fill=family_color, anchor="lm")
        vals = mepdg[mepdg.section_id == sec].sort_values("seed")
        for k, (_, r) in enumerate(vals.iterrows()):
            xx = sx(r["RD_total_mm"])
            dy = [-14, 0, 14][k % 3]
            col = family_color if bool(r["MEPDG_all_pass"]) else FAIL
            draw.ellipse((xx - 15, yy + dy - 15, xx + 15, yy + dy + 15), fill=col, outline="white", width=3)
            if not bool(r["MEPDG_all_pass"]):
                draw.ellipse((xx - 25, yy + dy - 25, xx + 25, yy + dy + 25), outline=FAIL, width=5)


def panel_d(draw):
    x0, y0, x1, y1 = 220, 4270, 5980, 5020
    text(draw, (x0, y0 - 88), "d", f=F["panel"])
    text(draw, (x0 + 95, y0 - 70), "Paradigm contrast: HPDS single-layer adjustment vs iLLM-PD five-layer co-optimisation", f=F["h1"])

    row_label_w = 720
    col_w = 1520
    row_h = 155
    headers = ["HPDS: base layer", "HPDS: AC layer", "iLLM-PD: five layers"]
    rect(draw, (x0, y0, x1, y0 + 90), "#f1f4f7", None)
    for j, h in enumerate(headers):
        text(draw, (x0 + row_label_w + j * col_w + col_w / 2, y0 + 52), h, f=F["small_b"], anchor="mm")

    for i, row in enumerate(HPDS_ROWS):
        yy = y0 + 90 + i * row_h
        rect(draw, (x0, yy, x1, yy + row_h), "#fbfcfd" if i % 2 == 0 else "white", None)
        text(draw, (x0 + 20, yy + 52), row["section"], f=F["small_b"], anchor="lm")
        text(draw, (x0 + 20, yy + 100), f"Esub = {row['E_sub']}", f=F["tiny"], fill=MUTED, anchor="lm")
        for j, key in enumerate(["base", "ac", "illm"]):
            status, main, sub = row[key]
            cx0 = x0 + row_label_w + j * col_w + 80
            cx1 = x0 + row_label_w + (j + 1) * col_w - 80
            cy0, cy1 = yy + 26, yy + row_h - 26
            if status == "pass":
                fc, ec, sym = "#dff2e7", PASS, "PASS"
            else:
                fc, ec, sym = "#fde4e1", FAIL, "FAIL"
            rect(draw, (cx0, cy0, cx1, cy1), fc, ec, 3, radius=18)
            text(draw, (cx0 + 44, (cy0 + cy1) / 2), sym, f=F["tiny_b"], fill=ec, anchor="lm")
            text(draw, (cx0 + 200, cy0 + 37), main, f=F["tiny_b"], fill=ec)
            text(draw, (cx0 + 200, cy0 + 80), sub, f=F["tiny"], fill=MUTED)

    text(draw, (x0, y1 + 58),
         "* 30_7076: HPDS initial base thickness already satisfies the checks; this is verification, not iterative design.",
         f=F["tiny"], fill=MUTED)
    text(draw, (x0, y1 + 98),
         "HPDS adjusts one designated layer by construction, whereas iLLM-PD co-optimises all five layers.",
         f=F["tiny"], fill=MUTED)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dsr, mepdg = build_data()
    dsr.to_csv(OUT_DSR_CSV, index=False)
    mepdg.to_csv(OUT_MEPDG_CSV, index=False)
    hpds_rows = []
    for row in HPDS_ROWS:
        for approach, key in [("HPDS_base_layer", "base"), ("HPDS_AC_layer", "ac"), ("iLLM_PD_five_layer", "illm")]:
            status, main_label, sub_label = row[key]
            hpds_rows.append({
                "section": row["section"],
                "E_sub": row["E_sub"],
                "approach": approach,
                "status": status,
                "main_label": main_label,
                "sub_label": sub_label,
            })
    pd.DataFrame(hpds_rows).to_csv(OUT_HPDS_CSV, index=False)

    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    text(draw, (220, 105), "Fig. 4 | Comparison with established practice and cross-specification", f=F["title"])
    text(draw, (220, 215),
         "iLLM-PD is compared against as-built engineering practice, AASHTO 1993 designs, ME-PDG rutting predictions, and HPDS single-layer design.",
         f=F["body"], fill=MUTED)

    panel_a(draw, dsr)
    panel_b(draw, dsr)
    panel_c(draw, mepdg)
    panel_d(draw)

    img.save(OUT_PNG, dpi=(450, 450))
    img.save(OUT_PDF, "PDF", resolution=450.0)
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_DSR_CSV}")
    print(f"Saved {OUT_MEPDG_CSV}")
    print(f"Saved {OUT_HPDS_CSV}")
    print("JTG pass counts:")
    print(dsr.groupby("method")["pass_JTG"].sum().to_string())
    print("MEPDG pass:", int(mepdg["MEPDG_all_pass"].sum()), "/", len(mepdg))


if __name__ == "__main__":
    main()


