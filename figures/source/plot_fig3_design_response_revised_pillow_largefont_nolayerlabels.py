from pathlib import Path
import json
import math

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path("D:/iLLM_PD_new")

LAYER_CSV = PROJECT / "experiments" / "core_results_2048_full.csv"
LATEST_LTPP_SUMMARY = PROJECT / "experiments" / "ltpp_data" / "deliverables" / "ltpp_inference_esal_20260808" / "ltpp_inference_summary_20260808_170223.csv"
STATE_GEOJSON = PROJECT / "plot" / "us-states.geojson"
NCAT_REAL_FEA_CSV = ROOT / "reports" / "ncat_design_benchmark_20260715" / "real_fea_eval" / "ncat_asbuilt_vs_design_real_fea_20260716_095838.csv"
NCAT_SENS_CSV = ROOT / "reports" / "ncat_design_benchmark_20260715" / "material_sensitivity" / "ncat_material_sensitivity_20260716_111322.csv"

OUT_DIR = ROOT / "output" / "fig3_revised"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PNG_OUT = OUT_DIR / "Fig3_design_response_revised_pillow_largefont_nolayerlabels.png"
PDF_OUT = OUT_DIR / "Fig3_design_response_revised_pillow_largefont_nolayerlabels.pdf"
CSV_OUT = OUT_DIR / "Fig3_revised_source_data.csv"

W, H = 5400, 6200

COL = {
    "flexible": "#0072B2",
    "semi_rigid": "#D55E00",
    "text": "#202020",
    "muted": "#666666",
    "line": "#d8d8d8",
    "water": "#f7fbfd",
    "state_fill": "#f3f1eb",
    "state_edge": "#9d9d9d",
    "gray": "#8d8d8d",
}
CLIMATE_COLORS = {
    "Dry-Freeze": "#4C78A8",
    "Dry-NoFreeze": "#F58518",
    "Wet-Freeze": "#54A24B",
    "Wet-NoFreeze": "#B279A2",
}
LAYERS = [
    ("AC-1 upper", "#252B48"),
    ("AC-2 mid", "#4C5C92"),
    ("AC-3 lower", "#8294C4"),
    ("Base", "#A76F2B"),
    ("Subbase", "#D8C39A"),
]
STATE_INFO = {
    "04": ("AZ", -111.5, 34.3),
    "06": ("CA", -119.5, 37.1),
    "12": ("FL", -82.5, 28.2),
    "16": ("ID", -114.5, 44.1),
    "27": ("MN", -94.5, 46.2),
    "30": ("MT", -110.5, 47.0),
    "48": ("TX", -99.2, 31.1),
}
OFFSETS = {
    "04_1034": (-0.55, 0.20),
    "04_1065": (0.45, -0.25),
    "12_1060": (-0.55, -0.10),
    "12_4097": (0.55, 0.20),
    "27_1085": (-0.50, -0.15),
    "27_2023": (0.50, 0.18),
    "48_0001": (-0.80, -0.18),
    "48_1076": (0.00, 0.42),
    "48_1109": (0.75, -0.05),
}
LABEL_OFFSETS = {
    "04_1034": (-188, -70),
    "04_1065": (52, -24),
    "12_1060": (-218, -48),
    "12_4097": (46, 62),
    "16_1010": (36, -10),
    "27_1085": (-178, -48),
    "27_2023": (42, 42),
    "30_7076": (42, -14),
    "48_0001": (62, 92),
    "48_1076": (-240, -88),
    "48_1109": (88, -6),
}
CLIMATE = {
    "16_1010": "Dry-Freeze",
    "27_1085": "Wet-Freeze",
    "04_1034": "Dry-NoFreeze",
    "48_1076": "Dry-NoFreeze",
    "12_1060": "Wet-NoFreeze",
    "48_0001": "Wet-NoFreeze",
    "30_7076": "Dry-Freeze",
    "04_1065": "Dry-NoFreeze",
    "48_1109": "Wet-NoFreeze",
    "06_2004": "Dry-NoFreeze",
    "27_2023": "Wet-Freeze",
    "12_4097": "Wet-NoFreeze",
}
EXCLUDE_NAMES = {"Alaska", "Hawaii", "Puerto Rico"}


def font(size, bold=False):
    name = "arialbd.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)


F = {
    "panel": font(98, True),
    "h2": font(74, True),
    "body": font(62),
    "small": font(55),
    "tiny": font(49),
    "mini": font(38),
    "micro": font(40),
    "table": font(54),
    "table_bold": font(54, True),
    "axis": font(52, True),
    "axis_big": font(60, True),
    "legend": font(50),
}


def t(draw, xy, s, f=None, fill=None, anchor=None):
    draw.text(xy, str(s), font=f or F["body"], fill=fill or COL["text"], anchor=anchor)


def rotated_text(img, xy, s, f=None, fill=None, angle=90):
    f = f or F["body"]
    fill = fill or COL["text"]
    bbox = f.getbbox(str(s))
    w, h = bbox[2] - bbox[0] + 12, bbox[3] - bbox[1] + 12
    tmp = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    td = ImageDraw.Draw(tmp)
    td.text((6 - bbox[0], 6 - bbox[1]), str(s), font=f, fill=fill)
    rot = tmp.rotate(angle, expand=True)
    img.paste(rot, (int(xy[0] - rot.width / 2), int(xy[1] - rot.height / 2)), rot)


def rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def shade(h, factor):
    r, g, b = rgb(h)
    if factor >= 1:
        r = int(255 - (255 - r) / factor)
        g = int(255 - (255 - g) / factor)
        b = int(255 - (255 - b) / factor)
    else:
        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


def header(draw, x, y, letter, title):
    t(draw, (x, y), letter, F["panel"])
    t(draw, (x + 70, y + 9), title, F["h2"])


def albers(lon, lat):
    lon = math.radians(float(lon))
    lat = math.radians(float(lat))
    lat1, lat2, lat0, lon0 = [math.radians(v) for v in (29.5, 45.5, 23.0, -96.0)]
    n = 0.5 * (math.sin(lat1) + math.sin(lat2))
    c = math.cos(lat1) ** 2 + 2 * n * math.sin(lat1)
    rho = math.sqrt(max(0, c - 2 * n * math.sin(lat))) / n
    rho0 = math.sqrt(max(0, c - 2 * n * math.sin(lat0))) / n
    theta = n * (lon - lon0)
    return rho * math.sin(theta), rho0 - rho * math.cos(theta)


def state_polys():
    data = json.loads(STATE_GEOJSON.read_text(encoding="utf-8"))
    for feat in data["features"]:
        name = feat.get("properties", {}).get("name") or feat.get("properties", {}).get("NAME")
        if name in EXCLUDE_NAMES:
            continue
        geom = feat["geometry"]
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for poly in polys:
            outer = poly[0]
            if len(outer) >= 3:
                yield name, outer


def map_transform(box):
    pts = []
    for _, poly in state_polys():
        for lon, lat in poly:
            pts.append(albers(lon, lat))
    minx, maxx = min(p[0] for p in pts), max(p[0] for p in pts)
    miny, maxy = min(p[1] for p in pts), max(p[1] for p in pts)
    x0, y0, x1, y1 = box
    pad = 42
    sx = (x1 - x0 - 2 * pad) / (maxx - minx)
    sy = (y1 - y0 - 2 * pad) / (maxy - miny)
    sc = min(sx, sy)
    ox = x0 + pad + (x1 - x0 - 2 * pad - (maxx - minx) * sc) / 2
    oy_extra = (y1 - y0 - 2 * pad - (maxy - miny) * sc) / 2

    def tr(lon, lat):
        x, y = albers(lon, lat)
        return ox + (x - minx) * sc, y1 - pad - (y - miny) * sc - oy_extra

    return tr


def load_ltp_data():
    layers = pd.read_csv(LAYER_CSV)
    summary = pd.read_csv(LATEST_LTPP_SUMMARY)
    summary = summary.sort_values(["section_id", "delivered_C_const_usd_m2"]).groupby("section_id", as_index=False).first()
    df = layers.merge(summary, left_on="section", right_on="section_id", how="left", suffixes=("", "_new"))
    df["_sort"] = df["type"].map({"flexible": 0, "semi_rigid": 1})
    df = df.sort_values(["_sort", "E_subgrade", "section"]).reset_index(drop=True)
    df["study_id"] = [f"Section {i}" for i in range(1, len(df) + 1)]
    df["climate"] = df["section"].map(CLIMATE)
    df["state_code"] = df["section"].str[:2]
    df["state"] = df["state_code"].map(lambda s: STATE_INFO[s][0])
    df["lon"] = df["state_code"].map(lambda s: STATE_INFO[s][1]) + df["section"].map(lambda s: OFFSETS.get(s, (0, 0))[0])
    df["lat"] = df["state_code"].map(lambda s: STATE_INFO[s][2]) + df["section"].map(lambda s: OFFSETS.get(s, (0, 0))[1])
    df["total_thickness_cm"] = df[[f"h{i}" for i in range(1, 6)]].sum(axis=1)
    df["cost_USD"] = df["delivered_C_const_usd_m2"].astype(float)
    df["LCC_USD"] = df["delivered_lcc_npv_usd_m2"].astype(float)
    df["DSR"] = df["delivered_dsr"].astype(float)
    df["SCR"] = df["compliance_rate_in_episode"].astype(float)
    df["E_sub"] = df["E_subgrade"].astype(float)
    return df


def load_ncat():
    real = pd.read_csv(NCAT_REAL_FEA_CSV)
    real["short"] = real["section"].str.replace("NCAT_CG_", "", regex=False)
    sens = pd.read_csv(NCAT_SENS_CSV)
    reps = ["NCAT_CG_N1", "NCAT_CG_N5", "NCAT_CG_S6"]
    cols = [
        ("Default", lambda x: (x["scenario"].eq("baseline")) & (x["R0_mm"].eq(1.5)) & (x["VFA_pct"].eq(65.0))),
        ("Stiffness only", lambda x: (x["scenario"].str.contains("stiffer")) & (x["R0_mm"].eq(1.5)) & (x["VFA_pct"].eq(65.0))),
        ("Rutting resistance", lambda x: (x["scenario"].eq("baseline")) & (x["R0_mm"].eq(1.2)) & (x["VFA_pct"].eq(65.0))),
        ("Combined", lambda x: (x["R0_mm"].eq(1.2)) & (x["VFA_pct"].eq(65.0))),
    ]
    heat = []
    for sec in reps:
        sub = sens[sens["section"].eq(sec)]
        row = {"section": sec.replace("NCAT_CG_", "")}
        for name, filt in cols:
            candidates = sub[filt(sub)]
            row[name] = float(candidates["dsr"].max()) if len(candidates) else float("nan")
        heat.append(row)
    return real, pd.DataFrame(heat)


def draw_map_and_lookup(draw, df, box):
    x0, y0, x1, y1 = box
    header(draw, x0, y0 - 105, "a", "Field-section locations and LTPP lookup")
    map_box = (x0, y0, x0 + 3500, y1)
    tab_box = (x0 + 3560, y0, x1, y1)
    draw.rounded_rectangle(map_box, radius=18, fill=COL["water"], outline="#d7dee2", width=2)
    tr = map_transform((map_box[0] + 20, map_box[1] + 20, map_box[2] - 20, map_box[3] - 132))
    climate_by_state = {}
    for _, r in df.iterrows():
        climate_by_state.setdefault(r["state"], r["climate"])
    state_name_to_abbr = {"Arizona": "AZ", "California": "CA", "Florida": "FL", "Idaho": "ID", "Minnesota": "MN", "Montana": "MT", "Texas": "TX", "Alabama": "AL"}
    for name, poly in state_polys():
        abbr = state_name_to_abbr.get(name)
        fill = CLIMATE_COLORS.get(climate_by_state.get(abbr, ""), COL["state_fill"])
        pts = [tr(lon, lat) for lon, lat in poly]
        draw.polygon(pts, fill=fill, outline=COL["state_edge"])
    for abbr, lon, lat in STATE_INFO.values():
        xx, yy = tr(lon, lat)
        t(draw, (xx, yy), abbr, F["tiny"], "#444444", "mm")
    for _, r in df.iterrows():
        xx, yy = tr(r["lon"], r["lat"])
        if r["type"] == "flexible":
            draw.ellipse([xx - 19, yy - 19, xx + 19, yy + 19], fill="white", outline=COL["flexible"], width=6)
        else:
            draw.rectangle([xx - 18, yy - 18, xx + 18, yy + 18], fill="white", outline=COL["semi_rigid"], width=6)
        dx, dy = LABEL_OFFSETS.get(r["section"], (30, -12))
        lx2, ly2 = xx + dx, yy + dy
        draw.line([(xx, yy), (lx2 - 6 if dx > 0 else lx2 + 112, ly2 + 11)], fill="#777777", width=1)
        t(draw, (lx2, ly2), r["study_id"], F["tiny"])
    # NCAT Test Track, Auburn, Alabama.
    nx, ny = tr(-85.49, 32.60)
    star = [(nx, ny - 28), (nx + 9, ny - 8), (nx + 30, ny - 8), (nx + 14, ny + 5), (nx + 20, ny + 26), (nx, ny + 14), (nx - 20, ny + 26), (nx - 14, ny + 5), (nx - 30, ny - 8), (nx - 9, ny - 8)]
    draw.polygon(star, fill="#111111", outline="white")
    t(draw, (nx + 38, ny - 8), "NCAT", F["tiny"])
    lx, ly = map_box[0] + 58, map_box[3] - 125
    for i, (name, c) in enumerate(CLIMATE_COLORS.items()):
        draw.rectangle([lx + i * 430, ly, lx + i * 430 + 44, ly + 32], fill=c, outline="#555555")
        t(draw, (lx + i * 430 + 58, ly - 1), name.replace("NoFreeze", "No-freeze"), F["micro"])
    tx = map_box[2] - 760
    draw.ellipse([tx, ly + 62, tx + 42, ly + 104], fill="white", outline=COL["flexible"], width=6)
    t(draw, (tx + 58, ly + 64), "Flexible", F["micro"])
    draw.rectangle([tx + 320, ly + 62, tx + 362, ly + 104], fill="white", outline=COL["semi_rigid"], width=6)
    t(draw, (tx + 378, ly + 64), "Semi-rigid", F["micro"])

    # Three-line lookup table, LTPP only.
    draw.rectangle([tab_box[0] - 24, tab_box[1] - 4, tab_box[2] + 24, tab_box[3] + 4], fill="white")
    t(draw, ((tab_box[0] + tab_box[2]) / 2, tab_box[1] + 34), "LTPP validation sections", F["table_bold"], anchor="mm")
    cx = [tab_box[0], tab_box[0] + 290, tab_box[0] + 610, tab_box[0] + 970, tab_box[0] + 1385, tab_box[2]]
    climate_short = {
        "Dry-Freeze": "Dry-F",
        "Dry-NoFreeze": "Dry-NF",
        "Wet-Freeze": "Wet-F",
        "Wet-NoFreeze": "Wet-NF",
    }
    heads = ["Study ID", "LTPP ID", "Family", "Climate", "Esub\n(MPa)"]
    yhead = tab_box[1] + 132
    table_bottom = tab_box[3] - 14
    draw.line([(tab_box[0], yhead - 58), (tab_box[2], yhead - 58)], fill="#222222", width=4)
    draw.line([(tab_box[0], yhead + 64), (tab_box[2], yhead + 64)], fill="#222222", width=4)
    for i, h in enumerate(heads):
        for k, line in enumerate(h.split("\n")):
            t(draw, ((cx[i] + cx[i + 1]) / 2, yhead - 24 + k * 46), line, F["table_bold"], anchor="mm")
    row_h = (table_bottom - (yhead + 84)) / len(df)
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
            t(draw, ((cx[j] + cx[j + 1]) / 2, yy), val, F["table"], fill=COL["text"], anchor="mm")
    ybot = table_bottom
    draw.line([(tab_box[0], ybot), (tab_box[2], ybot)], fill="#222222", width=3)


def cuboid(draw, x, base, w, dep, z0, dz, zscale, color):
    yb = base - z0 * zscale
    yt = base - (z0 + dz) * zscale
    dx, dy = dep, -int(dep * 0.55)
    edge = shade(color, 0.78)
    draw.polygon([(x + w, yt), (x + w + dx, yt + dy), (x + w + dx, yb + dy), (x + w, yb)], fill=shade(color, 0.72), outline=edge)
    draw.polygon([(x, yt), (x + w, yt), (x + w, yb), (x, yb)], fill=rgb(color), outline=edge)
    draw.polygon([(x, yt), (x + w, yt), (x + w + dx, yt + dy), (x + dx, yt + dy)], fill=shade(color, 1.25), outline=edge)


def draw_layer_callouts(draw, x, base, w, dep, zscale, row, side="right"):
    """CAD-style side labels for the five layer thicknesses of one structure."""
    if side == "left":
        bracket_x = x - 34
        text_x = bracket_x - 35
        text_anchor = "rm"
        tick0, tick1 = bracket_x - 16, bracket_x + 16
        lead0, lead1 = bracket_x - 16, text_x + 8
    else:
        bracket_x = x + w + dep + 38
        text_x = bracket_x + 31
        text_anchor = "lm"
        tick0, tick1 = bracket_x - 16, bracket_x + 16
        lead0, lead1 = bracket_x + 16, text_x - 8
    h1, h2, h3, h4, h5 = (float(row[f"h{i}"]) for i in range(1, 6))
    layer_specs = [
        ("h1", h1, LAYERS[0][1], h5 + h4 + h3 + h2),
        ("h2", h2, LAYERS[1][1], h5 + h4 + h3),
        ("h3", h3, LAYERS[2][1], h5 + h4),
        ("h4", h4, LAYERS[3][1], h5),
        ("h5", h5, LAYERS[4][1], 0.0),
    ]
    for _, h, color, z in layer_specs:
        yb = base - z * zscale
        yt = base - (z + h) * zscale
        ym = (yb + yt) / 2
        draw.line([(bracket_x, yt), (bracket_x, yb)], fill=color, width=4)
        draw.line([(tick0, yt), (tick1, yt)], fill=color, width=3)
        draw.line([(tick0, yb), (tick1, yb)], fill=color, width=3)
        draw.line([(lead0, ym), (lead1, ym)], fill=color, width=3)
        t(draw, (text_x, ym), f"{h:.1f}", F["mini"], COL["text"], text_anchor)
        z += h


def draw_structures(draw, df, box):
    x0, y0, x1, y1 = box
    header(draw, x0, y0 - 135, "b", "Delivered five-layer structures")
    row_base = {"flexible": y0 + 860, "semi_rigid": y0 + 1960}
    row_label_y = {"flexible": y0 + 80, "semi_rigid": y0 + 1180}
    labels = {"flexible": "Flexible", "semi_rigid": "Semi-rigid"}
    start = x0 + 300
    gap = (x1 - start - 280) / 5
    bar_w, dep, zscale = 255, 78, 9.2
    for ptype in ["flexible", "semi_rigid"]:
        part = df[df["type"].eq(ptype)].reset_index(drop=True)
        draw.line([(x0 + 20, row_base[ptype] + 2), (x1 - 25, row_base[ptype] + 2)], fill="#d0d0d0", width=2)
        for i, r in part.iterrows():
            xx = start + i * gap
            z = 0
            draw_order = [
                ("h5", LAYERS[4][1]),
                ("h4", LAYERS[3][1]),
                ("h3", LAYERS[2][1]),
                ("h2", LAYERS[1][1]),
                ("h1", LAYERS[0][1]),
            ]
            for key, c in draw_order:
                h = float(r[key])
                cuboid(draw, xx, row_base[ptype], bar_w, dep, z, h, zscale, c)
                z += h
            cx = xx + bar_w / 2 + dep / 2
            t(draw, (cx, row_base[ptype] + 72), r["study_id"], F["small"], anchor="mm")
            t(draw, (cx, row_base[ptype] + 130), f"h={float(r['total_thickness_cm']):.1f} cm", F["tiny"], COL["muted"], "mm")
        label_x = start + 2.5 * gap + bar_w / 2 + dep / 2
        t(draw, (label_x, row_label_y[ptype]), labels[ptype], F["h2"], COL[ptype], "mm")
    item_w = 650
    total_legend_w = len(LAYERS) * item_w
    lx, ly = (x0 + x1 - total_legend_w) / 2, y1 - 95
    for i, (name, c) in enumerate(LAYERS):
        item_x = lx + i * item_w
        draw.rectangle([item_x, ly, item_x + 50, ly + 30], fill=c)
        t(draw, (item_x + 64, ly - 4), name, F["tiny"])


def norm(v, lo, hi):
    return 1 if hi == lo else max(0, min(1, (float(v) - lo) / (hi - lo)))


def draw_bar_cell(draw, x0, x1, y, val, lo, hi, color, label):
    frac = norm(val, lo, hi)
    x_bar = x0 + (x1 - x0) * frac
    draw.rectangle([x0, y - 25, x1, y + 25], fill="#eeeeee")
    draw.rectangle([x0, y - 25, x_bar, y + 25], fill=color)
    label_font = F["table_bold" if label == "1.000" else "table"]
    if frac >= 0.58:
        text_fill = "white"
    else:
        text_fill = "#202020"
    t(draw, ((x0 + x1) / 2, y), label, label_font, text_fill, "mm")


def draw_matrix(draw, df, box):
    x0, y0, x1, y1 = box
    header(draw, x0, y0 - 125, "c", "LTPP delivered-design metrics")
    metrics = [
        ("E_sub", "Esub\n(MPa)", None),
        ("total_thickness_cm", "Total h\n(cm)", None),
        ("DSR", "DSR", "#59A14F"),
        ("SCR", "Episode\nSCR", None),
        ("cost_USD", "Cost\n(USD)", None),
        ("LCC_USD", "20-year LCC\n(USD)", None),
    ]
    cols = [x0 + 350, x0 + 1145, x0 + 1940, x0 + 2735, x0 + 3530, x0 + 4325]
    bar_w = 690
    yhead = y0 + 74
    t(draw, (x0 + 18, yhead - 24), "Section", F["table_bold"])
    for col, (_, label, _) in zip(cols, metrics):
        for k, line in enumerate(label.split("\n")):
            t(draw, (col + bar_w / 2, yhead - 56 + k * 54), line, F["table_bold"], anchor="mm")
    draw.line([(x0, yhead + 55), (x1, yhead + 55)], fill="#222222", width=3)
    row_h = 83
    for i, r in df.iterrows():
        yy = yhead + 110 + i * row_h
        bg = "#f7fbff" if r["type"] == "flexible" else "#fff8f2"
        draw.rectangle([x0, yy - 37, x1, yy + 37], fill=bg)
        if i == 6:
            draw.line([(x0, yy - 47), (x1, yy - 47)], fill="#777777", width=3)
        c = COL[r["type"]]
        t(draw, (x0 + 18, yy - 25), r["study_id"], F["table_bold"])
        for col, (m, _, mc) in zip(cols, metrics):
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
            draw_bar_cell(draw, col, col + bar_w, yy, val, lo, hi, c, lab)
    t(draw, (x0, y1 - 16), "Bars are min-max scaled within each metric; costs are USD m⁻².", F["micro"], COL["muted"])


def draw_ncat(img, draw, real, heat, box):
    x0, y0, x1, y1 = box
    header(draw, x0, y0 - 78, "d", "NCAT fixed-material boundary response")
    px0, py0, px1, py1 = x0 + 230, y0 + 150, x1 - 230, y1 - 150
    draw.line([(px0, py1), (px1, py1)], fill="#222222", width=4)
    draw.line([(px0, py0), (px0, py1)], fill="#222222", width=4)
    draw.line([(px1, py0), (px1, py1)], fill="#222222", width=4)
    for tick in [0, 0.2, 0.4, 0.6]:
        yy = py1 - (py1 - py0) * tick / 0.6
        draw.line([(px0, yy), (px1, yy)], fill="#e8e8e8", width=2)
        t(draw, (px0 - 24, yy), f"{tick:.1f}", F["axis_big"], COL["muted"], "ra")
    for tick in [0, 20, 40, 60]:
        yy = py1 - (py1 - py0) * tick / 60.0
        t(draw, (px1 + 28, yy), str(tick), F["axis_big"], COL["muted"], "lm")
    rotated_text(img, (px0 - 170, (py0 + py1) / 2), "DSR", F["axis_big"], COL["muted"], 90)
    rotated_text(img, (px1 + 210, (py0 + py1) / 2), "Rutting reduction (%)", F["axis_big"], COL["muted"], 270)

    sections = list(real["short"])
    n = len(sections)
    step = (px1 - px0) / n
    bar_w = step * 0.42
    as_pts, alt_pts = [], []
    for i, r in real.iterrows():
        cx = px0 + step * (i + 0.5)
        reduction = -float(r["pred_rut_change_pct"])
        bar_h = (py1 - py0) * reduction / 60.0
        draw.rectangle([cx - bar_w / 2, py1 - bar_h, cx + bar_w / 2, py1], fill="#4C5C92")
        t(draw, (cx, py1 - bar_h - 58), f"{reduction:.0f}%", F["axis"], "#252B48", "mm")
        y_as = py1 - (py1 - py0) * float(r["asbuilt_dsr"]) / 0.6
        y_alt = py1 - (py1 - py0) * float(r["alternative_dsr"]) / 0.6
        as_pts.append((cx, y_as))
        alt_pts.append((cx, y_alt))
        t(draw, (cx, py1 + 52), sections[i], F["axis_big"], anchor="mm")
    draw.line(as_pts, fill=COL["semi_rigid"], width=8)
    draw.line(alt_pts, fill="#0072B2", width=9)
    for x, y in as_pts:
        draw.ellipse([x - 15, y - 15, x + 15, y + 15], fill=COL["semi_rigid"], outline="white", width=3)
    for x, y in alt_pts:
        draw.ellipse([x - 17, y - 17, x + 17, y + 17], fill="#0072B2", outline="white", width=3)

    lx, ly = px0 + 80, py0 + 46
    draw.rectangle([lx, ly - 20, lx + 48, ly + 18], fill="#4C5C92")
    t(draw, (lx + 64, ly - 24), "Rutting-related response reduction", F["legend"])
    draw.line([(lx + 1080, ly), (lx + 1170, ly)], fill=COL["semi_rigid"], width=8)
    draw.ellipse([lx + 1125 - 15, ly - 15, lx + 1125 + 15, ly + 15], fill=COL["semi_rigid"])
    t(draw, (lx + 1190, ly - 24), "As-built DSR", F["legend"])
    draw.line([(lx + 1490, ly), (lx + 1580, ly)], fill="#0072B2", width=9)
    draw.ellipse([lx + 1535 - 17, ly - 17, lx + 1535 + 17, ly + 17], fill="#0072B2")
    t(draw, (lx + 1605, ly - 24), "Candidate DSR", F["legend"])


def main():
    if not STATE_GEOJSON.exists():
        raise FileNotFoundError(f"Missing state boundary file: {STATE_GEOJSON}")
    df = load_ltp_data()
    df.to_csv(CSV_OUT, index=False)
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    draw_map_and_lookup(draw, df, (230, 300, 5170, 1800))
    draw_structures(draw, df, (230, 2040, 5170, 4520))
    draw_matrix(draw, df, (230, 4780, 5170, 5930))
    img.save(PNG_OUT, dpi=(450, 450))
    img.save(PDF_OUT, resolution=450.0)
    print(f"Saved {PNG_OUT}")
    print(f"Saved {PDF_OUT}")
    print(f"Saved {CSV_OUT}")


if __name__ == "__main__":
    main()
