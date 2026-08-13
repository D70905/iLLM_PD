from __future__ import annotations

from base64 import b64encode
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import plot_fig3_design_response_revised_pillow_largefont as src


OUT_DIR = Path(r"C:\Users\Ivy\Documents\iLLM_PD\output\fig3_revised")
CSV = OUT_DIR / "Fig3_LTPP_only_true_delivered_source_data.csv"
PNG = OUT_DIR / "Fig3_LTPP_only_true_delivered_redraw.png"
SVG = OUT_DIR / "Fig3_LTPP_only_true_delivered_redraw_wrapped.svg"
W, H = 5400, 6200

COL = {
    "flexible": "#0072B2",
    "semi_rigid": "#D55E00",
    "text": "#202020",
    "muted": "#666666",
    "line": "#222222",
    "light": "#E7E7E7",
    "water": "#F6FBFF",
    "state_edge": "#AEBAC2",
    "state_fill": "#F4F4F4",
}
CLIMATE_COLORS = src.CLIMATE_COLORS
LAYERS = [
    ("AC-1 upper", "#252B48"),
    ("AC-2 mid", "#4C5C92"),
    ("AC-3 lower", "#8294C4"),
    ("Base", "#A76F2B"),
    ("Subbase", "#D8C39A"),
]


def font(size: int, bold: bool = True):
    for p in [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


F = {
    "panel": font(98),
    "title": font(74),
    "body": font(54),
    "small": font(47),
    "tiny": font(40),
}


def txt(draw, xy, s, f, fill=COL["text"], anchor="la"):
    draw.text(xy, str(s), font=f, fill=fill, anchor=anchor)


def header(draw, x, y, letter, title):
    txt(draw, (x, y), letter, F["panel"])
    txt(draw, (x + 70, y + 9), title, F["title"])


def shade(hex_color: str, factor: float):
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    if factor >= 1:
        r = int(255 - (255 - r) / factor)
        g = int(255 - (255 - g) / factor)
        b = int(255 - (255 - b) / factor)
    else:
        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)
    return f"#{max(0,min(255,r)):02x}{max(0,min(255,g)):02x}{max(0,min(255,b)):02x}"


def cuboid(draw, x, base, w, dep, z0, dz, zscale, color):
    yb = base - z0 * zscale
    yt = base - (z0 + dz) * zscale
    dx, dy = dep, -int(dep * 0.55)
    draw.polygon([(x + w, yt), (x + w + dx, yt + dy), (x + w + dx, yb + dy), (x + w, yb)], fill=shade(color, 0.72), outline=shade(color, 0.78))
    draw.polygon([(x, yt), (x + w, yt), (x + w, yb), (x, yb)], fill=color, outline=shade(color, 0.78))
    draw.polygon([(x, yt), (x + w, yt), (x + w + dx, yt + dy), (x + dx, yt + dy)], fill=shade(color, 1.25), outline=shade(color, 0.78))


def map_panel(draw, df, box):
    x0, y0, x1, y1 = box
    header(draw, x0, y0 - 80, "a", "Field-section locations and LTPP lookup")
    map_box = (x0, y0, x0 + 3500, y1)
    tab_box = (x0 + 3560, y0, x1, y1)
    draw.rounded_rectangle(map_box, radius=18, fill=COL["water"], outline="#D7DEE2", width=2)
    tr = src.map_transform((map_box[0] + 20, map_box[1] + 20, map_box[2] - 20, map_box[3] - 132))
    climate_by_state = {}
    for _, r in df.iterrows():
        climate_by_state.setdefault(r["state"], r["climate"])
    state_name_to_abbr = {"Arizona": "AZ", "California": "CA", "Florida": "FL", "Idaho": "ID", "Minnesota": "MN", "Montana": "MT", "Texas": "TX", "Alabama": "AL"}
    for name, poly in src.state_polys():
        abbr = state_name_to_abbr.get(name)
        fill = CLIMATE_COLORS.get(climate_by_state.get(abbr, ""), COL["state_fill"])
        draw.polygon([tr(lon, lat) for lon, lat in poly], fill=fill, outline=COL["state_edge"])
    for abbr, lon, lat in src.STATE_INFO.values():
        xx, yy = tr(lon, lat)
        txt(draw, (xx, yy), abbr, F["small"], "#444444", "mm")
    for _, r in df.iterrows():
        xx, yy = tr(r["lon"], r["lat"])
        if r["type"] == "flexible":
            draw.ellipse([xx - 19, yy - 19, xx + 19, yy + 19], fill="white", outline=COL["flexible"], width=6)
        else:
            draw.rectangle([xx - 18, yy - 18, xx + 18, yy + 18], fill="white", outline=COL["semi_rigid"], width=6)
        dx, dy = src.LABEL_OFFSETS.get(r["section"], (30, -12))
        lx, ly = xx + dx, yy + dy
        draw.line([xx, yy, lx - 6 if dx > 0 else lx + 112, ly + 11], fill="#777777", width=1)
        txt(draw, (lx, ly), r["study_id"], F["small"])

    lx, ly = map_box[0] + 58, map_box[3] - 125
    for i, (name, c) in enumerate(CLIMATE_COLORS.items()):
        xx = lx + i * 430
        draw.rectangle([xx, ly, xx + 44, ly + 32], fill=c, outline="#555555")
        txt(draw, (xx + 58, ly + 15), name.replace("NoFreeze", "No-freeze"), F["tiny"], anchor="lm")
    tx = map_box[2] - 760
    draw.ellipse([tx, ly + 62, tx + 42, ly + 104], fill="white", outline=COL["flexible"], width=6)
    txt(draw, (tx + 58, ly + 83), "Flexible", F["tiny"], anchor="lm")
    draw.rectangle([tx + 320, ly + 62, tx + 362, ly + 104], fill="white", outline=COL["semi_rigid"], width=6)
    txt(draw, (tx + 378, ly + 83), "Semi-rigid", F["tiny"], anchor="lm")

    # Redrawn table: long rules and wider Esub column.
    draw.rectangle([tab_box[0] - 42, tab_box[1] - 4, tab_box[2] + 42, tab_box[3] + 4], fill="white")
    txt(draw, ((tab_box[0] + tab_box[2]) / 2, tab_box[1] + 34), "LTPP validation sections", F["body"], anchor="mm")
    cx = [tab_box[0] - 14, tab_box[0] + 260, tab_box[0] + 585, tab_box[0] + 930, tab_box[0] + 1290, tab_box[2] + 42]
    heads = ["Study ID", "LTPP ID", "Family", "Climate", "Esub\n(MPa)"]
    yhead = tab_box[1] + 132
    bottom = tab_box[3] - 14
    draw.line([cx[0], yhead - 58, cx[-1], yhead - 58], fill=COL["line"], width=5)
    draw.line([cx[0], yhead + 64, cx[-1], yhead + 64], fill=COL["line"], width=5)
    for i, h in enumerate(heads):
        for k, line in enumerate(h.split("\n")):
            txt(draw, ((cx[i] + cx[i + 1]) / 2, yhead - 24 + k * 46), line, F["body"] if i != 4 else F["small"], anchor="mm")
    row_h = (bottom - (yhead + 84)) / len(df)
    climate_short = {"Dry-Freeze": "Dry-F", "Dry-NoFreeze": "Dry-NF", "Wet-Freeze": "Wet-F", "Wet-NoFreeze": "Wet-NF"}
    for i, r in df.iterrows():
        yy = yhead + 84 + i * row_h + row_h / 2
        vals = [r["study_id"], r["section"], "Flexible" if r["type"] == "flexible" else "Semi-rigid", climate_short.get(r["climate"], r["climate"]), f"{int(round(r['E_sub']))}"]
        for j, val in enumerate(vals):
            txt(draw, ((cx[j] + cx[j + 1]) / 2, yy), val, F["body"] if j != 4 else F["small"], anchor="mm")
    draw.line([cx[0], bottom, cx[-1], bottom], fill=COL["line"], width=4)


def structure_panel(draw, df, box):
    x0, y0, x1, y1 = box
    header(draw, x0, y0 - 135, "b", "Delivered five-layer structures")
    row_base = {"flexible": y0 + 860, "semi_rigid": y0 + 1960}
    row_label_y = {"flexible": y0 + 80, "semi_rigid": y0 + 1180}
    start = x0 + 300
    gap = (x1 - start - 280) / 5
    bar_w, dep, zscale = 255, 78, 9.2
    for ptype in ["flexible", "semi_rigid"]:
        part = df[df["type"].eq(ptype)].reset_index(drop=True)
        draw.line([x0 + 20, row_base[ptype] + 2, x1 - 25, row_base[ptype] + 2], fill="#D0D0D0", width=2)
        for i, r in part.iterrows():
            xx = start + i * gap
            z = 0.0
            for j, (_, c) in enumerate(LAYERS):
                h = float(r[f"h{j + 1}"])
                cuboid(draw, xx, row_base[ptype], bar_w, dep, z, h, zscale, c)
                z += h
            cx = xx + bar_w / 2 + dep / 2
            txt(draw, (cx, row_base[ptype] + 72), r["study_id"], F["small"], anchor="mm")
            txt(draw, (cx, row_base[ptype] + 130), f"h={float(r['total_thickness_cm']):.1f} cm", F["small"], COL["muted"], "mm")
        label_x = start + 2.5 * gap + bar_w / 2 + dep / 2
        txt(draw, (label_x, row_label_y[ptype]), "Flexible" if ptype == "flexible" else "Semi-rigid", F["title"], COL[ptype], "mm")
    item_w = 650
    lx, ly = (x0 + x1 - len(LAYERS) * item_w) / 2, y1 - 95
    for i, (name, c) in enumerate(LAYERS):
        item_x = lx + i * item_w
        draw.rectangle([item_x, ly, item_x + 50, ly + 30], fill=c)
        txt(draw, (item_x + 64, ly + 15), name, F["small"], anchor="lm")


def norm(v, lo, hi):
    return 1 if hi == lo else max(0, min(1, (float(v) - lo) / (hi - lo)))


def matrix_panel(draw, df, box):
    x0, y0, x1, y1 = box
    header(draw, x0, y0 - 125, "c", "LTPP delivered-design metrics")
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
    txt(draw, (x0 + 18, yhead - 24), "Section", F["body"])
    for col, (_, label) in zip(cols, metrics):
        for k, line in enumerate(label.split("\n")):
            txt(draw, (col + bar_w / 2, yhead - 56 + k * 54), line, F["body"], anchor="mm")
    draw.line([x0, yhead + 55, x1, yhead + 55], fill=COL["line"], width=3)
    for i, r in df.iterrows():
        yy = yhead + 110 + i * row_h
        bg = "#F7FBFF" if r["type"] == "flexible" else "#FFF8F2"
        c = COL[r["type"]]
        draw.rectangle([x0, yy - 37, x1, yy + 37], fill=bg)
        if i == 6:
            draw.line([x0, yy - 47, x1, yy - 47], fill="#777777", width=3)
        txt(draw, (x0 + 18, yy), r["study_id"], F["body"], anchor="lm")
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
            frac = norm(val, lo, hi)
            draw.rectangle([col, yy - 25, col + bar_w, yy + 25], fill="#EEEEEE")
            draw.rectangle([col, yy - 25, col + bar_w * frac, yy + 25], fill=c)
            fill = "white" if frac >= 0.38 else COL["text"]
            txt(draw, (col + bar_w / 2, yy), lab, F["body"], fill, "mm")
    txt(draw, (x0, y1 - 16), "Bars are min-max scaled within each metric; costs are USD m^-2.", F["tiny"], COL["muted"])


def write_wrapped_svg():
    img = Image.open(PNG)
    payload = b64encode(PNG.read_bytes()).decode("ascii")
    SVG.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{img.width}" height="{img.height}" viewBox="0 0 {img.width} {img.height}">\n'
        f'<image href="data:image/png;base64,{payload}" width="{img.width}" height="{img.height}"/>\n</svg>\n',
        encoding="utf-8",
    )


def main():
    df = pd.read_csv(CSV)
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    map_panel(draw, df, (230, 300, 5170, 1800))
    structure_panel(draw, df, (230, 2040, 5170, 4520))
    matrix_panel(draw, df, (230, 4780, 5170, 6100))
    img.save(PNG, dpi=(450, 450))
    write_wrapped_svg()
    print(PNG)
    print(SVG)


if __name__ == "__main__":
    main()
