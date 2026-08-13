from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd

import plot_fig3_design_response_revised_pillow_largefont as src


OUT = src.OUT_DIR / "Fig3_design_response_revised_editable.svg"
W, H = 5400, 8200


def e(s):
    return escape(str(s), quote=True)


class SVG:
    def __init__(self, w, h):
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<style><![CDATA[text{font-family:Arial,Helvetica,sans-serif;font-weight:700;fill:#202020;} .muted{fill:#666666;}]]></style>',
        ]

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=54, fill="#202020", anchor="start", weight=700, rotate=None):
        transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}" dominant-baseline="middle"{transform}>{e(s)}</text>'
        )

    def rect(self, x, y, w, h, fill, stroke="none", sw=1, rx=0):
        self.add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def line(self, x1, y1, x2, y2, stroke="#202020", sw=2):
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}"/>')

    def polygon(self, pts, fill, stroke="none", sw=1):
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.add(f'<polygon points="{p}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def circle(self, x, y, r, fill, stroke="none", sw=1):
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def path(self, d, fill="none", stroke="#202020", sw=2):
        self.add(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def save(self, path):
        self.parts.append("</svg>")
        path.write_text("\n".join(self.parts), encoding="utf-8")


def header(svg, x, y, panel, title):
    svg.text(x, y, panel, 98, "#202020", "start")
    svg.text(x + 58, y, title, 74, "#202020", "start")


def shade(hex_color, factor):
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


def norm(v, lo, hi):
    return 1 if hi == lo else max(0, min(1, (float(v) - lo) / (hi - lo)))


def draw_map(svg, df, box):
    x0, y0, x1, y1 = box
    header(svg, x0, y0 - 80, "a", "Field-section locations and LTPP lookup")
    map_box = (x0, y0, x0 + 3500, y1)
    tab_box = (x0 + 3560, y0, x1, y1)
    svg.rect(*map_box[:2], map_box[2] - map_box[0], map_box[3] - map_box[1], src.COL["water"], "#d7dee2", 2, 18)
    tr = src.map_transform((map_box[0] + 20, map_box[1] + 20, map_box[2] - 20, map_box[3] - 132))
    climate_by_state = {}
    for _, r in df.iterrows():
        climate_by_state.setdefault(r["state"], r["climate"])
    state_name_to_abbr = {"Arizona": "AZ", "California": "CA", "Florida": "FL", "Idaho": "ID", "Minnesota": "MN", "Montana": "MT", "Texas": "TX", "Alabama": "AL"}
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
    nx, ny = tr(-85.49, 32.60)
    star = [(nx, ny - 28), (nx + 9, ny - 8), (nx + 30, ny - 8), (nx + 14, ny + 5), (nx + 20, ny + 26), (nx, ny + 14), (nx - 20, ny + 26), (nx - 14, ny + 5), (nx - 30, ny - 8), (nx - 9, ny - 8)]
    svg.polygon(star, "#111111", "white", 1)
    svg.text(nx + 38, ny - 8, "NCAT", 47)
    lx, ly = map_box[0] + 58, map_box[3] - 125
    for i, (name, c) in enumerate(src.CLIMATE_COLORS.items()):
        svg.rect(lx + i * 430, ly, 44, 32, c, "#555555", 1)
        svg.text(lx + i * 430 + 58, ly + 15, name.replace("NoFreeze", "No-freeze"), 40)
    tx = map_box[2] - 760
    svg.circle(tx + 21, ly + 83, 21, "white", src.COL["flexible"], 6)
    svg.text(tx + 58, ly + 83, "Flexible", 40)
    svg.rect(tx + 320, ly + 62, 42, 42, "white", src.COL["semi_rigid"], 6)
    svg.text(tx + 378, ly + 83, "Semi-rigid", 40)

    svg.rect(tab_box[0] - 24, tab_box[1] - 4, tab_box[2] - tab_box[0] + 48, tab_box[3] - tab_box[1] + 8, "white")
    svg.text((tab_box[0] + tab_box[2]) / 2, tab_box[1] + 34, "LTPP validation sections", 54, anchor="middle")
    cx = [tab_box[0], tab_box[0] + 290, tab_box[0] + 610, tab_box[0] + 970, tab_box[0] + 1385, tab_box[2]]
    heads = ["Study ID", "LTPP ID", "Family", "Climate", "Esub\n(MPa)"]
    yhead = tab_box[1] + 132
    table_bottom = tab_box[3] - 14
    svg.line(tab_box[0], yhead - 58, tab_box[2], yhead - 58, "#222222", 4)
    svg.line(tab_box[0], yhead + 64, tab_box[2], yhead + 64, "#222222", 4)
    for i, h in enumerate(heads):
        for k, line in enumerate(h.split("\n")):
            svg.text((cx[i] + cx[i + 1]) / 2, yhead - 24 + k * 46, line, 54, anchor="middle")
    row_h = (table_bottom - (yhead + 84)) / len(df)
    climate_short = {"Dry-Freeze": "Dry-F", "Dry-NoFreeze": "Dry-NF", "Wet-Freeze": "Wet-F", "Wet-NoFreeze": "Wet-NF"}
    for i, r in df.iterrows():
        yy = yhead + 84 + i * row_h + row_h / 2
        vals = [r["study_id"], r["section"], "Flexible" if r["type"] == "flexible" else "Semi-rigid", climate_short.get(r["climate"], r["climate"]), f"{int(round(r['E_sub']))}"]
        for j, val in enumerate(vals):
            svg.text((cx[j] + cx[j + 1]) / 2, yy, val, 54, anchor="middle")
    svg.line(tab_box[0], table_bottom, tab_box[2], table_bottom, "#222222", 3)


def cuboid(svg, x, base, w, dep, z0, dz, zscale, color):
    yb = base - z0 * zscale
    yt = base - (z0 + dz) * zscale
    dx, dy = dep, -int(dep * 0.55)
    svg.polygon([(x + w, yt), (x + w + dx, yt + dy), (x + w + dx, yb + dy), (x + w, yb)], shade(color, 0.72), shade(color, 0.78), 1)
    svg.polygon([(x, yt), (x + w, yt), (x + w, yb), (x, yb)], color, shade(color, 0.78), 1)
    svg.polygon([(x, yt), (x + w, yt), (x + w + dx, yt + dy), (x + dx, yt + dy)], shade(color, 1.25), shade(color, 0.78), 1)


def draw_structures(svg, df, box):
    x0, y0, x1, y1 = box
    header(svg, x0, y0 - 135, "b", "Delivered five-layer structures")
    row_base = {"semi_rigid": y0 + 860, "flexible": y0 + 1960}
    row_label_y = {"semi_rigid": y0 + 80, "flexible": y0 + 1180}
    labels = {"flexible": "Flexible", "semi_rigid": "Semi-rigid"}
    start = x0 + 300
    gap = (x1 - start - 280) / 5
    bar_w, dep, zscale = 255, 78, 9.2
    for ptype in ["semi_rigid", "flexible"]:
        part = df[df["type"].eq(ptype)].reset_index(drop=True)
        svg.line(x0 + 20, row_base[ptype] + 2, x1 - 25, row_base[ptype] + 2, "#d0d0d0", 2)
        for i, r in part.iterrows():
            xx = start + i * gap
            z = 0
            for j, (_, c) in enumerate(src.LAYERS):
                h = float(r[f"h{j + 1}"])
                cuboid(svg, xx, row_base[ptype], bar_w, dep, z, h, zscale, c)
                z += h
            cx = xx + bar_w / 2 + dep / 2
            svg.text(cx, row_base[ptype] + 72, r["study_id"], 53, anchor="middle")
            svg.text(cx, row_base[ptype] + 130, f"h={float(r['total_thickness_cm']):.1f} cm", 47, src.COL["muted"], "middle")
        label_x = start + 2.5 * gap + bar_w / 2 + dep / 2
        svg.text(label_x, row_label_y[ptype], labels[ptype], 74, src.COL[ptype], "middle")
    item_w = 650
    total_legend_w = len(src.LAYERS) * item_w
    lx, ly = (x0 + x1 - total_legend_w) / 2, y1 - 95
    for i, (name, c) in enumerate(src.LAYERS):
        item_x = lx + i * item_w
        svg.rect(item_x, ly, 50, 30, c)
        svg.text(item_x + 64, ly + 15, name, 47)


def draw_matrix(svg, df, box):
    x0, y0, x1, y1 = box
    header(svg, x0, y0 - 125, "c", "LTPP delivered-design metrics")
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
            frac = norm(val, lo, hi)
            svg.rect(col, yy - 25, bar_w, 50, "#eeeeee")
            svg.rect(col, yy - 25, bar_w * frac, 50, c)
            fill = "white" if frac >= 0.58 else "#202020"
            svg.text(col + bar_w / 2, yy, lab, 54, fill, "middle")
    svg.text(x0, y1 - 16, "Bars are min-max scaled within each metric; costs are USD m⁻².", 40, src.COL["muted"])


def draw_ncat(svg, real, box):
    x0, y0, x1, y1 = box
    header(svg, x0, y0 - 78, "d", "NCAT fixed-material boundary response")
    px0, py0, px1, py1 = x0 + 230, y0 + 150, x1 - 230, y1 - 150
    svg.line(px0, py1, px1, py1, "#222222", 4)
    svg.line(px0, py0, px0, py1, "#222222", 4)
    svg.line(px1, py0, px1, py1, "#222222", 4)
    for tick in [0, 0.2, 0.4, 0.6]:
        yy = py1 - (py1 - py0) * tick / 0.6
        svg.line(px0, yy, px1, yy, "#e8e8e8", 2)
        svg.text(px0 - 24, yy, f"{tick:.1f}", 60, src.COL["muted"], "end")
    for tick in [0, 20, 40, 60]:
        yy = py1 - (py1 - py0) * tick / 60.0
        svg.text(px1 + 28, yy, str(tick), 60, src.COL["muted"])
    svg.text(px0 - 170, (py0 + py1) / 2, "DSR", 60, src.COL["muted"], "middle", rotate=-90)
    svg.text(px1 + 210, (py0 + py1) / 2, "Rutting reduction (%)", 60, src.COL["muted"], "middle", rotate=90)
    n = len(real)
    step = (px1 - px0) / n
    bar_w = step * 0.42
    as_pts, alt_pts = [], []
    for i, r in real.iterrows():
        cx = px0 + step * (i + 0.5)
        reduction = -float(r["pred_rut_change_pct"])
        bar_h = (py1 - py0) * reduction / 60.0
        svg.rect(cx - bar_w / 2, py1 - bar_h, bar_w, bar_h, "#4C5C92")
        svg.text(cx, py1 - bar_h - 58, f"{reduction:.0f}%", 52, "#252B48", "middle")
        y_as = py1 - (py1 - py0) * float(r["asbuilt_dsr"]) / 0.6
        y_alt = py1 - (py1 - py0) * float(r["alternative_dsr"]) / 0.6
        as_pts.append((cx, y_as))
        alt_pts.append((cx, y_alt))
        svg.text(cx, py1 + 52, r["short"], 60, anchor="middle")
    def polyline(pts, color, sw):
        d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        svg.path(d, "none", color, sw)
    polyline(as_pts, src.COL["semi_rigid"], 8)
    polyline(alt_pts, "#0072B2", 9)
    for x, y in as_pts:
        svg.circle(x, y, 15, src.COL["semi_rigid"], "white", 3)
    for x, y in alt_pts:
        svg.circle(x, y, 17, "#0072B2", "white", 3)
    lx, ly = px0 + 80, py0 + 46
    svg.rect(lx, ly - 20, 48, 38, "#4C5C92")
    svg.text(lx + 64, ly, "Rutting-related response reduction", 50)
    svg.line(lx + 1080, ly, lx + 1170, ly, src.COL["semi_rigid"], 8)
    svg.circle(lx + 1125, ly, 15, src.COL["semi_rigid"])
    svg.text(lx + 1190, ly, "As-built DSR", 50)
    svg.line(lx + 1490, ly, lx + 1580, ly, "#0072B2", 9)
    svg.circle(lx + 1535, ly, 17, "#0072B2")
    svg.text(lx + 1605, ly, "Candidate DSR", 50)


def main():
    df = src.load_ltp_data()
    real, _ = src.load_ncat()
    svg = SVG(W, H)
    draw_map(svg, df, (230, 300, 5170, 1800))
    draw_structures(svg, df, (230, 2040, 5170, 4520))
    draw_matrix(svg, df, (230, 4780, 5170, 5930))
    draw_ncat(svg, real, (230, 6260, 5170, 8060))
    svg.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
