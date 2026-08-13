"""Add S1.3 material prices + guard bounds table and S9 statistics to supplementary."""
from docx import Document
from docx.shared import RGBColor
from docx.oxml.ns import qn
from docx.oxml import OxmlElement as OE

BLUE = RGBColor(0x00, 0x55, 0xFF)

supp = Document(r'C:/Users/Ivy/Desktop/NC一审意见回复/文本/supplementary_material/iLLM_PD_Supplementary_Material_0609.docx')

body = supp.element.body

# Find S1.2 paragraph and table (Table S1.2 = index 2, the reward weights table)
# S1 section: paragraph "S1." then paragraphs about S1.1, table, S1.2, table.
# Find the paragraph after S1.2 content
s2_idx = None
for i, child in enumerate(body):
    if child.tag == qn('w:p'):
        texts = child.findall('.//' + qn('w:t'))
        combined = ''.join(t.text or '' for t in texts)
        if combined.strip().startswith('S2. Finite-element'):
            s2_idx = i
            break

if s2_idx is None:
    raise SystemExit("S2 not found")

def add_para(text, sz=10, bold=False, color='0055FF', insert_at=None):
    p_el = OE('w:p')
    if text:
        r_el = OE('w:r')
        rPr = OE('w:rPr')
        size = OE('w:sz'); size.set(qn('w:val'), str(int(sz * 2)))
        rPr.append(size)
        if bold: b = OE('w:b'); rPr.append(b)
        c = OE('w:color'); c.set(qn('w:val'), color); rPr.append(c)
        r_el.append(rPr)
        t_el = OE('w:t'); t_el.set(qn('xml:space'), 'preserve'); t_el.text = text
        r_el.append(t_el)
        p_el.append(r_el)
    if insert_at is not None:
        body.insert(insert_at, p_el)
    return p_el

# Insert S1.3 content before S2
insert_pos = s2_idx

# 1. S1.3 header
add_para('', 6, insert_at=insert_pos)
add_para('S1.3 Material unit prices and NumericalGuard bounds.', 10, bold=True, insert_at=insert_pos)

# 2. Price table caption
add_para('Table S1.3a | Material unit prices and construction cost parameters. Prices in CNY m⁻³ (2024 reference); 1 USD ≈ 7.20 CNY. Flexible base/subbase use unbound granular materials (graded crushed stone, sand-gravel); semi-rigid base/subbase use cement-stabilised aggregate and lime-flyash. Modulus-dependent effective price = base price × (1 + coeff × E_default).', 9, insert_at=insert_pos)

# Build price table
price_data = [
    ['Layer', 'Base price\n(CNY/m³)', 'Modulus\ncoeff (1/MPa)', 'Default E\n(MPa)', 'Effective price\n(CNY/m³)',
     'Base price\n(CNY/m³)', 'Modulus\ncoeff (1/MPa)', 'Default E\n(MPa)', 'Effective price\n(CNY/m³)'],
    ['', '', '', '', '', '', '', '', ''],
    ['Upper AC', '1800', '2.0×10⁻⁵', '14000', '2304', '1800', '2.0×10⁻⁵', '14000', '2304'],
    ['Mid AC', '1100', '1.8×10⁻⁵', '11000', '1318', '1100', '1.8×10⁻⁵', '11000', '1318'],
    ['Lower AC', '900', '1.5×10⁻⁵', '9000', '1022', '900', '1.5×10⁻⁵', '9000', '1022'],
    ['Base', '320', '6.0×10⁻⁵', '1500', '349', '100', '1.0×10⁻⁵', '350', '100'],
    ['Subbase', '180', '0', '400', '180', '80', '0', '250', '80'],
]
# Row 0 is the "Semi-rigid" / "Flexible" group header
price_data[1][0] = 'Semi-rigid pavement'
price_data[1][4] = 'Flexible pavement'

col_w = [1000, 1100, 1100, 900, 1100, 1100, 1100, 900, 1100]

tbl = OE('w:tbl')
tblPr = OE('w:tblPr')
tblW = OE('w:tblW'); tblW.set(qn('w:w'), '9360'); tblW.set(qn('w:type'), 'dxa')
tblPr.append(tblW)
tblB = OE('w:tblBorders')
for bn in ['top','left','bottom','right','insideH','insideV']:
    b = OE('w:'+bn); b.set(qn('w:val'),'single'); b.set(qn('w:sz'),'4'); b.set(qn('w:space'),'0'); b.set(qn('w:color'),'AAAAAA')
    tblB.append(b)
tblPr.append(tblB)
tbl.append(tblPr)
tblGrid = OE('w:tblGrid')
for w in col_w:
    gc = OE('w:gridCol'); gc.set(qn('w:w'), str(w)); tblGrid.append(gc)
tbl.append(tblGrid)

for ri, rd in enumerate(price_data):
    tr = OE('w:tr')
    for ci, ct in enumerate(rd):
        tc = OE('w:tc')
        tcPr = OE('w:tcPr')
        tcW = OE('w:tcW'); tcW.set(qn('w:w'), str(col_w[ci])); tcW.set(qn('w:type'), 'dxa')
        tcPr.append(tcW)
        if ri == 0:
            shd = OE('w:shd'); shd.set(qn('w:val'),'clear'); shd.set(qn('w:fill'),'E8E8E8')
            tcPr.append(shd)
        if ri == 1:  # group header row
            shd = OE('w:shd'); shd.set(qn('w:val'),'clear'); shd.set(qn('w:fill'),'F0F4F8')
            tcPr.append(shd)
        tc.append(tcPr)

        p_el = OE('w:p')
        r_el = OE('w:r')
        rPr = OE('w:rPr')
        sz = OE('w:sz'); sz.set(qn('w:val'), '14' if ri > 0 else '14')  # 7pt
        rPr.append(sz)
        if ri == 0:
            b_el = OE('w:b'); rPr.append(b_el)
        r_el.append(rPr)
        t_el = OE('w:t'); t_el.set(qn('xml:space'),'preserve'); t_el.text = ct
        r_el.append(t_el)
        p_el.append(r_el)
        tc.append(p_el)
        tr.append(tc)
    tbl.append(tr)

body.insert(insert_pos, tbl)

# 3. Guard bounds caption + table
add_para('', 6, insert_at=insert_pos)
add_para('Table S1.3b | NumericalGuard hard bounds by layer and pavement type. Source: rl/guards.py. The Guard rejects any action that would move any parameter outside these ranges before finite-element evaluation.', 9, insert_at=insert_pos)

guard_data = [
    ['Layer', 'h_min (m)', 'h_max (m)', 'E_min (MPa)', 'E_max (MPa)',
     'h_min (m)', 'h_max (m)', 'E_min (MPa)', 'E_max (MPa)'],
    ['', '', '', '', '', '', '', '', ''],
    ['Upper AC', '0.02', '0.10', '4000', '25000', '0.02', '0.10', '4000', '25000'],
    ['Mid AC', '0.03', '0.15', '3000', '18000', '0.03', '0.15', '3000', '18000'],
    ['Lower AC', '0.04', '0.25', '2000', '15000', '0.04', '0.25', '2000', '15000'],
    ['Base', '0.15', '0.50', '800', '3500', '0.15', '0.50', '150', '500'],
    ['Subbase', '0.10', '0.40', '200', '800', '0.10', '0.45', '100', '400'],
]
guard_data[1][0] = 'Semi-rigid'
guard_data[1][4] = 'Flexible'

tbl2 = OE('w:tbl')
tblPr2 = OE('w:tblPr')
tblW2 = OE('w:tblW'); tblW2.set(qn('w:w'), '9360'); tblW2.set(qn('w:type'), 'dxa')
tblPr2.append(tblW2)
tblB2 = OE('w:tblBorders')
for bn in ['top','left','bottom','right','insideH','insideV']:
    b = OE('w:'+bn); b.set(qn('w:val'),'single'); b.set(qn('w:sz'),'4'); b.set(qn('w:space'),'0'); b.set(qn('w:color'),'AAAAAA')
    tblB2.append(b)
tblPr2.append(tblB2)
tbl2.append(tblPr2)
tblGrid2 = OE('w:tblGrid')
for w in col_w:
    gc = OE('w:gridCol'); gc.set(qn('w:w'), str(w)); tblGrid2.append(gc)
tbl2.append(tblGrid2)

for ri, rd in enumerate(guard_data):
    tr = OE('w:tr')
    for ci, ct in enumerate(rd):
        tc = OE('w:tc')
        tcPr = OE('w:tcPr')
        tcW = OE('w:tcW'); tcW.set(qn('w:w'), str(col_w[ci])); tcW.set(qn('w:type'), 'dxa')
        tcPr.append(tcW)
        if ri == 0:
            shd = OE('w:shd'); shd.set(qn('w:val'),'clear'); shd.set(qn('w:fill'),'E8E8E8'); tcPr.append(shd)
        if ri == 1:
            shd = OE('w:shd'); shd.set(qn('w:val'),'clear'); shd.set(qn('w:fill'),'F0F4F8'); tcPr.append(shd)
        tc.append(tcPr)
        p_el = OE('w:p')
        r_el = OE('w:r')
        rPr = OE('w:rPr')
        sz = OE('w:sz'); sz.set(qn('w:val'), '14')
        rPr.append(sz)
        if ri == 0: b_el = OE('w:b'); rPr.append(b_el)
        r_el.append(rPr)
        t_el = OE('w:t'); t_el.set(qn('xml:space'),'preserve'); t_el.text = ct
        r_el.append(t_el)
        p_el.append(r_el)
        tc.append(p_el)
        tr.append(tc)
    tbl2.append(tr)

body.insert(insert_pos, tbl2)

# 4. Maintenance costs
add_para('', 6, insert_at=insert_pos)
add_para('Table S1.3c | Maintenance treatment unit costs (USD m⁻², 2023–2024). Sources: FHWA HIF-10-020; NCHRP Synthesis 495.', 9, insert_at=insert_pos)

maint_data = [
    ['Treatment', 'Cost (USD/m²)'],
    ['Crack seal', '1.20'],
    ['Chip seal', '1.80'],
    ['Slurry seal', '3.60'],
    ['Thin overlay (1.5 in)', '13.20'],
    ['Structural overlay (2 in)', '21.50'],
    ['Mill-inlay (4 in deep)', '29.90'],
    ['Full-depth reclamation', '41.90'],
    ['Full reconstruction', '65.80'],
    ['Routine minor maintenance', '2.00'],
]

tbl3 = OE('w:tbl')
tblPr3 = OE('w:tblPr')
tblW3 = OE('w:tblW'); tblW3.set(qn('w:w'), '5000'); tblW3.set(qn('w:type'), 'dxa')
tblPr3.append(tblW3)
tblB3 = OE('w:tblBorders')
for bn in ['top','left','bottom','right','insideH','insideV']:
    b = OE('w:'+bn); b.set(qn('w:val'),'single'); b.set(qn('w:sz'),'4'); b.set(qn('w:space'),'0'); b.set(qn('w:color'),'AAAAAA')
    tblB3.append(b)
tblPr3.append(tblB3)
tbl3.append(tblPr3)
tblGrid3 = OE('w:tblGrid')
gc1 = OE('w:gridCol'); gc1.set(qn('w:w'), '3500'); tblGrid3.append(gc1)
gc2 = OE('w:gridCol'); gc2.set(qn('w:w'), '1500'); tblGrid3.append(gc2)
tbl3.append(tblGrid3)

for ri, rd in enumerate(maint_data):
    tr = OE('w:tr')
    for ci, ct in enumerate(rd):
        tc = OE('w:tc')
        tcPr = OE('w:tcPr')
        tcW = OE('w:tcW'); tcW.set(qn('w:w'), '3500' if ci==0 else '1500'); tcW.set(qn('w:type'), 'dxa')
        tcPr.append(tcW)
        if ri == 0:
            shd = OE('w:shd'); shd.set(qn('w:val'),'clear'); shd.set(qn('w:fill'),'E8E8E8'); tcPr.append(shd)
        tc.append(tcPr)
        p_el = OE('w:p')
        r_el = OE('w:r')
        rPr = OE('w:rPr')
        sz = OE('w:sz'); sz.set(qn('w:val'), '14')
        rPr.append(sz)
        if ri == 0: b_el = OE('w:b'); rPr.append(b_el)
        r_el.append(rPr)
        t_el = OE('w:t'); t_el.set(qn('xml:space'),'preserve'); t_el.text = ct
        r_el.append(t_el)
        p_el.append(r_el)
        tc.append(p_el)
        tr.append(tc)
    tbl3.append(tr)

body.insert(insert_pos, tbl3)
add_para('', 6, insert_at=insert_pos)

# 5. Update S9 statistics
for p in supp.paragraphs:
    if 'Analysis covered n =' in p.text and '1,942' in p.text:
        for run in p.runs:
            if 'n = 1,942' in run.text:
                run.text = run.text.replace(
                    'n = 1,942 scored actions across 15 training runs.',
                    'n = 15,191 scored actions across 77 training runs (all audit chain records). '
                    'Spearman ρ = −0.028 (P = 0.0006; 95% CI [−0.044, −0.012]). '
                    'Non-compliance rate is 25–29% flat across all score bands (0–3, 3–5, 5–7, 7–10) '
                    '— no score band predicts compliance better than any other. '
                    'The test set spans both compliant and non-compliant designs (28.6% overall non-compliance).')
                run.font.color.rgb = BLUE
        print('Updated S9 statistics')
        break

supp.save(r'C:/Users/Ivy/Desktop/NC一审意见回复/文本/supplementary_material/iLLM_PD_Supplementary_Material_0609.docx')
print('Supplementary saved with S1.3 + S9 stats.')