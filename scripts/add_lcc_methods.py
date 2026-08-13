"""Add LCC (Life-Cycle Cost) subsection to Methods, after M1 and before M2."""
from docx import Document
from docx.shared import Pt, RGBColor
from docx.oxml.ns import qn
from docx.oxml import OxmlElement as OE

BLUE = RGBColor(0x00, 0x55, 0xFF)

doc = Document(r'C:/Users/Ivy/Desktop/NC一审意见回复/文本/manuscript/iLLM-PD_重写_0608.docx')

body = doc.element.body

# Find the M2 paragraph in body
body_idx = None
for i, child in enumerate(body):
    if child.tag == qn('w:p'):
        texts = child.findall('.//' + qn('w:t'))
        combined = ''.join(t.text or '' for t in texts)
        if combined.strip().startswith('M2. Input parsing'):
            body_idx = i
            break

if body_idx is None:
    raise SystemExit("M2 not found")

# Word namespaces
MATH_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/math'

def make_math_paragraph(formula_text, font_size=10):
    """
    Create a paragraph containing inline math as OMML.
    Inserts a proper Office Math equation element.
    """
    p_el = OE('w:p')
    pPr = OE('w:pPr')
    jc = OE('w:jc'); jc.set(qn('w:val'), 'center')
    pPr.append(jc)
    p_el.append(pPr)

    # OMML equation using m: namespace
    m_para = OE('m:oMathPara')

    m_oMath = OE('m:oMath')
    m_r = OE('m:r')
    m_rPr = OE('m:rPr')
    m_nor = OE('m:nor')
    m_rPr.append(m_nor)
    m_r.append(m_rPr)
    m_t = OE('m:t')
    m_t.text = formula_text
    m_r.append(m_t)
    m_oMath.append(m_r)
    m_para.append(m_oMath)

    p_el.append(m_para)

    return p_el

def make_text_paragraph(text, font_size=10, bold=False, color=None, italic=False, indent_left=0):
    """Create a simple text paragraph."""
    p_el = OE('w:p')
    pPr = OE('w:pPr')
    if indent_left:
        ind = OE('w:ind'); ind.set(qn('w:left'), str(indent_left))
        pPr.append(ind)

    r_el = OE('w:r')
    rPr = OE('w:rPr')
    sz = OE('w:sz'); sz.set(qn('w:val'), str(int(font_size * 2)))
    rPr.append(sz)
    if bold:
        b_el = OE('w:b'); rPr.append(b_el)
    if color:
        c_el = OE('w:color'); c_el.set(qn('w:val'), color); rPr.append(c_el)
    if italic:
        i_el = OE('w:i'); rPr.append(i_el)
    r_el.append(rPr)

    t_el = OE('w:t')
    t_el.set(qn('xml:space'), 'preserve')
    t_el.text = text
    r_el.append(t_el)
    p_el.append(r_el)

    return p_el

# Insert LCC subsection before M2
# 1. Blank line
body.insert(body_idx, make_text_paragraph('', 6))

# 2. Subsection header
body.insert(body_idx, make_text_paragraph(
    'Economic evaluation (life-cycle cost).',
    10, bold=True, color='0055FF'))

# 3. Construction cost + formula
body.insert(body_idx, make_text_paragraph(
    'Construction cost was computed from the five layer thicknesses and material unit prices '
    '(Supplementary Table S1.3). The 20-year life-cycle cost (LCC) was reported as net present '
    'value (NPV) following the FHWA LCCA framework (FHWA-IF-02-047). The formula is:',
    10, color='0055FF'))

# 4. Formula paragraph: C_LCC = C₀ + Σ C_maint,y / (1+r)^y
body.insert(body_idx, make_math_paragraph(
    'C_LCC = C₀ + Σ_{y=1}^{20} C_maint,y / (1 + r)^y', 10))

# 5. Formula explanation
body.insert(body_idx, make_text_paragraph(
    'where C₀ is the initial construction cost (USD m⁻²), r = 0.04 is the real discount rate '
    '(4%, FHWA recommendation; OMB Circular A-94), and C_maint,y denotes the maintenance cost '
    'incurred in year y. Maintenance events were scheduled from the AC fatigue margin (B1) and '
    'semi-rigid base fatigue margin (B2) following the AASHTO ME-PDG (2020) recommended '
    'intervals and NCHRP Synthesis 495 treatment selection guidance: routine preventive '
    'maintenance every 5 years with slurry seal at 15-year intervals; thin AC overlay at 5-year '
    'intervals if B1 < 1.0; structural overlay at progressively longer intervals for '
    'B1 ∈ [1.0, 2.0); and one preventive overlay at year 20 for B1 ≥ 2.0. For semi-rigid '
    'pavements with B2 < 1.5, additional base rehabilitation (mill-inlay or full-depth '
    'reclamation) was scheduled. Unit costs for each treatment were extracted from FHWA '
    'HIF-10-020 and US municipal DOT data (2023–2024), ranging from $1.20 m⁻² (crack seal) '
    'to $29.90 m⁻² (deep mill-inlay), and are tabulated in full in Supplementary Table S1.3.',
    10, color='0055FF', italic=True))

# 6. Usage note
body.insert(body_idx, make_text_paragraph(
    'The LCC is used to rank compliant candidate designs for reporting purposes only; '
    'structural compliance is determined solely by the finite-element responses and the '
    'JTG D50-2017 criteria (M5). The per-step reward (M3) includes a construction-cost term '
    'but not the full life-cycle NPV, which is computed off the critical path at episode end.',
    10, color='0055FF'))

# 7. Blank line
body.insert(body_idx, make_text_paragraph('', 6))

doc.save(r'C:/Users/Ivy/Desktop/NC一审意见回复/文本/manuscript/iLLM-PD_重写_0608.docx')
print('LCC subsection inserted.')