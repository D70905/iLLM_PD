# -*- coding: utf-8 -*-
"""
ABAQUS-side modeling script for fea.runner.run_fea (v0.4, 6-layer)
====================================================================
Runs inside ABAQUS CAE Python. DO NOT run from regular Python.

Reads:  pavement_input.json (from CWD)
Writes: pavement_result.json (to CWD)

UPGRADE v0.4 (Phase 2A-1):
    - 4-layer (AC + base + subbase + subgrade) → 6-layer:
        [0] Upper AC    (SMA-13 fine surface)
        [1] Mid AC      (AC-20 intermediate)
        [2] Lower AC    (AC-25 coarse bottom)
        [3] Base        (cement-stabilized aggregate)
        [4] Subbase     (graded aggregate)
        [5] Subgrade    (fixed semi-infinite)
    - thickness[] / modulus[] / poisson[] length: 3 → 5 (top-down)
    - Mesh: 5 internal partitions → 6 face regions
    - New extractions: vertical compressive stress at mid-depth of each
      AC sublayer (for JTG B.3.1 multi-sublayer permanent deformation)

Output structure (v0.4):
{
  # Backward-compat fields
  "D_FEA_mm":                ...,
  "sigma_FEA_MPa":           ...,
  "epsilon_FEA_microstrain": ...,
  "deflection_basin_mm":     {...},

  # Protocol-aware responses
  "responses": {
    "epsilon_a_microstrain":   ...,   # Lower AC bottom = AC bottom (JTG B.1)
    "sigma_t_MPa":             ...,   # Semi-rigid base bottom    (JTG B.2)
    "epsilon_z_microstrain":   ...,   # Subgrade top              (JTG B.4)
    "p_AC_upper_mid_MPa":      ...,   # Upper AC mid-depth vert. stress (JTG B.3)
    "p_AC_mid_mid_MPa":        ...,   # Mid AC mid-depth vert. stress   (JTG B.3)
    "p_AC_lower_mid_MPa":      ...,   # Lower AC mid-depth vert. stress (JTG B.3)
  },

  # Metadata
  "load":       {...},
  "pavement":   {...},
  ...
}
"""
from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
import json
import os
import subprocess

executeOnCaeStartup()


# 1. Read input ──────────────────────────────────────────────────────
INPUT_FILE = 'pavement_input.json'
if not os.path.exists(INPUT_FILE):
    raise IOError(
        'pavement_input.json not found in CWD ({}). '
        'This script must be launched by fea.runner.run_fea.'.format(os.getcwd()))

with open(INPUT_FILE, 'r') as f:
    inp = json.load(f)

thicknesses    = inp['thickness']      # length 5, top-down
moduli         = inp['modulus']        # length 5, top-down
poissons       = inp['poisson']        # length 5, top-down
E_subgrade     = inp['E_subgrade']
nu_subgrade    = inp['nu_subgrade']
P              = inp['load_pressure']
r_load         = inp['load_radius']
sensor_offsets = inp['sensor_offsets']
num_cpus       = inp.get('num_cpus', 4)

if len(thicknesses) != 5:
    raise ValueError(
        'Expected 5 layer thicknesses (upper_AC, mid_AC, lower_AC, base, '
        'subbase), got {}'.format(len(thicknesses)))

Z_PAVEMENT = sum(thicknesses)
Z_SUBGRADE = 8.0
Z_TOTAL    = Z_PAVEMENT + Z_SUBGRADE
R_DOMAIN   = 8.0

# z-boundaries (BOTTOM-UP): bottom of subgrade, top of subgrade, top of subbase,
# top of base, top of lower AC, top of mid AC, top of upper AC (= ground surface)
# Index:    [0]   [1]            [2]                [3]              [4]               [5]            [6]
z_boundaries = [
    0.0,                                                                                          # bottom
    Z_SUBGRADE,                                                                                   # subgrade top = subbase bot
    Z_SUBGRADE + thicknesses[4],                                                                  # subbase top = base bot
    Z_SUBGRADE + thicknesses[4] + thicknesses[3],                                                 # base top = lower AC bot
    Z_SUBGRADE + thicknesses[4] + thicknesses[3] + thicknesses[2],                                # lower AC top = mid AC bot
    Z_SUBGRADE + thicknesses[4] + thicknesses[3] + thicknesses[2] + thicknesses[1],               # mid AC top = upper AC bot
    Z_TOTAL,                                                                                      # top = ground surface
]

# Layer names BOTTOM-UP (so index matches z_boundaries[i] = bottom of layer i)
layer_names = ['Subgrade', 'Subbase', 'Base', 'LowerAC', 'MidAC', 'UpperAC']
# Material properties BOTTOM-UP (reverse of input top-down order)
E_all  = [E_subgrade,  moduli[4],   moduli[3],   moduli[2],   moduli[1],  moduli[0]]
nu_all = [nu_subgrade, poissons[4], poissons[3], poissons[2], poissons[1], poissons[0]]

print('=' * 70)
print('fea.abaqus_script v0.4 - Axisymmetric Pavement FEA (6-layer)')
for i in range(6):
    print('  {:<10}: z=[{:6.3f},{:6.3f}] m  E={:7.1f} MPa  nu={:.2f}'.format(
        layer_names[i], z_boundaries[i], z_boundaries[i+1], E_all[i], nu_all[i]))
print('=' * 70)


# 2. Build axisymmetric part ─────────────────────────────────────────
model_name = 'iLLM_PD_PaveFEA'
if model_name in mdb.models:
    del mdb.models[model_name]
m = mdb.Model(name=model_name)

s = m.ConstrainedSketch(name='ProfileMain', sheetSize=2.0 * max(R_DOMAIN, Z_TOTAL))
s.ConstructionLine(point1=(0.0, -10.0), point2=(0.0, 10.0))
s.FixedConstraint(entity=s.geometry.findAt((0.0, 0.0)))
s.rectangle(point1=(0.0, 0.0), point2=(R_DOMAIN, Z_TOTAL))

p = m.Part(name='Pavement', dimensionality=AXISYMMETRIC, type=DEFORMABLE_BODY)
p.BaseShell(sketch=s)
del m.sketches['ProfileMain']

the_face = p.faces[0]
t = p.MakeSketchTransform(sketchPlane=the_face, sketchPlaneSide=SIDE1,
                          origin=(0.0, 0.0, 0.0))
sp = m.ConstrainedSketch(name='PartitionSketch',
                         sheetSize=2.0 * max(R_DOMAIN, Z_TOTAL),
                         gridSpacing=0.5, transform=t)
p.projectReferencesOntoSketch(sketch=sp, filter=COPLANAR_EDGES)

# Partition by 5 internal horizontal lines (between 6 layers)
for zb in z_boundaries[1:-1]:    # 5 internal boundaries
    sp.Line(point1=(0.0, zb), point2=(R_DOMAIN, zb))
p.PartitionFaceBySketch(faces=(the_face,), sketch=sp)
del m.sketches['PartitionSketch']
if len(p.faces) != 6:
    raise RuntimeError('Expected 6 faces, got {}'.format(len(p.faces)))

dp_feat = p.DatumPointByCoordinate(coords=(r_load, Z_TOTAL, 0.0))
top_edge_to_split = p.edges.findAt((R_DOMAIN / 2.0, Z_TOTAL, 0.0))
p.PartitionEdgeByPoint(edge=top_edge_to_split, point=p.datums[dp_feat.id])


# 3. Materials & sections ────────────────────────────────────────────
for i in range(6):
    m.Material(name='Mat_' + layer_names[i]).Elastic(
        table=((E_all[i] * 1e6, nu_all[i]),))
    m.HomogeneousSolidSection(name='Sec_' + layer_names[i],
                              material='Mat_' + layer_names[i], thickness=None)

# Identify faces by their y-centroid; sort BOTTOM-UP to match layer_names order
faces_with_yc = sorted([(f.pointOn[0][1], f.pointOn[0]) for f in p.faces],
                       key=lambda x: x[0])
for i, (yc, pt) in enumerate(faces_with_yc):
    face_set = p.Set(faces=p.faces.findAt((pt,)), name='Set_' + layer_names[i])
    p.SectionAssignment(region=face_set,
                        sectionName='Sec_' + layer_names[i],
                        offset=0.0, offsetType=MIDDLE_SURFACE,
                        offsetField='', thicknessAssignment=FROM_SECTION)


# 4. Assembly, step, output ─────────────────────────────────────────
a = m.rootAssembly
a.DatumCsysByDefault(CARTESIAN)
inst = a.Instance(name='Inst', part=p, dependent=ON)

m.StaticStep(name='LoadStep', previous='Initial',
             timePeriod=1.0, initialInc=1.0, nlgeom=OFF)
m.fieldOutputRequests['F-Output-1'].setValues(
    variables=('S', 'E', 'U', 'COORD'), frequency=LAST_INCREMENT)


# 5. BCs ─────────────────────────────────────────────────────────────
TOL = 1.0e-4
axis_edges  = inst.edges.getByBoundingBox(-TOL, -TOL, -TOL, TOL, Z_TOTAL+TOL, TOL)
bot_edges   = inst.edges.getByBoundingBox(-TOL, -TOL, -TOL, R_DOMAIN+TOL, TOL, TOL)
right_edges = inst.edges.getByBoundingBox(R_DOMAIN-TOL, -TOL, -TOL,
                                          R_DOMAIN+TOL, Z_TOTAL+TOL, TOL)
m.DisplacementBC(name='BC_Axis',   createStepName='Initial',
                 region=a.Set(edges=axis_edges,  name='SetAxis'),  u1=0.0)
m.DisplacementBC(name='BC_Bottom', createStepName='Initial',
                 region=a.Set(edges=bot_edges,   name='SetBottom'),
                 u1=0.0, u2=0.0)
m.DisplacementBC(name='BC_Right',  createStepName='Initial',
                 region=a.Set(edges=right_edges, name='SetRight'), u1=0.0)


# 6. Wheel load ──────────────────────────────────────────────────────
P_Pa = P * 1.0e6
top_load_edges = inst.edges.getByBoundingBox(-TOL, Z_TOTAL-TOL, -TOL,
                                             r_load+TOL, Z_TOTAL+TOL, TOL)
if len(top_load_edges) == 0:
    raise RuntimeError('No load edges found at top surface.')
m.Pressure(name='WheelLoad', createStepName='LoadStep',
           region=a.Surface(side1Edges=top_load_edges, name='SurfLoad'),
           distributionType=UNIFORM, magnitude=P_Pa, amplitude=UNSET)


# 7. Mesh ────────────────────────────────────────────────────────────
# Global seed (coarse, far-field)
p.seedPart(size=0.10, deviationFactor=0.1, minSizeFactor=0.1)

# Refine near the load contact (very fine, ≤ 2 cm)
part_top_load = p.edges.getByBoundingBox(-TOL, Z_TOTAL-TOL, -TOL,
                                         r_load+TOL, Z_TOTAL+TOL, TOL)
if len(part_top_load) > 0:
    p.seedEdgeBySize(edges=part_top_load, size=0.02,
                     deviationFactor=0.1, constraint=FINER)

# Refine the entire AC zone (3 sublayers) — finer than far-field
ac_zone_bot = z_boundaries[3]    # = top of base = bottom of AC zone
ac_zone_edges = p.edges.getByBoundingBox(
    -TOL, ac_zone_bot-TOL, -TOL,
    R_DOMAIN+TOL, Z_TOTAL+TOL, TOL)
if len(ac_zone_edges) > 0:
    p.seedEdgeBySize(edges=ac_zone_edges, size=0.04,
                     deviationFactor=0.1, constraint=FINER)

# Refine the upper AC (4cm typical, mesh needs to resolve thin layer)
upper_ac_bot = z_boundaries[5]    # = bottom of upper AC
upper_ac_edges = p.edges.getByBoundingBox(
    -TOL, upper_ac_bot-TOL, -TOL,
    R_DOMAIN+TOL, Z_TOTAL+TOL, TOL)
if len(upper_ac_edges) > 0:
    p.seedEdgeBySize(edges=upper_ac_edges, size=0.025,
                     deviationFactor=0.1, constraint=FINER)

p.setElementType(regions=(p.faces,),
                 elemTypes=(mesh.ElemType(elemCode=CAX4R, elemLibrary=STANDARD),
                            mesh.ElemType(elemCode=CAX3,  elemLibrary=STANDARD)))
p.generateMesh()
print('[MESH] {} nodes, {} elements (6-layer)'.format(
    len(p.nodes), len(p.elements)))


# 8. Job: writeInput + subprocess ───────────────────────────────────
job_name = 'iLLM_PD_FEA'
if job_name in mdb.jobs:
    del mdb.jobs[job_name]
myJob = mdb.Job(name=job_name, model=model_name, type=ANALYSIS,
                memory=80, memoryUnits=PERCENTAGE,
                getMemoryFromAnalysis=True,
                explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE,
                resultsFormat=ODB)
myJob.writeInput(consistencyChecking=OFF)

for ext in ['.odb', '.lck', '.sta', '.msg', '.dat', '.log',
            '.com', '.prt', '.sim']:
    stale = job_name + ext
    if os.path.exists(stale):
        try: os.remove(stale)
        except: pass

cmd = 'abaqus job={} interactive ask_delete=OFF'.format(job_name)
return_code = subprocess.Popen(cmd, shell=True, cwd=os.getcwd()).wait()
print('[SOLVER] exit code = {}'.format(return_code))
if not os.path.exists(job_name + '.odb'):
    raise RuntimeError('Solver did not produce .odb file.')


# 9. Extract from ODB ───────────────────────────────────────────────
from odbAccess import openOdb

odb = openOdb(path=job_name + '.odb')
step = odb.steps['LoadStep']
frame = step.frames[-1]
inst_odb = odb.rootAssembly.instances['INST']

# Node coordinates dict
node_coords = {}
for n in inst_odb.nodes:
    node_coords[n.label] = (n.coordinates[0], n.coordinates[1])

# Displacement: NODAL position by default
U_vals = {}
for v in frame.fieldOutputs['U'].values:
    U_vals[v.nodeLabel] = v.data


def find_node(r_t, z_t, r_tol=0.05, z_tol=0.02):
    best_label, best_dist = None, float('inf')
    for label, (r, z) in node_coords.items():
        if abs(r - r_t) < r_tol and abs(z - z_t) < z_tol:
            d = ((r - r_t) ** 2 + (z - z_t) ** 2) ** 0.5
            if d < best_dist:
                best_dist, best_label = d, label
    return best_label


# Critical z-locations (6-layer)
# Note: thicknesses is TOP-DOWN: [upper_AC, mid_AC, lower_AC, base, subbase]
h_upper_AC = thicknesses[0]
h_mid_AC   = thicknesses[1]
h_lower_AC = thicknesses[2]
h_base     = thicknesses[3]
h_subbase  = thicknesses[4]

z_subgrade_top  = Z_SUBGRADE                                               # subgrade top
z_base_bot      = Z_SUBGRADE + h_subbase                                   # base bottom (= subbase top)
z_AC_bot        = Z_SUBGRADE + h_subbase + h_base                          # AC bottom (= lower AC bottom)
z_lower_AC_mid  = z_AC_bot + h_lower_AC / 2.0
z_mid_AC_mid    = z_AC_bot + h_lower_AC + h_mid_AC / 2.0
z_upper_AC_mid  = z_AC_bot + h_lower_AC + h_mid_AC + h_upper_AC / 2.0

# Build ELEMENT_NODAL subsets once (extrapolation to corner nodes)
S_nodal = frame.fieldOutputs['S'].getSubset(position=ELEMENT_NODAL)
E_nodal = frame.fieldOutputs['E'].getSubset(position=ELEMENT_NODAL)


def collect_at(field_nodal, z_target, r_max=0.3, z_tol=0.025,
               comp_idx=0, sign='positive'):
    """
    Collect a stress/strain component across ELEMENT_NODAL values near
    (r<r_max, |z-z_target|<z_tol).

    comp_idx:
      For axisymmetric CAX4R, data = [S11=radial, S22=axial(vertical),
                                       S33=hoop(circumferential), S12=shear_rz]
      comp_idx=0 → radial (horizontal)
      comp_idx=1 → axial (vertical)
      comp_idx=2 → hoop

    sign:
      'positive'     → max value (tensile in ABAQUS convention)
      'negative_abs' → max |value| where value<0 (compressive magnitude)
      'abs'          → max |value| regardless of sign

    Returns:  scalar or 0.0 if no nodes match.
    """
    vals = []
    for v in field_nodal.values:
        if v.nodeLabel == 0 or v.nodeLabel not in node_coords:
            continue
        r, z = node_coords[v.nodeLabel]
        if r < r_max and abs(z - z_target) < z_tol:
            x = v.data[comp_idx]
            if sign == 'positive' and x > 0:
                vals.append(x)
            elif sign == 'negative_abs' and x < 0:
                vals.append(abs(x))
            elif sign == 'abs':
                vals.append(abs(x))
    return max(vals) if vals else 0.0


# ─── 9.1  Surface deflection (backward compat) ────────────────────
nl_centre = find_node(0.0, Z_TOTAL)
D_FEA = abs(U_vals[nl_centre][1]) * 1000.0 if nl_centre else 0.0

basin = {}
for sr in sensor_offsets:
    nl_b = find_node(sr if sr > 0 else 0.0, Z_TOTAL)
    basin['r_{:.2f}m'.format(sr)] = (
        round(abs(U_vals[nl_b][1]) * 1000.0, 4) if nl_b else None)


# ─── 9.2  AC bottom (z = z_AC_bot, JTG B.1 ε_a) ───────────────────
# Radial (S11) tensile strain at the BOTTOM of the lowest AC sublayer
eps_a_strain = collect_at(E_nodal, z_AC_bot,
                          r_max=0.3, z_tol=0.025,
                          comp_idx=0, sign='positive')
epsilon_a_microstrain = eps_a_strain * 1.0e6

# Backward-compat: max principal stress at AC bottom (old sigma_FEA)
sigma_list = []
for v in S_nodal.values:
    if v.nodeLabel == 0 or v.nodeLabel not in node_coords:
        continue
    r, z = node_coords[v.nodeLabel]
    if r < 0.3 and abs(z - z_AC_bot) < 0.025:
        sr_v  = v.data[0] / 1e6
        sz_v  = v.data[1] / 1e6
        st_v  = v.data[2] / 1e6
        srz_v = v.data[3] / 1e6
        sigma_p_rz = (0.5 * (sr_v + sz_v)
                      + ((0.5 * (sr_v - sz_v)) ** 2 + srz_v ** 2) ** 0.5)
        sigma_list.append(max(sigma_p_rz, st_v, sr_v))
sigma_FEA = max(sigma_list) if sigma_list else 0.0


# ─── 9.3  Semi-rigid base bottom (z = z_base_bot, JTG B.2 σ_t) ────
sigma_t_Pa = collect_at(S_nodal, z_base_bot,
                       r_max=0.35, z_tol=0.025,
                       comp_idx=0, sign='positive')
sigma_t_MPa = sigma_t_Pa / 1.0e6

# Backward-compat: max principal strain at base bottom (old epsilon_FEA)
eps_list = []
for v in E_nodal.values:
    if v.nodeLabel == 0 or v.nodeLabel not in node_coords:
        continue
    r, z = node_coords[v.nodeLabel]
    if r < 0.35 and abs(z - z_base_bot) < 0.025:
        er  = v.data[0] * 1e6
        ez  = v.data[1] * 1e6
        et  = v.data[2] * 1e6
        erz = v.data[3] * 1e6
        eps_p_rz = (0.5 * (er + ez)
                    + ((0.5 * (er - ez)) ** 2 + (erz / 2.0) ** 2) ** 0.5)
        eps_list.append(max(eps_p_rz, et, er))
epsilon_FEA = max(eps_list) if eps_list else 0.0


# ─── 9.4  Subgrade top (z = z_subgrade_top, JTG B.4 ε_z) ──────────
eps_z_strain = collect_at(E_nodal, z_subgrade_top,
                         r_max=0.40, z_tol=0.030,
                         comp_idx=1, sign='negative_abs')
epsilon_z_microstrain = eps_z_strain * 1.0e6


# ─── 9.5  AC sublayer mid-depth vertical stresses (NEW, JTG B.3) ──
# For B.3.1 multi-sublayer permanent deformation, we need p_i for each
# AC sublayer. p_i is the vertical (S22) compressive stress at the
# sublayer's mid-depth, in MPa (magnitude).
def get_p_AC_mid(z_target):
    """Vertical compressive stress at AC sublayer mid-depth, MPa."""
    val_Pa = collect_at(S_nodal, z_target,
                       r_max=0.20, z_tol=0.030,
                       comp_idx=1, sign='negative_abs')
    return val_Pa / 1.0e6

p_AC_upper_mid_MPa = get_p_AC_mid(z_upper_AC_mid)
p_AC_mid_mid_MPa   = get_p_AC_mid(z_mid_AC_mid)
p_AC_lower_mid_MPa = get_p_AC_mid(z_lower_AC_mid)
# Vertical pressures at the two NCAT earth-pressure-cell interfaces.
# Stress continuity makes S22 suitable at the shared interface. These fields
# are separate from the JTG AC-midpoint stresses and are intended for direct
# comparison with NCAT Cracking Group EPC observations.
p_AC_base_interface_MPa = collect_at(
    S_nodal, z_AC_bot, r_max=0.20, z_tol=0.020,
    comp_idx=1, sign='negative_abs') / 1.0e6
p_base_subgrade_interface_MPa = collect_at(
    S_nodal, z_base_bot, r_max=0.25, z_tol=0.020,
    comp_idx=1, sign='negative_abs') / 1.0e6



# ─── 9.6  AC sublayer mid-depth vertical ELASTIC strains (NEW) ────
# True vertical compressive elastic (resilient) strain at each AC
# sublayer mid-depth, microstrain (magnitude). This is the correct
# input for the NCHRP 1-37A AC rutting model: it accounts for triaxial
# confinement, unlike a uniaxial sigma_v/E approximation (which ignores
# the lateral stresses and therefore overestimates the vertical strain).
def get_eps_AC_mid(z_target):
    """Vertical (axial, comp_idx=1) compressive elastic strain at AC
    sublayer mid-depth, microstrain (magnitude)."""
    val = collect_at(E_nodal, z_target,
                     r_max=0.20, z_tol=0.030,
                     comp_idx=1, sign='negative_abs')
    return val * 1.0e6

eps_AC_upper_mid_micro = get_eps_AC_mid(z_upper_AC_mid)
eps_AC_mid_mid_micro   = get_eps_AC_mid(z_mid_AC_mid)
eps_AC_lower_mid_micro = get_eps_AC_mid(z_lower_AC_mid)

odb.close()


# 10. Write result JSON ─────────────────────────────────────────────
result = {
    'success': True,

    # Backward-compat fields
    'D_FEA_mm':                round(D_FEA, 4),
    'sigma_FEA_MPa':           round(sigma_FEA, 4),
    'epsilon_FEA_microstrain': round(epsilon_FEA, 2),
    'deflection_basin_mm':     basin,

    # Protocol-aware responses (6-layer extension)
    'responses': {
        'epsilon_a_microstrain':   round(epsilon_a_microstrain, 2),
        'sigma_t_MPa':             round(sigma_t_MPa, 4),
        'epsilon_z_microstrain':   round(epsilon_z_microstrain, 2),
        # B.3 multi-sublayer inputs
        'p_AC_upper_mid_MPa':      round(p_AC_upper_mid_MPa, 4),
        'p_AC_mid_mid_MPa':        round(p_AC_mid_mid_MPa, 4),
        'p_AC_lower_mid_MPa':      round(p_AC_lower_mid_MPa, 4),
        'p_AC_base_interface_MPa': round(p_AC_base_interface_MPa, 4),
        'p_base_subgrade_interface_MPa': round(
            p_base_subgrade_interface_MPa, 4),
        # AC sublayer mid-depth vertical ELASTIC strains (microstrain).
        # Correct (confinement-aware) input for the NCHRP 1-37A AC rutting
        # model; replaces the uniaxial sigma_v/E approximation.
        'eps_AC_upper_mid_microstrain': round(eps_AC_upper_mid_micro, 2),
        'eps_AC_mid_mid_microstrain':   round(eps_AC_mid_mid_micro, 2),
        'eps_AC_lower_mid_microstrain': round(eps_AC_lower_mid_micro, 2),
        '_notes': {
            'epsilon_a': 'AC bottom radial tensile strain (JTG D50-2017 eps_a). '
                         'Position: z = Z_SUBGRADE + h_subbase + h_base (lower AC bottom).',
            'sigma_t':   'Semi-rigid base bottom radial tensile stress '
                         '(JTG D50-2017 sigma_t). Position: z = Z_SUBGRADE + h_subbase.',
            'epsilon_z': 'Subgrade top vertical compressive strain magnitude '
                         '(JTG D50-2017 eps_z). Position: z = Z_SUBGRADE.',
            'p_AC_mid':  'AC sublayer mid-depth vertical compressive stress (MPa). '
                         'For JTG B.3.1 multi-sublayer permanent deformation R_a calculation.',
            'ncat_interface_pressures':
                'Vertical pressures at the AC/base and base/subgrade '
                'interfaces for comparison with NCAT EPC observations.',

            'sublayer_z_mid_m': {
                'upper': round(z_upper_AC_mid, 4),
                'mid':   round(z_mid_AC_mid,   4),
                'lower': round(z_lower_AC_mid, 4),
            },
        }
    },

    'num_nodes':        len(p.nodes),
    'num_elements':     len(p.elements),
    'model_type':       'ABAQUS_2024_CAX4R_axisymmetric_v0.4_6layer',
    'solver_exit_code': return_code,
    'load':             {'P_MPa': P, 'r_m': r_load},
    'pavement': {
        'layer_order_top_down': ['upper_AC', 'mid_AC', 'lower_AC', 'base', 'subbase'],
        'thicknesses_m':        thicknesses,
        'moduli_MPa':           moduli,
        'poissons':             poissons,
        'E_subgrade_MPa':       E_subgrade,
        'nu_subgrade':          nu_subgrade,
    },
}
with open('pavement_result.json', 'w') as f:
    json.dump(result, f, indent=2)

print('=' * 70)
print('FEA RESULT (v0.4, 6-layer)')
print('  ─── Legacy fields ───')
print('    D       = {:.4f} mm'.format(D_FEA))
print('    sigma   = {:.4f} MPa (max principal at AC bottom)'.format(sigma_FEA))
print('    epsilon = {:.2f} ue (max principal at base bottom)'.format(epsilon_FEA))
print('  ─── Protocol responses ───')
print('    eps_a   (lower AC bot, radial tensile) = {:.2f} ue'
      .format(epsilon_a_microstrain))
print('    sigma_t (base bot, radial tensile)     = {:.4f} MPa'
      .format(sigma_t_MPa))
print('    eps_z   (subgrade top, vert. comp.)    = {:.2f} ue'
      .format(epsilon_z_microstrain))
print('  ─── AC sublayer mid-depth vertical stresses ───')
print('    p_upper_AC_mid = {:.4f} MPa'.format(p_AC_upper_mid_MPa))
print('    p_mid_AC_mid   = {:.4f} MPa'.format(p_AC_mid_mid_MPa))
print('    p_lower_AC_mid = {:.4f} MPa'.format(p_AC_lower_mid_MPa))
print('=' * 70)
