# -*- coding: utf-8 -*-

'''Example 06

Solves a plane stress 2D problem using a structured mesh.
Shows how to draw von Mises effective stress as an element value with
drawElementValues(). Shows use of GmshMesher attribute 'nodesOnCurve'
(dictionary that says which nodes are on a given geometry curve)
'''

import calfem.geometry as cfg
import calfem.mesh as cfm

import calfem.utils as cfu
import calfem.core as cfc
import numpy as np
import calfem.vis as cfv

from math import sqrt, ceil


def get_el_on_curve(line, edge_lenght, points):
    pt1 = points[line[0]]
    pt2 = points[line[1]]
    line_lenght = sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
    elements_on_curve = ceil(line_lenght / edge_lenght)
    return elements_on_curve


cfu.enableLogging()

# ---- Define problem variables ---------------------------------------------

layers = [1, 0, 1]
t = 0.02
E = 12e9
G = 0.45*690*1e6
ptype = 1
ep = [ptype, 1]
Dx = cfc.ortho_hooke(ptype, E, 0, 0, 0, G)
Dy = cfc.ortho_hooke(ptype, 0, E, 0, 0, G)
D = np.zeros(shape=[3, 3])
for layer in layers:
    if layer == 1:
        D += Dx * t
    else:
        D += Dy * t

# ---- Define geometry ------------------------------------------------------

cfu.info("Creating geometry...")

g = cfg.geometry()


points = [[0, 0], [10, 0], [10, 3], [0, 3], [3, 1], [6, 1], [3, 2.5], [6, 2.5]]  # 20-24

for xp, yp in points:
    g.point([xp * 0.1, yp * 0.1])

splines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 7], [7, 6], [6, 4]]  # 25
top_lenght = 0
for i, s in enumerate(splines):
    if i == 6:
        el_num = get_el_on_curve(s, 0.05, points)
    elif i == 2:
        el_num = get_el_on_curve(s, 0.1, points)
        pt1 = points[s[0]]
        pt2 = points[s[1]]
        line_lenght = sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
        top_lenght = line_lenght/el_num
    elif i == 5 or i == 7:
        el_num = get_el_on_curve(s, 0.05, points)
    else:
        el_num = get_el_on_curve(s, 0.3, points)
    g.spline(s, el_on_curve=el_num)

g.curveMarker(ID=0, marker=5)
g.curveMarker(ID=2, marker=7)

# Points in circle arcs are [start, center, end]

g.addSurface([0, 1, 2, 3], [[4, 5, 6, 7]])  # 0

# ---- Create mesh ----------------------------------------------------------

cfu.info("Meshing geometry...")

# Create mesh

mesh = cfm.GmshMesh(geometry=g)
mesh.el_type = 3
mesh.dofs_per_node = 2
mesh.maxsize = 0.1

coords, edof, dofs, bdofs, elementmarkers = mesh.create()

# ---- Solve problem --------------------------------------------------------

cfu.info("Assembling system matrix...")

nDofs = np.size(dofs)
ex, ey = cfc.coordxtr(edof, coords, dofs)
K = np.zeros([nDofs, nDofs])

for eltopo, elx, ely in zip(edof, ex, ey):
    if mesh.el_type == 2:
        Ke = cfc.plante(elx, ely, ep, D)
    else:
        Ke = cfc.planqe(elx, ely, ep, D)
    cfc.assem(eltopo, K, Ke)

cfu.info("Solving equation system...")

f = np.zeros([nDofs, 1])

bc = np.array([], 'i')
bcVal = np.array([], 'f')

bc, bcVal = cfu.applybc(bdofs, bc, bcVal, 5, 0.0, 0)

cfu.applyforce(bdofs, f, 7, -2000 * top_lenght, 2)

a, r = cfc.solveq(K, f, bc, bcVal)
displacement = str(np.max(np.abs(a)))
cfu.info(displacement)
cfu.info("Computing element forces...")

ed = cfc.extractEldisp(edof, a)
nx = []
ny = []
txy = []

# For each element:

for i in range(edof.shape[0]):
    # Determine element stresses and strains in the element.

    if mesh.el_type == 2:
        es, et = cfc.plants(ex[i, :], ey[i, :], ep, D, ed[i, :])
        es=es[0]
        et=et[0]
    else:
        es, et = cfc.planqs(ex[i, :], ey[i, :], ep, D, ed[i, :])

    # Calc and append effective stress to list.
    nx.append(float(es[0]))
    ny.append(float(es[1]))
    txy.append(float(es[2]))

    ## es: [sigx sigy tauxy]

# ---- Visualise results ----------------------------------------------------

cfu.info("Visualising...")
cfu.info(str(np.max(np.abs(nx))))
cfu.info(str(np.max(np.abs(ny))))
cfu.info(str(np.max(np.abs(txy))))
cfv.drawGeometry(g, draw_points=False, label_curves=True)

cfv.figure()
cfv.draw_element_values(nx, coords, edof, mesh.dofs_per_node, mesh.el_type, a, draw_elements=True,
                        draw_undisplaced_mesh=False)

cfv.figure()
cfv.draw_element_values(ny, coords, edof, mesh.dofs_per_node, mesh.el_type, a, draw_elements=True,
                        draw_undisplaced_mesh=False)

cfv.figure()
cfv.draw_element_values(txy, coords, edof, mesh.dofs_per_node, mesh.el_type, a, draw_elements=True,
                        draw_undisplaced_mesh=False)

cfv.figure()
cfv.draw_displacements(a, coords, edof, mesh.dofs_per_node, mesh.el_type, draw_undisplaced_mesh=True,
                       title="Example 06")

cfv.show_and_wait()

print("Done.")
