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

from math import sqrt



def in_plane_orthotropic_analysis(external_polygon, holes, support_edges, support_points, load_edges, load_points,
                                  cross_section):
    cfu.enableLogging()

    # ---- Define problem variables ---------------------------------------------

    layers = [1, 0, 1, 0, 1]
    t = 0.3
    v = 0.35
    E = 2.1e9
    G = 2.1e9 / 16
    ptype = 1
    ep = [ptype, t]
    Dx = cfc.ortho_hooke(ptype, E, 0, 0, 0, G)
    Dy = cfc.ortho_hooke(ptype, 0, E, 0, 0, G)

    # ---- Define geometry ------------------------------------------------------

    cfu.info("Creating geometry...")

    g = cfg.geometry()

    # Just a shorthand. We use this to make the circle arcs.

    s2 = 1 / sqrt(2)

    points = [[0, 0], [10, 0], [10, 3], [0, 3], [3, 1], [6, 1], [3, 2.5], [6, 2.5]]  # 20-24

    for xp, yp in points:
        g.point([xp * 0.1, yp * 0.1])

    splines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 7], [7, 6], [6, 4]]  # 25

    for s in splines:
        g.spline(s, el_on_curve=10)

    g.curveMarker(ID=0, marker=5)
    g.curveMarker(ID=2, marker=7)

    # Points in circle arcs are [start, center, end]

    g.addSurface([0, 1, 2, 3], [[4, 5, 6, 7]])  # 0

    # ---- Create mesh ----------------------------------------------------------

    cfu.info("Meshing geometry...")

    # Create mesh

    mesh = cfm.GmshMesh(geometry=g)
    mesh.el_type = 2
    mesh.dofs_per_node = 2
    mesh.maxsize = 0.1

    coords, edof, dofs, bdofs, elementmarkers = mesh.create()

    # ---- Solve problem --------------------------------------------------------

    cfu.info("Assembling system matrix...")

    nDofs = np.size(dofs)
    ex, ey = cfc.coordxtr(edof, coords, dofs)
    K = np.zeros([nDofs, nDofs])

    for eltopo, elx, ely in zip(edof, ex, ey):
        Ke = np.zeros(shape=[6, 6])
        for layer in layers:
            if layer == 1:
                Ke += cfc.plante(elx, ely, ep, Dx)
            else:
                Ke += cfc.plante(elx, ely, ep, Dy)
        cfc.assem(eltopo, K, Ke)

    cfu.info("Solving equation system...")

    f = np.zeros([nDofs, 1])

    bc = np.array([], 'i')
    bcVal = np.array([], 'f')

    bc, bcVal = cfu.applybc(bdofs, bc, bcVal, 5, 0.0, 0)

    cfu.applyforce(bdofs, f, 7, -1e5, 2)

    a, r = cfc.solveq(K, f, bc, bcVal)

    cfu.info("Computing element forces...")

    ed = cfc.extractEldisp(edof, a)
    vonMises = []

    # For each element:

    for i in range(edof.shape[0]):
        # Determine element stresses and strains in the element.

        es, et = cfc.plants(ex[i, :], ey[i, :], ep, Dx, ed[i, :])

        # Calc and append effective stress to list.

        vonMises.append(es[0])

        ## es: [sigx sigy tauxy]

    # ---- Visualise results ----------------------------------------------------

    cfu.info("Visualising...")

    cfv.drawGeometry(g, draw_points=False, label_curves=True)

    cfv.figure()
    # cfv.draw_element_values(vonMises, coords, edof, mesh.dofs_per_node, mesh.el_type, a, draw_elements=True,
    #                        draw_undisplaced_mesh=True)

    cfv.figure()
    cfv.draw_displacements(a, coords, edof, mesh.dofs_per_node, mesh.el_type, draw_undisplaced_mesh=True,
                           title="Example 06")

    cfv.show_and_wait()

    print("Done.")
