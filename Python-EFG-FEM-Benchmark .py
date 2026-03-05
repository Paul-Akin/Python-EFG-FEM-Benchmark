#!/usr/bin/env python3
"""
=============================================================================
Meshless vs. Mesh-Based FEA: Side-by-Side Comparison
=============================================================================
Three solutions to the same problem on the same node grid:

    1. Timoshenko Exact  - Analytical elasticity solution (ground truth)
    2. Traditional FEM   - Bilinear Q4 elements (a la Bergstrom/NASTRAN CQUAD4)
    3. Element-Free Galerkin (EFG) - MLS meshless method

Benchmark: Cantilever beam under parabolic end shear (plane stress)

References:
    [1] Timoshenko, S.P. & Goodier, J.N. (1970) "Theory of Elasticity"
        3rd Ed., McGraw-Hill. Section 2.8.
    [2] Bergstrom, J. (2022) "FEA in 100 Lines of Python" (GPL v2)
        Adapted from http://compmech.lab.asu.edu/codes.php
    [3] Belytschko, T., Lu, Y.Y., Gu, L. (1994) "Element-Free Galerkin
        Methods" Int. J. Numer. Meth. Engng., 37, 229-256.
    [4] Liu, G.R. (2010) "Meshfree Methods: Moving Beyond the Finite
        Element Method" 2nd Ed., CRC Press.

Units: US Customary (lbf, in, psi)

Author: Paul Akin
=============================================================================
"""

import numpy as np
from numpy.linalg import solve, inv, det
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.patches import Polygon as MplPolygon
from time import perf_counter
import math

# ============================================================================
# MATERIAL & GEOMETRY (shared by all three solutions)
# ============================================================================
E       = 10.0e6        # Young's modulus [psi]
nu      = 0.30          # Poisson's ratio [-]
P       = -1000.0       # End shear load [lbf] (downward)
L       = 48.0          # Beam length [in]
D       = 12.0          # Beam depth [in]
t_thick = 1.0           # Thickness for plane stress [in]
I_mom   = t_thick * D**3 / 12.0  # Second moment of area [in^4]

# Plane stress constitutive matrix [C]
# Ref: Timoshenko & Goodier, or any FEM textbook (e.g., Cook et al.)
C_mat = (E / (1.0 - nu**2)) * np.array([
    [1.0,  nu,  0.0          ],
    [nu,   1.0, 0.0          ],
    [0.0,  0.0, (1.0 - nu)/2.0]
])

# ============================================================================
# TIMOSHENKO EXACT SOLUTION
# ============================================================================
def exact_displacement(x, y):
    """Exact displacement for cantilever under parabolic end shear."""
    ux = (P / (6*E*I_mom)) * (
            y * (6*L - 3*x) * x + (2 + nu) * y * (y**2 - D**2/4)
    )
    uy = -(P / (6*E*I_mom)) * (
            3*nu * y**2 * (L - x) + x**2 * (3*L - x) +
            (4 + 5*nu) * D**2 * x / 4
    )
    return ux, uy

def exact_stress(x, y):
    """Exact stress field."""
    sxx = P * (L - x) * y / I_mom
    syy = 0.0 * np.ones_like(x) if isinstance(x, np.ndarray) else 0.0
    sxy = -P / (2*I_mom) * (D**2/4 - y**2)
    return sxx, syy, sxy


# ============================================================================
# EULER-BERNOULLI BEAM THEORY SOLUTION
# ============================================================================
# The introductory Mechanics of Materials solution (Hibbeler, Gere & Goodno).
# Neglects shear deformation and cross-section warping.
#
#   v(x)     = Px^2(3L - x) / (6EI)           (double integration of EI*v'' = M)
#   sigma_xx = P(L-x)*y / I  = M*y/I          (same as elasticity - exact for this case)
#   tau_xy   = -P/(2I)*(D^2/4 - y^2) = VQ/Ib  (same as elasticity - exact for this case)
#
# The ONLY difference from the Timoshenko solution is in the displacement field:
#   - No shear deformation correction term: (4+5*nu)*P*D^2*x / (24EI)
#   - No cross-section warping:  (2+nu)*y*(y^2 - D^2/4) / (6EI)

def beam_theory_deflection(x):
    """
    Euler-Bernoulli centerline deflection.

    The Timoshenko sign convention in this code defines P as the shear
    coefficient where the applied traction is tau = -P/(2I)*(D^2/4 - y^2).
    For P = -1000, the net applied force is -P = +1000 lbf (upward),
    producing upward (positive) deflection.

    Standard textbook formula: delta = F*L^3/(3EI) with F = -P (upward).
    So: v(x) = (-P)*x^2*(3L - x) / (6EI) = -P*x^2*(3L-x)/(6EI)

    Ref: Hibbeler, "Mechanics of Materials", or Roark's Table 8.1, Case 1a.
    """
    return -P * x**2 * (3*L - x) / (6*E*I_mom)

def beam_theory_slope(x):
    """Beam theory slope: dv/dx = -Px(2L-x)/(2EI)"""
    return -P * x * (2*L - x) / (2*E*I_mom)


# ============================================================================
#  SHARED NODE / MESH GENERATION
# ============================================================================
def generate_nodes(nx, ny):
    """Regular grid of nodes. x in [0,L], y in [-D/2, D/2]."""
    xs = np.linspace(0, L, nx)
    ys = np.linspace(-D/2, D/2, ny)
    xx, yy = np.meshgrid(xs, ys)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])
    return nodes

def generate_connectivity(nx, ny):
    """
    Build Q4 element connectivity for a structured grid.
    Same logic as Bergstrom's code, adapted for our node ordering.

    Node ordering (per row of y, sweeping x):
        Row j:  j*nx, j*nx+1, ..., j*nx+(nx-1)

    Element (i,j) connects:
        n0 = i + j*nx          (bottom-left)
        n1 = i + 1 + j*nx      (bottom-right)
        n2 = i + 1 + (j+1)*nx  (top-right)
        n3 = i + (j+1)*nx      (top-left)
    """
    conn = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = i + j * nx
            conn.append([n0, n0 + 1, n0 + 1 + nx, n0 + nx])
    return conn


# ############################################################################
#
#  PART 1: TRADITIONAL FEM SOLVER (Q4 bilinear elements)
#
#  This is essentially what Bergstrom's "FEA in 100 lines" does, and what
#  NASTRAN's CQUAD4 element does at its core. Bilinear shape functions on
#  a 4-node quadrilateral with 2x2 Gauss quadrature.
#
# ############################################################################

def fem_shape(xi):
    """
    Q4 bilinear shape functions in natural coordinates (xi, eta).
    N_I = (1/4)(1 + xi_I * xi)(1 + eta_I * eta)

    This is the CQUAD4 element formulation. Same as Bergstrom's shape().
    """
    x, y = xi[0], xi[1]
    N = np.array([
        (1.0 - x) * (1.0 - y),
        (1.0 + x) * (1.0 - y),
        (1.0 + x) * (1.0 + y),
        (1.0 - x) * (1.0 + y)
    ]) * 0.25
    return N

def fem_gradshape(xi):
    """
    Derivatives of Q4 shape functions w.r.t. natural coordinates.
    Returns dN/dxi and dN/deta as a (2 x 4) matrix.
    Same as Bergstrom's gradshape().
    """
    x, y = xi[0], xi[1]
    dN = np.array([
        [-(1.0-y),  (1.0-y), (1.0+y), -(1.0+y)],
        [-(1.0-x), -(1.0+x), (1.0+x),  (1.0-x)]
    ]) * 0.25
    return dN


def fem_solve(nodes, nx, ny):
    """
    Traditional FEM solver using Q4 elements.

    Steps (same as any NASTRAN-like solver):
        1. Loop over elements
        2. For each element, loop over 2x2 Gauss points
        3. Compute Jacobian, B-matrix, element stiffness Ke = B^T C B |J|
        4. Assemble into global K via connectivity (guide vectors!)
        5. Apply BCs, solve Ku = f

    This mirrors Bergstrom's code almost exactly, adapted for:
        - Plane stress (not plane strain)
        - Our cantilever geometry and loading
        - Penalty BCs using exact Timoshenko displacement at x=0
    """
    print("\n  [FEM] Traditional Q4 Solver")
    print("  " + "-"*50)

    conn = generate_connectivity(nx, ny)
    n_nodes = len(nodes)
    n_dof = 2 * n_nodes
    n_elem = len(conn)

    print(f"  {n_nodes} nodes, {n_elem} elements, {n_dof} DOFs")

    t_start = perf_counter()

    # Global stiffness matrix
    K = np.zeros((n_dof, n_dof))

    # 2x2 Gauss quadrature points (same as Bergstrom's q4 list)
    g = 1.0 / math.sqrt(3.0)
    gauss_pts = [[-g, -g], [g, -g], [g, g], [-g, g]]

    # B-matrix template (3 x 8) for plane stress
    B = np.zeros((3, 8))

    # --- Element loop (the core of any FEM code) ---
    for elem_nodes in conn:
        # Element nodal coordinates (4 x 2)
        xe = nodes[elem_nodes, :]

        # Element stiffness (8 x 8)
        Ke = np.zeros((8, 8))

        for qpt in gauss_pts:
            # Shape function derivatives in natural coords
            dN = fem_gradshape(qpt)

            # Jacobian: J = dN * xe  (maps natural -> physical coords)
            # J = [dx/dxi   dy/dxi ]
            #     [dx/deta  dy/deta]
            J = dN @ xe
            detJ = np.linalg.det(J)

            # Shape function derivatives in physical coords:
            # dN/dx = J^{-1} * dN/dxi
            dN_phys = np.linalg.inv(J) @ dN

            # Strain-displacement matrix B (Voigt notation)
            # {epsilon} = {du/dx, dv/dy, du/dy + dv/dx}^T = [B]{u_e}
            #
            # For node I (DOFs 2I, 2I+1):
            #   B[:,2I]   = [dN_I/dx,    0,     dN_I/dy]^T
            #   B[:,2I+1] = [   0,    dN_I/dy,  dN_I/dx]^T
            B[0, 0::2] = dN_phys[0, :]   # dN/dx -> epsilon_xx
            B[1, 1::2] = dN_phys[1, :]   # dN/dy -> epsilon_yy
            B[2, 0::2] = dN_phys[1, :]   # dN/dy -> gamma_xy (du/dy part)
            B[2, 1::2] = dN_phys[0, :]   # dN/dx -> gamma_xy (dv/dx part)

            # Ke += B^T * C * B * |J| * thickness
            Ke += B.T @ C_mat @ B * detJ * t_thick

        # --- Global assembly using guide vector concept ---
        # This is exactly what the guide vectors in your FEM textbook do:
        # map local element DOFs (0..7) to global DOFs (2*I, 2*I+1)
        for i, I in enumerate(elem_nodes):
            for j, J in enumerate(elem_nodes):
                K[2*I,   2*J  ] += Ke[2*i,   2*j  ]
                K[2*I+1, 2*J  ] += Ke[2*i+1, 2*j  ]
                K[2*I,   2*J+1] += Ke[2*i,   2*j+1]
                K[2*I+1, 2*J+1] += Ke[2*i+1, 2*j+1]

    t_assembly = perf_counter() - t_start
    print(f"  Assembly time: {t_assembly:.3f} sec")

    # --- Force vector: parabolic end shear at x = L ---
    F = np.zeros(n_dof)
    tol = 1e-6
    tip_nodes = np.where(np.abs(nodes[:, 0] - L) < tol)[0]
    tip_sorted = tip_nodes[np.argsort(nodes[tip_nodes, 1])]
    y_vals = nodes[tip_sorted, 1]
    n_tip = len(tip_sorted)

    for i, I in enumerate(tip_sorted):
        y_I = nodes[I, 1]
        if i == 0:
            h_trib = (y_vals[1] - y_vals[0]) / 2.0
        elif i == n_tip - 1:
            h_trib = (y_vals[-1] - y_vals[-2]) / 2.0
        else:
            h_trib = (y_vals[i+1] - y_vals[i-1]) / 2.0

        tau = -P / (2*I_mom) * (D**2/4 - y_I**2)
        F[2*I + 1] += tau * h_trib * t_thick

    # --- Essential BCs via penalty at x = 0 ---
    alpha = 1.0e15
    fixed_nodes = np.where(np.abs(nodes[:, 0]) < tol)[0]
    for I in fixed_nodes:
        ux_ex, uy_ex = exact_displacement(nodes[I, 0], nodes[I, 1])
        K[2*I, 2*I]     += alpha
        K[2*I+1, 2*I+1] += alpha
        F[2*I]           += alpha * ux_ex
        F[2*I+1]         += alpha * uy_ex

    # --- Solve ---
    t_solve_start = perf_counter()
    U = solve(K, F)
    t_solve = perf_counter() - t_solve_start
    print(f"  Solve time:    {t_solve:.3f} sec")

    # --- Stress recovery (at element centroids, then averaged to nodes) ---
    # FEM stresses are naturally computed at Gauss points or centroids.
    # For a fair comparison, we extrapolate to nodes using weighted averaging.
    t_stress_start = perf_counter()
    node_stress = np.zeros((n_nodes, 3))
    node_count = np.zeros(n_nodes)

    for elem_nodes_list in conn:
        xe = nodes[elem_nodes_list, :]
        ue = np.zeros(8)
        for i, I in enumerate(elem_nodes_list):
            ue[2*i]   = U[2*I]
            ue[2*i+1] = U[2*I+1]

        # Evaluate stress at element centroid (xi=0, eta=0)
        dN = fem_gradshape([0.0, 0.0])
        J = dN @ xe
        dN_phys = np.linalg.inv(J) @ dN

        B_cent = np.zeros((3, 8))
        B_cent[0, 0::2] = dN_phys[0, :]
        B_cent[1, 1::2] = dN_phys[1, :]
        B_cent[2, 0::2] = dN_phys[1, :]
        B_cent[2, 1::2] = dN_phys[0, :]

        strain = B_cent @ ue
        stress = C_mat @ strain

        # Distribute centroidal stress to nodes (simple averaging)
        for I in elem_nodes_list:
            node_stress[I] += stress
            node_count[I] += 1.0

    # Average
    for I in range(n_nodes):
        if node_count[I] > 0:
            node_stress[I] /= node_count[I]

    t_stress = perf_counter() - t_stress_start
    t_total = perf_counter() - t_start
    print(f"  Stress time:   {t_stress:.3f} sec")
    print(f"  Total time:    {t_total:.3f} sec")

    return U, node_stress


# ############################################################################
#
#  PART 2: ELEMENT-FREE GALERKIN (EFG) MESHLESS SOLVER
#
#  No mesh, no connectivity, no elements. Just a cloud of nodes and
#  Moving Least Squares shape functions with background cell integration.
#
# ############################################################################

# --- Weight function (cubic spline kernel) ---
def cubic_spline_weight(r):
    """Cubic spline weight function (Liu, 2010, Eq. 5.23). C^2 continuous."""
    w = np.zeros_like(r)
    mask1 = (r >= 0) & (r <= 0.5)
    mask2 = (r > 0.5) & (r < 1.0)
    w[mask1] = 2.0/3.0 - 4.0*r[mask1]**2 + 4.0*r[mask1]**3
    w[mask2] = 4.0/3.0 - 4.0*r[mask2] + 4.0*r[mask2]**2 - (4.0/3.0)*r[mask2]**3
    return w

def cubic_spline_dweight(r):
    """Derivative of cubic spline w.r.t. normalized distance r."""
    dw = np.zeros_like(r)
    mask1 = (r >= 0) & (r <= 0.5)
    mask2 = (r > 0.5) & (r < 1.0)
    dw[mask1] = -8.0*r[mask1] + 12.0*r[mask1]**2
    dw[mask2] = -4.0 + 8.0*r[mask2] - 4.0*r[mask2]**2
    return dw

# --- MLS shape functions ---
def compute_mls_shape(x_eval, nodes, d_max_factor=2.5):
    """
    Compute MLS shape functions and gradients at evaluation point.
    Linear basis p = [1, x, y]^T.
    Returns phi, dphi_dx, dphi_dy, neighbor indices.
    """
    n_nodes = len(nodes)
    bbox = nodes.max(axis=0) - nodes.min(axis=0)
    avg_spacing = np.sqrt(bbox[0] * bbox[1] / n_nodes)
    d_max = d_max_factor * avg_spacing

    dx = x_eval[0] - nodes[:, 0]
    dy = x_eval[1] - nodes[:, 1]
    dist = np.sqrt(dx**2 + dy**2)

    neighbors = np.where(dist < d_max)[0]
    if len(neighbors) < 3:
        neighbors = np.argsort(dist)[:6]

    n_nb = len(neighbors)
    r = dist[neighbors] / d_max
    w = cubic_spline_weight(r)
    dw_dr = cubic_spline_dweight(r)

    safe_dist = np.maximum(dist[neighbors], 1e-14)
    dr_dx = dx[neighbors] / (safe_dist * d_max)
    dr_dy = dy[neighbors] / (safe_dist * d_max)
    zero_mask = dist[neighbors] < 1e-12
    dr_dx[zero_mask] = 0.0
    dr_dy[zero_mask] = 0.0

    dw_dx = dw_dr * dr_dx
    dw_dy = dw_dr * dr_dy

    m = 3  # linear basis
    p_eval = np.array([1.0, x_eval[0], x_eval[1]])
    dp_dx = np.array([0.0, 1.0, 0.0])
    dp_dy = np.array([0.0, 0.0, 1.0])

    A = np.zeros((m, m))
    dA_dx = np.zeros((m, m))
    dA_dy = np.zeros((m, m))
    B = np.zeros((m, n_nb))
    dB_dx = np.zeros((m, n_nb))
    dB_dy = np.zeros((m, n_nb))

    for i_loc in range(n_nb):
        i_glob = neighbors[i_loc]
        p_I = np.array([1.0, nodes[i_glob, 0], nodes[i_glob, 1]])
        pp = np.outer(p_I, p_I)
        A += w[i_loc] * pp
        dA_dx += dw_dx[i_loc] * pp
        dA_dy += dw_dy[i_loc] * pp
        B[:, i_loc] = w[i_loc] * p_I
        dB_dx[:, i_loc] = dw_dx[i_loc] * p_I
        dB_dy[:, i_loc] = dw_dy[i_loc] * p_I

    A_inv = inv(A)
    gamma = A_inv @ p_eval
    phi_nb = gamma @ B

    dgamma_dx = A_inv @ dp_dx - A_inv @ dA_dx @ A_inv @ p_eval
    dgamma_dy = A_inv @ dp_dy - A_inv @ dA_dy @ A_inv @ p_eval
    dphi_nb_dx = dgamma_dx @ B + gamma @ dB_dx
    dphi_nb_dy = dgamma_dy @ B + gamma @ dB_dy

    phi = np.zeros(n_nodes)
    dphi_dx = np.zeros(n_nodes)
    dphi_dy = np.zeros(n_nodes)
    phi[neighbors] = phi_nb
    dphi_dx[neighbors] = dphi_nb_dx
    dphi_dy[neighbors] = dphi_nb_dy

    return phi, dphi_dx, dphi_dy, neighbors

# --- Background integration ---
def gauss_points_2x2():
    g = 1.0 / np.sqrt(3.0)
    pts = np.array([[-g, -g], [g, -g], [g, g], [-g, g]])
    wts = np.array([1.0, 1.0, 1.0, 1.0])
    return pts, wts

def generate_background_cells(ncx, ncy):
    dx_cell = L / ncx
    dy_cell = D / ncy
    cells = []
    for i in range(ncx):
        for j in range(ncy):
            x0 = i * dx_cell
            y0 = -D/2 + j * dy_cell
            cells.append((x0, y0, dx_cell, dy_cell))
    return cells


def efg_solve(nodes, ncx, ncy, d_max_factor=3.0):
    """
    Element-Free Galerkin solver.
    Same problem, same nodes, no mesh -- just a point cloud + MLS.
    """
    print("\n  [EFG] Element-Free Galerkin Solver")
    print("  " + "-"*50)

    n_nodes = len(nodes)
    n_dof = 2 * n_nodes

    cells = generate_background_cells(ncx, ncy)
    n_cells = len(cells)
    print(f"  {n_nodes} nodes, {n_cells} background cells, {n_dof} DOFs")

    t_start = perf_counter()

    # --- Assemble stiffness ---
    K = np.zeros((n_dof, n_dof))
    gp_ref, gw = gauss_points_2x2()

    for ic, (x0, y0, dx_c, dy_c) in enumerate(cells):
        jac = dx_c * dy_c / 4.0
        for igp in range(4):
            xi, eta = gp_ref[igp]
            x_gp = x0 + dx_c * (1 + xi) / 2.0
            y_gp = y0 + dy_c * (1 + eta) / 2.0
            wt = gw[igp] * jac * t_thick

            phi, dphi_dx, dphi_dy, nbrs = compute_mls_shape(
                np.array([x_gp, y_gp]), nodes, d_max_factor
            )

            for I in nbrs:
                B_I = np.array([
                    [dphi_dx[I], 0.0        ],
                    [0.0,        dphi_dy[I] ],
                    [dphi_dy[I], dphi_dx[I] ]
                ])
                for J in nbrs:
                    B_J = np.array([
                        [dphi_dx[J], 0.0        ],
                        [0.0,        dphi_dy[J] ],
                        [dphi_dy[J], dphi_dx[J] ]
                    ])
                    k_ij = B_I.T @ C_mat @ B_J * wt
                    K[2*I,   2*J  ] += k_ij[0, 0]
                    K[2*I,   2*J+1] += k_ij[0, 1]
                    K[2*I+1, 2*J  ] += k_ij[1, 0]
                    K[2*I+1, 2*J+1] += k_ij[1, 1]

    t_assembly = perf_counter() - t_start
    print(f"  Assembly time: {t_assembly:.2f} sec")

    # --- Force vector ---
    F = np.zeros(n_dof)
    tol = 1e-6
    tip_nodes = np.where(np.abs(nodes[:, 0] - L) < tol)[0]
    tip_sorted = tip_nodes[np.argsort(nodes[tip_nodes, 1])]
    y_vals = nodes[tip_sorted, 1]
    n_tip = len(tip_sorted)

    for i, I_node in enumerate(tip_sorted):
        y_I = nodes[I_node, 1]
        if i == 0:
            h_trib = (y_vals[1] - y_vals[0]) / 2.0
        elif i == n_tip - 1:
            h_trib = (y_vals[-1] - y_vals[-2]) / 2.0
        else:
            h_trib = (y_vals[i+1] - y_vals[i-1]) / 2.0
        tau = -P / (2*I_mom) * (D**2/4 - y_I**2)
        F[2*I_node + 1] += tau * h_trib * t_thick

    # --- Penalty BCs at x=0 ---
    alpha = 1.0e15
    fixed_nodes = np.where(np.abs(nodes[:, 0]) < tol)[0]
    for I in fixed_nodes:
        ux_ex, uy_ex = exact_displacement(nodes[I, 0], nodes[I, 1])
        K[2*I, 2*I]     += alpha
        K[2*I+1, 2*I+1] += alpha
        F[2*I]           += alpha * ux_ex
        F[2*I+1]         += alpha * uy_ex

    # --- Solve ---
    t_solve_start = perf_counter()
    U = solve(K, F)
    t_solve = perf_counter() - t_solve_start
    print(f"  Solve time:    {t_solve:.3f} sec")

    # --- Stress recovery at nodes ---
    t_stress_start = perf_counter()
    node_stress = np.zeros((n_nodes, 3))

    for I in range(n_nodes):
        phi, dphi_dx, dphi_dy, nbrs = compute_mls_shape(
            nodes[I], nodes, d_max_factor
        )
        eps = np.zeros(3)
        for J in nbrs:
            eps[0] += dphi_dx[J] * U[2*J]
            eps[1] += dphi_dy[J] * U[2*J+1]
            eps[2] += dphi_dy[J] * U[2*J] + dphi_dx[J] * U[2*J+1]
        node_stress[I] = C_mat @ eps

    t_stress = perf_counter() - t_stress_start
    t_total = perf_counter() - t_start
    print(f"  Stress time:   {t_stress:.2f} sec")
    print(f"  Total time:    {t_total:.2f} sec")

    return U, node_stress


# ############################################################################
#
#  COMPARISON & VISUALIZATION
#
# ############################################################################

def _draw_fixed_bc(ax, x, y, size=1.0, color='#2E7D32'):
    """Draw a fixed-support triangle symbol at a boundary node."""
    from matplotlib.patches import Polygon as MplPoly
    tri = MplPoly([
        [x, y],
        [x - size*0.8, y + size*0.45],
        [x - size*0.8, y - size*0.45]
    ], closed=True, facecolor=color, edgecolor='k',
        linewidth=0.5, alpha=0.7, zorder=5)
    ax.add_patch(tri)


def _draw_ground_hatch(ax, x_pos, y_min, y_max, size=1.0):
    """Draw ground hatching lines behind fixed-support symbols."""
    for yh in np.linspace(y_min - 0.3, y_max + 0.3, 14):
        ax.plot([x_pos - size*1.3, x_pos - size*0.5],
                [yh - size*0.35, yh + size*0.35],
                'k-', lw=0.5, alpha=0.45)
    ax.plot([x_pos - size*0.8, x_pos - size*0.8],
            [y_min - 0.6, y_max + 0.6], 'k-', lw=1.0, alpha=0.6)


def _draw_parabolic_shear(ax, x_pos, y_min, y_max, n_arrows=9,
                          max_len=3.5, color='#D32F2F', label=True):
    """Draw parabolic shear traction arrows at the free end."""
    ys = np.linspace(y_min, y_max, n_arrows)
    tau_max = abs(P) / (2*I_mom) * (D**2/4)
    for y_i in ys:
        tau = abs(-P / (2*I_mom) * (D**2/4 - y_i**2))
        arrow_len = max_len * tau / tau_max
        if arrow_len < 0.15:
            continue
        ax.annotate('', xy=(x_pos, y_i - arrow_len), xytext=(x_pos, y_i),
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=1.4, mutation_scale=10))
    if label:
        ax.text(x_pos + 1.8, 0,
                f'P = {abs(P):.0f} lbf\n(parabolic\nend shear)',
                fontsize=7.5, ha='left', va='center', color=color,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=color, alpha=0.9))


def plot_setup(nodes, nx, ny, ncx, ncy, d_max_factor):
    """
    Plot the undeformed problem setup for both methods:
        Top:    FEM mesh with element edges, nodes, BCs, loads
        Bottom: EFG node cloud with support domain examples, BCs, loads
    """
    from matplotlib.patches import Polygon as MplPoly

    conn = generate_connectivity(nx, ny)
    n_nodes = len(nodes)
    tol = 1e-6

    # EFG support radius
    bbox_vals = nodes.max(axis=0) - nodes.min(axis=0)
    avg_spacing = np.sqrt(bbox_vals[0] * bbox_vals[1] / n_nodes)
    d_max = d_max_factor * avg_spacing

    # Identify BC / load nodes
    fixed_idx = np.where(np.abs(nodes[:, 0]) < tol)[0]
    tip_idx   = np.where(np.abs(nodes[:, 0] - L) < tol)[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle('Problem Setup: Undeformed Discretization with BCs & Loads\n'
                 f'E = {E/1e6:.0f} Mpsi,  $\\nu$ = {nu},  '
                 f'P = {abs(P):.0f} lbf,  Plane Stress,  t = {t_thick:.1f} in',
                 fontsize=14, fontweight='bold', y=0.99)

    # Interior node mask (for distinct coloring)
    interior_mask = (~np.isin(np.arange(n_nodes), fixed_idx) &
                     ~np.isin(np.arange(n_nodes), tip_idx))

    # ==================================================================
    # TOP PANEL: FEM Q4 MESH
    # ==================================================================
    ax = ax1
    ax.set_title(f'Traditional FEM:  {len(conn)} CQUAD4 Elements,  '
                 f'{n_nodes} Nodes,  {2*n_nodes} DOFs',
                 fontsize=11, fontweight='bold', color='#1565C0', pad=8)

    # Draw element faces + edges (the mesh!)
    for el in conn:
        xe = nodes[el]
        poly = MplPoly(list(zip(xe[:, 0], xe[:, 1])), closed=True,
                       facecolor='#E3F2FD', edgecolor='#1565C0',
                       linewidth=0.6, alpha=0.55, zorder=2)
        ax.add_patch(poly)

    # Interior nodes
    ax.scatter(nodes[interior_mask, 0], nodes[interior_mask, 1],
               s=14, c='#1565C0', zorder=4,
               edgecolors='#0D47A1', linewidths=0.4)

    # Fixed end nodes (green squares)
    ax.scatter(nodes[fixed_idx, 0], nodes[fixed_idx, 1],
               s=28, c='#2E7D32', zorder=6, marker='s',
               edgecolors='#1B5E20', linewidths=0.7, label='Fixed nodes')

    # Loaded end nodes (red triangles)
    ax.scatter(nodes[tip_idx, 0], nodes[tip_idx, 1],
               s=28, c='#D32F2F', zorder=6, marker='^',
               edgecolors='#B71C1C', linewidths=0.7, label='Loaded nodes')

    # Fixed BC symbols
    for idx in fixed_idx:
        _draw_fixed_bc(ax, nodes[idx, 0], nodes[idx, 1], size=1.0, color='#2E7D32')
    _draw_ground_hatch(ax, 0, -D/2, D/2, size=1.0)
    ax.text(-3.8, 0, 'Fixed\n($u_x = u_y = 0$)', fontsize=7,
            ha='center', va='center', color='#2E7D32', fontweight='bold',
            rotation=90,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor='#2E7D32', alpha=0.9))

    # Parabolic end shear
    _draw_parabolic_shear(ax, L, -D/2, D/2, max_len=3.0)

    # Dimension lines
    dim_y_pos = -D/2 - 2.2
    ax.annotate('', xy=(0, dim_y_pos), xytext=(L, dim_y_pos),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=1.0))
    ax.text(L/2, dim_y_pos - 0.6, f'L = {L:.0f} in', fontsize=8,
            ha='center', color='gray')

    ax.legend(loc='upper right', fontsize=7.5, framealpha=0.9)
    ax.set_xlim(-6, L + 7)
    ax.set_ylim(-D/2 - 4.5, D/2 + 3)
    ax.set_aspect('equal')
    ax.set_xlabel('x [in]', fontsize=9)
    ax.set_ylabel('y [in]', fontsize=9)
    ax.grid(False)

    # ==================================================================
    # BOTTOM PANEL: EFG NODE CLOUD
    # ==================================================================
    ax = ax2
    ax.set_title(f'EFG Meshless:  {n_nodes} Nodes (no elements),  '
                 f'{2*n_nodes} DOFs,  '
                 f'$d_{{max}}$ = {d_max:.1f} in  ({d_max_factor:.1f}$\\times$spacing)',
                 fontsize=11, fontweight='bold', color='#C62828', pad=8)

    # All interior nodes (just dots, no element edges!)
    ax.scatter(nodes[interior_mask, 0], nodes[interior_mask, 1],
               s=14, c='#C62828', zorder=4,
               edgecolors='#B71C1C', linewidths=0.4)

    # Show a few representative MLS support domains
    demo_ids = [
        int(3 * nx + 5),     # lower-left
        int(4 * nx + 10),    # mid-beam center
        int(6 * nx + 16),    # upper-right
        int(2 * nx + 18),    # lower-right
    ]
    s_colors = ['#FF9800', '#9C27B0', '#00897B', '#5C6BC0']
    for nid, sc in zip(demo_ids, s_colors):
        xn, yn = nodes[nid]
        circle = plt.Circle((xn, yn), d_max, fill=False,
                            edgecolor=sc, linewidth=1.5,
                            linestyle='--', alpha=0.65, zorder=3)
        ax.add_patch(circle)
        ax.scatter([xn], [yn], s=60, c=sc, zorder=7,
                   edgecolors='k', linewidths=0.8, marker='*')

        # Circled neighbor nodes
        dxn = xn - nodes[:, 0]
        dyn = yn - nodes[:, 1]
        dist_n = np.sqrt(dxn**2 + dyn**2)
        nbrs = np.where((dist_n < d_max) & (dist_n > 1e-10))[0]
        ax.scatter(nodes[nbrs, 0], nodes[nbrs, 1], s=20,
                   facecolors='none', edgecolors=sc, linewidths=0.9,
                   zorder=5, alpha=0.5)

    # Callout for one support domain
    ref_nid = demo_ids[1]
    xr, yr = nodes[ref_nid]
    angle = 30 * np.pi / 180
    rx = xr + d_max * np.cos(angle)
    ry = yr + d_max * np.sin(angle)
    ax.annotate(f'$d_{{max}}$ = {d_max:.1f} in',
                xy=(rx, ry), xytext=(rx + 2.5, ry + 1.5),
                fontsize=8, color='#9C27B0', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#9C27B0', lw=1.0),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='#9C27B0', alpha=0.9))

    # Note about meshless
    ax.text(L/2, D/2 + 2.0,
            'No element edges, no connectivity table.  '
            'Nodes interact through overlapping MLS support domains.',
            fontsize=8, ha='center', va='bottom', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4',
                      edgecolor='#FFA000', alpha=0.9))

    # Fixed end nodes
    ax.scatter(nodes[fixed_idx, 0], nodes[fixed_idx, 1],
               s=28, c='#2E7D32', zorder=6, marker='s',
               edgecolors='#1B5E20', linewidths=0.7,
               label='Fixed nodes (penalty)')

    # Loaded end nodes
    ax.scatter(nodes[tip_idx, 0], nodes[tip_idx, 1],
               s=28, c='#D32F2F', zorder=6, marker='^',
               edgecolors='#B71C1C', linewidths=0.7, label='Loaded nodes')

    # Fixed BC symbols
    for idx in fixed_idx:
        _draw_fixed_bc(ax, nodes[idx, 0], nodes[idx, 1], size=1.0, color='#2E7D32')
    _draw_ground_hatch(ax, 0, -D/2, D/2, size=1.0)
    ax.text(-3.8, 0, 'Fixed\n(penalty)', fontsize=7,
            ha='center', va='center', color='#2E7D32', fontweight='bold',
            rotation=90,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor='#2E7D32', alpha=0.9))

    # Parabolic end shear
    _draw_parabolic_shear(ax, L, -D/2, D/2, max_len=3.0)

    # Dimension lines
    ax.annotate('', xy=(0, dim_y_pos), xytext=(L, dim_y_pos),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=1.0))
    ax.text(L/2, dim_y_pos - 0.6, f'L = {L:.0f} in', fontsize=8,
            ha='center', color='gray')

    ax.legend(loc='upper right', fontsize=7.5, framealpha=0.9)
    ax.set_xlim(-6, L + 7)
    ax.set_ylim(-D/2 - 4.5, D/2 + 3)
    ax.set_aspect('equal')
    ax.set_xlabel('x [in]', fontsize=9)
    ax.set_ylabel('y [in]', fontsize=9)
    ax.grid(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(r'C:\Users\pakin\Documents\Python\Python FEA\outputs\problem_setup.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: problem_setup.png")


def compute_errors(label, nodes, U, stress):
    """Compute and print error metrics for either solver."""
    n = len(nodes)
    ux, uy = U[0::2], U[1::2]
    ux_ex, uy_ex = exact_displacement(nodes[:, 0], nodes[:, 1])
    sxx_ex, _, _ = exact_stress(nodes[:, 0], nodes[:, 1])

    # L2 displacement error
    err_u = np.sqrt(np.sum((ux - ux_ex)**2 + (uy - uy_ex)**2))
    norm_u = np.sqrt(np.sum(ux_ex**2 + uy_ex**2))
    rel_err_u = err_u / norm_u if norm_u > 0 else err_u

    # Interior stress error
    interior = ((nodes[:, 0] > L*0.1) & (nodes[:, 0] < L*0.9) &
                (nodes[:, 1] > -D/2*0.8) & (nodes[:, 1] < D/2*0.8))
    err_sxx = np.sqrt(np.mean((stress[interior, 0] - sxx_ex[interior])**2))
    norm_sxx = np.sqrt(np.mean(sxx_ex[interior]**2))
    rel_err_sxx = err_sxx / norm_sxx if norm_sxx > 0 else err_sxx

    # Tip centerline deflection
    tip_center = np.argmin(np.abs(nodes[:, 0] - L) + np.abs(nodes[:, 1]))
    _, uy_tip_exact = exact_displacement(L, 0.0)
    uy_tip = uy[tip_center]
    tip_err = abs(uy_tip - uy_tip_exact) / abs(uy_tip_exact)

    # Tip corner displacement magnitude (for comparison with NASTRAN |u| output)
    # Find nodes at (L, +D/2) and (L, -D/2)
    tol = 1e-6
    tip_top = np.argmin(np.abs(nodes[:, 0] - L) + np.abs(nodes[:, 1] - D/2))
    tip_bot = np.argmin(np.abs(nodes[:, 0] - L) + np.abs(nodes[:, 1] + D/2))

    mag_top = np.sqrt(ux[tip_top]**2 + uy[tip_top]**2)
    mag_bot = np.sqrt(ux[tip_bot]**2 + uy[tip_bot]**2)
    mag_max = max(mag_top, mag_bot)

    # Exact corner magnitude
    ux_corner_ex, uy_corner_ex = exact_displacement(L, D/2)
    mag_corner_exact = np.sqrt(ux_corner_ex**2 + uy_corner_ex**2)
    corner_err = abs(mag_max - mag_corner_exact) / mag_corner_exact

    print(f"\n  [{label}] Accuracy:")
    print(f"    L2 displacement error:  {rel_err_u:.4%}")
    print(f"    L2 stress error (int):  {rel_err_sxx:.4%}")
    print(f"    Tip deflection (y=0):   {uy_tip*1000:.4f} x10^-3 in  "
          f"(exact: {uy_tip_exact*1000:.4f} x10^-3 in, err: {tip_err:.4%})")
    print(f"    Tip corner |u| max:     {mag_max:.6f} in  "
          f"(exact: {mag_corner_exact:.6f} in, err: {corner_err:.4%})")

    return rel_err_u, rel_err_sxx, tip_err


def plot_setup(nodes, nx, ny, ncx, ncy, d_max_factor):
    """
    Plot the undeformed problem setup for both methods side by side:
      - FEM: Q4 mesh with element edges, nodes, BCs, and loads
      - EFG: Node cloud (no connectivity), BCs, and loads

    This is the "what does the solver actually see?" figure.
    """
    conn = generate_connectivity(nx, ny)
    n_nodes = len(nodes)
    x = nodes[:, 0]
    y = nodes[:, 1]
    tol = 1e-6

    # Support radius for EFG
    bbox_dims = nodes.max(axis=0) - nodes.min(axis=0)
    avg_sp = np.sqrt(bbox_dims[0] * bbox_dims[1] / n_nodes)
    d_max = d_max_factor * avg_sp

    # Identify BC / load nodes
    fixed_idx = np.where(np.abs(x) < tol)[0]
    tip_idx = np.where(np.abs(x - L) < tol)[0]
    tip_sorted = tip_idx[np.argsort(y[tip_idx])]

    # Parabolic traction magnitudes at tip nodes (for arrow scaling)
    tau_max = abs(P) / (2 * I_mom) * (D**2 / 4)

    # ---------------------------------------------------------------
    #  Helper: draw BC triangles, ground hatching, load arrows
    # ---------------------------------------------------------------
    def draw_fixed_supports(ax, node_indices, tri_size=0.9):
        """Draw standard fixed-support triangles and ground hatch."""
        for idx in node_indices:
            xi, yi = nodes[idx]
            s = tri_size
            tri = MplPolygon([
                [xi, yi],
                [xi - s * 0.85, yi + s * 0.45],
                [xi - s * 0.85, yi - s * 0.45]
            ], closed=True, facecolor='#2E7D32', edgecolor='k',
                linewidth=0.5, alpha=0.7, zorder=6)
            ax.add_patch(tri)
        # Ground hatch line + hash marks
        y_min_bc = y[node_indices].min()
        y_max_bc = y[node_indices].max()
        hatch_x = -tri_size * 0.85
        ax.plot([hatch_x, hatch_x],
                [y_min_bc - 0.6, y_max_bc + 0.6],
                'k-', lw=1.3, zorder=5)
        for yh in np.linspace(y_min_bc - 0.5, y_max_bc + 0.5, 14):
            ax.plot([hatch_x - tri_size * 0.55, hatch_x],
                    [yh - tri_size * 0.35, yh + tri_size * 0.35],
                    'k-', lw=0.5, alpha=0.55, zorder=5)

    def draw_load_arrows(ax, tip_sorted_idx, arrow_color='#D32F2F'):
        """Draw parabolic shear traction arrows at the free end."""
        y_tip = nodes[tip_sorted_idx, 1]
        for i, idx in enumerate(tip_sorted_idx):
            yi = nodes[idx, 1]
            tau_i = abs(-P / (2 * I_mom) * (D**2 / 4 - yi**2))
            arrow_len = 4.0 * tau_i / tau_max
            if arrow_len < 0.2:
                continue
            # Arrows point downward (net shear is in -y for P < 0)
            ax.annotate(
                '', xy=(L + 0.4, yi - arrow_len),
                xytext=(L + 0.4, yi),
                arrowprops=dict(arrowstyle='->', color=arrow_color,
                                lw=1.6, mutation_scale=11))
        # Resultant label
        ax.text(L + 3.5, 0,
                f'P = {abs(P):.0f} lbf\n(parabolic\nend shear)',
                fontsize=8, ha='left', va='center', color=arrow_color,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=arrow_color, alpha=0.92))

    def draw_dimensions(ax):
        """Draw L and D dimension annotations."""
        dim_y = -D / 2 - 2.2
        ax.annotate('', xy=(0, dim_y), xytext=(L, dim_y),
                    arrowprops=dict(arrowstyle='<->', color='gray', lw=1.0))
        ax.text(L / 2, dim_y - 0.7, f'L = {L:.0f} in',
                fontsize=8.5, ha='center', color='gray')
        dim_x = -3.2
        ax.annotate('', xy=(dim_x, -D / 2), xytext=(dim_x, D / 2),
                    arrowprops=dict(arrowstyle='<->', color='gray', lw=1.0))
        ax.text(dim_x - 1.3, 0, f'D = {D:.0f} in', fontsize=8.5,
                ha='center', va='center', color='gray', rotation=90)

    # ---------------------------------------------------------------
    #  Build the figure
    # ---------------------------------------------------------------
    fig, (ax_fem, ax_efg) = plt.subplots(2, 1, figsize=(16, 11))
    fig.suptitle(
        'Problem Setup: Undeformed Discretization with BCs & Loads\n'
        f'E = {E/1e6:.0f} Mpsi,  $\\nu$ = {nu},  '
        f'P = {P:.0f} lbf,  Plane Stress,  t = {t_thick:.1f} in',
        fontsize=13, fontweight='bold')

    # ==================================================================
    #  TOP: FEM mesh
    # ==================================================================
    ax = ax_fem
    ax.set_title(
        f'Traditional FEM:  {(nx-1)*(ny-1)} CQUAD4 Elements,  '
        f'{n_nodes} Nodes,  {2*n_nodes} DOFs',
        fontsize=11, fontweight='bold', color='#1565C0', pad=8)

    # Element faces (light fill + edges)
    for el in conn:
        xe = nodes[el]
        poly = MplPolygon(
            list(zip(xe[:, 0], xe[:, 1])), closed=True,
            facecolor='#E3F2FD', edgecolor='#1565C0',
            linewidth=0.6, alpha=0.55, zorder=2)
        ax.add_patch(poly)

    # Interior nodes
    interior_mask = ((x > tol) & (x < L - tol))
    ax.scatter(x[interior_mask], y[interior_mask], s=14, c='#1565C0',
               edgecolors='#0D47A1', linewidths=0.4, zorder=4)

    # Fixed-end nodes (green, larger)
    ax.scatter(x[fixed_idx], y[fixed_idx], s=36, c='#2E7D32',
               edgecolors='k', linewidths=0.6, zorder=7, label='Fixed nodes')

    # Tip nodes (red, larger)
    ax.scatter(x[tip_sorted], y[tip_sorted], s=36, c='#D32F2F',
               edgecolors='k', linewidths=0.6, zorder=7, label='Loaded nodes')

    # BCs, loads, dims
    draw_fixed_supports(ax, fixed_idx)
    draw_load_arrows(ax, tip_sorted)
    draw_dimensions(ax)

    # BC label
    ax.text(-4.3, 0, 'Fixed\n($u_x=u_y=0$)', fontsize=7.5, ha='center',
            va='center', color='#2E7D32', fontweight='bold', rotation=90,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor='#2E7D32', alpha=0.9))

    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.set_xlim(-6, L + 10)
    ax.set_ylim(-D / 2 - 4, D / 2 + 3)
    ax.set_aspect('equal')
    ax.set_xlabel('x [in]')
    ax.set_ylabel('y [in]')
    ax.grid(False)

    # ==================================================================
    #  BOTTOM: EFG node cloud
    # ==================================================================
    ax = ax_efg
    ax.set_title(
        f'EFG Meshless:  {n_nodes} Nodes (no elements),  '
        f'{2*n_nodes} DOFs,  '
        f'$d_{{max}}$ = {d_max:.1f} in  ({d_max_factor:.1f}$\\times$spacing)',
        fontsize=11, fontweight='bold', color='#C62828', pad=8)

    # Nodes as a bare point cloud -- the ONLY thing the EFG solver "sees"
    interior_mask_efg = ((x > tol) & (x < L - tol))
    ax.scatter(x[interior_mask_efg], y[interior_mask_efg], s=22, c='#C62828',
               edgecolors='#B71C1C', linewidths=0.4, zorder=4)

    # Fixed-end nodes
    ax.scatter(x[fixed_idx], y[fixed_idx], s=40, c='#2E7D32',
               edgecolors='k', linewidths=0.6, zorder=7, label='Fixed nodes (penalty)')

    # Tip nodes
    ax.scatter(x[tip_sorted], y[tip_sorted], s=40, c='#D32F2F',
               edgecolors='k', linewidths=0.6, zorder=7, label='Loaded nodes')

    # Show a few example MLS support domains (dashed circles)
    demo_nodes = [
        int(4 * nx + 3),        # left-center interior
        int(4 * nx + 10),       # mid-beam
        int(4 * nx + nx - 4),   # right interior
        int(1 * nx + 10),       # lower
        int(7 * nx + 10),       # upper
    ]
    for nid in demo_nodes:
        xn, yn = nodes[nid]
        circle = plt.Circle((xn, yn), d_max, fill=False,
                            edgecolor='#FF9800', linewidth=1.2,
                            linestyle='--', alpha=0.45, zorder=3)
        ax.add_patch(circle)
        # highlight center
        ax.scatter([xn], [yn], s=55, marker='*', c='#FF9800',
                   edgecolors='k', linewidths=0.5, zorder=8)

    # MLS domain label (one callout)
    demo_x, demo_y = nodes[demo_nodes[1]]
    ax.annotate(
        f'MLS support\n$d_{{max}}$ = {d_max:.1f} in',
        xy=(demo_x + d_max * 0.7, demo_y + d_max * 0.7),
        xytext=(demo_x + d_max + 3, demo_y + d_max + 1.5),
        fontsize=7.5, ha='left', color='#FF9800', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='#FF9800', lw=1.2),
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                  edgecolor='#FF9800', alpha=0.9))

    # BCs, loads, dims
    draw_fixed_supports(ax, fixed_idx)
    draw_load_arrows(ax, tip_sorted)
    draw_dimensions(ax)

    ax.text(-4.3, 0, 'Fixed\n(penalty)', fontsize=7.5, ha='center',
            va='center', color='#2E7D32', fontweight='bold', rotation=90,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor='#2E7D32', alpha=0.9))

    # "No mesh" annotation
    ax.text(L / 2, D / 2 + 1.5,
            'No element edges, no connectivity table.  '
            'Nodes interact through overlapping MLS support domains.',
            fontsize=8, ha='center', va='bottom', style='italic',
            color='#616161',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4',
                      edgecolor='#FFB74D', alpha=0.85))

    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.set_xlim(-6, L + 10)
    ax.set_ylim(-D / 2 - 4, D / 2 + 3)
    ax.set_aspect('equal')
    ax.set_xlabel('x [in]')
    ax.set_ylabel('y [in]')
    ax.grid(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(r'C:\Users\pakin\Documents\Python\Python FEA\outputs\problem_setup.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: problem_setup.png")


def plot_comparison(nodes, U_fem, stress_fem, U_efg, stress_efg):
    """Generate comprehensive comparison plots."""
    x = nodes[:, 0]
    y = nodes[:, 1]

    # Exact at nodes
    ux_ex, uy_ex = exact_displacement(x, y)
    sxx_ex, _, sxy_ex = exact_stress(x, y)

    # Triangulation for contours (purely visualization, not used by either solver)
    tri = Triangulation(x, y)

    # ========================= FIGURE 1: Contour comparison =================
    fig1, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig1.suptitle(
        'Meshless vs. Mesh-Based FEA: Cantilever Beam Benchmark\n'
        f'E = {E/1e6:.0f} Mpsi, $\\nu$ = {nu}, P = {P:.0f} lbf, '
        f'L = {L:.0f} in, D = {D:.0f} in, {len(nodes)} nodes',
        fontsize=13, fontweight='bold'
    )

    # Compute shared color limits for fair comparison
    uy_fem = U_fem[1::2]
    uy_efg = U_efg[1::2]
    uy_all = np.concatenate([uy_fem, uy_efg, uy_ex])
    uy_min, uy_max = uy_all.min(), uy_all.max()

    sxx_all = np.concatenate([stress_fem[:, 0], stress_efg[:, 0], sxx_ex])
    sxx_min, sxx_max = sxx_all.min(), sxx_all.max()

    sxy_all = np.concatenate([stress_fem[:, 2], stress_efg[:, 2], sxy_ex])
    sxy_min, sxy_max = sxy_all.min(), sxy_all.max()

    plot_data = [
        # Row 0: Exact
        (0, 0, uy_ex,            uy_min, uy_max,   'Exact: $u_y$ [in]',            'RdBu_r'),
        (0, 1, sxx_ex,           sxx_min, sxx_max, 'Exact: $\\sigma_{xx}$ [psi]',  'RdBu_r'),
        (0, 2, sxy_ex,           sxy_min, sxy_max, 'Exact: $\\tau_{xy}$ [psi]',    'RdBu_r'),
        # Row 1: FEM
        (1, 0, uy_fem,           uy_min, uy_max,   'FEM (Q4): $u_y$ [in]',         'RdBu_r'),
        (1, 1, stress_fem[:, 0], sxx_min, sxx_max, 'FEM (Q4): $\\sigma_{xx}$ [psi]','RdBu_r'),
        (1, 2, stress_fem[:, 2], sxy_min, sxy_max, 'FEM (Q4): $\\tau_{xy}$ [psi]', 'RdBu_r'),
        # Row 2: EFG
        (2, 0, uy_efg,           uy_min, uy_max,   'EFG: $u_y$ [in]',              'RdBu_r'),
        (2, 1, stress_efg[:, 0], sxx_min, sxx_max, 'EFG: $\\sigma_{xx}$ [psi]',    'RdBu_r'),
        (2, 2, stress_efg[:, 2], sxy_min, sxy_max, 'EFG: $\\tau_{xy}$ [psi]',      'RdBu_r'),
    ]

    for (row, col, data, vmin, vmax, title, cmap) in plot_data:
        ax = axes[row, col]
        cf = ax.tricontourf(tri, data, levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlabel('x [in]', fontsize=8)
        ax.set_ylabel('y [in]', fontsize=8)
        ax.tick_params(labelsize=7)
        fig1.colorbar(cf, ax=ax, shrink=0.7)

    plt.tight_layout()
    plt.savefig(r'C:\Users\pakin\Documents\Python\Python FEA\outputs\comparison_contours.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    # ======================== FIGURE 2: Line plots ==========================
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5.5))
    fig2.suptitle('Quantitative Comparison: FEM vs EFG vs Exact vs Beam Theory',
                  fontsize=13, fontweight='bold')

    # --- 2a: Centerline deflection ---
    ax = axes2[0]
    center_mask = np.abs(y) < (D / 20)
    x_c = x[center_mask]
    sort_c = np.argsort(x_c)

    x_exact = np.linspace(0, L, 200)
    _, uy_exact_line = exact_displacement(x_exact, np.zeros_like(x_exact))
    uy_beam_line = beam_theory_deflection(x_exact)

    ax.plot(x_exact, uy_exact_line * 1000, 'k-', lw=2.5,
            label='Exact (Timoshenko)', zorder=1)
    ax.plot(x_exact, uy_beam_line * 1000, '--', color='#2E7D32', lw=2.0,
            label='Beam Theory (E-B)', zorder=1)
    ax.plot(x_c[sort_c], uy_fem[center_mask][sort_c] * 1000,
            's', color='#2196F3', ms=7, mfc='none', mew=1.5, label='FEM (Q4)', zorder=3)
    ax.plot(x_c[sort_c], uy_efg[center_mask][sort_c] * 1000,
            'o', color='#F44336', ms=6, label='EFG', zorder=2)
    ax.set_title('Centerline Deflection $u_y(x, 0)$')
    ax.set_xlabel('x [in]')
    ax.set_ylabel('$u_y$ [$\\times 10^{-3}$ in]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Annotate the shear deformation gap at the tip
    uy_tip_exact = uy_exact_line[-1] * 1000
    uy_tip_beam = uy_beam_line[-1] * 1000
    mid_y = (uy_tip_exact + uy_tip_beam) / 2
    ax.annotate('', xy=(L - 0.5, uy_tip_exact), xytext=(L - 0.5, uy_tip_beam),
                arrowprops=dict(arrowstyle='<->', color='#2E7D32', lw=1.2))
    ax.text(L - 4, mid_y,
            f'Shear\ncorrection\n({abs(uy_tip_exact - uy_tip_beam):.2f})',
            fontsize=7, ha='right', va='center', color='#2E7D32',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='#2E7D32', alpha=0.9))

    # --- 2b: Bending stress at x = L/2 ---
    ax = axes2[1]
    mid_mask = np.abs(x - L/2) < (L / 30)
    y_m = y[mid_mask]
    sort_m = np.argsort(y_m)

    y_exact = np.linspace(-D/2, D/2, 200)
    sxx_exact_line, _, _ = exact_stress(L/2 * np.ones_like(y_exact), y_exact)

    # Beam theory sigma_xx = My/I = P(L-x)y/I -- identical to elasticity
    sxx_beam_line = P * (L - L/2) * y_exact / I_mom

    ax.plot(sxx_exact_line, y_exact, 'k-', lw=2.5, label='Exact (Timoshenko)', zorder=1)
    ax.plot(sxx_beam_line, y_exact, '--', color='#2E7D32', lw=2.0, dashes=(6, 3),
            label='Beam Theory ($My/I$)', zorder=1)
    ax.plot(stress_fem[mid_mask, 0][sort_m], y_m[sort_m],
            's', color='#2196F3', ms=7, mfc='none', mew=1.5, label='FEM (Q4)', zorder=3)
    ax.plot(stress_efg[mid_mask, 0][sort_m], y_m[sort_m],
            'o', color='#F44336', ms=6, label='EFG', zorder=2)
    ax.set_title('Bending Stress at $x = L/2$')
    ax.set_xlabel('$\\sigma_{xx}$ [psi]')
    ax.set_ylabel('y [in]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Note that beam theory stresses are exact
    ax.text(0.98, 0.02, '$\\sigma_{xx}$: Beam theory\n= Elasticity (identical)',
            transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
            color='#2E7D32', style='italic',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='#2E7D32', alpha=0.85))

    # --- 2c: Shear stress at x = L/2 ---
    ax = axes2[2]
    _, _, sxy_exact_line = exact_stress(L/2 * np.ones_like(y_exact), y_exact)

    # Beam theory tau = -VQ/(Ib) = -P/(2I)*(D^2/4 - y^2) -- identical to elasticity
    sxy_beam_line = -P / (2*I_mom) * (D**2/4 - y_exact**2)

    ax.plot(sxy_exact_line, y_exact, 'k-', lw=2.5, label='Exact (Timoshenko)', zorder=1)
    ax.plot(sxy_beam_line, y_exact, '--', color='#2E7D32', lw=2.0, dashes=(6, 3),
            label='Beam Theory ($VQ/Ib$)', zorder=1)
    ax.plot(stress_fem[mid_mask, 2][sort_m], y_m[sort_m],
            's', color='#2196F3', ms=7, mfc='none', mew=1.5, label='FEM (Q4)', zorder=3)
    ax.plot(stress_efg[mid_mask, 2][sort_m], y_m[sort_m],
            'o', color='#F44336', ms=6, label='EFG', zorder=2)
    ax.set_title('Shear Stress at $x = L/2$')
    ax.set_xlabel('$\\tau_{xy}$ [psi]')
    ax.set_ylabel('y [in]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Note that beam theory shear stress is exact
    ax.text(0.98, 0.02, '$\\tau_{xy}$: Beam theory\n= Elasticity (identical)',
            transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
            color='#2E7D32', style='italic',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='#2E7D32', alpha=0.85))

    plt.tight_layout()
    plt.savefig(r'C:\Users\pakin\Documents\Python\Python FEA\outputs\comparison_line_plots.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    # ======================== FIGURE 3: Error distributions ==================
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
    fig3.suptitle('Error Distribution: |Numerical - Exact|',
                  fontsize=13, fontweight='bold')

    # Displacement error
    err_uy_fem = np.abs(uy_fem - uy_ex)
    err_uy_efg = np.abs(uy_efg - uy_ex)
    vmax_uy = max(err_uy_fem.max(), err_uy_efg.max())

    ax = axes3[0]
    cf = ax.tricontourf(tri, err_uy_fem, levels=20, cmap='hot_r', vmin=0, vmax=vmax_uy)
    ax.set_title('FEM: $|u_y - u_{y,exact}|$ [in]', fontsize=10)
    ax.set_aspect('equal')
    ax.set_xlabel('x [in]', fontsize=8)
    fig3.colorbar(cf, ax=ax, shrink=0.7)

    ax = axes3[1]
    cf = ax.tricontourf(tri, err_uy_efg, levels=20, cmap='hot_r', vmin=0, vmax=vmax_uy)
    ax.set_title('EFG: $|u_y - u_{y,exact}|$ [in]', fontsize=10)
    ax.set_aspect('equal')
    ax.set_xlabel('x [in]', fontsize=8)
    fig3.colorbar(cf, ax=ax, shrink=0.7)

    # Stress error (sigma_xx)
    err_sxx_fem = np.abs(stress_fem[:, 0] - sxx_ex)
    err_sxx_efg = np.abs(stress_efg[:, 0] - sxx_ex)
    vmax_sxx = max(err_sxx_fem.max(), err_sxx_efg.max())

    ax = axes3[2]
    # Bar chart of peak errors
    metrics = ['$u_y$ L2 err', '$\\sigma_{xx}$ L2 err', 'Tip $u_y$ err']

    # Recompute for the bar chart
    ux_fem, ux_efg = U_fem[0::2], U_efg[0::2]

    def rel_l2_u(ux_num, uy_num):
        err = np.sqrt(np.sum((ux_num - ux_ex)**2 + (uy_num - uy_ex)**2))
        norm = np.sqrt(np.sum(ux_ex**2 + uy_ex**2))
        return err / norm * 100

    def rel_l2_sxx(s_num):
        interior = ((nodes[:, 0] > L*0.1) & (nodes[:, 0] < L*0.9) &
                    (nodes[:, 1] > -D/2*0.8) & (nodes[:, 1] < D/2*0.8))
        err = np.sqrt(np.mean((s_num[interior] - sxx_ex[interior])**2))
        norm = np.sqrt(np.mean(sxx_ex[interior]**2))
        return err / norm * 100

    def tip_err(uy_num):
        tip = np.argmin(np.abs(nodes[:, 0] - L) + np.abs(nodes[:, 1]))
        _, uy_tip_exact = exact_displacement(L, 0.0)
        return abs(uy_num[tip] - uy_tip_exact) / abs(uy_tip_exact) * 100

    fem_vals = [rel_l2_u(ux_fem, uy_fem), rel_l2_sxx(stress_fem[:, 0]), tip_err(uy_fem)]
    efg_vals = [rel_l2_u(ux_efg, uy_efg), rel_l2_sxx(stress_efg[:, 0]), tip_err(uy_efg)]

    # Beam theory 2D displacement field for error computation:
    #   ux_beam = -y * dv/dx = -y * Px(2L-x)/(2EI)  (plane sections remain plane)
    #   uy_beam = v(x)       = Px^2(3L-x)/(6EI)      (centerline deflection at all y)
    ux_beam = -nodes[:, 1] * beam_theory_slope(nodes[:, 0])
    uy_beam = beam_theory_deflection(nodes[:, 0])

    # Beam theory stresses are identical to exact, so stress L2 error = 0
    # (for this particular load case -- not true in general!)
    beam_tip = beam_theory_deflection(L)
    _, uy_tip_exact_val = exact_displacement(L, 0.0)
    beam_tip_err = abs(beam_tip - uy_tip_exact_val) / abs(uy_tip_exact_val) * 100
    beam_vals = [rel_l2_u(ux_beam, uy_beam), 0.0, beam_tip_err]

    x_bar = np.arange(len(metrics))
    width = 0.25
    bars1 = ax.bar(x_bar - width, fem_vals, width, label='FEM (Q4)',
                   color='#2196F3', alpha=0.8)
    bars2 = ax.bar(x_bar,         efg_vals, width, label='EFG',
                   color='#F44336', alpha=0.8)
    bars3 = ax.bar(x_bar + width, beam_vals, width, label='Beam Theory',
                   color='#4CAF50', alpha=0.8)

    # Value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.05,
                        f'{h:.2f}%', ha='center', va='bottom', fontsize=7.5)
            else:
                ax.text(bar.get_x() + bar.get_width()/2., 0.08,
                        '0%', ha='center', va='bottom', fontsize=7.5,
                        color='#4CAF50', fontweight='bold')

    ax.set_ylabel('Relative Error [%]')
    ax.set_title('Summary: Error Comparison')
    ax.set_xticks(x_bar)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(r'C:\Users\pakin\Documents\Python\Python FEA\outputs\comparison_errors.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    print("\n  Plots saved:")
    print("    problem_setup.png         - Undeformed mesh/cloud + BCs + loads")
    print("    comparison_contours.png   - 3x3 contour grid")
    print("    comparison_line_plots.png - Centerline & stress profiles")
    print("    comparison_errors.png     - Error distributions & summary")
    print("    deformed_shapes.png       - FEM/EFG/Exact deformed shapes")


def plot_deformed(nodes, nx, ny, U_fem, U_efg, scale=50.0):
    """
    Plot deformed shapes for FEM, EFG, and Exact solutions.
    Top:    FEM deformed mesh with element edges
    Middle: EFG deformed node cloud
    Bottom: Exact deformed shape
    """
    from matplotlib.patches import Polygon as MplPoly

    conn = generate_connectivity(nx, ny)
    n_nodes = len(nodes)
    x = nodes[:, 0]
    y = nodes[:, 1]

    # Displacements
    ux_fem, uy_fem = U_fem[0::2], U_fem[1::2]
    ux_efg, uy_efg = U_efg[0::2], U_efg[1::2]
    ux_ex, uy_ex = exact_displacement(x, y)

    # Deformed coordinates
    xd_fem = x + scale * ux_fem
    yd_fem = y + scale * uy_fem
    xd_efg = x + scale * ux_efg
    yd_efg = y + scale * uy_efg
    xd_ex  = x + scale * ux_ex
    yd_ex  = y + scale * uy_ex

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 15))
    fig.suptitle(f'Deformed Shape Comparison  (displacement magnified {scale:.0f}$\\times$)\n'
                 f'E = {E/1e6:.0f} Mpsi,  $\\nu$ = {nu},  '
                 f'P = {P:.0f} lbf,  {n_nodes} nodes',
                 fontsize=14, fontweight='bold', y=0.99)

    # Shared axis limits (use exact deformed extents + padding)
    all_xd = np.concatenate([xd_fem, xd_efg, xd_ex, x])
    all_yd = np.concatenate([yd_fem, yd_efg, yd_ex, y])
    xlim = (min(all_xd.min(), x.min()) - 3, max(all_xd.max(), x.max()) + 3)
    ylim = (min(all_yd.min(), y.min()) - 3, max(all_yd.max(), y.max()) + 3)

    # ==================================================================
    # TOP: FEM deformed mesh
    # ==================================================================
    ax = ax1
    ax.set_title('FEM (Q4): Deformed Mesh',
                 fontsize=11, fontweight='bold', color='#1565C0', pad=8)

    # Undeformed mesh (faded)
    for el in conn:
        xe_u = nodes[el]
        verts_u = list(zip(xe_u[:, 0], xe_u[:, 1]))
        poly_u = MplPoly(verts_u, closed=True,
                         facecolor='none', edgecolor='#90CAF9',
                         linewidth=0.4, alpha=0.5, zorder=1)
        ax.add_patch(poly_u)

    # Undeformed nodes
    ax.scatter(x, y, s=8, c='#90CAF9', zorder=2, alpha=0.5)

    # Deformed mesh
    nodes_d_fem = np.column_stack([xd_fem, yd_fem])
    for el in conn:
        xe_d = nodes_d_fem[el]
        verts_d = list(zip(xe_d[:, 0], xe_d[:, 1]))
        poly_d = MplPoly(verts_d, closed=True,
                         facecolor='#E3F2FD', edgecolor='#1565C0',
                         linewidth=0.7, alpha=0.6, zorder=3)
        ax.add_patch(poly_d)

    # Deformed nodes colored by displacement magnitude
    disp_mag_fem = np.sqrt(ux_fem**2 + uy_fem**2)
    sc1 = ax.scatter(xd_fem, yd_fem, s=14, c=disp_mag_fem, cmap='hot_r',
                     zorder=4, edgecolors='#0D47A1', linewidths=0.3)
    fig.colorbar(sc1, ax=ax, shrink=0.6, label='|u| [in]', pad=0.02)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_xlabel('x [in]', fontsize=9)
    ax.set_ylabel('y [in]', fontsize=9)
    ax.grid(False)

    # Legend
    ax.scatter([], [], s=10, c='#90CAF9', label='Undeformed')
    ax.scatter([], [], s=10, c='#1565C0', label=f'Deformed ($\\times${scale:.0f})')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

    # ==================================================================
    # MIDDLE: EFG deformed node cloud
    # ==================================================================
    ax = ax2
    ax.set_title('EFG Meshless: Deformed Node Cloud',
                 fontsize=11, fontweight='bold', color='#C62828', pad=8)

    # Undeformed nodes (faded)
    ax.scatter(x, y, s=8, c='#FFCDD2', zorder=1, alpha=0.5)

    # Deformed nodes colored by displacement magnitude
    disp_mag_efg = np.sqrt(ux_efg**2 + uy_efg**2)
    sc2 = ax.scatter(xd_efg, yd_efg, s=14, c=disp_mag_efg, cmap='hot_r',
                     zorder=4, edgecolors='#B71C1C', linewidths=0.3)
    fig.colorbar(sc2, ax=ax, shrink=0.6, label='|u| [in]', pad=0.02)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_xlabel('x [in]', fontsize=9)
    ax.set_ylabel('y [in]', fontsize=9)
    ax.grid(False)

    ax.scatter([], [], s=10, c='#FFCDD2', label='Undeformed')
    ax.scatter([], [], s=10, c='#C62828', label=f'Deformed ($\\times${scale:.0f})')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

    # ==================================================================
    # BOTTOM: Exact deformed shape
    # ==================================================================
    ax = ax3
    ax.set_title('Exact (Timoshenko Elasticity): Deformed Shape',
                 fontsize=11, fontweight='bold', color='#2E7D32', pad=8)

    # Undeformed
    ax.scatter(x, y, s=8, c='#C8E6C9', zorder=1, alpha=0.5)

    # Exact deformed
    disp_mag_ex = np.sqrt(ux_ex**2 + uy_ex**2)
    sc3 = ax.scatter(xd_ex, yd_ex, s=14, c=disp_mag_ex, cmap='hot_r',
                     zorder=4, edgecolors='#1B5E20', linewidths=0.3)
    fig.colorbar(sc3, ax=ax, shrink=0.6, label='|u| [in]', pad=0.02)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_xlabel('x [in]', fontsize=9)
    ax.set_ylabel('y [in]', fontsize=9)
    ax.grid(False)

    ax.scatter([], [], s=10, c='#C8E6C9', label='Undeformed')
    ax.scatter([], [], s=10, c='#2E7D32', label=f'Deformed ($\\times${scale:.0f})')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(r'C:\Users\pakin\Documents\Python\Python FEA\outputs\deformed_shapes.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: deformed_shapes.png")


def print_centerline_table(nodes, U_fem, stress_fem, U_efg, stress_efg):
    """
    Print tabulated centerline results at every L/8 for easy comparison
    with Simcenter NASTRAN nodal queries.

    Reports: uy deflection and sigma_xx bending stress
    for Exact, Beam Theory, FEM, and EFG.

    NOTE: The node grid may not land exactly on L/8 stations.
          Table shows actual node x-position used and computes
          the exact/beam solutions at that same x for fair comparison.
    """
    ux_fem, uy_fem = U_fem[0::2], U_fem[1::2]
    ux_efg, uy_efg = U_efg[0::2], U_efg[1::2]

    # Station targets: 0, L/8, 2L/8, ..., L
    n_stations = 9
    x_targets = np.array([i * L / 8.0 for i in range(n_stations)])

    # Find nearest centerline and top-fiber nodes to each station
    center_nodes = []     # (station_label, node_idx, actual_x)
    top_nodes = []

    for i, xs in enumerate(x_targets):
        # Centerline (y=0): heavily penalize y-offset
        dist_c = np.abs(nodes[:, 0] - xs) + np.abs(nodes[:, 1]) * 10
        idx_c = np.argmin(dist_c)
        center_nodes.append((i, idx_c, nodes[idx_c, 0], nodes[idx_c, 1]))

        # Top fiber (y=+D/2)
        dist_t = np.abs(nodes[:, 0] - xs) + np.abs(nodes[:, 1] - D/2) * 10
        idx_t = np.argmin(dist_t)
        top_nodes.append((i, idx_t, nodes[idx_t, 0], nodes[idx_t, 1]))

    print("\n" + "="*108)
    print("  CENTERLINE RESULTS AT EVERY L/8  (nearest node to target station)")
    print("="*108)

    # --- Deflection table (centerline, y~0) ---
    print(f"\n  {'DEFLECTION uy [x10^-3 in] along neutral axis (y ~ 0)':^104}")
    print(f"  {'-'*104}")
    hdr = (f"  {'Station':>8}  {'x_node':>8}  {'y_node':>6}  "
           f"{'Exact':>10}  {'Beam(E-B)':>10}  "
           f"{'FEM(Q4)':>10}  {'FEM err':>8}  "
           f"{'EFG':>10}  {'EFG err':>8}")
    print(hdr)
    print(f"  {'-'*104}")

    for i, idx, xn, yn in center_nodes:
        # Evaluate exact and beam theory at the ACTUAL node position
        _, uy_ex = exact_displacement(xn, yn)
        uy_bt = beam_theory_deflection(xn)

        uy_f = uy_fem[idx]
        uy_e = uy_efg[idx]

        if abs(uy_ex) > 1e-15:
            err_f = (uy_f - uy_ex) / abs(uy_ex) * 100
            err_e = (uy_e - uy_ex) / abs(uy_ex) * 100
        else:
            err_f = 0.0
            err_e = 0.0

        label = f"{i}/8 L"
        print(f"  {label:>8}  {xn:8.2f}  {yn:6.2f}  "
              f"{uy_ex*1000:10.4f}  {uy_bt*1000:10.4f}  "
              f"{uy_f*1000:10.4f}  {err_f:+7.2f}%  "
              f"{uy_e*1000:10.4f}  {err_e:+7.2f}%")

    print(f"  {'-'*104}")

    # --- Bending stress table (top fiber, y~+D/2) ---
    print(f"\n  {'BENDING STRESS sigma_xx [psi] at top fiber (y ~ +D/2)':^104}")
    print(f"  {'-'*104}")
    hdr2 = (f"  {'Station':>8}  {'x_node':>8}  {'y_node':>6}  "
            f"{'Exact':>10}  {'Beam(My/I)':>10}  "
            f"{'FEM(Q4)':>10}  {'FEM err':>8}  "
            f"{'EFG':>10}  {'EFG err':>8}")
    print(hdr2)
    print(f"  {'-'*104}")

    for i, idx, xn, yn in top_nodes:
        sxx_ex, _, _ = exact_stress(xn, yn)
        sxx_bt = P * (L - xn) * yn / I_mom

        sxx_f = stress_fem[idx, 0]
        sxx_e = stress_efg[idx, 0]

        if abs(sxx_ex) > 1e-10:
            err_f = (sxx_f - sxx_ex) / abs(sxx_ex) * 100
            err_e = (sxx_e - sxx_ex) / abs(sxx_ex) * 100
        else:
            err_f = 0.0
            err_e = 0.0

        label = f"{i}/8 L"
        print(f"  {label:>8}  {xn:8.2f}  {yn:6.2f}  "
              f"{sxx_ex:10.2f}  {sxx_bt:10.2f}  "
              f"{sxx_f:10.2f}  {err_f:+7.2f}%  "
              f"{sxx_e:10.2f}  {err_e:+7.2f}%")

    print(f"  {'-'*104}")

    # --- Shear stress table (neutral axis, y~0) ---
    print(f"\n  {'SHEAR STRESS tau_xy [psi] at neutral axis (y ~ 0)':^104}")
    print(f"  {'-'*104}")
    hdr3 = (f"  {'Station':>8}  {'x_node':>8}  {'y_node':>6}  "
            f"{'Exact':>10}  {'Beam(VQ/Ib)':>10}  "
            f"{'FEM(Q4)':>10}  {'FEM err':>8}  "
            f"{'EFG':>10}  {'EFG err':>8}")
    print(hdr3)
    print(f"  {'-'*104}")

    for i, idx, xn, yn in center_nodes:
        _, _, sxy_ex = exact_stress(xn, yn)
        sxy_bt = -P / (2*I_mom) * (D**2/4 - yn**2)

        sxy_f = stress_fem[idx, 2]
        sxy_e = stress_efg[idx, 2]

        if abs(sxy_ex) > 1e-10:
            err_f = (sxy_f - sxy_ex) / abs(sxy_ex) * 100
            err_e = (sxy_e - sxy_ex) / abs(sxy_ex) * 100
        else:
            err_f = 0.0
            err_e = 0.0

        label = f"{i}/8 L"
        print(f"  {label:>8}  {xn:8.2f}  {yn:6.2f}  "
              f"{sxy_ex:10.2f}  {sxy_bt:10.2f}  "
              f"{sxy_f:10.2f}  {err_f:+7.2f}%  "
              f"{sxy_e:10.2f}  {err_e:+7.2f}%")

    print(f"  {'-'*104}")
    print()


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("="*60)
    print("  MESHLESS vs MESH-BASED FEA: HEAD-TO-HEAD COMPARISON")
    print("  Cantilever Beam - Timoshenko Exact Benchmark")
    print("="*60)

    # --- Shared node grid ---
    nx, ny = 21, 9    # 189 nodes
    ncx, ncy = 20, 8  # EFG background cells (same density as FEM elements)
    d_max_factor = 3.0

    nodes = generate_nodes(nx, ny)
    print(f"\n  Shared grid: {nx} x {ny} = {len(nodes)} nodes")
    print(f"  FEM elements: {(nx-1)*(ny-1)}")
    print(f"  EFG background cells: {ncx*ncy}")

    # --- Pre-solve visualization: show what each method "sees" ---
    print("\n  Generating problem setup figure...")
    plot_setup(nodes, nx, ny, ncx, ncy, d_max_factor)

    # --- Run both solvers ---
    U_fem, stress_fem = fem_solve(nodes, nx, ny)
    U_efg, stress_efg = efg_solve(nodes, ncx, ncy, d_max_factor)

    # --- Error analysis ---
    print("\n" + "="*60)
    print("  ACCURACY COMPARISON (vs. Timoshenko Exact)")
    print("="*60)
    compute_errors("FEM", nodes, U_fem, stress_fem)
    compute_errors("EFG", nodes, U_efg, stress_efg)

    _, uy_tip_exact = exact_displacement(L, 0.0)
    ux_corner, uy_corner = exact_displacement(L, D/2)
    mag_corner = np.sqrt(ux_corner**2 + uy_corner**2)
    uy_tip_beam = beam_theory_deflection(L)
    beam_tip_err = abs(uy_tip_beam - uy_tip_exact) / abs(uy_tip_exact)
    print(f"\n  [Beam Theory] Euler-Bernoulli Reference:")
    print(f"    Tip deflection:         {uy_tip_beam*1000:.4f} x10^-3 in  "
          f"(exact: {uy_tip_exact*1000:.4f} x10^-3 in, err: {beam_tip_err:.4%})")
    print(f"    Stresses:               Identical to exact for this load case")
    print(f"    Missing:                Shear deformation correction = "
          f"{(uy_tip_exact - uy_tip_beam)*1000:.4f} x10^-3 in")

    print(f"\n  {'='*58}")
    print(f"  REFERENCE VALUES")
    print(f"  {'='*58}")
    print(f"    Exact tip uy (y=0):       {uy_tip_exact*1000:.4f} x10^-3 in")
    print(f"    Exact tip corner ux:      {ux_corner*1000:.4f} x10^-3 in")
    print(f"    Exact tip corner uy:      {uy_corner*1000:.4f} x10^-3 in")
    print(f"    Exact tip corner |u|:     {mag_corner:.6f} in")
    print(f"    Beam theory (PL^3/3EI):   {uy_tip_beam*1000:.4f} x10^-3 in")
    print(f"    Simcenter NASTRAN ref:    0.0272 in (max |u|, 160 CQUAD4)")
    print("="*60)

    # --- Tabulated centerline results at every L/8 ---
    print_centerline_table(nodes, U_fem, stress_fem, U_efg, stress_efg)

    # --- Plots ---
    print("\n  Generating comparison plots...")
    plot_comparison(nodes, U_fem, stress_fem, U_efg, stress_efg)

    print("  Generating deformed shape plots...")
    plot_deformed(nodes, nx, ny, U_fem, U_efg, scale=50.0)

    print("\n  Done!")
    print("="*60)


if __name__ == "__main__":
    main()
