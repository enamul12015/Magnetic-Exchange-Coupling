#!/usr/bin/env python3
# ============================================================
# Full analytic tetrahedron method (η-free)
# References:
#   G. Lehmann & M. Taut, Phys. Stat. Sol. (b) 54, 469 (1972)
#   P. E. Blöchl, O. Jepsen, O. K. Andersen,
#       Phys. Rev. B 49, 16223 (1994)
# ============================================================
##############################@Enamul Haque######################
#if you use this code in your calculations, please cite this paper:
#Enamul Haque;  Yuefeng Yin;  Mark T. Edmonds;  David Cortie;  Nikhil V. Medhekar, "Highly #Confined Magnetophononic Polaritons and Electron–Phonon Interactions in Atomically Thin #MnBi2Te4", ACS Nano 20 (44): 29481–29497 (2026)
#################################################
##############################@Enamul Haque######################
#if you use this code in your calculations, please cite this paper:
#Enamul Haque;  Yuefeng Yin;  Mark T. Edmonds;  David Cortie;  Nikhil V. Medhekar, "Highly #Confined Magnetophononic Polaritons and Electron–Phonon Interactions in Atomically Thin #MnBi2Te4", ACS Nano 20 (44): 29481–29497 (2026)
#################################################
import numpy as np

# ------------------------------------------------------------
# Build tetrahedra from Monkhorst-Pack mesh (uniform grid)
# ------------------------------------------------------------
def build_tetrahedra(nkx, nky, nkz):
    """
    Return list of tetrahedra, each as 4 indices into flattened k-grid.
    k-grid is assumed to be uniform Monkhorst-Pack.
    """
    tetras = []
    def idx(ix, iy, iz):
        return ix + nkx * (iy + nky * iz)

    for ix in range(nkx - 1):
        for iy in range(nky - 1):
            for iz in range(nkz - 1):
                # cube corner indices
                c = [idx(ix, iy, iz),
                     idx(ix+1, iy, iz),
                     idx(ix, iy+1, iz),
                     idx(ix+1, iy+1, iz),
                     idx(ix, iy, iz+1),
                     idx(ix+1, iy, iz+1),
                     idx(ix, iy+1, iz+1),
                     idx(ix+1, iy+1, iz+1)]
                # 6 tetrahedra per cube (standard decomposition)
                tetras += [
                    [c[0], c[1], c[3], c[7]],
                    [c[0], c[3], c[2], c[7]],
                    [c[0], c[2], c[6], c[7]],
                    [c[0], c[6], c[4], c[7]],
                    [c[0], c[4], c[5], c[7]],
                    [c[0], c[5], c[1], c[7]],
                ]
    return np.array(tetras, dtype=int)

##############################@Enamul Haque######################
#if you use this code in your calculations, please cite this paper:
#Enamul Haque;  Yuefeng Yin;  Mark T. Edmonds;  David Cortie;  Nikhil V. Medhekar, "Highly #Confined Magnetophononic Polaritons and Electron–Phonon Interactions in Atomically Thin #MnBi2Te4", ACS Nano 20 (44): 29481–29497 (2026)
#################################################
# ------------------------------------------------------------
# Full Blöchl weights
# ------------------------------------------------------------
def tetra_weights(eps_tetra, E_F):
    """
    Compute analytic integration weights for one tetrahedron
    (Blöchl, Jepsen, Andersen, PRB 49, 16223 (1994)).

    eps_tetra : array (4,) eigenvalues at tetrahedron corners
    E_F       : Fermi energy

    Returns: array (4,) of vertex weights
    """
    # Sort by energy
    order = np.argsort(eps_tetra)
    e = eps_tetra[order]
    w = np.zeros(4)

    # Convenience
    def frac(x,y): return (E_F - x) / (y - x + 1e-14)

    # Case 1: below lowest
    if E_F <= e[0]:
        return np.zeros(4)

    # Case 2: above highest
    if E_F >= e[3]:
        return np.ones(4) / 4.0

    V = 1.0 / 24.0  # normalized volume (arbitrary scaling)

    # Case 3: E between e0 and e1
    if e[0] <= E_F < e[1]:
        a = frac(e[0], e[1])
        b = frac(e[0], e[2])
        c = frac(e[0], e[3])
        p = a * b * c
        w[order[0]] = V * (1 - 3*p)
        w[order[1]] = V * p
        w[order[2]] = V * p
        w[order[3]] = V * p
        return w

    # Case 4: E between e1 and e2
    if e[1] <= E_F < e[2]:
        a = frac(e[1], e[0])
        b = frac(e[1], e[2])
        c = frac(e[1], e[3])
        q = b * c
        w[order[0]] = V * a
        w[order[1]] = V * (1 - 2*q - a)
        w[order[2]] = V * q
        w[order[3]] = V * q
        return w

    # Case 5: E between e2 and e3
    if e[2] <= E_F < e[3]:
        a = frac(e[3], e[0])
        b = frac(e[3], e[1])
        c = frac(e[3], e[2])
        r = a * b * c
        w[order[0]] = V * r
        w[order[1]] = V * r
        w[order[2]] = V * r
        w[order[3]] = V * (1 - 3*r)
        return w

    return w


# ========================= 2D support (nkz == 1) =========================
def build_triangles_2d(nkx, nky):
    """
    Return list of triangles, each as 3 indices into flattened k-grid (z=0).
    Two triangles per square: (A,B,D) and (A,D,C).
    """
    def idx(ix, iy):
        return ix + nkx * iy  # same flattening as build_tetrahedra with z=0
    tris = []
    for ix in range(nkx - 1):
        for iy in range(nky - 1):
            A = idx(ix,     iy)
            B = idx(ix + 1, iy)
            C = idx(ix,     iy + 1)
            D = idx(ix + 1, iy + 1)
            tris.append([A, B, D])
            tris.append([A, D, C])
    return np.array(tris, dtype=int)


def triangle_weights(eps_tri, E_F):
    """
    2D analogue of Blöchl weights for one triangle's 3 vertices.
    Returns 3 weights that:
      - are all zero if E_F <= min(e)
      - are all 1/3 if E_F >= max(e)
      - otherwise give the exact linear (area) contribution up to E_F.
    """
    order = np.argsort(eps_tri)
    e = eps_tri[order]
    w_sorted = np.zeros(3)
    eps = 1e-14

    # empty
    if E_F <= e[0]:
        pass
    # full
    elif E_F >= e[2]:
        w_sorted[:] = 1.0 / 3.0
    # partial near lowest: E in [e0, e1)
    elif e[0] <= E_F < e[1]:
        t01 = (E_F - e[0]) / (e[1] - e[0] + eps)
        t02 = (E_F - e[0]) / (e[2] - e[0] + eps)
        f = t01 * t02  # occupied area fraction of the small triangle
        # average of a linear field over a triangle equals the average of its vertices
        # weights partition that area linearly along the cut
        w0 = f * (3.0 - t01 - t02) / 3.0
        w1 = f * (t01 / 3.0)
        w2 = f * (t02 / 3.0)
        w_sorted[:] = [w0, w1, w2]
    # partial near highest: E in [e1, e2)
    else:
        s12 = (e[2] - E_F) / (e[2] - e[1] + eps)
        s02 = (e[2] - E_F) / (e[2] - e[0] + eps)
        fcap = s12 * s02  # unfilled cap fraction near the highest-energy vertex
        full = 1.0 / 3.0
        cap_w2 = fcap * (3.0 - s12 - s02) / 3.0
        cap_w1 = fcap * (s12 / 3.0)
        cap_w0 = fcap * (s02 / 3.0)
        w0 = full - cap_w0
        w1 = full - cap_w1
        w2 = full - cap_w2
        w_sorted[:] = [w0, w1, w2]

    w = np.zeros(3)
    w[order] = w_sorted
    return w
# ========================================================================

##############################@Enamul Haque######################
#if you use this code in your calculations, please cite this paper:
#Enamul Haque;  Yuefeng Yin;  Mark T. Edmonds;  David Cortie;  Nikhil V. Medhekar, "Highly #Confined Magnetophononic Polaritons and Electron–Phonon Interactions in Atomically Thin #MnBi2Te4", ACS Nano 20 (44): 29481–29497 (2026)
#################################################

# ------------------------------------------------------------
# Tetrahedron integration driver
# ------------------------------------------------------------
def tetra_integral(eigenvals, quantities, nkmesh, E_F):
    """
    Integrate quantities over the BZ using tetrahedron method.

    eigenvals   : (N_k, N_bands) eigenvalues
    quantities  : (N_k, N_bands, ...) integrand to be averaged
    nkmesh      : (nkx, nky, nkz)
    E_F         : Fermi energy

    Returns: integrated quantity with shape quantities.shape[2:]
    """
    nkx, nky, nkz = nkmesh
    N_k, N_bands = eigenvals.shape
    assert quantities.shape[0] == N_k
    assert quantities.shape[1] == N_bands

    # ensure float math (prevents integer truncation to zero)
    eigenvals = np.asarray(eigenvals, dtype=float)
    quantities = np.asarray(quantities, dtype=float)

    # ---------- 2D path (nkz == 1): triangle method ----------
    if nkz == 1:
        tris = build_triangles_2d(nkx, nky)
        if len(tris) == 0:
            return np.zeros_like(quantities[0,0], dtype=float)
        result = np.zeros_like(quantities[0,0], dtype=float)
        for tri in tris:
            for n in range(N_bands):
                eps = eigenvals[tri, n]          # (3,)
                qvals = quantities[tri, n]       # (3, ...)
                w = triangle_weights(eps, E_F)   # (3,)
                result += np.tensordot(w, qvals, axes=([0],[0]))
        return result / len(tris)
    # ---------------------------------------------------------

    # ---------- original 3D tetra path (unchanged math) ------
    tetras = build_tetrahedra(nkx, nky, nkz)
    if len(tetras) == 0:
        return np.zeros_like(quantities[0,0], dtype=float)
    result = np.zeros_like(quantities[0,0], dtype=float)

    for tet in tetras:
        for n in range(N_bands):
            eps = eigenvals[tet, n]
            qvals = quantities[tet, n]
            w = tetra_weights(eps, E_F)
            result += np.tensordot(w, qvals, axes=([0],[0]))

    return result / len(tetras)


##############################@Enamul Haque######################
#if you use this code in your calculations, please cite this paper:
#Enamul Haque;  Yuefeng Yin;  Mark T. Edmonds;  David Cortie;  Nikhil V. Medhekar, "Highly #Confined Magnetophononic Polaritons and Electron–Phonon Interactions in Atomically Thin #MnBi2Te4", ACS Nano 20 (44): 29481–29497 (2026)
#################################################
