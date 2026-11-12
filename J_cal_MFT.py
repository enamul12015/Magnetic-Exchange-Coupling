#!/usr/bin/env python3
"""
##############################@Enamul Haque######################
#if you use this code in your calculations, please cite this paper:
#Enamul Haque;  Yuefeng Yin;  Mark T. Edmonds;  David Cortie;  Nikhil V. Medhekar, "Highly #Confined Magnetophononic Polaritons and Electron–Phonon Interactions in Atomically Thin #MnBi2Te4", ACS Nano 20 (44): 29481–29497 (2026)
#################################################
"""
from __future__ import annotations
import numpy as np
import argparse
import math
from typing import List, Tuple, Dict
from ase.io import read as ase_read
from scipy.linalg import solve, eigh
from scipy.integrate import simpson
from tetra import tetra_integral

# ------------------- Basic IO --------------------------------------

from ase.io import read as ase_read
import os

import numpy as np
import os

def read_poscar(path: str):
    """
    Robust POSCAR reader for VASP 4.x / 5.x formats.
    Returns:
        cell (3x3 ndarray)
        symbols (list of str)
        positions (Nx3 ndarray)
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"POSCAR file not found: {path}")

    try:
        with open(path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        # scaling factor
        scale = float(lines[1])

        # lattice vectors
        cell = np.array([[float(x) for x in lines[i].split()] for i in range(2,5)]) * scale

        # element symbols and counts
        try:
            symbols = lines[5].split()
            counts = [int(x) for x in lines[6].split()]
            pos_start = 7
        except ValueError:
            # sometimes counts are on the same line
            counts = [int(x) for x in lines[5].split()]
            symbols = ['X']*len(counts)  # unknown symbols
            pos_start = 6

        # total number of atoms
        n_atoms = sum(counts)

        # check for selective dynamics
        selective = False
        if lines[pos_start].lower().startswith('s'):
            selective = True
            pos_start += 1

        # check for Cartesian / Direct
        coord_type = lines[pos_start].lower()
        if coord_type.startswith('d'):
            direct = True
        elif coord_type.startswith('c'):
            direct = False
        else:
            raise RuntimeError("Cannot detect coordinate type in POSCAR")
        pos_start += 1

        # read positions
        positions = np.array([[float(x) for x in lines[pos_start+i].split()[:3]] for i in range(n_atoms)])
        if direct:
            # convert to Cartesian
            positions = positions @ cell

        # repeat symbols according to counts
        symbols_full = []
        for sym, count in zip(symbols, counts):
            symbols_full.extend([sym]*count)

        return cell, symbols_full, positions

    except Exception as e:
        raise RuntimeError(f"Failed to read POSCAR {path}: {e}")




# ------------------- Wannier90 HR parser (spinor-aware) ------------

def read_wannier90_hr(hr_path: str):
    """Parse wannier90_hr.dat (format: Rx Ry Rz i j Re Im)."""

    with open(hr_path, 'r') as f:
        # skip any header lines until we hit an integer
        line = f.readline()
        while not line.strip().isdigit():
            line = f.readline()
        n_wann = int(line.strip())

        # next: number of R vectors
        line = f.readline()
        while not line.strip().isdigit():
            line = f.readline()
        n_R = int(line.strip())

        # read degeneracies (n_R integers, possibly wrapped)
        degs = []
        while len(degs) < n_R:
            line = f.readline().split()
            if not line:
                continue
            degs.extend([int(x) for x in line])
        degs = np.array(degs)

        # read the Hamiltonian lines
        entries = []
        for line in f:
            parts = line.split()
            if len(parts) < 7:
                continue
            Rx, Ry, Rz = map(int, parts[0:3])
            i = int(parts[3]) - 1
            j = int(parts[4]) - 1
            re = float(parts[5])
            im = float(parts[6])
            entries.append((i, j, (Rx, Ry, Rz), re + 1j * im))

        if not entries:
            raise RuntimeError("No Hamiltonian entries found in HR file")

        dim = max(max(i, j) for i, j, _, _ in entries) + 1
        ham_R: dict[tuple[int,int,int], np.ndarray] = {}
        Rset = sorted({r for (_, _, r, _) in entries})
        for R in Rset:
            ham_R[R] = np.zeros((dim, dim), dtype=complex)
        for i, j, R, val in entries:
            ham_R[R][i, j] = val

        return ham_R, dim, n_wann


##############################@Enamul Haque######################
#if you use this code in your calculations, please cite this paper:
#Enamul Haque;  Yuefeng Yin;  Mark T. Edmonds;  David Cortie;  Nikhil V. Medhekar, "Highly #Confined Magnetophononic Polaritons and Electron–Phonon Interactions in Atomically Thin #MnBi2Te4", ACS Nano 20 (44): 29481–29497 (2026)
#################################################
# ------------------- Bloch Hamiltonian -----------------------------

def h_k_from_hr(ham_R: Dict[Tuple[int,int,int], np.ndarray], kvec: np.ndarray) -> np.ndarray:
    dims = next(iter(ham_R.values())).shape[0]
    Hk = np.zeros((dims, dims), dtype=complex)
    for R, H in ham_R.items():
        # note: many HR files use exp(i k·R) with k in cartesian reciprocal units;
        # keep this convention (consistent with original script)
        phase = np.exp(1j * np.dot(kvec, np.array(R, dtype=float)))
        Hk += H * phase
    return Hk

# ------------------- Greens solver (numerically stable) -----------------

def greens_from_eig(eigvals: np.ndarray, eigvecs: np.ndarray, energy: float, eta: float):
    """
    Return G(E) from spectral decomposition G = sum_n |n><n| /(E + i eta - eps_n)
    (we won't form full G in general, but this helper exists if needed)
    """
    z = energy + 1j*eta
    # Instead of building full matrix (costly), the caller will use band projectors
    # and band denominators directly.
    D = 1.0 / (z - eigvals)   # shape (nbands,)
    # If caller wants full G: G = eigvecs @ np.diag(D) @ eigvecs.conj().T
    return D

# ------------------- Pauli matrices helper ------------------------
_pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_pauli_y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
_pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

# ------------------- Vector rotation helper ----------------------

def rotate_vector(vec: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate vector `vec` around `axis` by `angle` radians using Rodrigues' formula."""
    axis = axis / np.linalg.norm(axis)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return vec * cos_a + np.cross(axis, vec) * sin_a + axis * np.dot(axis, vec) * (1 - cos_a)


# ------------------- On-site exchange extraction ------------------

def extract_onsite_field_vectors(H0: np.ndarray, wf_indices_per_site: List[List[int]]) -> List[np.ndarray]:
    dim = H0.shape[0]
    if dim % 2 != 0:
        raise RuntimeError('HR dimension is not even: cannot assume spinor blocks of size 2')
    n_wf = dim // 2

    site_b_vectors = []
    for wf_list in wf_indices_per_site:
        b_tot = np.zeros(3, dtype=float)
        for wf in wf_list:
            if wf < 0 or wf >= n_wf:
                raise IndexError(f'WF index {wf} out of range (0..{n_wf-1})')
            s0 = 2 * wf; s1 = s0 + 2
            hblock = H0[s0:s1, s0:s1]
            hblock = 0.5 * (hblock + hblock.conj().T)
            bx = 0.5 * np.trace(hblock @ _pauli_x).real
            by = 0.5 * np.trace(hblock @ _pauli_y).real
            bz = 0.5 * np.trace(hblock @ _pauli_z).real
            b_tot += np.array([bx, by, bz])
        site_b_vectors.append(b_tot)
    return site_b_vectors

# ------------------- Build Sigma operators ------------------------
def build_sigma_operators(site_wf_lists: List[List[int]], dim: int) -> List[List[np.ndarray]]:
    n_wf = dim // 2
    sigmas_per_site = []
    for wf_list in site_wf_lists:
        Sigma_x = np.zeros((dim, dim), dtype=complex)
        Sigma_y = np.zeros((dim, dim), dtype=complex)
        Sigma_z = np.zeros((dim, dim), dtype=complex)
        for wf in wf_list:
            s0 = 2 * wf; s1 = s0 + 2
            Sigma_x[s0:s1, s0:s1] = _pauli_x
            Sigma_y[s0:s1, s0:s1] = _pauli_y
            Sigma_z[s0:s1, s0:s1] = _pauli_z
        sigmas_per_site.append([Sigma_x, Sigma_y, Sigma_z])
    return sigmas_per_site
def build_sigma_operators_rotated(site_wf_lists: List[List[int]], dim: int,
                                  onsite_b_vectors: List[np.ndarray],
                                  cant_site: int = 1,
                                  cant_angle: float = 0.0,
                                  cant_axis: np.ndarray = np.array([1,0,0])) -> List[List[np.ndarray]]:
    """Build Σ operators with optional finite canting for one site."""
    n_wf = dim // 2
    sigmas_per_site = []
    for p, wf_list in enumerate(site_wf_lists):
        bvec = onsite_b_vectors[p].copy()
        if p == cant_site and cant_angle != 0.0:
            bvec = rotate_vector(bvec, cant_axis, cant_angle)
        Sigma_x = np.zeros((dim, dim), dtype=complex)
        Sigma_y = np.zeros((dim, dim), dtype=complex)
        Sigma_z = np.zeros((dim, dim), dtype=complex)
        for wf in wf_list:
            s0 = 2*wf; s1 = s0+2
            Sigma_x[s0:s1, s0:s1] = _pauli_x * bvec[0]
            Sigma_y[s0:s1, s0:s1] = _pauli_y * bvec[1]
            Sigma_z[s0:s1, s0:s1] = _pauli_z * bvec[2]
        sigmas_per_site.append([Sigma_x, Sigma_y, Sigma_z])
    return sigmas_per_site



# ------------------- Exchange tensor calculation (band decomposed) ------------------

# ---------- Replace compute_exchange_tensor with this η-free version ----------
def compute_exchange_tensor(ham_R: Dict[Tuple[int,int,int], np.ndarray], kmesh: Tuple[int,int,int],
                            fermi_e: float, egrid: np.ndarray, temp: float,
                            wf_indices_per_site: List[List[int]], eta: float=1e-3) -> Dict:
    """
    η-free spectral formula evaluation using band-off-diagonal kernel and tetrahedral k-integration.
    This replaces the real-axis + eta integration with the analytic result:
      J_ab = sum_k sum_{m != n} M_mn^{ab}(k) * (f(e_n)-f(e_m)) / (e_n - e_m)
    Then integrate the k-dependent C_ab(k) with tetrahedron integration (BZ average).
    """

    def integrate_k_tetrahedral_vertex_values_local(vertex_vals: np.ndarray, kmesh_local: Tuple[int,int,int]):
     """
     Average vertex-valued quantity over the BZ.
     - 3D (nx,ny,nz>1): uniform tetrahedra (your original path)
     - 2D (nx,ny,1): split each cell into 2 triangles and integrate linearly
     """
     import math
     nx, ny, nz = kmesh_local
     n_k = nx * ny * nz
     if vertex_vals.shape[0] != n_k:
        raise RuntimeError("vertex_vals first dim does not match kmesh product")

     out_shape = vertex_vals.shape[1:]
     integrated = np.zeros(out_shape, dtype=float)

     # ---------------- 2D path (nkz == 1): triangles ----------------
     if nz == 1:
        if nx < 2 or ny < 2:
            return np.zeros(out_shape, dtype=float)

        # BZ "area" for 2D
        Abz = (2.0 * math.pi) ** 2
        cell_area = Abz / float(nx * ny)     # each grid cell area
        tri_area = 0.5 * cell_area           # two triangles per cell

        # flattening matches your 3D idx with nz=1
        def idx2d(ix, iy):
            return (ix * ny) + iy

        # build two triangles per cell: (A,B,D) and (A,D,C)
        for ix in range(nx - 1):
            for iy in range(ny - 1):
                A = idx2d(ix,     iy)
                B = idx2d(ix + 1, iy)
                C = idx2d(ix,     iy + 1)
                D = idx2d(ix + 1, iy + 1)

                # triangle 1
                vals = vertex_vals[[A, B, D]]   # (3, ...)
                avg  = np.mean(vals, axis=0)
                integrated += avg * tri_area

                # triangle 2
                vals = vertex_vals[[A, D, C]]
                avg  = np.mean(vals, axis=0)
                integrated += avg * tri_area

        # return BZ-averaged value
        return integrated / Abz

     # ---------------- 3D path (original tetrahedra) ----------------
     Vbz = (2.0 * math.pi) ** 3
     cube_vol = Vbz / float(n_k)
     tet_vol = cube_vol / 6.0

     def idx(ix, iy, iz):
        return (ix * ny * nz) + (iy * nz) + iz

     tets = []
     for ix in range(nx - 1):
        for iy in range(ny - 1):
            for iz in range(nz - 1):
                v000 = idx(ix  , iy  , iz  )
                v100 = idx(ix+1, iy  , iz  )
                v010 = idx(ix  , iy+1, iz  )
                v110 = idx(ix+1, iy+1, iz  )
                v001 = idx(ix  , iy  , iz+1)
                v101 = idx(ix+1, iy  , iz+1)
                v011 = idx(ix  , iy+1, iz+1)
                v111 = idx(ix+1, iy+1, iz+1)
                tets.append((v000, v100, v010, v001))
                tets.append((v100, v110, v010, v111))
                tets.append((v001, v010, v011, v111))
                tets.append((v100, v001, v101, v111))
                tets.append((v010, v100, v111, v001))
                tets.append((v101, v100, v111, v001))

     for (v0, v1, v2, v3) in tets:
        vals = vertex_vals[[v0, v1, v2, v3]]   # (4, ...)
        avg  = np.mean(vals, axis=0)
        integrated += avg * tet_vol

     return integrated / Vbz

    if len(wf_indices_per_site) < 2:
        raise RuntimeError('Need at least two sites to compute pairwise J tensor')

    nkx, nky, nkz = kmesh
    kpts = []
    for ikx in range(nkx):
        for iky in range(nky):
            for ikz in range(nkz):
                kx = (ikx + 0.5) / nkx * 2.0 * math.pi
                ky = (iky + 0.5) / nky * 2.0 * math.pi
                kz = (ikz + 0.5) / nkz * 2.0 * math.pi
                kpts.append(np.array([kx, ky, kz]))
    n_kpoints = len(kpts)

    #H0 = h_k_from_hr(ham_R, np.array([0.0, 0.0, 0.0]))
    #site_b = extract_onsite_field_vectors(H0, wf_indices_per_site)
    H00 = ham_R[(0,0,0)]  # only the R=0 matrix
    site_b = extract_onsite_field_vectors(H00, wf_indices_per_site)
    dim = H00.shape[0]
    sigmas = build_sigma_operators(wf_indices_per_site, dim)

    Sigma_ops = []
    for p, bvec in enumerate(site_b):
        Sigma_px = sigmas[p][0] * bvec[0]
        Sigma_py = sigmas[p][1] * bvec[1]
        Sigma_pz = sigmas[p][2] * bvec[2]
        Sigma_ops.append([Sigma_px, Sigma_py, Sigma_pz])

    # temperature and fermi
    kb = 8.617333262145e-5
    beta = 1.0 / (kb * max(1e-6, temp)) if temp > 0 else 1e6
    def fermi_fn(E):
        if temp <= 0:
            return 1.0 if E < fermi_e else 0.0
        x = (E - fermi_e) * beta
        if x > 700:
            return 0.0
        if x < -700:
            return 1.0
        return 1.0 / (1.0 + math.exp(x))

    # For each k, compute C_ab(k)
    # store C_k as shape (n_kpoints, 3, 3)
    C_k = np.zeros((n_kpoints, 3, 3), dtype=float)

    for ik, kvec in enumerate(kpts):
        Hk = h_k_from_hr(ham_R, kvec)
        eps, U = eigh(Hk)   # eps: (nb,), U: (dim, nb)

        nb = eps.size
        # compute M_A and M_B overlap matrices as in previous implementation
        M_A = [None, None, None]
        M_B = [None, None, None]
        for a in range(3):
            Aop = Sigma_ops[0][a]
            Bop = Sigma_ops[1][a]
            tmpA = Aop @ U
            tmpB = Bop @ U
            M_A[a] = (U.conj().T @ tmpA)   # shape (nb, nb)
            M_B[a] = (U.conj().T @ tmpB)

        # Build M_ab matrices: M_ab[a][b] is nb x nb with entries for m (row) n (col)
        M_ab = [[None]*3 for _ in range(3)]
        for a in range(3):
            for b in range(3):
                # Mtmp[m,n] = M_A[a][n,m] * M_B[b][m,n]
                # Using elementwise: Mtmp = M_B[b] * (M_A[a].T)
                Mtmp = M_B[b] * (M_A[a].T)
                M_ab[a][b] = Mtmp

        # Now compute C_ab(k) = sum_{m != n} M_ab[m,n] * (f(e_n)-f(e_m)) / (e_n - e_m)
        # Vectorize using broadcasting
        eps_row = eps.reshape((1, nb))   # shape (1, nb)
        eps_col = eps.reshape((nb, 1))   # shape (nb, 1)
        denom = eps_row - eps_col        # (nb, nb) with denom[n]-denom[m] = e_n - e_m
        # avoid division by zero on the diagonal; we will zero diagonal contributions explicitly
        # compute f differences
        fvals = np.array([fermi_fn(E) for E in eps])
        f_row = fvals.reshape((1, nb))
        f_col = fvals.reshape((nb, 1))
        fdiff = f_row - f_col   # shape (nb, nb): f(en) - f(em)

        # Build mask for off-diagonal
        offdiag = ~np.eye(nb, dtype=bool)

        Cab = np.zeros((3,3), dtype=float)
        for a in range(3):
            for b in range(3):
                Mmat = M_ab[a][b]   # complex nb x nb
                # compute contribution only off-diagonal
                numer = fdiff * Mmat    # shape (nb, nb) complex
                # safe divide only off-diagonals
                contrib = np.zeros_like(numer, dtype=complex)
                contrib[offdiag] = numer[offdiag] / denom[offdiag]
                # sum over m,n
                val = np.sum(contrib)
                Cab[a,b] = val.real   # final contribution should be real
        C_k[ik] = Cab

    # Integrate C_k over k with tetrahedral integrator (average over BZ)
    try:
        C_integrated = integrate_k_tetrahedral_vertex_values(C_k, kmesh)  # expects shape (n_kpoints, 3,3)
    except NameError:
        # fallback to local helper if global not present
        C_integrated = integrate_k_tetrahedral_vertex_values_local(C_k, kmesh)

    J_tensor = C_integrated   # already BZ-averaged integrand (units consistent with previous code)
    # normalization: keep original factor if you had S^2 etc.
    Nwi = len(wf_indices_per_site[0])
    Nwj = len(wf_indices_per_site[1])
    J_iso = (np.trace(J_tensor) / (3.0 * 6.25*(Nwi * Nwj)*2))*1000
    J_sym = 0.5 * (J_tensor + J_tensor.T) - np.eye(3) * J_iso
    D = np.array([0.5*(J_tensor[1,2] - J_tensor[2,1]),
                  0.5*(J_tensor[2,0] - J_tensor[0,2]),
                  0.5*(J_tensor[0,1] - J_tensor[1,0])])

    return {
        'J_tensor': J_tensor,
        'J_iso': J_iso,
        'J_sym': J_sym,
        'D': D,
        'site_b_vectors': site_b
    }


def compute_exchange_tensor_canted(ham_R: Dict[Tuple[int,int,int], np.ndarray],
                                   kmesh: Tuple[int,int,int],
                                   wf_indices_per_site: List[List[int]],
                                   fermi_e: float,
                                   temp: float,
                                   theta_deg: float = 1.0) -> Dict:
    """
    Compute interlayer exchange J_tensor using finite canting (small rotation)
    method. Captures dynamical effects, works with tetrahedron k-integration.

    Arguments:
    - ham_R: HR matrix dictionary
    - kmesh: (nkx, nky, nkz)
    - wf_indices_per_site: list of Wannier function indices for site A and B
    - fermi_e: Fermi energy
    - temp: temperature in eV
    - theta_deg: small canting angle in degrees
    """

    import numpy as np
    import math
    from scipy.linalg import eigh

    theta = np.deg2rad(theta_deg)  # convert to radians

    # --- Prepare k-points ---
    nkx, nky, nkz = kmesh
    kpts = []
    for ikx in range(nkx):
        for iky in range(nky):
            for ikz in range(nkz):
                kx = (ikx + 0.5)/nkx * 2*math.pi
                ky = (iky + 0.5)/nky * 2*math.pi
                kz = (ikz + 0.5)/nkz * 2*math.pi
                kpts.append(np.array([kx,ky,kz]))
    n_kpoints = len(kpts)

    # --- Onsite fields and sigma operators ---
    H0 = h_k_from_hr(ham_R, np.array([0.0,0.0,0.0]))
    dim = H0.shape[0]
    site_b = extract_onsite_field_vectors(H0, wf_indices_per_site)
    sigmas = build_sigma_operators(wf_indices_per_site, dim)

    # Apply small canting rotation to site B
    Sigma_ops_canted = []
    for p, bvec in enumerate(site_b):
        bx, by, bz = bvec
        if p == 1:  # site B is canted
            bz_new = bz * math.cos(theta)
            bx_new = bx + bz * math.sin(theta)  # rotate around x
            by_new = by
            bvec = np.array([bx_new, by_new, bz_new])
        Sigma_px = sigmas[p][0] * bvec[0]
        Sigma_py = sigmas[p][1] * bvec[1]
        Sigma_pz = sigmas[p][2] * bvec[2]
        Sigma_ops_canted.append([Sigma_px, Sigma_py, Sigma_pz])

    # --- Compute C_k for each k ---
    C_k = np.zeros((n_kpoints, 3,3), dtype=float)

    kb = 8.617333262145e-5
    beta = 1.0 / (kb*max(1e-6, temp)) if temp>0 else 1e6
    def fermi_fn(E):
        if temp<=0: return 1.0 if E<fermi_e else 0.0
        x = (E-fermi_e)*beta
        if x>700: return 0.0
        if x<-700: return 1.0
        return 1.0/(1.0+math.exp(x))

    for ik, kvec in enumerate(kpts):
        Hk = h_k_from_hr(ham_R, kvec)
        eps, U = eigh(Hk)
        nb = eps.size

        # Build M_ab with canted Sigma_ops
        M_ab = [[None]*3 for _ in range(3)]
        for a in range(3):
            for b in range(3):
                Aop = Sigma_ops_canted[0][a]
                Bop = Sigma_ops_canted[1][b]
                Mtmp = (U.conj().T @ Bop @ U) * ((U.conj().T @ Aop @ U).T)
                M_ab[a][b] = Mtmp

        # Off-diagonal spectral sum
        eps_row = eps.reshape((1, nb))
        eps_col = eps.reshape((nb, 1))
        denom = eps_row - eps_col
        fvals = np.array([fermi_fn(E) for E in eps])
        fdiff = fvals.reshape((1,nb)) - fvals.reshape((nb,1))
        offdiag = ~np.eye(nb, dtype=bool)

        Cab = np.zeros((3,3), dtype=float)
        for a in range(3):
            for b in range(3):
                numer = fdiff * M_ab[a][b]
                contrib = np.zeros_like(numer, dtype=complex)
                contrib[offdiag] = numer[offdiag] / denom[offdiag]
                Cab[a,b] = np.sum(contrib).real
        C_k[ik] = Cab

    # --- Integrate over k (tetrahedron method) ---
    try:
        C_integrated = integrate_k_tetrahedral_vertex_values(C_k, kmesh)
    except NameError:
        # fallback to local helper if global not defined
        def integrate_k_tetrahedral_vertex_values_local(vertex_vals: np.ndarray, kmesh_local: Tuple[int,int,int]):
            nx,ny,nz = kmesh_local
            n_k = nx*ny*nz
            Vbz = (2*math.pi)**3
            cube_vol = Vbz/float(n_k)
            tet_vol = cube_vol/6.0
            def idx(ix,iy,iz): return (ix*ny*nz)+(iy*nz)+iz
            tets=[]
            for ix in range(nx-1):
                for iy in range(ny-1):
                    for iz in range(nz-1):
                        v000=idx(ix,iy,iz); v100=idx(ix+1,iy,iz)
                        v010=idx(ix,iy+1,iz); v110=idx(ix+1,iy+1,iz)
                        v001=idx(ix,iy,iz+1); v101=idx(ix+1,iy,iz+1)
                        v011=idx(ix,iy+1,iz+1); v111=idx(ix+1,iy+1,iz+1)
                        tets.append((v000,v100,v010,v001)); tets.append((v100,v110,v010,v111))
                        tets.append((v001,v010,v011,v111)); tets.append((v100,v001,v101,v111))
                        tets.append((v010,v100,v111,v001)); tets.append((v101,v100,v111,v001))
            out_shape = vertex_vals.shape[1:]
            integrated = np.zeros(out_shape, dtype=float)
            for tet in tets:
                vals = vertex_vals[list(tet)]
                avg = np.mean(vals, axis=0)
                integrated += avg*tet_vol
            return integrated / Vbz
        C_integrated = integrate_k_tetrahedral_vertex_values_local(C_k, kmesh)

    # --- J tensor extraction ---
    Nwi = len(wf_indices_per_site[0])
    Nwj = len(wf_indices_per_site[1])
    J_iso = np.trace(C_integrated)/(3.0*6.25*(Nwi*Nwj)*2)*1000
    J_sym = 0.5*(C_integrated + C_integrated.T) - np.eye(3)*J_iso
    D = np.array([0.5*(C_integrated[1,2]-C_integrated[2,1]),
                  0.5*(C_integrated[2,0]-C_integrated[0,2]),
                  0.5*(C_integrated[0,1]-C_integrated[1,0])])

    return {
        'J_tensor': C_integrated,
        'J_iso': J_iso,
        'J_sym': J_sym,
        'D': D,
        'site_b_vectors': site_b
    }

##############################@Enamul Haque######################
#if you use this code in your calculations, please cite this paper:
#Enamul Haque;  Yuefeng Yin;  Mark T. Edmonds;  David Cortie;  Nikhil V. Medhekar, "Highly #Confined Magnetophononic Polaritons and Electron–Phonon Interactions in Atomically Thin #MnBi2Te4", ACS Nano 20 (44): 29481–29497 (2026)
#################################################
# ------------------- CLI / Runner ---------------------------------

def parse_site_wfs_args(arg_list: List[str]) -> List[List[int]]:
    sites = []
    for token in arg_list:
        if token.strip() == '':
            continue
        parts = token.split(',')
        ints = [int(p) for p in parts if p.strip()!='']
        sites.append(ints)
    return sites

def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Compute noncollinear exchange tensor from wannier90 HR (spinor)')
    p.add_argument('--poscar', required=True)
    p.add_argument('--hr', required=True)
    p.add_argument('--fE', type=float, required=True)
    p.add_argument('--kmesh', nargs=3, type=int, default=[6,6,6])
    p.add_argument('--emin', type=float, default=-10.0)
    p.add_argument('--emax', type=float, default=10.0)
    p.add_argument('--ne', type=int, default=201)
    p.add_argument('--temp', type=float, default=300.0)
    p.add_argument('--eta', type=float, default=1e-3)
    p.add_argument('--site-wfs', nargs='+', required=True,
                   help='Provide comma-separated WF indices per magnetic site: e.g. "0,1" "5,6,7"')
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    cell, symbols, positions = read_poscar(args.poscar)
    print('Read POSCAR:', len(symbols), 'atoms')
    ham_R, dim, n_wann = read_wannier90_hr(args.hr)
    print('Parsed HR: dim=', dim, 'n_wann_guess=', n_wann)
    if dim % 2 != 0:
        raise RuntimeError('Parsed HR matrix dimension is odd: non-spinor data or parsing error')

    wf_sites = parse_site_wfs_args(args.site_wfs)
    if len(wf_sites) < 2:
        raise RuntimeError('Provide at least two site WF lists to compute a pairwise J tensor')

    egrid = np.linspace(args.emin, args.emax, args.ne)
    kmesh = tuple(args.kmesh)

    out = compute_exchange_tensor(ham_R, kmesh, args.fE, egrid, args.temp, wf_sites, eta=args.eta)

    print()
    print('Isotropic J (1/3 trace) [meV]:', out['J_iso'])
    #H0 = h_k_from_hr(ham_R, np.array([0.0,0.0,0.0]))
    #dim = H0.shape[0]
    #site_b = extract_onsite_field_vectors(H0, wf_sites)
    #angles = [0.0, 45.0, 90.00, 180.0]  # radians
    #print("\n=== Finite-canting GF J scan ===")
    #for theta in angles:
    #    J_out = compute_exchange_tensor_canted(ham_R, kmesh, wf_sites, args.fE, args.temp, theta_deg=theta)
    #    J_iso = J_out['J_iso']
    #    print(f"theta = {theta:.3f} rad -> J_iso = {J_iso:.6f} meV")


##############################@Enamul Haque######################
#if you use this code in your calculations, please cite this paper:
#Enamul Haque;  Yuefeng Yin;  Mark T. Edmonds;  David Cortie;  Nikhil V. Medhekar, "Highly #Confined Magnetophononic Polaritons and Electron–Phonon Interactions in Atomically Thin #MnBi2Te4", ACS Nano 20 (44): 29481–29497 (2026)
#################################################
if __name__ == '__main__':
    main()
##############################@Enamul Haque######################
#if you use this code in your calculations, please cite this paper:
#Enamul Haque;  Yuefeng Yin;  Mark T. Edmonds;  David Cortie;  Nikhil V. Medhekar, "Highly #Confined Magnetophononic Polaritons and Electron–Phonon Interactions in Atomically Thin #MnBi2Te4", ACS Nano 20 (44): 29481–29497 (2026)
#################################################
