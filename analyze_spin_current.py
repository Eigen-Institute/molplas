#    CTC 2026
#    http://eigeninstitute.org
"""
Spin Current Density Analysis for Coronene Radical Anion RT-TDDFT

Implements the full analysis pipeline from plan.md:
  Steps 0-3: File discovery, cube parsing, coordinate grid, spatial regions
  Step 4:    Snapshot loop — regional dipole moments + azimuthal Fourier
  Step 5:    Dipole angular momenta via cross product
  Step 6:    Radial profile of spin current density (m=+1 Fourier)
  Step 7:    Validation against bulk dipole time series
  Step 8:    Visualization (5 figure types)
"""

import os
import glob
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import time as timer

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

CUBE_DIR      = "cube_dir"
RT_DATA_PATH  = "rt_data.dat"
OUTPUT_DIR    = "outputs_dir"

# File naming pattern
ALPHA_PREFIX = "CUBE.alpha"
BETA_PREFIX  = "CUBE.beta"

DT_SIM        = 0.2        # au per simulation step
CUBE_INTERVAL = 25         # steps between cube dumps
DT_CUBE       = DT_SIM * CUBE_INTERVAL  # 5.0 au

R_CUT         = 2.68       # Bohr — hub/rim boundary
R_MAX         = 10.0       # Bohr — max radius for radial bins
N_RADIAL_BINS = 50
T_POST_PULSE  = 500.0      # au — start of free-oscillation window

AU_TO_FS      = 0.02418884
AU_TO_DEBYE   = 2.5417464

# ═══════════════════════════════════════════════════════════════════════
# Step 0 — File Discovery and Sorting
# ═══════════════════════════════════════════════════════════════════════

def discover_cube_pairs(cube_dir):
    """Find and pair alpha/beta cube files, sorted by step number."""
    alpha_files = glob.glob(os.path.join(cube_dir, ALPHA_PREFIX + ".*.cube"))
    beta_files  = glob.glob(os.path.join(cube_dir, BETA_PREFIX  + ".*.cube"))

    # Parse step numbers
    step_re = re.compile(r'\.(\d+)\.cube$')

    alpha_by_step = {}
    for f in alpha_files:
        m = step_re.search(f)
        if m:
            alpha_by_step[int(m.group(1))] = f

    beta_by_step = {}
    for f in beta_files:
        m = step_re.search(f)
        if m:
            beta_by_step[int(m.group(1))] = f

    # Pair by step number
    common_steps = sorted(set(alpha_by_step) & set(beta_by_step))
    
    # Filter by CUBE_INTERVAL
    common_steps = [s for s in common_steps if s % CUBE_INTERVAL == 0]
    
    unpaired = (set(alpha_by_step) | set(beta_by_step)) - set(common_steps)
    if unpaired:
        # Don't count filtered-out steps as "unpaired" warnings if they were actually paired but just filtered
        actual_unpaired = (set(alpha_by_step) ^ set(beta_by_step))
        if actual_unpaired:
             print(f"  Warning: {len(actual_unpaired)} unpaired cube files skipped")
        filtered = len(unpaired) - len(actual_unpaired)
        if filtered > 0:
             print(f"  Note: {filtered} snapshots skipped due to CUBE_INTERVAL={CUBE_INTERVAL}")
    pairs = []
    for step in common_steps:
        t_au = step * DT_SIM
        pairs.append((t_au, alpha_by_step[step], beta_by_step[step]))

    print(f"  Found {len(pairs)} snapshot pairs")
    if pairs:
        print(f"  Time range: {pairs[0][0]:.1f} – {pairs[-1][0]:.1f} au "
              f"({pairs[0][0]*AU_TO_FS:.3f} – {pairs[-1][0]*AU_TO_FS:.3f} fs)")
    return pairs


# ═══════════════════════════════════════════════════════════════════════
# Step 1 — Fast Cube File Parser
# ═══════════════════════════════════════════════════════════════════════

def parse_cube(filepath):
    """Parse a Gaussian cube file. Returns (origin, axes, natoms, atom_data, data_3d, dV)."""
    with open(filepath, 'r') as f:
        # Lines 1-2: comments
        f.readline()
        f.readline()

        # Line 3: natoms, origin
        parts = f.readline().split()
        natoms = int(parts[0])
        origin = np.array([float(parts[1]), float(parts[2]), float(parts[3])])

        # Lines 4-6: axis definitions
        axes = []
        for _ in range(3):
            parts = f.readline().split()
            n = int(parts[0])
            dvec = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            axes.append((n, dvec))

        # Atom records
        atom_data = np.empty((natoms, 5))
        for i in range(natoms):
            parts = f.readline().split()
            atom_data[i] = [float(x) for x in parts]

        # Volumetric data — read entire remaining block as one string
        data_text = f.read()

    # Fast conversion: split on whitespace, convert to float array, reshape
    Nx, Ny, Nz = axes[0][0], axes[1][0], axes[2][0]
    data_flat = np.fromstring(data_text, dtype=np.float64, sep=' ')
    data_3d = data_flat.reshape((Nx, Ny, Nz))

    # Volume element (axis-aligned grid)
    dV = abs(axes[0][1][0] * axes[1][1][1] * axes[2][1][2])

    return origin, axes, natoms, atom_data, data_3d, dV


def parse_cube_data_only(filepath, header_lines):
    """Fast path: skip header, read only the volumetric data block.
    header_lines = 6 + natoms (known after first parse)."""
    with open(filepath, 'r') as f:
        for _ in range(header_lines):
            f.readline()
        data_text = f.read()
    return np.fromstring(data_text, dtype=np.float64, sep=' ')


# ═══════════════════════════════════════════════════════════════════════
# Step 2 — Build Coordinate Grid
# ═══════════════════════════════════════════════════════════════════════

def build_grid(origin, axes):
    """Construct 3D Cartesian and cylindrical coordinate arrays."""
    Nx, dx_vec = axes[0]
    Ny, dy_vec = axes[1]
    Nz, dz_vec = axes[2]

    x_1d = origin[0] + np.arange(Nx) * dx_vec[0]
    y_1d = origin[1] + np.arange(Ny) * dy_vec[1]
    z_1d = origin[2] + np.arange(Nz) * dz_vec[2]

    X, Y, Z = np.meshgrid(x_1d, y_1d, z_1d, indexing='ij')

    R_perp = np.sqrt(X**2 + Y**2)
    Theta  = np.arctan2(Y, X)
    exp_neg_itheta = np.exp(-1j * Theta)

    return dict(X=X, Y=Y, Z=Z, R_perp=R_perp, Theta=Theta,
                exp_neg_itheta=exp_neg_itheta,
                Nx=Nx, Ny=Ny, Nz=Nz)


# ═══════════════════════════════════════════════════════════════════════
# Step 3 — Define Spatial Regions
# ═══════════════════════════════════════════════════════════════════════

def define_regions(grid, R_cut=R_CUT, R_max=R_MAX, n_bins=N_RADIAL_BINS):
    """Create boolean masks for hub/rim and radial bin masks."""
    R = grid['R_perp']
    hub_mask = R < R_cut
    rim_mask = ~hub_mask

    # Radial bins
    bin_edges = np.linspace(0, R_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Precompute digitized bin indices (0 = below first edge, n_bins = above last)
    bin_idx = np.digitize(R, bin_edges)  # 1-indexed: bin i → values in [edges[i-1], edges[i])

    return dict(hub_mask=hub_mask, rim_mask=rim_mask,
                bin_edges=bin_edges, bin_centers=bin_centers, bin_idx=bin_idx)


# ═══════════════════════════════════════════════════════════════════════
# Step 4 — Snapshot Loop: Regional Dipole Moments + Fourier
# ═══════════════════════════════════════════════════════════════════════

def process_snapshots(pairs, grid, regions, dV):
    """Loop over all snapshots; compute regional dipoles and azimuthal Fourier."""
    n_times = len(pairs)
    n_bins  = len(regions['bin_centers'])
    Nx, Ny, Nz = grid['Nx'], grid['Ny'], grid['Nz']
    header_lines = None  # determined on first parse

    X = grid['X']
    Y = grid['Y']
    Z = grid['Z']
    exp_neg_itheta = grid['exp_neg_itheta']
    hub = regions['hub_mask']
    rim = regions['rim_mask']
    bin_idx = regions['bin_idx']

    # Pre-allocate: dipoles[spin][region][component] = 1D array(n_times)
    spins   = ['alpha', 'beta']
    regs    = ['hub', 'rim', 'total']
    comps   = ['x', 'y', 'z']
    coord   = {'x': X, 'y': Y, 'z': Z}
    masks   = {'hub': hub, 'rim': rim, 'total': np.ones_like(hub)}

    mu = {}
    for s in spins:
        mu[s] = {}
        for r in regs:
            mu[s][r] = {}
            for c in comps:
                mu[s][r][c] = np.zeros(n_times)

    # Azimuthal Fourier coefficients
    c1 = {s: np.zeros((n_times, n_bins), dtype=np.complex128) for s in spins}

    times = np.zeros(n_times)

    # Precompute coordinate*mask and exp*mask products for each region (saves repeated masking)
    # For radial bins, we'll use vectorized digitize approach
    print(f"\n  Processing {n_times} snapshots...")
    t_start = timer.time()

    for idx, (t_au, alpha_path, beta_path) in enumerate(pairs):
        times[idx] = t_au

        if idx == 0:
            # Full parse on first file to get header info
            origin, axes, natoms, atom_data, data_alpha, dV_check = parse_cube(alpha_path)
            header_lines = 6 + natoms
            _, _, _, _, data_beta, _ = parse_cube(beta_path)
        else:
            data_alpha = parse_cube_data_only(alpha_path, header_lines).reshape((Nx, Ny, Nz))
            data_beta  = parse_cube_data_only(beta_path,  header_lines).reshape((Nx, Ny, Nz))

        for s, data in [('alpha', data_alpha), ('beta', data_beta)]:
            rho_dV = data * dV

            # Regional dipole moments (electron convention: μ = -∫ r ρ dV)
            for rname, mask in masks.items():
                masked = rho_dV * mask
                mu[s][rname]['x'][idx] = -np.sum(X * masked)
                mu[s][rname]['y'][idx] = -np.sum(Y * masked)
                mu[s][rname]['z'][idx] = -np.sum(Z * masked)

            # Azimuthal Fourier: c_{+1}(r_i) = sum_{ijk in bin} rho * exp(-i*theta) * dV
            rho_exp = rho_dV * exp_neg_itheta
            # Use bincount for fast binned summation
            flat_bin = bin_idx.ravel()
            flat_rho_exp_real = np.real(rho_exp).ravel()
            flat_rho_exp_imag = np.imag(rho_exp).ravel()
            bc_real = np.bincount(flat_bin, weights=flat_rho_exp_real, minlength=n_bins+2)
            bc_imag = np.bincount(flat_bin, weights=flat_rho_exp_imag, minlength=n_bins+2)
            # bin_idx is 1-indexed, so bins 1..n_bins correspond to our radial bins
            c1[s][idx, :] = bc_real[1:n_bins+1] + 1j * bc_imag[1:n_bins+1]

        if (idx + 1) % 20 == 0 or idx == n_times - 1:
            elapsed = timer.time() - t_start
            rate = (idx + 1) / elapsed
            eta = (n_times - idx - 1) / rate
            print(f"    {idx+1}/{n_times}  ({rate:.1f} snapshots/s, ETA {eta:.0f}s)")

    elapsed = timer.time() - t_start
    print(f"  Done in {elapsed:.1f}s ({n_times/elapsed:.1f} snapshots/s)")

    return times, mu, c1


# ═══════════════════════════════════════════════════════════════════════
# Step 5 — Dipole Angular Momenta
# ═══════════════════════════════════════════════════════════════════════

def finite_diff(y, dt):
    """Central finite differences with forward/backward at endpoints."""
    dy = np.empty_like(y)
    if len(y) < 3:
        dy[:] = 0.0
        if len(y) == 2:
            dy[:] = (y[1] - y[0]) / dt
        return dy
    dy[1:-1] = (y[2:] - y[:-2]) / (2 * dt)
    dy[0]    = (y[1] - y[0]) / dt
    dy[-1]   = (y[-1] - y[-2]) / dt
    return dy


def compute_angular_momenta(times, mu):
    """Compute L_z = mu_x * dmu_y/dt - mu_y * dmu_x/dt for each spin/region."""
    dt = DT_CUBE
    spins = ['alpha', 'beta']
    regs  = ['hub', 'rim', 'total']

    Lz = {}
    for s in spins:
        Lz[s] = {}
        for r in regs:
            dmux_dt = finite_diff(mu[s][r]['x'], dt)
            dmuy_dt = finite_diff(mu[s][r]['y'], dt)
            Lz[s][r] = mu[s][r]['x'] * dmuy_dt - mu[s][r]['y'] * dmux_dt

    # Derived: spin = alpha - beta, charge = alpha + beta
    Lz['spin']   = {r: Lz['alpha'][r] - Lz['beta'][r] for r in regs}
    Lz['charge'] = {r: Lz['alpha'][r] + Lz['beta'][r] for r in regs}

    return Lz


# ═══════════════════════════════════════════════════════════════════════
# Step 6 — Azimuthal Fourier Radial Profile
# ═══════════════════════════════════════════════════════════════════════

def compute_radial_profiles(times, c1, bin_centers):
    """Compute angular momentum density per radial shell from m=+1 Fourier."""
    dt = DT_CUBE
    n_times, n_bins = c1['alpha'].shape

    # dc1/dt via finite differences (per bin)
    dc1 = {}
    for s in ['alpha', 'beta']:
        dc1[s] = np.zeros_like(c1[s])
        for j in range(n_bins):
            dc1[s][:, j] = finite_diff(c1[s][:, j].real, dt) + \
                           1j * finite_diff(c1[s][:, j].imag, dt)

    # ell_z^sigma(r,t) = Im[ c1* . dc1/dt ]
    ell_z = {}
    for s in ['alpha', 'beta']:
        ell_z[s] = np.imag(np.conj(c1[s]) * dc1[s])

    ell_z['spin']   = ell_z['alpha'] - ell_z['beta']
    ell_z['charge'] = ell_z['alpha'] + ell_z['beta']

    # Time-averaged profiles (post-pulse)
    post_mask = times > T_POST_PULSE
    n_post = np.sum(post_mask)
    print(f"\n  Time-averaging: {n_post} snapshots with t > {T_POST_PULSE} au")

    avg = {}
    std = {}
    rms = {}
    for key in ['spin', 'charge', 'alpha', 'beta']:
        if n_post > 0:
            data = ell_z[key][post_mask]
        else:
            data = ell_z[key]
        avg[key] = np.mean(data, axis=0)
        std[key] = np.std(data, axis=0)
        rms[key] = np.sqrt(np.mean(data**2, axis=0))

    return ell_z, avg, std, rms


# ═══════════════════════════════════════════════════════════════════════
# Step 7 — Validation Against Bulk Dipoles
# ═══════════════════════════════════════════════════════════════════════

def validate_dipoles(times, mu, rt_data_path):
    """Compare cube-integrated total dipoles against rt_data time series."""
    print(f"\n  Loading {rt_data_path}...")
    rt = np.loadtxt(rt_data_path)
    # Columns (0-indexed): 0=time, 13=mu_alpha_x, 14=mu_alpha_y, 15=mu_alpha_z,
    #                       16=mu_beta_x, 17=mu_beta_y, 18=mu_beta_z
    rt_time = rt[:, 0]
    rt_mu = {
        'alpha': {'x': rt[:, 13], 'y': rt[:, 14], 'z': rt[:, 15]},
        'beta':  {'x': rt[:, 16], 'y': rt[:, 17], 'z': rt[:, 18]}
    }
    # Electric field columns (0-indexed): 7=Ex, 8=Ey, 9=Ez
    rt_field = {'x': rt[:, 7], 'y': rt[:, 8], 'z': rt[:, 9]}

    # Ground-state values (t=0)
    gs = {s: {c: rt_mu[s][c][0] for c in 'xyz'} for s in ['alpha', 'beta']}

    # Subsample rt_data at cube snapshot times
    # Each cube step corresponds to step = t_au / dt_sim, and we need row index = step
    results = []
    report_lines = ["Validation: cube-derived total dipoles vs rt_data (ground-state subtracted)\n"]
    report_lines.append(f"{'Spin':>6} {'Comp':>5} {'MaxAbsDev':>12} {'RMS':>12} {'MaxSignal':>12} {'%Err':>8}")
    report_lines.append("-" * 60)

    for s in ['alpha', 'beta']:
        for c in 'xyz':
            cube_vals = mu[s]['total'][c]  # these are delta-mu from cubes
            # Find matching rt_data rows
            rt_matched = np.zeros(len(times))
            for i, t in enumerate(times):
                row = int(round(t / DT_SIM))
                if row < len(rt_time):
                    rt_matched[i] = rt_mu[s][c][row] - gs[s][c]

            dev = cube_vals - rt_matched
            max_abs = np.max(np.abs(dev))
            rms = np.sqrt(np.mean(dev**2))
            max_signal = np.max(np.abs(rt_matched))
            pct = (rms / max_signal * 100) if max_signal > 0 else 0

            flag = " ***" if pct > 1.0 else ""
            report_lines.append(f"{s:>6} {c:>5} {max_abs:12.6e} {rms:12.6e} {max_signal:12.6e} {pct:7.2f}%{flag}")
            results.append((s, c, max_abs, rms, max_signal, pct))

    report = "\n".join(report_lines)
    print(report)

    # Save report
    with open(os.path.join(OUTPUT_DIR, "validation.txt"), 'w') as f:
        f.write(report + "\n")

    # Return rt_data for use in plotting
    return rt_time, rt_mu, rt_field, gs


# ═══════════════════════════════════════════════════════════════════════
# Step 8 — Visualization
# ═══════════════════════════════════════════════════════════════════════

def _save(fig, name):
    fig.savefig(os.path.join(OUTPUT_DIR, name + ".pdf"), bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, name + ".png"), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"    Saved {name}.pdf/.png")


def plot_dipoles(times, mu, rt_time, rt_field):
    """(a) Regional spin-resolved dipole moments vs time — 2×2 panel."""
    t_fs = times * AU_TO_FS

    # Save regional dipoles to text files
    for reg in ['hub', 'rim']:
        filename = os.path.join(OUTPUT_DIR, f"dip_{reg}.dat")
        data = np.column_stack([
            t_fs,
            mu['alpha'][reg]['x'] * AU_TO_DEBYE,
            mu['alpha'][reg]['y']* AU_TO_DEBYE,
            mu['beta'][reg]['x']* AU_TO_DEBYE,
            mu['beta'][reg]['y']* AU_TO_DEBYE
        ])
        header = "Time (fs) | dipole-x, alpha | dipole-y, alpha | dipole-x, beta | dipole-y, beta (dipoles in au)"
        np.savetxt(filename, data, header=header, fmt='%15.8e')
        print(f"    Saved {filename}")

    fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    regions = ['hub', 'rim']
    components = ['x', 'y']
    titles_row = ['Hub', 'Rim']
    titles_col = [r'$\mu_x$', r'$\mu_y$']

    # Field envelope for shading (use magnitude of field at cube times)
    field_col = {'x': 7, 'y': 8}

    for i, reg in enumerate(regions):
        for j, comp in enumerate(components):
            ax = axs[i, j]
            # Shaded field envelope on twin axis
            ax2 = ax.twinx()
            # Subsample field to full rt_data resolution for smooth curve
            ft_fs = rt_time * AU_TO_FS
            field = rt_field[comp] * 514200  # mV/nm
            ax2.fill_between(ft_fs, field, alpha=0.15, color='gray')
            ax2.set_ylim(-120, 120)
            if j == 1:
                ax2.set_ylabel('E (mV/nm)', fontsize=12, color='gray')
            else:
                ax2.set_yticklabels([])

            # Dipole traces
            mu_a = mu['alpha'][reg][comp] * AU_TO_DEBYE
            mu_b = mu['beta'][reg][comp]  * AU_TO_DEBYE
            ax.plot(t_fs, mu_a, color='C0', lw=1.2, label=r'$\alpha$')
            ax.plot(t_fs, mu_b, color='C1', lw=1.2, label=r'$\beta$')
            ax.set_ylabel('Dipole (D)', fontsize=12)
            ax.legend(fontsize=12, loc='upper right')
            ax.set_title(f'{titles_row[i]} — {titles_col[j]}', fontsize=12)

    for ax in axs[1]:
        ax.set_xlabel('Time (fs)')
    #fig.suptitle('Regional Spin-Resolved Dipole Moments', fontsize=12)
    fig.tight_layout()
    _save(fig, 'fig_dipoles')


def plot_dipoles_2(times, mu, rt_time, rt_field):
    """Regional dipole moments: rows=hub/rim, cols=alpha/beta, both x&y per panel."""
    t_fs = times * AU_TO_FS
    ft_fs = rt_time * AU_TO_FS

    fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    regions = ['hub', 'rim']
    spins = ['alpha', 'beta']
    titles_row = ['Hub', 'Rim']
    titles_col = [r'$\alpha$', r'$\beta$']

    for i, reg in enumerate(regions):
        for j, spin in enumerate(spins):
            ax = axs[i, j]
            # Shaded field envelope on twin axis
            ax2 = ax.twinx()
            field_x = rt_field['x'] * 514200
            ax2.fill_between(ft_fs, field_x, alpha=0.15, color='gray')
            ax2.set_ylim(-120, 120)
            if j == 1:
                ax2.set_ylabel('E (mV/nm)', fontsize=9, color='gray')
            else:
                ax2.set_yticklabels([])

            # Dipole traces — both x and y components
            mu_x = mu[spin][reg]['x'] * AU_TO_DEBYE
            mu_y = mu[spin][reg]['y'] * AU_TO_DEBYE
            ax.plot(t_fs, mu_x, color='C0', lw=1.2, label=r'$\mu_x$')
            ax.plot(t_fs, mu_y, color='C1', lw=1.2, label=r'$\mu_y$')
            ax.set_ylabel('Dipole (D)', fontsize=9)
            ax.legend(fontsize=8, loc='upper right')
            ax.set_title(f'{titles_row[i]} — {titles_col[j]}', fontsize=10)

    for ax in axs[1]:
        ax.set_xlabel('Time (fs)')
    fig.tight_layout()
    _save(fig, 'fig_dipoles_2')


def plot_angular_momentum(times, Lz):
    """(b) Angular momentum time traces — 3 stacked panels."""
    t_fs = times * AU_TO_FS
    fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    regions = ['hub', 'rim', 'total']
    titles = ['Hub', 'Rim', 'Total']

    for i, (reg, title) in enumerate(zip(regions, titles)):
        ax = axs[i]
        ax.plot(t_fs, Lz['alpha'][reg],  color='C0', lw=1.0, label=r'$L_z^{\alpha}$')
        ax.plot(t_fs, Lz['beta'][reg],   color='C1', lw=1.0, label=r'$L_z^{\beta}$')
        ax.plot(t_fs, Lz['spin'][reg],   color='C3', lw=1.5, label=r'$L_z^{\mathrm{spin}}$')
        ax.plot(t_fs, Lz['charge'][reg], color='C2', lw=1.5, label=r'$L_z^{\mathrm{charge}}$', ls='--')
        ax.axvline(T_POST_PULSE * AU_TO_FS, color='gray', ls=':', lw=0.8)
        ax.set_ylabel(r'$L_z$ (a.u.)', fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=12, ncol=2, loc='upper right')

    axs[-1].set_xlabel('Time (fs)')
    #fig.suptitle('Dipole Angular Momenta', fontsize=12)
    fig.tight_layout()
    _save(fig, 'fig_angular_momentum')


def plot_radial_profile(bin_centers, avg, std, rms):
    """(c) Radial profile of angular momentum density — RMS and mean."""
    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.12})

    # --- Top panel: RMS ---
    ax = axes[0]
    ax.plot(bin_centers, rms['spin'],   color='C3', lw=2,
            label=r'$\mathrm{RMS}(\ell_z^{\mathrm{spin}})$')
    ax.plot(bin_centers, rms['charge'], color='C2', lw=2, ls='--',
            label=r'$\mathrm{RMS}(\ell_z^{\mathrm{charge}})$')
    ax.plot(bin_centers, rms['alpha'],  color='C0', lw=1, ls=':',
            label=r'$\mathrm{RMS}(\ell_z^{\alpha})$')
    ax.plot(bin_centers, rms['beta'],   color='C1', lw=1, ls=':',
            label=r'$\mathrm{RMS}(\ell_z^{\beta})$')

    ax.set_ylabel(r'RMS $\ell_z$ (a.u.)')
    #ax.set_title('Radial Profile of Angular Momentum Density')
    ax.legend(fontsize=12, ncol=2)

    yhi = ax.get_ylim()[1]
    for r, lbl in [(2.68, 'hub C'), (4.65, 'spoke C')]:
        ax.axvline(r, color='gray', ls='--', lw=0.7, alpha=0.7)
        ax.text(r + 0.1, yhi * 0.92, lbl, fontsize=11, color='gray', va='top')
    ax.axvspan(6.6, 8.7, alpha=0.06, color='blue')
    ax.text(7.6, yhi * 0.92, 'rim C', fontsize=11, color='gray', ha='center', va='top')

    # --- Bottom panel: Mean ± std (for reference) ---
    ax2 = axes[1]
    ax2.plot(bin_centers, avg['spin'],   color='C3', lw=2,
             label=r'$\langle \ell_z^{\mathrm{spin}} \rangle$')
    ax2.fill_between(bin_centers,
                     avg['spin'] - std['spin'],
                     avg['spin'] + std['spin'],
                     color='C3', alpha=0.2)

    ax2.plot(bin_centers, avg['charge'], color='C2', lw=2, ls='--',
             label=r'$\langle \ell_z^{\mathrm{charge}} \rangle$')
    ax2.fill_between(bin_centers,
                     avg['charge'] - std['charge'],
                     avg['charge'] + std['charge'],
                     color='C2', alpha=0.2)

    ax2.set_xlabel(r'$r_\perp$ (Bohr)')
    ax2.set_ylabel(r'$\langle \ell_z \rangle$ (a.u.)')
    ax2.legend(fontsize=11)
    ax2.set_xlim(0, R_MAX)

    ylo2, yhi2 = ax2.get_ylim()
    for r, lbl in [(2.68, 'hub C'), (4.65, 'spoke C')]:
        ax2.axvline(r, color='gray', ls='--', lw=0.7, alpha=0.7)
        ax2.text(r + 0.1, yhi2 - 0.05*(yhi2-ylo2), lbl, fontsize=11, color='gray', va='top')
    ax2.axvspan(6.6, 8.7, alpha=0.06, color='blue')
    ax2.text(7.6, yhi2 - 0.05*(yhi2-ylo2), 'rim C', fontsize=11, color='gray',
             ha='center', va='top')

    fig.tight_layout()
    _save(fig, 'fig_radial_profile')


def plot_parametric(times, mu):
    """(d) Parametric dipole trajectories — 2×2 panel."""
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    regions = ['hub', 'rim']
    spins   = ['alpha', 'beta']
    titles_r = ['Hub', 'Rim']
    titles_s = [r'$\alpha$', r'$\beta$']
    colors_s = ['C0', 'C1']

    t_fs = times * AU_TO_FS
    norm = Normalize(vmin=t_fs[0], vmax=t_fs[-1])
    post = times > T_POST_PULSE
    axis_ratios = {}
    ellipse_areas = {}

    for i, reg in enumerate(regions):
        for j, spin in enumerate(spins):
            ax = axs[i, j]
            mx = mu[spin][reg]['x'] * AU_TO_DEBYE
            my = mu[spin][reg]['y'] * AU_TO_DEBYE

            # Colored line segments
            points = np.column_stack([mx, my]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap='viridis', norm=norm)
            lc.set_array(t_fs[:-1])
            lc.set_linewidth(1.2)
            ax.add_collection(lc)
            ax.autoscale()

            # Direction arrow near the middle
            mid = len(mx) // 2
            ax.annotate('', xy=(mx[mid+1], my[mid+1]), xytext=(mx[mid], my[mid]),
                        arrowprops=dict(arrowstyle='->', color=colors_s[j], lw=1.5))

            # --- Ellipticity from covariance of post-pulse trajectory ---
            axis_ratio = np.nan
            ell_area = np.nan
            if np.sum(post) > 2:
                mx_post = mx[post]
                my_post = my[post]
                cov = np.cov(mx_post, my_post)
                eigvals = np.linalg.eigvalsh(cov)
                eigvals = np.sort(eigvals)[::-1]  # descending
                semi_a = np.sqrt(eigvals[0])  # major semi-axis (std dev)
                semi_b = np.sqrt(eigvals[1])  # minor semi-axis (std dev)
                if eigvals[0] > 0:
                    axis_ratio = semi_b / semi_a  # b/a: 1=circle, 0=line
                else:
                    axis_ratio = 0.0
                ell_area = np.pi * semi_a * semi_b  # D^2
                ax.text(0.03, 0.97,
                        f'$b/a = {axis_ratio:.2f}$',#\n$A = {ell_area:.1e}$ D$^2$',
                        transform=ax.transAxes, fontsize=11, va='top',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            axis_ratios[(reg, spin)] = axis_ratio
            ellipse_areas[(reg, spin)] = ell_area

            ax.set_xlabel(r'$\mu_x$ (D)', fontsize=12)
            if j == 0:
                ax.set_ylabel(r'$\mu_y$ (D)', fontsize=12)
            #ax.set_title(f'{titles_r[i]} — {titles_s[j]}', fontsize=11)
            ax.set_aspect('equal', adjustable='datalim')

            # Hub dipole values are O(10^-3) D — use scientific notation
            if reg == 'hub':
                ax.ticklabel_format(style='scientific', scilimits=(0, 0),
                                    axis='both', useMathText=True)
            if reg == 'rim':
                ax.ticklabel_format(style='scientific', scilimits=(0, 0),
                                    axis='both', useMathText=True)

    #fig.suptitle('Parametric Dipole Trajectories', fontsize=12)
    fig.tight_layout()
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    fig.colorbar(sm, ax=axs, label='Time (fs)', shrink=0.6)
    _save(fig, 'fig_parametric')

    # Print summary table
    print("\n  Elliptical axis ratio b/a (1 = circle, 0 = line)")
    print("  and covariance ellipse area A = π·σ_a·σ_b (D²):")
    print(f"  {'':12s}  {'b/a α':>8s}  {'b/a β':>8s}  {'A α (D²)':>12s}  {'A β (D²)':>12s}")
    for reg in regions:
        ba_a = axis_ratios[(reg, 'alpha')]
        ba_b = axis_ratios[(reg, 'beta')]
        ea_a = ellipse_areas[(reg, 'alpha')]
        ea_b = ellipse_areas[(reg, 'beta')]
        print(f"  {reg:12s}  {ba_a:8.3f}  {ba_b:8.3f}  {ea_a:12.3e}  {ea_b:12.3e}")
    return axis_ratios, ellipse_areas


def plot_phase_evolution(times, c1, bin_centers):
    """(e) Phase of c_{+1} at selected radii vs time."""
    t_fs = times * AU_TO_FS
    radii_target = [2.3, 4.6, 6.9]

    fig, axs = plt.subplots(len(radii_target), 1, figsize=(8, 2.5 * len(radii_target)), sharex=True)
    if len(radii_target) == 1:
        axs = [axs]

    for k, r_target in enumerate(radii_target):
        j = np.argmin(np.abs(bin_centers - r_target))
        r_actual = bin_centers[j]
        ax = axs[k]

        # Amplitude threshold: mask where |c1| < 5% of its peak,
        # then unwrap within each contiguous valid segment
        for spin, color, label in [
            ('alpha', 'C0', r'$\alpha$'),
            ('beta',  'C1', r'$\beta$')
        ]:
            amp = np.abs(c1[spin][:, j])
            thresh = 0.05 * np.max(amp)
            valid = amp > thresh
            phase_raw = np.angle(c1[spin][:, j])
            phase_out = np.full_like(phase_raw, np.nan)
            # Find contiguous valid segments and unwrap each independently
            segments = np.diff(np.concatenate([[0], valid.astype(int), [0]]))
            starts = np.where(segments == 1)[0]
            ends   = np.where(segments == -1)[0]
            for s, e in zip(starts, ends):
                phase_out[s:e] = np.unwrap(phase_raw[s:e])
            ax.plot(t_fs[:], phase_out[:], color=color, lw=1.2, label=label)

        ax.axvline(T_POST_PULSE * AU_TO_FS, color='gray', ls=':', lw=0.8)
        ax.set_ylabel('Phase (rad)', fontsize=12)
        ax.set_title(f'r = {r_actual:.2f} Bohr', fontsize=12)
        ax.legend(fontsize=12, loc='upper left')

    axs[-1].set_xlabel('Time (fs)')
    #fig.suptitle(r'Phase Evolution of $c_{+1}^{\sigma}(r, t)$', fontsize=12)
    fig.tight_layout()
    _save(fig, 'fig_phase_evolution')


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("Spin Current Density Analysis — Coronene Anion")
    print("=" * 60)

    # Step 0
    print("\nStep 0: File discovery")
    pairs = discover_cube_pairs(CUBE_DIR)
    if not pairs:
        print("ERROR: No cube file pairs found!")
        return

    # Steps 1-2: Parse first cube to build grid
    print("\nSteps 1-2: Building coordinate grid from first cube file")
    origin, axes, natoms, atom_data, _, dV = parse_cube(pairs[0][1])
    grid = build_grid(origin, axes)
    print(f"  Grid: {grid['Nx']}×{grid['Ny']}×{grid['Nz']}, dV = {dV:.6e} Bohr³")
    print(f"  X range: [{grid['X'].min():.2f}, {grid['X'].max():.2f}] Bohr")
    print(f"  Y range: [{grid['Y'].min():.2f}, {grid['Y'].max():.2f}] Bohr")

    # Step 3
    print("\nStep 3: Defining spatial regions")
    regions = define_regions(grid)
    n_hub = np.sum(regions['hub_mask'])
    n_tot = regions['hub_mask'].size
    print(f"  Hub voxels: {n_hub} ({100*n_hub/n_tot:.1f}%)")
    print(f"  Rim voxels: {n_tot - n_hub} ({100*(n_tot-n_hub)/n_tot:.1f}%)")
    print(f"  Radial bins: {len(regions['bin_centers'])}, dr = {regions['bin_edges'][1]:.3f} Bohr")

    # Step 4
    print("\nStep 4: Processing snapshots")
    times, mu, c1 = process_snapshots(pairs, grid, regions, dV)

    # Step 5
    print("\nStep 5: Computing dipole angular momenta")
    Lz = compute_angular_momenta(times, mu)
    # Print some summary stats for post-pulse region
    post = times > T_POST_PULSE
    if np.any(post):
        for reg in ['hub', 'rim', 'total']:
            lz_s = np.mean(Lz['spin'][reg][post])
            lz_c = np.mean(Lz['charge'][reg][post])
            print(f"  {reg:>5}: <L_z^spin> = {lz_s:+.6e}, <L_z^charge> = {lz_c:+.6e}")

    # Step 6
    print("\nStep 6: Computing radial profiles")
    ell_z, avg, std, rms = compute_radial_profiles(times, c1, regions['bin_centers'])

    # Step 7
    print("\nStep 7: Validation")
    rt_time, rt_mu, rt_field, gs = validate_dipoles(times, mu, RT_DATA_PATH)

    # Save all computed data
    print("\n  Saving data_summary.npz...")
    np.savez(os.path.join(OUTPUT_DIR, "data_summary.npz"),
             times=times,
             bin_centers=regions['bin_centers'],
             # Dipole moments (flatten the nested dict)
             **{f"mu_{s}_{r}_{c}": mu[s][r][c]
                for s in ['alpha','beta'] for r in ['hub','rim','total'] for c in 'xyz'},
             # Angular momenta
             **{f"Lz_{k}_{r}": Lz[k][r]
                for k in ['alpha','beta','spin','charge'] for r in ['hub','rim','total']},
             # Fourier coefficients
             c1_alpha=c1['alpha'], c1_beta=c1['beta'],
             # Radial profiles
             **{f"avg_{k}": avg[k] for k in ['spin','charge','alpha','beta']},
             **{f"std_{k}": std[k] for k in ['spin','charge','alpha','beta']},
             **{f"rms_{k}": rms[k] for k in ['spin','charge','alpha','beta']})

    # Step 8
    print("\nStep 8: Generating plots")
    try:
        plot_dipoles(times, mu, rt_time, rt_field)
        plot_dipoles_2(times, mu, rt_time, rt_field)
        plot_angular_momentum(times, Lz)
        plot_radial_profile(regions['bin_centers'], avg, std, rms)
        axis_ratios, ellipse_areas = plot_parametric(times, mu)
        plot_phase_evolution(times, c1, regions['bin_centers'])
        print("\nAll plots generated successfully.")
    except Exception as e:
        print(f"Warning: Some plotting functions failed due to missing matplotlib dependency:\n{e}")

    print("\n" + "=" * 60)
    print("Analysis complete. Outputs in:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()
