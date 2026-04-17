######################################################################################
#    Calculates the plasmonicity index (PI) and spin plasmonicity index (SPI)
#    from alpha and beta transition density cube files.
#
#    PI  uses the charge density:      rho   = rho_alpha + rho_beta
#    SPI uses the spin density:        rho_s = rho_alpha - rho_beta
#
#    See:
#      Bursi et al., ACS Photonics 2016, 3, 520-525  (PI definition, Eq. 5)
#
#    CTC 2026
#    http://eigeninstitute.org
#
#    usage: plasmonicity.py alpha.cube beta.cube
######################################################################################
import sys
import numpy as np


def solve_poisson_fft(density_grid, voxel_size):
    """
    Solves V(r) = integral [ rho(r') / |r-r'| ] d3r' using FFT.
    Uses zero-padding to simulate an isolated (non-periodic) system.
    """
    nx, ny, nz = density_grid.shape
    # Zero-pad to double size to prevent periodic wrap-around
    pad_x, pad_y, pad_z = 2 * nx, 2 * ny, 2 * nz
    padded_grid = np.zeros((pad_x, pad_y, pad_z))
    padded_grid[:nx, :ny, :nz] = density_grid

    rho_k = np.fft.fftn(padded_grid)

    kx = 2 * np.pi * np.fft.fftfreq(pad_x, d=voxel_size[0])
    ky = 2 * np.pi * np.fft.fftfreq(pad_y, d=voxel_size[1])
    kz = 2 * np.pi * np.fft.fftfreq(pad_z, d=voxel_size[2])
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')

    K2 = KX**2 + KY**2 + KZ**2
    K2[0, 0, 0] = np.inf  # Remove DC divergence

    # Green's function in k-space: V(k) = rho(k) / k^2  (Bursi convention)
    v_k = rho_k / K2

    v_total = np.real(np.fft.ifftn(v_k))

    return v_total[:nx, :ny, :nz]


def read_cube(cube_file):
    """
    Reads a Gaussian cube file.
    Returns (density_grid, voxel_size, voxel_volume).
    Handles both row-major Gaussian ordering and standard reshape.
    """
    with open(cube_file, 'r') as f:
        lines = f.readlines()

    # Header lines 2-5 (0-indexed): natoms+origin, nx+dx, ny+dy, nz+dz
    natoms = int(lines[2].split()[0])
    nx, dx = int(lines[3].split()[0]), float(lines[3].split()[1])
    ny, dy = int(lines[4].split()[0]), float(lines[4].split()[2])
    nz, dz = int(lines[5].split()[0]), float(lines[5].split()[3])

    voxel_vol = dx * dy * dz

    data_start = 6 + natoms
    data = np.fromstring(" ".join(lines[data_start:]), sep=' ')
    rho = data.reshape((nx, ny, nz))

    return rho, (dx, dy, dz), voxel_vol


def plasmonicity_index(density, voxel_size, voxel_vol):
    """
    Computes the Bursi plasmonicity index (Eq. 5) for a given density grid:

        PI = Integral(|V_ind|^2 dV) / Integral(|rho|^2 dV)

    where V_ind is the Coulomb potential of rho.
    """
    v_ind = solve_poisson_fft(density, voxel_size)
    numerator   = np.sum(v_ind**2)   * voxel_vol
    denominator = np.sum(density**2) * voxel_vol
    return numerator / denominator, v_ind


def main():
    if len(sys.argv) != 3:
        print("Usage: plasmonicity.py alpha.cube beta.cube")
        sys.exit(1)

    alpha_file, beta_file = sys.argv[1], sys.argv[2]

    print(f"Reading alpha density: {alpha_file}")
    rho_a, voxel_size, voxel_vol = read_cube(alpha_file)

    print(f"Reading beta  density: {beta_file}")
    rho_b, voxel_size_b, voxel_vol_b = read_cube(beta_file)

    # Sanity check: grids must match
    if rho_a.shape != rho_b.shape:
        raise ValueError(
            f"Grid shape mismatch: alpha {rho_a.shape} vs beta {rho_b.shape}"
        )
    if not np.allclose(voxel_size, voxel_size_b, rtol=1e-5):
        raise ValueError(
            f"Voxel size mismatch: alpha {voxel_size} vs beta {voxel_size_b}"
        )

    # --- Charge density (alpha + beta) ---
    rho_charge = rho_a + rho_b
    pi_charge, v_ind_charge = plasmonicity_index(rho_charge, voxel_size, voxel_vol)

    # --- Spin density (alpha - beta) ---
    rho_spin = rho_a - rho_b
    pi_spin, v_ind_spin = plasmonicity_index(rho_spin, voxel_size, voxel_vol)

    # --- Individual channel PIs (for reference) ---
    pi_alpha, _ = plasmonicity_index(rho_a, voxel_size, voxel_vol)
    pi_beta,  _ = plasmonicity_index(rho_b, voxel_size, voxel_vol)

    # --- Cross-channel Coulomb coupling term (numerator cross term of full PI) ---
    v_ind_a = solve_poisson_fft(rho_a, voxel_size)
    v_ind_b = solve_poisson_fft(rho_b, voxel_size)
    cross_numerator = 2 * np.sum(v_ind_a * v_ind_b) * voxel_vol

    print()
    print("=" * 52)
    print("  Plasmonicity Results")
    print("=" * 52)
    print(f"  PI  (charge, rho_a + rho_b) : {pi_charge:.6f}")
    print(f"  SPI (spin,   rho_a - rho_b) : {pi_spin:.6f}")
    print("-" * 52)
    print(f"  PI  alpha channel only       : {pi_alpha:.6f}")
    print(f"  PI  beta  channel only       : {pi_beta:.6f}")
    print(f"  Cross-coupling (2*<Va|Vb>)   : {cross_numerator:.6e}")
    print("=" * 52)

    return pi_charge, pi_spin


if __name__ == "__main__":
    main()
