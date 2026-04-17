######################################################################################
#    Calculates the plasmonicity index, PI, of a transition density cube file
#    See Bursi, L.; Calzolari, A.; Corni, S.; Molinari, E. Quantifying the Plasmonic Character of Optical Excitations in Nanostructures. ACS Photonics 2016, 3, 520−525
#    
#    CTC 2026
#    http://eigeninstitute.org
#
#    usage: plasmonicity.py file.cube
######################################################################################
import sys
import numpy as np

def solve_poisson_fft(density_grid, voxel_size):
    """
    Solves V(r) = integral [ rho(r') / |r-r'| ] d3r' using FFT.
    Uses zero-padding to simulate an isolated system.
    """
    nx, ny, nz = density_grid.shape
    # 1. Zero-pad the grid to double size to prevent periodic wrap-around
    pad_x, pad_y, pad_z = 2*nx, 2*ny, 2*nz
    padded_grid = np.zeros((pad_x, pad_y, pad_z))
    padded_grid[:nx, :ny, :nz] = density_grid

    # 2. Compute FFT of the density
    rho_k = np.fft.fftn(padded_grid)

    # 3. Define k-vectors
    kx = 2 * np.pi * np.fft.fftfreq(pad_x, d=voxel_size[0])
    ky = 2 * np.pi * np.fft.fftfreq(pad_y, d=voxel_size[1])
    kz = 2 * np.pi * np.fft.fftfreq(pad_z, d=voxel_size[2])
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    
    K2 = KX**2 + KY**2 + KZ**2
    # Avoid division by zero at the DC component (k=0)
    K2[0, 0, 0] = np.inf 

    # 4. Green's function in k-space: V(k) = rho(k) / k^2 (Bursi convention)
    v_k = rho_k / K2

    # 5. Inverse FFT to get potential in real space
    v_total = np.real(np.fft.ifftn(v_k))

    # Return the original region (unpadded)
    return v_total[:nx, :ny, :nz]

def calculate_pi(cube_file):
    """
    Reads cube, calculates induced potential, and returns Bursi PI.
    """
    # Simple parser for cube data (atoms and headers skipped for brevity)
    with open(cube_file, 'r') as f:
        lines = f.readlines()
        
    header = lines[2:6]
    nx, dx = int(header[1].split()[0]), float(header[1].split()[1])
    ny, dy = int(header[2].split()[0]), float(header[2].split()[2])
    nz, dz = int(header[3].split()[0]), float(header[3].split()[3])
    
    voxel_vol = dx * dy * dz
    # Flattened data starts after atoms (line 6 + number of atoms)
    num_atoms = int(header[0].split()[0])
    data = np.fromstring(" ".join(lines[6+num_atoms:]), sep=' ')
    rho = data.reshape((nx, ny, nz))

    # Solve for induced potential
    v_ind = solve_poisson_fft(rho, (dx, dy, dz))

    # Bursi Eq 5: PI = Integral(|V_ind|^2) / Integral(|rho|^2)
    # Numerical integration: sum(val^2 * dV)
    numerator = np.sum(v_ind**2) * voxel_vol
    denominator = np.sum(rho**2) * voxel_vol
    
    return numerator / denominator

# Execute
pi_index = calculate_pi(sys.argv[1]) 
print(f"Plasmonicity Index: {pi_index}")
#plot_slice()
