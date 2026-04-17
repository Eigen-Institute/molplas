**Input and analysis scripts for "Spin and Charge Plasmons in Polycyclic Aromatic Hydrocarbons: Topology-Controlled Ring Currents and Coherent Wavepacket Dynamics with Real-Time Time-Dependent Density Functional Theory"**

# DFT/TDDFT/RT-TDDFT
Running gpu4pyscf calculations, including rt-G4PS, involves a python wrapper and json input pair.
#### Wrappers:
- `dft_json.py`: single-point DFT
- `geom_json.py`: geometry optimization
- `tddft_json.py`: LR-TDDFT, including transition density cube file generation
- `rttddft_json.py`: RT-TDDFT simulation

#### Input Files:
- `dft.json`: single-point DFT
- `geom.json`: geometry optimization
- `tddft.json`: LR-TDDFT, including transition density cube file generation
- `rttddft.json`: RT-TDDFT simulation

#### Usage:
`python dft_json.py -i dft.json 2>&1 | tee dft.out`
# Analysis scripts
- `analyze_spin_current.py`: analyzes spin density angular momentum, radial distribution, parametric dipole trajectories for a series of cube files generated using `rttddft_json.py` with `rttddft.json`.
- `plasmonicity.py`: plasmonicity index from transition density cube file of total charge density ($\rho=\rho_\alpha+\rho_\beta$)
- `spi.py`: spin plasmonicity index from spin-polarized transition density cubes generated using open-shell `tddft_json.py` with `tddft.json`
- `pyscf_spectrum.py`: absorption spectrum from TDDFT output from `tddft_json.py` and `tddft.json`
