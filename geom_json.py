#===================================================================
# Geometry optimization using gpu4pyscf version 1.6.0 and PySCF version 2.12.0
#
# see: 
#       Sun, Q.; Zhang, X.; Banerjee, S.; Bao, P.; Barbry, M.; Blunt, N. S.; Bogdanov, N. A.; Booth, G. H.; Chen, J.; Cui, Z.-H.; Eriksen, J. J.; Gao, Y.; Guo, S.; Hermann, J.; Hermes, M. R.; Koh, K.; Koval, P.; Lehtola, S.; Li, Z.; Liu, J. et al. Recent developments in the PySCF program package, The Journal of Chemical Physics 2020, 153, 024109.
#
#       Li, R.; Sun, Q.; Zhang, X.; Chan, G. K.-L. Introducing GPU Acceleration into the Python-Based Simulations of Chemistry Framework, The Journal of Physical Chemistry A 2025, 129, 1459-1468
#
# usage: geom_json.py -i geom.json 
#===================================================================

import numpy as np
import cupy
import sys, time
import json
import argparse
from pyscf import gto
from pyscf.tools import cubegen
from gpu4pyscf import dft,tdscf
from pyscf.geomopt import geometric_solver
startTime=time.time()

# Parse Command Line Arguments
parser = argparse.ArgumentParser(description='GPU4PySCF Geometry Optimization')
parser.add_argument('--input','-i', type=str, help='Path to input JSON file')
args = parser.parse_args()

# Load Input Data
with open(args.input, 'r') as f:
    input_data = json.load(f)

# Name of the calculation
calcName = input_data.get('calcName', 'geom_opt')
calcName = calcName.replace(" ","_").lower()

# Define Molecule
mol_data = input_data.get('molecule', {})
mol = gto.M(
    atom=mol_data.get('atom', 'opt.xyz'),
    basis=mol_data.get('basis', '3-21g'),
    verbose=mol_data.get('verbose', 4),
    charge=mol_data.get('charge', 0),
    spin=mol_data.get('spin', 0)
)

# Define Theory
theory_data = input_data.get('theory', {})
if theory_data["shell"] == "open":
    ks = dft.UKS(mol)
    isOs = True
if theory_data["shell"] == "closed":
    ks =  dft.RKS(mol)
    isCs = True
ks.xc = theory_data.get('xc', 'pbe0')
ks.chkfile = theory_data.get("initial guess",calcName+'.chk')
ks.init_guess = 'chk'
ks.kernel()

# Geometry Optimization
geomopt_data = input_data.get('geomopt', {})
maxsteps = geomopt_data.get('maxsteps', 100)
optimizedFile = geomopt_data.get('output_file', calcName + '_opt.xyz')

mol_eq = geometric_solver.optimize(ks, maxsteps=maxsteps)
print(mol_eq.tostring())

# Get optimized coordinates in Angstroms
coords = mol_eq.atom_coords(unit='Ang')
symbols = [atom[0] for atom in mol_eq.atom]

# Print in XYZ format
print(len(symbols))
print("Optimized geometry")
for sym, (x, y, z) in zip(symbols, coords):
    print(f"{sym:2s} {x:12.6f} {y:12.6f} {z:12.6f}")
# Write in XYZ format
with open(optimizedFile, 'w') as f:
    f.write(str(len(symbols))+'\n')
    f.write("Optimized geometry "+calcName+" @"+mol.basis+"/"+ks.xc+"\n")
    for sym, (x, y, z) in zip(symbols, coords):
        f.write(f"{sym:2s} {x:12.6f} {y:12.6f} {z:12.6f}\n")



endTime=time.time()
print("\n\n     wall time:",str(endTime-startTime)+" s")

