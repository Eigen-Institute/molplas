#===================================================================
# Single point DFT using gpu4pyscf version 1.6.0 and PySCF version 2.12.0
#
# see: 
#       Sun, Q.; Zhang, X.; Banerjee, S.; Bao, P.; Barbry, M.; Blunt, N. S.; Bogdanov, N. A.; Booth, G. H.; Chen, J.; Cui, Z.-H.; Eriksen, J. J.; Gao, Y.; Guo, S.; Hermann, J.; Hermes, M. R.; Koh, K.; Koval, P.; Lehtola, S.; Li, Z.; Liu, J. et al. Recent developments in the PySCF program package, The Journal of Chemical Physics 2020, 153, 024109.
#
#       Li, R.; Sun, Q.; Zhang, X.; Chan, G. K.-L. Introducing GPU Acceleration into the Python-Based Simulations of Chemistry Framework, The Journal of Physical Chemistry A 2025, 129, 1459-1468
#
# usage: python dft_json.py -i dft.json 
#===================================================================

import numpy as np
import cupy
import sys, time
import json
import argparse
import h5py
from pyscf.lib import H5FileWrap
from pyscf import gto, symm
from pyscf.tools import cubegen, molden
from gpu4pyscf import dft, tdscf
import pandas as pd

# Parse Command Line Arguments
parser = argparse.ArgumentParser(description='GPU4PySCF Print MOs')
parser.add_argument('--input','-i', type=str, help='Path to input JSON file')
args = parser.parse_args()

# Load Input Data
if args.input is None:
    print("Error: Input file required. Use --input <file.json>")
    sys.exit(1)

with open(args.input, 'r') as f:
    input_data = json.load(f)
properties = input_data.get("property", {})
mol_data = input_data.get('molecule', {})
theory_data = input_data.get('theory', {})
cube_data = input_data.get("cube", {})

# Print Input Parameters
print('')
print("*"*100)
print("Reading calculation input from:"+args.input)
print("")
print(input_data)
print("*"*100)

# Name of the calculation
calcName = input_data.get('calcName', 'dft_calc')
calcName = calcName.replace(" ","_").lower()

# Define Molecule
mol = gto.M(
    atom=mol_data.get('atom', 'tet.xyz'),
    basis=mol_data.get('basis', '3-21g'),
    verbose=mol_data.get('verbose', 4),
    charge=mol_data.get('charge', 0),
    spin=mol_data.get('spin', 0),
    symmetry=mol_data.get('symmetry', False)
)

# Define Theory
isOs = False
if theory_data.get("shell", "closed").lower() == "open":
    ks = dft.UKS(mol)
    isOs = True
else:
    ks = dft.RKS(mol)

ks.xc = theory_data.get('xc', 'pbe0')
ks.chkfile = theory_data.get('initial guess', calcName+'.chk')
if h5py.is_hdf5(ks.chkfile):
    ks.init_guess = 'chkfile' 
else:
    print(" **** No initial checkpoint file found. Proceeding without it.")
ks.kernel()

#--------------------------------------
#   SCF finished
#   Begin Post-SCF analysis
#--------------------------------------

#--------------------------------------
# Print MO energies and occupations
#--------------------------------------
if properties.get("print MO energies",False):
    print("\n**** MO energies and occupations ****")
    
    if isOs:
        print("MO    energy (alpha)    occ (alpha)    energy (beta)    occ (beta)")
        # Handle Alpha and Beta separately
        e_alpha = ks.mo_energy[0]
        e_beta = ks.mo_energy[1]
        occ_alpha = ks.mo_occ[0]
        occ_beta = ks.mo_occ[1]
    
        for i in range(len(e_alpha)):
            print (i+1, e_alpha[i],"  ",occ_alpha[i],"      ", e_beta[i], "  ",occ_beta[i])
            if e_alpha[i] < 0:
                if e_alpha[i+1] > 0:
                    print("----------------------------------------------------------")
        print("\n")
    #    for spin_idx, spin_label in enumerate(['Alpha', 'Beta']):
    #        e = ks.mo_energy[spin_idx]
    #        occ = ks.mo_occ[spin_idx]
    #        
    #        for i in range(int(0.5*len(e))):
    #            print(i+1, e[i],"  ", occ[i],"      ", e[i+int(0.5*len(e))],"  ", occ[i+int(0.5*len(e))])
    else:
        print("MO    energy (alpha)    occ (alpha)")
        # Handle RKS (Closed Shell)
        e = ks.mo_energy
        occ = ks.mo_occ
        for i in range(len(e)):
            print(i+1, e[i], occ[i], spin_label)

#--------------------------------------
# Perform Analyze
#--------------------------------------
if properties.get("analyze", False):
    ks.analyze()
if properties.get("scf summary", False):
    ks.dump_scf_summary()

#--------------------------------------
# Perform Symmetry Analysis
#--------------------------------------

if mol.symmetry:
    mo_coeff = ks.mo_coeff
    if isOs:
        # For UKS, mo_coeff is [mo_a, mo_b]
        mo_a = mo_coeff[0].get() if hasattr(mo_coeff[0], 'get') else mo_coeff[0]
        mo_b = mo_coeff[1].get() if hasattr(mo_coeff[1], 'get') else mo_coeff[1]
        irreps_a = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo_a)
        irreps_b = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo_b)
        print("\n   Alpha MO irreps:", irreps_a)
        print("   Beta  MO irreps:", irreps_b, "\n")
    else:
        mo = mo_coeff.get() if hasattr(mo_coeff, 'get') else mo_coeff
        irreps = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo)
        print("\n   MO irreps:", irreps, "\n")

#--------------------------------------
# Dump MO Cube Files
#--------------------------------------

cube_spec = cube_data.get('MO cube', None)
cube_resolution = cube_data.get('cube resolution', None)
margin = cube_data.get("margin", 3.0)   

if cube_spec is not None:
    if isOs:
        nmo = ks.mo_coeff[0].shape[1]
    else:
        nmo = ks.mo_coeff.shape[1]

    if isinstance(cube_spec, (int, str)):
        cube_spec = [cube_spec]
    
    cubeList = []
    for item in cube_spec:
        if isinstance(item, str):
            if item.lower() == 'all':
                cubeList = list(range(nmo))
                break
            elif '-' in item:
                try:
                    start, end = map(int, item.split('-'))
                    cubeList.extend(range(start, end + 1))
                except ValueError:
                    print(f"Warning: Invalid range format '{item}'. Skipping.")
            else:
                try:
                    cubeList.append(int(item))
                except ValueError:
                    print(f"Warning: Invalid MO index '{item}'. Skipping.")
        elif isinstance(item, int):
            cubeList.append(item)
    
    # Remove duplicates and sort
    cubeList = sorted(list(set(cubeList)))

    for mo_idx in cubeList:
        if isOs:
            # Alpha
            mo_name_a = f"{calcName}_mo{mo_idx}_alpha.cube"
            print(f"Writing Alpha MO cube: {mo_name_a}")
            mo_coeff_a = ks.mo_coeff[0][:, mo_idx]
            if hasattr(mo_coeff_a, 'get'): mo_coeff_a = mo_coeff_a.get()
            cubegen.orbital(mol, mo_name_a, mo_coeff_a, margin=margin, resolution=cube_resolution)
            
            # Beta
            mo_name_b = f"{calcName}_mo{mo_idx}_beta.cube"
            print(f"Writing Beta MO cube: {mo_name_b}")
            mo_coeff_b = ks.mo_coeff[1][:, mo_idx]
            if hasattr(mo_coeff_b, 'get'): mo_coeff_b = mo_coeff_b.get()
            cubegen.orbital(mol, mo_name_b, mo_coeff_b, margin=margin, resolution=cube_resolution)
        else:
            mo_name = f"{calcName}_mo{mo_idx}.cube"
            print(f"Writing MO cube: {mo_name}")
            mo_coeff_val = ks.mo_coeff[:, mo_idx]
            if hasattr(mo_coeff_val, 'get'): mo_coeff_val = mo_coeff_val.get()
            cubegen.orbital(mol, mo_name, mo_coeff_val, margin=margin, resolution=cube_resolution)

print("\n **** Calculation finished.\n")
