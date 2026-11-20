###From FlowER-OM, only from verify_valid_BE.py

import torch
from torch.utils.data import Dataset
from rdkit import Chem
import numpy as np
import sys
from rdkit import RDLogger
from rdkit.Chem import AllChem, rdchem
RDLogger.DisableLog('rdApp.*') 
from multiprocessing import Pool, cpu_count

np.set_printoptions(threshold=sys.maxsize, linewidth=500)
torch.set_printoptions(profile="full")

# changed Pd to 2.1
# a_en = {0: '0', 1: '2.2', 2: 'no data', 3: '0.98', 4: '1.57', 5: '2.04', 6: '2.55', 7: '3.04', 8: '3.44', 9: '3.98', 10: 'no data', 11: '0.93', 12: '1.31', 13: '1.61', 14: '1.9', 15: '2.19', 16: '2.58', 
#         17: '3.16', 18: 'no data', 19: '0.82', 20: '1', 21: '1.36', 22: '1.54', 23: '1.63', 24: '1.66', 25: '1.55', 26: '1.83', 27: '1.88', 28: '1.91', 29: '1.9', 30: '1.65', 31: '1.81', 32: '2.01', 
#         33: '2.18', 34: '2.55', 35: '2.96', 36: '3', 37: '0.82', 38: '0.95', 39: '1.22', 40: '1.33', 41: '1.6', 42: '2.16', 43: '1.9', 44: '2.2', 45: '2.28', 46: '2.1', 47: '1.93', 48: '1.69', 
#         49: '1.78', 50: '1.96', 51: '2.05', 52: '2.1', 53: '2.66', 54: '2.6', 55: '0.79', 56: '0.89', 57: '1.1', 58: '1.12', 59: '1.13', 60: '1.14', 61: '1.2', 62: '1.17', 63: '1.2', 64: '1.2', 
#         65: '1.1', 66: '1.22', 67: '1.23', 68: '1.24', 69: '1.25', 70: '1.2', 71: '1.27', 72: '1.3', 73: '1.5', 74: '2.36', 75: '1.9', 76: '2.2', 77: '2.2', 78: '2.28', 79: '2.54', 80: '2', 81: '1.62', 
#         82: '2.33', 83: '2.02', 84: '2', 85: '2.2', 86: 'no data', 87: 'no data', 88: '0.9', 89: '1.1', 90: '1.3', 91: '1.5', 92: '1.38', 93: '1.36', 94: '1.28', 95: '1.3', 96: '1.3', 97: '1.3', 
#         98: '1.3', 99: '1.3', 100: '1.3', 101: '1.3', 102: '1.3'}

#all tm in sanitize list en changed to 1
a_en = {0: '0', 1: '2.2', 2: 'no data', 3: '0.98', 4: '1.57', 5: '2.04', 6: '2.55', 7: '3.04', 8: '3.44', 9: '3.98', 10: 'no data', 11: '0.93', 12: '1.31', 13: '1.61', 14: '1.9', 15: '2.19', 16: '2.58', 17: '3.16', 18: 'no data', 19: '0.82', 20: '1', 21: 1, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1, 27: 1, 28: 1, 29: 1, 30: '1.65', 31: '1.81', 32: '2.01', 33: '2.18', 34: '2.55', 35: '2.96', 36: '3', 37: '0.82', 38: '0.95', 39: 1, 40: 1, 41: 1, 42: 1, 43: '1.9', 44: 1, 45: 1, 46: 1, 47: 1, 48: '1.69', 49: '1.78', 50: '1.96', 51: '2.05', 52: '2.1', 53: '2.66', 54: '2.6', 55: '0.79', 56: '0.89', 57: 1, 58: '1.12', 59: '1.13', 60: '1.14', 61: '1.2', 62: '1.17', 63: '1.2', 64: '1.2', 65: '1.1', 66: '1.22', 67: '1.23', 68: '1.24', 69: '1.25', 70: '1.2', 71: '1.27', 72: 1, 73: 1, 74: 1, 75: 1, 76: 1, 77: 1, 78: 1, 79: 1, 80: '2', 81: '1.62', 82: '2.33', 83: '2.02', 84: '2', 85: '2.2', 86: 'no data', 87: 'no data', 88: '0.9', 89: '1.1', 90: '1.3', 91: '1.5', 92: '1.38', 93: '1.36', 94: '1.28', 95: '1.3', 96: '1.3', 97: '1.3', 98: '1.3', 99: '1.3', 100: '1.3', 101: '1.3', 102: '1.3'}

max_allowed_valences = {
    7 : 3, 
    8 : 2,
}

#metals = [
#            3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
#            31, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 55, 56,
#            57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
#            74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92, 93,
#            94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
#            113, 114, 115, 116
#]

en_lookup = np.zeros(130, dtype=float)

for k, v in a_en.items():
    if v == 'no data':
        en_lookup[k] = 0.0
    else:
        try:
            en_lookup[k] = float(v)
        except (ValueError, TypeError):
            print(f"Could not convert value '{v}' for key {k}")
            en_lookup[k] = 0.0

#print(en_lookup)

###old
ELEM_LIST = ['PAD', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', \
            'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', \
            'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Sr', 'Y', 'Zr', 'Mo', 'Tc', 'Ru', \
            'Rh', 'Pd', 'Ag', 'In', 'Sn', 'Sb', 'Te', 'I', 'Cs', 'Ba', 'La', 'Ce', 'Eu', \
             'Yb', 'Ta', 'W', 'Os', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'V', 'Sm', 'Cd', 'Ir', 'Rb', 'Xe', 'Kr']

transition_metals = np.array([
     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,  # Sc to Zn
     39, 40, 41, 42, 43, 44, 45, 46, 47, 48,  # Y to Cd
     57, 72, 73, 74, 75, 76, 77, 78, 79, 80,  # La, Hf to Hg (La often included, sometimes Lu)
     89,104,105,106,107,108,109,110,111,112   # Ac, Rf to Cn (Actinide transition analogs)
])

MATRIX_PAD = -30
bt_to_electron = {Chem.rdchem.BondType.DATIVE: 2,
                  Chem.rdchem.BondType.SINGLE: 2, 
                 Chem.rdchem.BondType.DOUBLE: 4,
                 Chem.rdchem.BondType.TRIPLE: 6,
                 Chem.rdchem.BondType.QUADRUPLE: 8,
                 Chem.rdchem.BondType.AROMATIC: 3}

tbl = Chem.GetPeriodicTable()

outer_electron_lookup = {i: tbl.GetNOuterElecs(i) for i in range(1, 119)}
outer_electron_lookup[0] = 0

def get_mat_formal_charge(matrix, a_num):
    v = [outer_electron_lookup[int(i)] for i in a_num] # probably make dict of outer elecs beforehand and just lookup
    return v - np.sum(matrix, axis=0)

def bond_features(bond):
    bt = bond.GetBondType()
    
    return bt_to_electron[bt]

def count_lone_pairs(a):
    v=outer_electron_lookup[a.GetAtomicNum()]
    c=a.GetFormalCharge()
    b=sum([bond.GetBondTypeAsDouble() for bond in a.GetBonds()])
    h=a.GetTotalNumHs()
    return v-c-b-h

ps = Chem.SmilesParserParams()
ps.removeHs = False
ps.sanitize = True

def get_BE_matrix(r):
    rmol = Chem.MolFromSmiles(r, ps)
    Chem.Kekulize(rmol)
    max_natoms = len(rmol.GetAtoms())
    f = np.zeros((max_natoms,max_natoms))
    
    for atom in rmol.GetAtoms():
        lone_pair = count_lone_pairs(atom)
        f[atom.GetIntProp('molAtomMapNumber') - 1, atom.GetIntProp('molAtomMapNumber') - 1] = lone_pair

    for bond in rmol.GetBonds():
        a1 = bond.GetBeginAtom().GetIntProp('molAtomMapNumber') - 1
        a2 = bond.GetEndAtom().GetIntProp('molAtomMapNumber') - 1
        if bond.GetBondType() == Chem.rdchem.BondType.DATIVE:
            f[(a1,a1)] -= 1
            f[(a2,a2)] += 1
        f[(a1,a2)] = f[(a2,a1)] = bond_features(bond)/2 # so that bond electron diff matrix sums up to 0

    return f

electron_to_bo = {val:key for key, val in bt_to_electron.items()}

def get_formal_charge(a, electron):
    v=tbl.GetNOuterElecs(a.GetAtomicNum())
    b=sum([bond.GetBondTypeAsDouble() for bond in a.GetBonds()])
    h=a.GetTotalNumHs()
    f =v -  electron  - b - h
    return f

def mol_prop_compute(matrix, dative_donors):
    """
    vectorized way of computing atom dict and bond dict from matrix
    """
    n = matrix.shape[0]

    # 1) Compute symmetric bond sums once:
    Mplus = matrix + matrix.T

    # Redistribute electrons from dative donors
    try:
        donors = np.array(list(dative_donors.keys()))
        acceptors = np.array(list(dative_donors.values()))
    
        bond_remains = Mplus[donors, acceptors] != 0
    
        donors = donors[bond_remains]
        acceptors = acceptors[bond_remains]
        
        np.subtract.at(matrix, (acceptors, acceptors), 1)
        np.add.at(matrix, (donors, donors), 1)

    except Exception as e:
        pass
    
    # 2) Extract all off-diagonal i<j where there's at least one bond
    iu, ju = np.triu_indices(n, k=1)
    vals = Mplus[iu, ju]
    mask = vals != 0
    
    # 3) Build bond_dict
    bond_dict = {
        (i + 1, j + 1): int(val)
        for i, j, val in zip(iu[mask], ju[mask], vals[mask])
    }
    # 5) Build atom_dict from the diagonal
    diag = matrix.diagonal()
    atom_dict = {
            (i + 1, i + 1): int(diag_val)
        for i, diag_val in enumerate(diag)
    }
    return atom_dict, bond_dict

def BEmatrix_to_mol(rmol, matrix, dative_donors, not_donors = [], idxfunc=lambda x:x.GetIdx()):
    atom_dict, bond_dict = mol_prop_compute(matrix, dative_donors)
    # print(dative_donors)
    # print(not_donors)
    #print(atom_dict, bond_dict)
    new_mol = Chem.RWMol(rmol)
    new_mol.UpdatePropertyCache(strict=False)
    
    amap = {}
    for atom in new_mol.GetAtoms():
        amap[atom.GetIntProp('molAtomMapNumber') - 1] = atom.GetIdx()

    for bond in rmol.GetBonds():
        a1 = idxfunc(bond.GetBeginAtom())
        a2 = idxfunc(bond.GetEndAtom())
        new_mol.RemoveBond(a1, a2)

    #print("Bond dict", bond_dict)
    for (a1, a2), electron in bond_dict.items():
        new_mol.AddBond(amap[a1-1], amap[a2-1], electron_to_bo[electron])

    #print("Atom dict", atom_dict)
    for (a1, a1), electron in atom_dict.items():
        a =  new_mol.GetAtomWithIdx(amap[a1-1])
        # if a.GetAtomicNum() in transition_metals:
        #     fc = get_formal_charge(a, electron)
        fc = get_formal_charge(a, electron)
        if fc < 0 and a.GetAtomicNum() in transition_metals:
            bonded_atoms = [x for k in bond_dict.keys() if a1 in k for x in k if x != a1]
            bonded_atoms = [x for x in bonded_atoms if get_formal_charge(new_mol.GetAtomWithIdx(amap[x-1]), atom_dict[(x,x)]) > 0]
            bonded_atoms = [x for x in bonded_atoms if (x-1) not in not_donors]

            fc += len(bonded_atoms)
            atom_dict[(a1,a1)] -= len(bonded_atoms)
            for a2 in bonded_atoms:
                if a2 < a1:
                    temp_a = new_mol.GetAtomWithIdx(amap[a2-1])
                    temp_a.SetFormalCharge(temp_a.GetFormalCharge() - 1)
                    atom_dict[(a2,a2)] += 1
                else:
                    atom_dict[(a2,a2)] += 1


        elif fc > 0  and a.GetAtomicNum() in transition_metals: 
            bonded_atoms = [x for k in bond_dict.keys() if a1 in k for x in k if x != a1]
            bonded_atoms = [x for x in bonded_atoms if get_formal_charge(new_mol.GetAtomWithIdx(amap[x-1]), atom_dict[(x,x)]) < 0]
            bonded_atoms = [x for x in bonded_atoms if (x-1) in dative_donors.keys()]

            fc -= len(bonded_atoms)
            atom_dict[(a1,a1)] += len(bonded_atoms)
            for a2 in bonded_atoms:
                if a2 < a1:
                    temp_a = new_mol.GetAtomWithIdx(amap[a2-1])
                    temp_a.SetFormalCharge(temp_a.GetFormalCharge() + 1)
                    atom_dict[(a2,a2)] -= 1
                else:
                    atom_dict[(a2,a2)] -= 1
        # if fc:
        #     print(fc)
        a.SetFormalCharge(int(fc))
    return new_mol

atom2idx_dict = {elem:i for i, elem in enumerate(ELEM_LIST)}

def smi2vocabid(smi):

    mol = Chem.MolFromSmiles(smi, ps)
    smi_vocab_id_list = np.zeros(len(mol.GetAtoms()))
    atomic_nums = np.zeros(len(mol.GetAtoms()), dtype=np.int64)
    for atom in mol.GetAtoms():
        idx = atom2idx_dict[atom.GetSymbol()]
        a_num = atom.GetAtomicNum()
        atomic_nums[atom.GetIntProp('molAtomMapNumber') - 1] = a_num
        smi_vocab_id_list[atom.GetIntProp('molAtomMapNumber') - 1] = idx
    
    if np.any(atomic_nums < 1) or np.any(atomic_nums > 119):
        invalid_mask = (atomic_nums < 1) | (atomic_nums > 119)
        raise ValueError(f"Atomic numbers must be between 1 and 119, got {atomic_nums[invalid_mask]}")
    return smi_vocab_id_list, len(smi_vocab_id_list), atomic_nums

def get_donors(smi):
    # takes rxn smi, returns dict of donor : acceptor locations in mat.

    mol = Chem.MolFromSmiles(smi, ps)
    dative_donors = {}
    not_donors = []

    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.DATIVE:
            dative_donors[bond.GetBeginAtom().GetIntProp('molAtomMapNumber') - 1] = bond.GetEndAtom().GetIntProp('molAtomMapNumber') - 1
            # tm = bond.GetEndAtom()
            # for tm_bonds in tm.GetBonds():
            #     if not tm_bonds.GetBondType() == Chem.rdchem.BondType.DATIVE:
            #         not_donors.append(tm_bonds.GetEndAtom().GetIntProp('molAtomMapNumber') - 1)

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in transition_metals:
            for tm_bonds in atom.GetBonds():
                if not tm_bonds.GetBondType() == Chem.rdchem.BondType.DATIVE:
                    not_donors.append(tm_bonds.GetEndAtom().GetIntProp('molAtomMapNumber') - 1)

    return dative_donors, not_donors

def process_smiles(args):
    idx, smiles = args
    src_smi, tgt_smi = smiles.strip().split('|')[0].split('>>')

    error = ""
    try:
        # _ = get_BE_matrix(src_smi)
        # _ = get_BE_matrix(tgt_smi)
        src_vocab_id_list, src_len, src_a_nums = smi2vocabid(src_smi)
        tgt_vocab_id_list, tgt_len, tgt_a_nums = smi2vocabid(tgt_smi)
        assert (src_vocab_id_list == tgt_vocab_id_list).all()
    except Exception as e:
        error = e
        print(f"ERROR: {e} \n {smiles}")
        src_smi, tgt_smi = '', ''
        src_vocab_id_list, src_len, src_a_nums = [], 0, []
        tgt_vocab_id_list, tgt_len, tgt_a_nums = [], 0, []

    # Return a tuple of results for this smiles pair
    return {
        'src_smi': src_smi,
        'tgt_smi': tgt_smi,
        'src_vocab_id_list': src_vocab_id_list,
        'tgt_vocab_id_list': tgt_vocab_id_list,
        'src_len': src_len,
        'tgt_len': tgt_len,
        'src_a_nums' : src_a_nums,
        'tgt_a_nums' : tgt_a_nums,
        'line_num': idx,
        "error": error
    }

def standardize_smiles(mol):
    [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
    return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True, allHsExplicit=True)

def fullSanitize (smi):
    '''
    Takes molecule smi -> fixes valence errors associated with charge/coordination by using dative bonds
    '''
    m = ("Ag", "Au", "Co", "Cr", "Cu", "Fe", "Ir", "La", "Mn", 
    "Mo", "Ni", "Os", "Pd", "Pt", "Rh", "Ru", "Sc", 
    "Ta", "Ti", "V", "W", "Y", "Zr")

    if any(i in smi for i in m):
        try:
            # print(f"Sanitizing SMILES: {smi}")
            settings = Chem.SmilesParserParams()
            settings.removeHs = False
            settings.sanitize = False
            mol = Chem.RWMol(Chem.MolFromSmiles(smi, settings))
            mol.UpdatePropertyCache(strict=False)
            for atom in mol.GetAtoms():
                Z = atom.GetAtomicNum()
                valence = sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())
                max_val = max_allowed_valences.get(Z, None)
                if max_val is not None and valence > (max_val + atom.GetFormalCharge()) or atom.GetFormalCharge() < 0:
                    for bond in atom.GetBonds():
                        neighbor = bond.GetOtherAtom(atom)
                        if neighbor.GetAtomicNum() in transition_metals:
                            begin_idx = atom.GetIdx()
                            end_idx = neighbor.GetIdx()
                            mol.RemoveBond(begin_idx, end_idx)
                            mol.AddBond(begin_idx, end_idx, rdchem.BondType.DATIVE)
                if atom.GetAtomicNum() in [47, 79, 27, 24, 29, 26, 72, 77, 57, 25, 42, 41, 28, 76, 46, 78, 75, 45, 44, 21, 73, 22, 23, 74, 39, 40]:
                    for bond in atom.GetBonds():
                        neighbor = bond.GetOtherAtom(atom)
                        if neighbor.GetAtomicNum() == 15:
                            begin_idx = neighbor.GetIdx()
                            end_idx = atom.GetIdx()
                            mol.RemoveBond(begin_idx, end_idx)
                            mol.AddBond(begin_idx, end_idx, rdchem.BondType.DATIVE)
                        elif neighbor.GetAtomicNum() == 8 and neighbor.GetFormalCharge() == 1 and any(nb.GetAtomicNum() == 5 for nb in neighbor.GetNeighbors()):
                            begin_idx = neighbor.GetIdx()
                            end_idx = atom.GetIdx()
                            mol.RemoveBond(begin_idx, end_idx)
                            mol.AddBond(begin_idx, end_idx, Chem.rdchem.BondType.SINGLE)
                        elif neighbor.GetAtomicNum() == 6 and neighbor.GetTotalValence() == 3:
                            begin_idx = neighbor.GetIdx()
                            end_idx = atom.GetIdx()
                            mol.RemoveBond(begin_idx, end_idx)
                            mol.AddBond(begin_idx, end_idx, rdchem.BondType.DATIVE)
                        elif neighbor.GetAtomicNum() == 7 and neighbor.GetIsAromatic():
                            begin_idx = neighbor.GetIdx()
                            end_idx = atom.GetIdx()
                            mol.RemoveBond(begin_idx, end_idx)
                            mol.AddBond(begin_idx, end_idx, rdchem.BondType.DATIVE)
                            
            quit = False
            while not quit:
                try:
                    Chem.SanitizeMol(mol)
                    return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True, allHsExplicit=True)
                except Exception as e:
                    try:
                        Chem.SanitizeMol(mol)
                        return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True, allHsExplicit=True)
                    except Exception:
                        dative_bonds = [bond for bond in mol.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.DATIVE]
                        
                        if not dative_bonds:
                            # No more dative bonds to fix — give up
                            #print(e)
                            return None
                        
                        for bond in dative_bonds:
                            begin = bond.GetBeginAtomIdx()
                            end = bond.GetEndAtomIdx()

                            # Temporarily replace dative bond with single bond
                            mol.RemoveBond(begin, end)
                            mol.AddBond(begin, end, Chem.rdchem.BondType.SINGLE)

                            try:
                                Chem.SanitizeMol(mol)
                                # Success! Keep this change and return
                                return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True, allHsExplicit=True)
                            except Exception:
                                # Failed, revert to dative bond
                                mol.RemoveBond(begin, end)
                                mol.AddBond(begin, end, Chem.rdchem.BondType.DATIVE)

                        quit = True

        except Exception as e:
            return None
            
    else: 
        return smi
            #print(f"Error sanitizing SMILES {Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)}: {e}")

def calculate_oxd_states(matrix, a_num, cutoff=0.05, e_neg=[]) -> np.array:
    '''
    Takes BE matrix and list of atomic numbers of atoms in matrix and returns a list of oxidation states
    Cutoff sets a lower bound on electronegativity differences to actually consider oxidation state changes
    '''
    if not e_neg:
        e_neg = en_lookup[a_num]
    
    n = matrix.shape[0]
    oxd_states = np.zeros(n)
    formal_charges = get_mat_formal_charge(matrix, a_num)
    
    Mplus = matrix + matrix.T
    
    i_upper, j_upper = np.triu_indices(n, k=1)
    vals = Mplus[i_upper, j_upper]
    nzero_mask = vals != 0
    
    i, j = i_upper[nzero_mask], j_upper[nzero_mask]
    v = vals[nzero_mask]
    
    diff = e_neg[i] - e_neg[j]

    #0.05 = arbitrary en cutoff
    mask_pos = diff > cutoff
    mask_neg = diff < -cutoff

    np.add.at(oxd_states, i[mask_pos], -v[mask_pos] / 2)
    np.add.at(oxd_states, j[mask_pos], v[mask_pos] / 2)

    np.add.at(oxd_states, j[mask_neg], -v[mask_neg] / 2)
    np.add.at(oxd_states, i[mask_neg], v[mask_neg] / 2)
            
    if not np.isclose(np.sum(oxd_states), 0, atol=1e-6):
        print("Oxidation states: ", oxd_states)
        raise Exception("Oxidation states don't sum to 0")
    
    return oxd_states + formal_charges

class ReactionBatch:
    def __init__(self,
                src_data_indices: torch.Tensor,
                src_token_ids: torch.Tensor,
                src_lens: torch.Tensor,
                src_matrices: torch.Tensor,
                tgt_matrices: torch.Tensor,
                matrix_masks: torch.Tensor,
                src_smiles_list: list,
                tgt_smiles_list: list,
                src_d_donors: list,
                src_not_donors: list,
                oxd_states = False,
                # src_line_nums: torch.Tensor = None,
                src_oxd_states: torch.Tensor = None, 
                tgt_oxd_states: torch.Tensor = None,
                src_a_nums: torch.Tensor = None,
                src_a_masks: torch.Tensor = None,
                ):
        self.src_data_indices = src_data_indices
        self.src_token_ids = src_token_ids
        self.src_lens = src_lens
        self.src_matrices = src_matrices
        self.tgt_matrices = tgt_matrices
        self.matrix_masks = matrix_masks
        self.src_smiles_list = src_smiles_list
        self.tgt_smiles_list = tgt_smiles_list
        self.oxd_states = oxd_states
        self.src_d_donors = src_d_donors
        self.src_not_donors = src_not_donors
        # self.src_line_nums = src_line_nums
        if self.oxd_states:
            self.src_oxd_states = src_oxd_states
            self.tgt_oxd_states = tgt_oxd_states
            self.src_a_nums = src_a_nums
            self.src_a_masks = src_a_masks
        # self.src_a_nums = src_a_nums
        # self.tgt_a_nums = tgt_a_nums

    def to(self, device):
        self.src_data_indices = self.src_data_indices.to(device)
        self.src_token_ids = self.src_token_ids.to(device)
        self.src_lens = self.src_lens.to(device)
        self.src_matrices = self.src_matrices.to(device)
        self.tgt_matrices = self.tgt_matrices.to(device)
        self.matrix_masks = self.matrix_masks.to(device)
        # self.src_a_nums = self.src_a_nums.to(device)
        # self.tgt_a_nums = self.tgt_a_nums.to(device)
        # if self.src_line_nums is not None:
        #     self.src_line_nums = self.src_line_nums.to(device)

        if self.oxd_states:
            self.src_oxd_states = self.src_oxd_states.to(device)
            self.tgt_oxd_states = self.tgt_oxd_states.to(device)
            self.src_a_nums = self.src_a_nums.to(device)
            self.src_a_masks = self.src_a_masks.to(device)

    def pin_memory(self):
        self.src_data_indices = self.src_data_indices.pin_memory()
        self.src_token_ids = self.src_token_ids.pin_memory()
        self.src_lens = self.src_lens.pin_memory()
        self.src_matrices = self.src_matrices.pin_memory()
        self.tgt_matrices = self.tgt_matrices.pin_memory()
        self.matrix_masks = self.matrix_masks.pin_memory()
        if self.oxd_states:
            self.src_oxd_states = self.src_oxd_states.pin_memory()
            self.tgt_oxd_states = self.tgt_oxd_states.pin_memory()
            self.src_a_nums = self.src_a_nums.pin_memory()
            self.src_a_masks = self.src_a_masks.pin_memory()
        # self.src_a_nums = self.src_a_nums.pin_memory()
        # self.tgt_a_nums = self.tgt_a_nums.pin_memory()

        return self

class ReactionDataset(Dataset):
    def __init__(self, args, smiles_list, parallel=True, reactant_only=False, line_offset=0):
        self.args = args
        self.device = args.device

        # self.line_offset = line_offset
        # self.src_line_nums = []

        self.reactant_only = reactant_only
        self.smiles_list = smiles_list
        self.src_smis = []
        self.tgt_smis = []

        self.src_token_ids = []
        self.tgt_token_ids = []

        self.src_lens = []
        self.tgt_lens = []

        self.src_a_nums = []
        self.tgt_a_nums = []

        if reactant_only:
            self.parse_reactant_only()
        else:
            if parallel:
                self.parse_data_parallel()
            else:
                self.parse_data()
        
        self.src_lens = np.asarray(self.src_lens)

        self.data_size = len(self.src_smis)
        self.data_indices = np.arange(self.data_size)

    def parse_reactant_only(self):
        for src_smi in self.smiles_list:
            src_smi = src_smi.strip()
            try:
                _ = get_BE_matrix(src_smi)
                src_vocab_id_list, src_len, src_a_num = smi2vocabid(src_smi)
            except Exception as e:
                print(e)
                continue

            self.src_smis.append(src_smi)
            self.src_token_ids.append(src_vocab_id_list)
            self.src_lens.append(src_len)
            self.tgt_lens.append(src_len)
            self.src_a_nums.append(src_a_num)

        assert len(self.src_smis) > 0, "Empty Data"

    def parse_data(self):
        for smiles in self.smiles_list:
            src_smi, tgt_smi = smiles.strip().split('|')[0].split('>>')

            try:
                # src_matrix = get_BE_matrix(src_smi)
                # tgt_matrix = get_BE_matrix(tgt_smi)    
                src_vocab_id_list, src_len, src_a_num = smi2vocabid(src_smi)
                tgt_vocab_id_list, tgt_len, tgt_a_num = smi2vocabid(tgt_smi)
                assert (src_vocab_id_list == tgt_vocab_id_list).all()
                assert src_len == tgt_len, "src len and tgt len should be the same"
            except Exception as e:
                print("ERROR HERE: ", e)
                print(smiles)
                continue

            self.src_smis.append(src_smi)
            self.tgt_smis.append(tgt_smi)
            self.src_token_ids.append(src_vocab_id_list)
            self.tgt_token_ids.append(tgt_vocab_id_list)
            self.src_lens.append(src_len)
            self.tgt_lens.append(tgt_len)
            self.src_a_nums.append(src_a_num)
            self.tgt_a_nums.append(tgt_a_num)

        assert len(self.src_smis) == len(self.tgt_smis) == len(self.src_lens) == len(self.tgt_lens)  \
              == len(self.tgt_lens) == len(self.src_token_ids) == len(self.tgt_token_ids)

    def parse_data_parallel(self):

        p = Pool(cpu_count())
        results = p.imap(process_smiles, enumerate(self.smiles_list))

        total = 0
        count = 0

        for result in results:
            total += 1
            if result['src_vocab_id_list'] is [] or result['src_len'] == 0: 
                print(f"{result['src_smi']}>>{result['tgt_smi']}")
                print(result['error'])
                count += 1
                continue
            else: 
                # self.src_line_nums.append(self.line_offset + result['line_num'])
                self.src_smis.append(result['src_smi'])
                self.tgt_smis.append(result['tgt_smi'])
                self.src_token_ids.append(result['src_vocab_id_list'])
                self.tgt_token_ids.append(result['tgt_vocab_id_list'])
                self.src_lens.append(result['src_len'])
                self.tgt_lens.append(result['tgt_len'])
                self.src_a_nums.append(result['src_a_nums'])
                self.tgt_a_nums.append(result['tgt_a_nums'])

        # Prepare the final data structures

            

        p.close()
        p.join()

        print(f"{count*100/total}% data is unparseable")

    def sort(self):
        self.data_indices = np.argsort(self.src_lens)

    def shuffle_in_bucket(self, bucket_size: int, seed: int = 0):
        rng = np.random.RandomState(seed)
        for i in range(0, self.data_size, bucket_size):
            rng.shuffle(self.data_indices[i:i + bucket_size])

    def batch(self, batch_type: str, batch_size: int, verbose=False):

        self.batch_sizes = []
        if batch_type.startswith("tokens"):
            sample_size = 0
            max_batch_src_len = 0
            max_batch_tgt_len = 0

            for data_idx in self.data_indices:
                src_len = self.src_lens[data_idx]
                tgt_len = self.tgt_lens[data_idx]

                max_batch_src_len = max(src_len, max_batch_src_len)
                max_batch_tgt_len = max(tgt_len, max_batch_tgt_len)

                if batch_type == "tokens" and \
                        max_batch_src_len * (sample_size + 1) <= batch_size:
                    sample_size += 1
                elif batch_type == "tokens_sum" and \
                        (max_batch_src_len + max_batch_tgt_len) * (sample_size + 1) <= batch_size:
                    sample_size += 1
                else:
                    self.batch_sizes.append(sample_size)

                    sample_size = 1
                    max_batch_src_len = src_len
                    max_batch_tgt_len = tgt_len

            # lastly
            self.batch_sizes.append(sample_size)
            self.batch_sizes = np.array(self.batch_sizes)
            assert np.sum(self.batch_sizes) == self.data_size, \
                f"Size mismatch! Data size: {self.data_size}, sum batch sizes: {np.sum(self.batch_sizes)}"

            self.batch_ends = np.cumsum(self.batch_sizes)
            self.batch_starts = np.concatenate([[0], self.batch_ends[:-1]])

        else:
            raise ValueError(f"batch_type {batch_type} not supported!")


    def __len__(self):
        return len(self.batch_sizes)

    def __getitem__(self, idx : int):
        batch_index = idx
        
        batch_start = self.batch_starts[batch_index]
        batch_end = self.batch_ends[batch_index]

        data_indices = self.data_indices[batch_start:batch_end]

        # print(self.src_lens[data_indices], data_indices)
        max_len = max(self.src_lens[data_indices])

        src_token_id_batch = []
        src_len_batch = []
        src_matrix_batch = []
        tgt_matrix_batch = []
        src_smiles_batch = []
        tgt_smiles_batch = []
        src_dative_donors_batch = []
        src_not_donors_batch = []
        if self.args.oxd_states:
            src_a_nums_batch = []
            src_oxd_states_batch = []
            tgt_oxd_states_batch = []
        for data_index in data_indices:
            # src_token_id, _ = smi2vocabid(self.src_smis[data_index])
            src_token_id = self.src_token_ids[data_index]
            src_len = self.src_lens[data_index]
            src_token_id = np.pad(src_token_id, (0, max_len - src_len),
                                   mode='constant', constant_values=0) # constant value 0 based on 'PAD' in ELEM_LIST
            src_token_id =  torch.as_tensor(src_token_id, dtype=torch.long)

            src_token_id_batch.append(src_token_id)

            src_matrix = get_BE_matrix(self.src_smis[data_index])
            src_d_donors, src_not_donors = get_donors(self.src_smis[data_index])
            src_dative_donors_batch.append(src_d_donors)
            src_not_donors_batch.append(src_not_donors)
            
            if self.args.oxd_states:
                #print(self.src_a_nums[data_index])
                #print(self.src_a_nums[data_index].dtype)
                src_a_nums = np.pad(self.src_a_nums[data_index], (0, max_len - src_len), mode='constant', constant_values=0) # i probably don't need to pad these???
                src_a_nums_batch.append(src_a_nums)
                src_oxd_states = calculate_oxd_states(src_matrix, self.src_a_nums[data_index])
                src_oxd_states = np.pad(src_oxd_states, (0, max_len - src_len), mode='constant', constant_values=0)
                src_oxd_states_batch.append(src_oxd_states)

                if not self.reactant_only:
                    tgt_oxd_states = calculate_oxd_states(src_matrix, self.tgt_a_nums[data_index])
                    tgt_oxd_states = np.pad(tgt_oxd_states, (0, max_len - src_len), mode='constant', constant_values=0)
                    tgt_oxd_states_batch.append(tgt_oxd_states)

            src_matrix = np.pad(src_matrix, ((0, max_len - src_len), (0, max_len - src_len)), 
                       mode='constant', constant_values=MATRIX_PAD)
            src_len_batch.append(src_len)
            src_matrix_batch.append(src_matrix)
            src_smiles_batch.append(self.src_smis[data_index])

            if not self.reactant_only:
                tgt_matrix = get_BE_matrix(self.tgt_smis[data_index])
                tgt_matrix = np.pad(tgt_matrix, ((0, max_len - src_len), (0, max_len - src_len)), 
                        mode='constant', constant_values=MATRIX_PAD)
                tgt_matrix_batch.append(tgt_matrix)
                tgt_smiles_batch.append(self.tgt_smis[data_index])

        # if len(self.src_line_nums) > 0:
        #     src_line_nums = torch.as_tensor([self.src_line_nums[i] for i in data_indices], dtype=torch.int)
        # else:
        #     src_line_nums = torch.zeros(len(data_indices), dtype=torch.int)
        src_data_indices = torch.as_tensor(data_indices, dtype=torch.long)
        src_len_batch = torch.as_tensor(src_len_batch, dtype=torch.long)
        src_token_id_batch = torch.stack(src_token_id_batch)
        src_matrix_batch = torch.as_tensor(np.stack(src_matrix_batch), dtype=torch.float)
        src_oxd_states_batch = torch.as_tensor(np.stack(src_oxd_states_batch), dtype=torch.float) if self.args.oxd_states else None
        src_a_nums_batch = torch.as_tensor(np.stack(src_a_nums_batch), dtype=torch.long) if self.args.oxd_states else None

        if not self.reactant_only: 
            tgt_matrix_batch = torch.as_tensor(np.stack(tgt_matrix_batch), dtype=torch.float)
            tgt_oxd_states_batch = torch.as_tensor(np.stack(tgt_oxd_states_batch), dtype=torch.float) if self.args.oxd_states else None
        else: 
            tgt_matrix_batch = src_matrix_batch
            tgt_oxd_states_batch = src_oxd_states_batch if self.args.oxd_states else None
        
        node_mask = (src_matrix_batch[:, :, 0] != MATRIX_PAD)
        matrix_masks = (node_mask.unsqueeze(1) * node_mask.unsqueeze(2)).long()

        if self.args.oxd_states:
            a_mask = (src_a_nums_batch != 0).long()
            reaction_batch = ReactionBatch(
                src_data_indices=src_data_indices,
                src_token_ids=src_token_id_batch,
                src_lens=src_len_batch,
                src_matrices=src_matrix_batch,
                tgt_matrices=tgt_matrix_batch,
                matrix_masks=matrix_masks,
                src_smiles_list=src_smiles_batch,
                tgt_smiles_list=tgt_smiles_batch,
                src_d_donors = src_dative_donors_batch,
                src_not_donors = src_not_donors_batch,
                oxd_states = True,
                # src_line_nums=src_line_nums,
                src_oxd_states=src_oxd_states_batch,
                tgt_oxd_states=tgt_oxd_states_batch,
                src_a_nums=src_a_nums_batch,
                src_a_masks=a_mask,
            )
        else:
            reaction_batch = ReactionBatch(
                src_data_indices=src_data_indices,
                src_token_ids=src_token_id_batch,
                src_lens=src_len_batch,
                src_matrices=src_matrix_batch,
                tgt_matrices=tgt_matrix_batch,
                matrix_masks=matrix_masks,
                src_smiles_list=src_smiles_batch,
                tgt_smiles_list=tgt_smiles_batch,
                src_d_donors = src_dative_donors_batch,
                src_not_donors = src_not_donors_batch,
                oxd_states = False,
                # src_line_nums=src_line_nums,
            )
        
        return reaction_batch
