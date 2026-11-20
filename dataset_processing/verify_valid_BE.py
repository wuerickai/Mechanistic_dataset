import argparse
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count

from rdkit import Chem, RDLogger
from tqdm import tqdm

# Allow direct script execution as well as python -m dataset_processing.verify_valid_BE.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.data_utils import get_BE_matrix, BEmatrix_to_mol, get_donors, fullSanitize

RDLogger.DisableLog('rdApp.*')


ps = Chem.SmilesParserParams()
ps.removeHs = False
ps.sanitize = True

def standardize_smiles(mol):
    [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
    return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True, allHsExplicit=True)


def test_line(smiles_line: str):
    rxn = smiles_line.strip().split('|')[0]
    reactants, products = rxn.split('>>')

    # Generate BE matrices and atom numbers
    try: 
        donors, not_donors = get_donors(reactants)
        # print(donors, not_donors)
        product_BE = get_BE_matrix(products)
        # print(product_BE)
        
        product_smi = BEmatrix_to_mol(Chem.MolFromSmiles(reactants, ps), product_BE, donors, not_donors=not_donors)
        pred_smi = standardize_smiles(product_smi)
        # print(pred_smi)
        pred_smi = fullSanitize(pred_smi)
        # print(pred_smi)
        pred_smi = Chem.MolToSmiles(Chem.MolFromSmiles(pred_smi, ps), isomericSmiles=False, canonical=True, allHsExplicit=True)

        tgt_mol = Chem.MolFromSmiles(products, ps)
        tgt_smiles = standardize_smiles(tgt_mol)
        tgt_smiles = fullSanitize(tgt_smiles)
        tgt_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(tgt_smiles, ps), isomericSmiles=False, canonical=True, allHsExplicit=True)

        if not tgt_smiles == pred_smi:
            #print(tgt_smiles)
            return (smiles_line, "")
        else:
            return ("", smiles_line)
    except Exception as e:
        #print(e)
        return (smiles_line, "")

def check_be_calc(in_file, out_file1, out_file2):
    with open(in_file, 'r') as f:
        lines = f.readlines()

    with Pool(min(cpu_count(), max(1, len(lines)))) as pool:
        results = list(tqdm(pool.imap(test_line, lines), total=len(lines)))
    # with Pool(cpu_count()) as pool:
    #     results = list(tqdm(islice(pool.imap(test_line, lines), 1000), total=1000))


    with open(out_file1, 'w') as w1, open(out_file2, 'w') as w2:
        for res in results:
            if res[0]:
                w1.write(res[0] if res[0].endswith("\n") else res[0] + "\n")
            else:
                w2.write(res[1] if res[1].endswith("\n") else res[1] + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify product BE-matrix reconstruction.")
    parser.add_argument("--input_file", required=True, help="Input mechanistic reaction TXT")
    parser.add_argument("--output_file", required=True, help="Verified reaction TXT")
    parser.add_argument("--failed_file", default="failed.txt", help="Rejected reaction TXT (default: failed.txt)")
    args = parser.parse_args()
    check_be_calc(args.input_file, args.failed_file, args.output_file)

