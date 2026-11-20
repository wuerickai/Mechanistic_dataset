import argparse
import logging
import json
import pickle
from multiprocessing import Pool, Value
import traceback
from itertools import product

# Third-party libraries
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdchem, rdmolops
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from tqdm import tqdm
if __package__:
    from .replace import reactions
else:
    from replace import reactions

def compileReactions(rxns): 
    compiledRxns = {}
    for key, value in rxns.items():
        if type(key) is str:
            reactants = (Chem.MolFromSmarts(key),)
        else:
            reactants = tuple(Chem.MolFromSmarts(mol) for mol in key)
        reaction = AllChem.ReactionFromSmarts(value)
        compiledRxns[reactants] = reaction
    
    return compiledRxns

def find_matched_reactions(mols, rxns):
    matched_reactions = {}

    for reactants, reaction in rxns.items():
        candidates = [
            [mol for mol in mols if mol.GetSubstructMatch(pattern)]
            for pattern in reactants
        ]
        for matched_mols in product(*candidates):
            if len(set(matched_mols)) == len(reactants):
                matched_reactions[reaction] = list(matched_mols)
                break

    return matched_reactions

def preprocess(input, output, rxns):
    """
    Reads a txt file where each line contains:
    <reaction_smiles> <reaction_description>
    """
    with open(input, 'r') as f, open(output, 'a') as o:
        for i, line in enumerate(tqdm(f)):
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # skip empty lines or comments
            # Split on first whitespace or tab
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                logging.warning(f"Line skipped (not enough parts): {line}")
                continue
            smiles = parts[0]
            r, r1, products = smiles.split(">")
            mols = [
                Chem.MolFromSmiles(smi, sanitize=False)
                for smi in r.split(".") + r1.split(".") if smi
            ]
            for mol in mols:
                mol.UpdatePropertyCache(strict=False)
            reactions = find_matched_reactions(mols, rxns)

            if reactions: 
                used_reactants = []
                new_reactants = []
                
                for reaction, reactants in reactions.items():
                    try:
                        products_sets = reaction.RunReactants(reactants)
                        if not products_sets:
                            continue
                        replacement_smiles = [
                            Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
                            for mol in products_sets[0]
                        ]
                    except Exception:
                        continue
                    new_reactants += replacement_smiles
                    used_reactants += reactants
            
                unused_reactants = [Chem.MolToSmiles(mol) for mol in mols if mol not in used_reactants]
                if used_reactants:
                    new_reaction = '.'.join(new_reactants + unused_reactants) + ">>" + products
                    line = new_reaction + " " + parts[1]
                o.write(line.replace("P(", "[P](").replace("[PH+]", "[P]").replace("[CH2-]", "[CH3]") + '\n')
                    #o.write(new_reaction + " " + parts[1] + '\n')
                    #print(new_reaction + " " + parts[1] + '\n')

            else:
                o.write(line.replace("P(", "[P](").replace("[PH+]", "[P]").replace("[CH2-]", "[CH3]") + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Standardize catalysts and reagents in SMILES/label TXT files.")
    parser.add_argument("input_file", help="Input SMILES/label TXT path")
    parser.add_argument("output_file", help="Output TXT path (appends if the file exists)")
    args = parser.parse_args()
    rxns = compileReactions(reactions)
    preprocess(args.input_file, args.output_file, rxns)
