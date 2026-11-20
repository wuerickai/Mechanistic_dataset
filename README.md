# Mechanistic Dataset generation updated for FlowER-OM
A python code for mechanistic dataset generation with organometallic templates for FlowER-OM contained in new_templates/. The corresponding model repo for which this code generated training data for can be found [here](https://github.com/wuerickai/FlowER_OM). An overview of the general use of this repo is found below and instructions for recreating the FlowER-OM dataset are the second half of the README.

## Step 0 - Installation
```shell
git clone https://github.com/wuerickai/Mechanistic_dataset
cd Mechanistic_dataset
conda env create --file environment.yml
conda activate mechdata2
```

## Step 1 - Data preparation
In order to generate mechanistic dataset from the overall reactions, reaction SMILES and its reaction class name as in NameRXN are required. The reaction SMILES and its reaction class name should be separated by a space as shown below.
It doesn't require atom mapping, a code will automatically assign all the atom mapping number.
```text
BrCCCOC1=CC(OCO2)=C2C=C1.OC3C(CCN(C3)C(OC(C)(C)C)=O)C4=CC=C(O)C=C4.[OH-]>>OC1C(CCN(C1)C(OC(C)(C)C)=O)C2=CC=C(OCCCOC3=CC(OCO4)=C4C=C3)C=C2 Williamson ether synthesis
```

## Step 2 - Set configuation parameters
This section outlines the configuration parameters used in the script, detailing their purpose and functionality.

- **num_combination**: Limits the number of reactant combinations per template to prevent combinatorial explosion.
- **uni_rxn**: Adds a unimolecular version of a bimolecular reaction template while retaining the original.
- **proton**: Ensures proton balance by adjusting templates on-the-fly based on the presence of acids or bases.
- **strongest_only** (default `True`): During proton balancing, uses the strongest available matching acid or base that satisfies the template's pKa requirement, ranked using the acid/base lookup table. Set to `False` to consider all available qualifying acids or bases.
- **max_num_temp**: Limits the number of possible templates when modifications like uni_rxn or proton are applied.
- **stoichiometry**: Allows duplicated reactants to be added if stoichiometry errors are detected.
- **do_not_pruning**: Controls whether to prune the reaction network to retain only those that lead to the final product.
- **max_pathways**: Limits the number of paths enumerated when converting the reaction network to elementary reaction SMARTS.
- **num_reaction_node**: Limits the number of reactions in the reaction network to prevent excessive processing time.
- **byproduct**: Adds byproducts from previous reactions when converting to elementary reaction SMARTS.
- **spectator**: Includes all chemical species in the flask at the current step when converting to SMARTS.
- **full**: Determines whether to return overall reaction SMARTS instead of elementary reaction SMARTS.
- **end**: Adds reactions that generate a product from an already existing product.
- **plain**: Generates reaction SMARTS without atom mapping numbers.
- **explicit_H**: Explicitly represents hydrogens in the reaction SMARTS.
- **reagent**: Positions non-reacting species between '>>' in Reaction SMARTS (Reactants > Reagents > Products).
- **remapping**: Remaps atom mapping in SMARTS to start from 1.
- **data**: Path to the file containing overall reactions.
- **save**: Path to the file where results will be saved.
- **debug**: Logs reactions with errors for debugging purposes.
- **all_info**: Saves all reactions, including the reaction network.
- **stat**: Generates statistics for the reactions.
- **rxn_class**: Specifies a particular reaction class to extract elementary reactions from.
- **process**: Number of processors to use in multiprocessing.
- **verbosity**: Sets the logging level.

## Step 3 - Run a code to generate mechanistic data

Mechanistic data can be generated as follows:

```shell
python ver2.py
```
---

## Step 4 - Visualize reaction templates

If you want to interactively **browse and visualize the reaction templates** (reaction class, mechanism, SMARTS for each step),
you can open the Colab notebook below:

<a href="https://colab.research.google.com/github/wuerickai/Mechanistic_dataset/blob/main/Mechanistic_dataset_visualization.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

This notebook allows you to:
- Select a reaction class and mechanism from dropdown menus
- View allowed and excluded reaction SMARTS for each mechanism
- Visualize each elementary reaction template
- Choose how atom features (like H count or charge) are shown in the molecule view

# To reproduce the FlowER-OM dataset

The scripts in `dataset_processing/` are necessary for the preparation and curation of the organometallic reaction dataset. Run these scripts in the following order on reactions from 2025 Q1 Pistachio from NextMove (which should already be named by NameRXN).

Run all commands below from the repository root in the environment above. Create `data/` first (`mkdir -p data`); the processing scripts do not create output directories. Run generation on Linux, macOS, or thorugh WSL. 

## Step 0 - Download and deduplicate Pistachio

The Pistachio dataset can be downloaded from NextMove with a license. Many of the reactions in Pistachio are duplicated, so canonicalizing and deduplicating by reaction field is necessary.

The commands below start from the deduplicated txt. Reformat the dataset into a txt file such that each line is in the format:
```shell
ReactantSMILES>ReagentSMILES>ProductSMILES NameRXNLabel
```

## Step 1 - Downselect reactions

[diversity_sampling.py](dataset_processing/diversity_sampling.py) selects diverse reaction subsets using DRFP fingerprints and a cosine-distance cutoff. Suzuki, Sonogashira, Stille, Miyaura borylation, and Buchwald-Hartwig C--N couplings are downselected to 12,000 parent reactions **before templating**. None of the other reaction classes are downselected.

Family cutoffs are Suzuki `0.45`, Sonogashira `0.33`, Stille `0.17`, Miyaura `0.14`, and Hartwig `0.4`.

For a file containing multiple reaction classes (likely what you will get out of Step 0), use `--labels` to downselect only one exact reaction class:

`--label` (alias `--labels`) accepts one or more exact NameRXN labels. Matching reactions form one combined pool with a **shared** `--target` and cutoff without touching other rules. For example, to select up to 12,000 reactions across the Suzuki reaction family:

```shell
python dataset_processing/diversity_sampling.py data/input.txt data/output.txt \
  --labels "Bromo Suzuki coupling" "Chloro Suzuki coupling" \
    "Iodo Suzuki coupling" "Triflyloxy Suzuki coupling" \
    "Tosyloxy Suzuki coupling" "Mesyloxy Suzuki coupling" "Triflyl Suzuki coupling" \
  --target 12000 --cutoff 0.45
```

To downselect families with different targets or cutoffs, run the script successively, passing each output as the next input.

| Argument | Default | Meaning |
| --- | --- | --- |
| `input_file` | Required | Input SMILES/label TXT path. |
| `output_file` | Required | Output TXT path; an existing file is overwritten. |
| `--target` | `12000` | Maximum number of selected reactions, shared across all matching labels when `--labels` is provided. |
| `--cutoff` | `0.17` | Minimum cosine distance between accepted DRFP fingerprints; larger values enforce greater separation. |
| `--batch-size` | `3000` | Number of candidate reactions processed per batch. |
| `--n-jobs` | `-1` | Parallel encoding workers; `-1` uses all cores. |
| `--label`, `--labels` | Omitted | One or more quoted exact labels to pool; omission samples the entire input. |

The script may select fewer than `--target` reactions if the input is exhausted under the chosen cutoff.

## Step 2 - Preprocess reactions


- [preprocess_legacy.py](dataset_processing/preprocess_legacy.py) is the preprocessing version used to create the FlowER-OM dataset. It applies the SMARTS replacement rules in [replace.py](dataset_processing/replace.py) to standardize catalyst and reagent representations.

```shell
python dataset_processing/preprocess_legacy.py data/input.txt data/processed.txt
```

The legacy version has a minor matching bug that can select a replacement rule without all required components. The corrected [preprocess.py](dataset_processing/preprocess.py) fixes these cases and can produce different outputs for affected rows. Use the legacy command above to reproduce the original preprocessing behavior.

Input and output use the same SMILES/label TXT format. Output is appended if the file already exists; use a fresh output path for each run to avoid duplicating rows.

## Step 3 - Prepare train/test/val splits

- [split_dataset.py](dataset_processing/split_dataset.py) shuffles the data with a fixed seed and writes training, validation, and test splits.

```shell
python dataset_processing/split_dataset.py data/processed.txt data/train.txt data/val.txt data/test.txt
```

## Step 4 - Generate mechanistic steps

Set `data`, `save`, and `not_templated` in [scripts/setting.py](scripts/setting.py) separately for each split. For the training split, use:

```python
self.data = 'data/train.txt'
self.save = 'data/train_mechanistic.txt'
self.not_templated = 'data/train_untemplated.txt'
```

```shell
python ver2.py
```

Generated rows contain five pipe-separated fields:

```text
reaction_smiles|reaction_class|mechanism_variant|step_description|source_reaction_number
```

See the [new_templates README](new_templates/README.md) for template details.


## Step 5 - Deduplicate elementary steps

[deduplicate_steps.py](dataset_processing/deduplicate_steps.py) removes symmetry-equivalent steps within each source reaction, preserves atom maps, and converts metal bonds to dative form for FlowER-OM. Run it on the mechanistic output for each split:

```shell
python -m dataset_processing.deduplicate_steps --input_file data/train_mechanistic.txt --output_file data/train_deduplicated.txt
```

## Step 6 - Verify BE matrices

[verify_valid_BE.py](dataset_processing/verify_valid_BE.py) checks that product BE matrices reconstruct the correct molecules and dative bonds. Run this final filter for each split:

```shell
python -m dataset_processing.verify_valid_BE --input_file data/train_deduplicated.txt --output_file data/train_verified.txt --failed_file data/train_failed.txt
```



## Citation
```
Coming soon!
```
