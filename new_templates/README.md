# Mechanistic reaction templates

The three Python modules build a library of staged reaction SMARTS. Start with
`reaction_templates.py` to find a reaction class and trace its mechanism back
to the elementary steps and SMARTS fragments it uses.

| File | Role |
| --- | --- |
| [groups.py](groups.py) | Defines `fg`, a shared dictionary of SMARTS fragments for functional groups, reagents, and ligands. Entries such as `fg["Halides"]` can be interpolated into larger SMARTS patterns. |
| [elem_steps.py](elem_steps.py) | Defines `steps`, a registry of parameterized elementary transformations such as ligand association, oxidative addition, transmetalation, and reductive elimination. The builders return atom-mapped reaction SMARTS. `build_step` fills placeholders using `fg` and explicit arguments, and helper functions adjust atom constraints such as charge and hydrogen count. |
| [reaction_templates.py](reaction_templates.py) | Combines elementary transformations into numbered stages. The `catalyst_preparation` and `catalytic_cycles` registries supply reusable stage sequences. The module exports `class_reaction_templates`, which maps reaction-class labels to named mechanism variants. |

## Template structure

A class key can be a tuple of labels that share the same mechanism variants.
Each variant has a `Stages` dictionary keyed by stage number and can specify
`Reagent` and `Exclude_reagent` SMARTS to describe which inputs it applies to.
A stage contains a `Templates` list of reaction SMARTS and a `Description`.
When present, entries in the stage's `pKa` list correspond to templates by
position. A `None` entry indicates that the template has no pKa annotation.

For example, the Suzuki entry includes a `Pd(0) with ligand - borate` variant.
It uses the ligand attachment builder to populate stages 0 and 1 and starts
the Suzuki borate cycle at stage 2.

## Inspect a template

From the repository root, run this in Python to print the first stage for
that variant.

```python
from new_templates.reaction_templates import class_reaction_templates

class_key = next(
    key for key in class_reaction_templates
    if key == "Bromo Suzuki coupling"
    or (isinstance(key, tuple) and "Bromo Suzuki coupling" in key)
)
mechanism = class_reaction_templates[class_key]["Pd(0) with ligand - borate"]
stage = mechanism["Stages"][0]
print(stage["Description"])
print(stage["Templates"][0])
```

Use [template_testing.ipynb](template_testing.ipynb) to browse reaction
classes, mechanisms, and stages interactively.

## Where to make changes

Edit `groups.py` for a shared SMARTS fragment, `elem_steps.py` for a reusable
elementary transformation, and `reaction_templates.py` for a mechanism's
reagent criteria or stage sequence. When combining stage dictionaries, check
their starting offsets so one sequence does not overwrite another's stages.

(Written with GPT-6.1 Sol)