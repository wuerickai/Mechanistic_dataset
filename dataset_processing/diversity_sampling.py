
import argparse
import os
import time
import pandas as pd
import numpy as np
from drfp import DrfpEncoder
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed


def encode_batch(smiles_list):
    """Encode a list of SMILES reactions directly (no per-item overhead)."""
    fps = DrfpEncoder.encode(smiles_list, mapping=False)
    return np.array(fps)


def filter_new_against_existing(existing_fps, new_fps, cutoff):
    n = len(new_fps)

    # --- Step 1: bulk check all new vs all existing (one cdist call) ---
    if len(existing_fps) > 0:
        dist_to_existing = cdist(new_fps, existing_fps, metric='cosine')  # (n_new, n_existing)
        fails_existing = np.any(dist_to_existing < cutoff, axis=1)        # (n_new,) boolean
    else:
        fails_existing = np.zeros(n, dtype=bool)

    candidates = np.where(~fails_existing)[0]
    if len(candidates) == 0:
        return []

    # --- Step 2: precompute all pairwise distances among candidates (one cdist call) ---
    candidate_fps = new_fps[candidates]
    dist_among = cdist(candidate_fps, candidate_fps, metric='cosine')  # (k, k)

    # --- Step 3: greedy selection on the precomputed matrix (no more cdist inside loop) ---
    rejected = np.zeros(len(candidates), dtype=bool)
    for i in range(len(candidates)):
        if rejected[i]:
            continue
        # vectorized: reject everything too close to i in one shot
        too_close = dist_among[i] < cutoff
        too_close[:i+1] = False        # only look forward
        rejected |= too_close

    return candidates[~rejected].tolist()


def iterative_random_sampling(input_file, target=12000, cutoff=0.85, batch_size=500, n_jobs=-1,
                              output_file="output.txt", label=None):
    """
    Read whitespace-separated reaction SMILES and labels, and write selected lines.
    Labels may contain spaces; original line contents are preserved.
    label accepts one exact label or an iterable of exact labels. Sample their
    combined pool with one shared target and retain every other row.
    In label mode, output and the returned DataFrame retain original input order.
    Incrementally samples and filters reactions using cosine distance.
    Only compares new reactions against the existing accepted set — O(n*k) not O(n²).
    """
    start = time.time()
    if target < 0 or batch_size <= 0:
        raise ValueError("target must be nonnegative and batch_size must be positive")
    labels = None if label is None else ([label] if isinstance(label, str) else list(label))
    if labels is not None and not labels:
        raise ValueError("label must contain at least one label")
    rows = []
    with open(input_file, encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"{input_file}:{line_number}: expected reaction SMILES followed by a label")
            rows.append({"rxn": parts[0], "label": parts[1].strip(), "line": line})
    all_rxns = pd.DataFrame(rows, columns=["rxn", "label", "line"])
    matching = all_rxns["label"].isin(labels) if labels is not None else None
    rxns_space = all_rxns if labels is None else all_rxns.loc[matching]
    rxns_space = rxns_space.sample(frac=1, random_state=42)

    accepted_rxns = []
    accepted_fps = np.empty((0, 2048), dtype=np.uint8)  # adjust 2048 to your DRFP dim
    rng = np.random.default_rng(6023)
    remaining = rxns_space.copy()
    steps = 0

    while len(accepted_fps) < target:
        if len(remaining) == 0:
            print("Ran out of reactions to sample.")
            break

        n_sample = min(batch_size, len(remaining))
        batch_idx = rng.choice(len(remaining), size=n_sample, replace=False)
        batch = remaining.iloc[batch_idx]
        remaining = remaining.drop(remaining.index[batch_idx])

        # --- Encode batch (parallelized across chunks with joblib) ---
        t0 = time.time()
        effective_jobs = os.cpu_count() or 1 if n_jobs < 0 else max(1, n_jobs)
        chunk_size = max(1, n_sample // effective_jobs)
        smiles_chunks = [batch.rxn.tolist()[i:i+chunk_size] for i in range(0, n_sample, chunk_size)]
        results = Parallel(n_jobs=n_jobs)(delayed(encode_batch)(chunk) for chunk in smiles_chunks)
        new_fps = np.vstack(results)
        print(f"Step {steps}: encoded {n_sample} in {time.time()-t0:.2f}s")

        # --- Incremental filtering: only new vs. accepted ---
        t1 = time.time()
        keep_indices = filter_new_against_existing(accepted_fps, new_fps, cutoff)
        print(f"  Kept {len(keep_indices)}/{n_sample} | filtered in {time.time()-t1:.2f}s")

        if keep_indices:
            accepted_fps = np.vstack([accepted_fps, new_fps[keep_indices]]) if len(accepted_fps) else new_fps[keep_indices]
            accepted_rxns.append(batch.iloc[keep_indices])

        steps += 1
        print(f"  Total accepted so far: {len(accepted_fps)}/{target}")

    selected = (pd.concat(accepted_rxns).iloc[:target]
              if accepted_rxns else rxns_space.iloc[:0].copy())
    if labels is None:
        result = selected.reset_index(drop=True)
    else:
        result = all_rxns.loc[~matching |
                              all_rxns.index.isin(selected.index)].reset_index(drop=True)
    with open(output_file, "w", encoding="utf-8") as destination:
        for line in result["line"]:
            destination.write(line + "\n")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s | {len(selected)} selected | {len(result)} output reactions | {steps} steps")
    with open('info.txt', 'a') as f:
        f.write(f"\nTime: {elapsed:.2f}s | Accepted: {len(selected)} | Output: {len(result)} | Steps: {steps}\n")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select diverse reactions from a SMILES/label TXT file.")
    parser.add_argument("input_file", help="TXT with reaction SMILES followed by whitespace and a label")
    parser.add_argument("output_file", help="Destination TXT; preserves selected input lines")
    parser.add_argument("--target", type=int, default=12000)
    parser.add_argument("--cutoff", type=float, default=0.17)
    parser.add_argument("--batch-size", type=int, default=3000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--label", "--labels", nargs="+",
                        help="Exact labels to pool under one shared target; retain all other labels")
    args = parser.parse_args()
    iterative_random_sampling(
        args.input_file,
        target=args.target,
        cutoff=args.cutoff,
        batch_size=args.batch_size,
        n_jobs=args.n_jobs,
        output_file=args.output_file,
        label=args.label,
    )



#Cutoffs
#miyaura .14
#hartwig .4
#suzuki .45
#sonogashira .33
#stille 0.17
