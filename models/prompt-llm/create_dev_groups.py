"""
Stratify the dev se into N_GROUPS folds,
preserving label balance. Randomly pick N_PICK groups and
save them as CSVs (dev_group_i.csv). Requires `label_text`.

Outputs:
- Prints how many folds created (~1/N_GROUPS each).
- For each saved group: row count, % of full dev, and per-label distribution.
- CSV files: dev_group_1.csv ... dev_group_N_PICK.csv

User TODO:
- Update `dev = pd.read_csv("path/to/the/dev/dataset")` with actual path.
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold
import random

SEED = 42
N_GROUPS = 27  # Split into 27 equal stratified groups
N_PICK = 4     # Pick 4 groups for prompting


# TODO (user):  update this path to your dev dataset
dev = pd.read_csv("path/to/the/dev/dataset") 

skf = StratifiedKFold(n_splits=N_GROUPS, shuffle=True, random_state=SEED)

groups = []
for group_idx, (_, idx) in enumerate(skf.split(dev, dev["label_text"])):
    split = dev.iloc[idx].reset_index(drop=True)
    groups.append(split)

print(f"Created {len(groups)} stratified groups (~{100 / N_GROUPS:.2f}% each)")

# Pick 4 groups randomly
random.seed(SEED)
selected_groups = random.sample(groups, N_PICK)

# Save the selected groups
for i, group in enumerate(selected_groups, 1):
    group.to_csv(f"./dev_group_{i}.csv", index=False)
    print(f"Group {i}: {len(group)} rows "
          f"({len(group)/len(dev):.1%} of full dev)")
    print(group["label_text"].value_counts(normalize=True).round(3))
    print("----")