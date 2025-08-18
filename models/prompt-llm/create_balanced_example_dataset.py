"""
Create a balanced example dataset for few-shot prompting.

- Samples up to N_PER_GROUP examples per (lang, framework, label_text) group.
- Produces ~1k examples for prompt engineering.
- Outputs to ./balanced_example_dataset.csv

User action required:
- Replace 'path/to/the/train/dataset' with the actual training dataset path.
"""
import pandas as pd

SEED = 42
N_PER_GROUP = 3  # number of examples per (lang, framework, label_text) group

# TODO (user): update this path to your training dataset
train = pd.read_csv("path/to/the/train/dataset")

balanced_example_dataset = (
    train.groupby(["lang", "framework", "label_text"])
         .apply(lambda x: x.sample(min(len(x), N_PER_GROUP), random_state=SEED))
         .reset_index(drop=True)
)

# Distribution sanity check
label_dist = train["label_text"].value_counts(normalize=True)
print(label_dist)

print(f"Number of groups: {train.groupby(['lang', 'framework', 'label_text']).ngroups}")
print(f"Balanced dataset size: {len(balanced_example_dataset)} examples.")

# Save output next to this script
balanced_example_dataset.to_csv("./balanced_example_dataset.csv", index=False)