# Build DISRPT 2025 Task 3 Dataset

This script extracts relation pairs from the official DISRPT 2025 shared task data
(`.rels` files) and produces clean CSVs (`train.csv` and `dev.csv`) with the
columns needed for Task 3 (relation classification).

⚠️ **Important Licensing Note**  
The following corpora are **not included** in this repository and require
separate licenses/permissions:

```python
CORPORA_WITH_UNDERSCORED_TEXT = {
    "eng.pdtb.pdtb",
    "eng.rst.rstdt",
    "tur.pdtb.tdb",
    "zho.pdtb.cdtb"
}
```
This script will automatically skip these corpora if present.

## Usage

```bash
# Clone the DISRPT 2025 data repository
git clone https://github.com/disrpt/sharedtask2025.git

# Navigate to this folder (where build_dataset.py lives)
cd scripts

# Run the script
python3 build_dataset.py
```

## Output

- **train.csv**
- **dev.csv** 

The script will skip underscored corpora and print progress logs as it processes each corpus.
