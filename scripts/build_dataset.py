import os
import pandas as pd

CORPORA_WITH_UNDERSCORED_TEXT = {
    "eng.pdtb.pdtb",
    "eng.rst.rstdt",
    "tur.pdtb.tdb",
    "zho.pdtb.cdtb"
}

def is_underscored_instance(row):
    # return all(c == '_' or c.isspace() for c in row["unit1_txt"]) and \
    #        all(c == '_' or c.isspace() for c in row["unit2_txt"])
    return "__" in row["unit1_txt"] and "__" in row["unit2_txt"]

def load_rel_file(rel_path):
  data = []
  co = os.path.basename(rel_path)
  corpus_name = os.path.basename(os.path.dirname(rel_path))
  language = corpus_name.split('.')[0] if '.' in corpus_name else 'unknown'
  framework = corpus_name.split('.')[1] if '.' in corpus_name else 'unknown'
  subcorpus = corpus_name.split('.')[2] if '.' in corpus_name else 'unknown'
  file_name = os.path.basename(rel_path)

  with open(rel_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

    start_index = 1 if lines and lines[0].strip().startswith('doc\t') else 0  # skipping header
    count = 0

    for line_num, line in enumerate(lines[start_index:], start=start_index+1):
      line = line.strip()
      if not line:
          continue

      parts = line.split('\t')
      if len(parts) == 15:
        row = {
              'corpus_name': corpus_name,
              'lang': language,
              'framework': framework,
              'subcorpus': subcorpus,
              'file': file_name,
              'doc': parts[0],
              'unit1_toks': parts[1],
              'unit2_toks': parts[2],
              'unit1_txt': parts[3],
              'unit2_txt': parts[4],
              'unit1_raw_text': parts[5],
              'unit2_raw_text': parts[6],
              's1_toks': parts[7],
              's2_toks': parts[8],
              'unit1_sent': parts[9],
              'unit2_sent': parts[10],
              'dir': parts[11],
              'type': parts[12],
              'orig_label': parts[13],
              'label_text': parts[14]
        }

        row["unordered_arg1"] = row['unit1_txt']
        row["unordered_arg2"] = row['unit2_txt']

        row["ordered_arg1"] = row['unit2_txt'] if row['dir'] == "1<2" else row['unit1_txt']
        row["ordered_arg2"] = row['unit1_txt'] if row['dir'] == "1<2" else row['unit2_txt']

        if is_underscored_instance(row):
          continue

        data.append(row)
        count += 1
      else:
        print(f'{file_name} at line {line_num} has only {len(parts)} columns ⛔⛔⛔⛔⛔⛔⛔⛔⛔')

    print(f"  ✅{file_name[:-5]}: Loaded {len(data)} examples")
    return data

def load_global_splits(corpora_dirs):
  train_data, dev_data= [], []
  processed_corpus_num = 0

  for corpus_path in corpora_dirs:
    corpus_name = os.path.basename(corpus_path)
    if corpus_name in CORPORA_WITH_UNDERSCORED_TEXT:
      print(f"⛔ Skipping underscored corpus: {corpus_name}")
      continue

    processed_corpus_num += 1
    rel_count = 0
    print(f'\nProcessing corpus: {corpus_name}')

    for file_name in os.listdir(corpus_path):
      if not file_name.endswith('.rels'):
        continue

      rel_path = os.path.join(corpus_path, file_name)
      data = load_rel_file(rel_path)

      rel_count += len(data) 

      if 'train' in file_name.lower():
        train_data.extend(data)
      elif 'dev' in file_name.lower():
        dev_data.extend(data)
      elif 'test' in file_name.lower():
        print('skipping test set')
      else:
        print(f'Unknown split in {file_name} ❌❌❌')
    print(f"🔢Total relations from {corpus_name}: {rel_count}")

  print(f"\nTotal corpus: {processed_corpus_num}")
  print(f"Total train: {len(train_data)}")
  print(f"Total dev: {len(dev_data)}")

  return train_data, dev_data

datapath = 'sharedtask2025/data'
corpora_dirs = [os.path.join(datapath, d)
          for d in os.listdir(datapath)
          if os.path.isdir(os.path.join(datapath,d))
          ]
global_train, global_dev = load_global_splits(corpora_dirs)

# Convert global_train list of dicts to DataFrame
df_train = pd.DataFrame(global_train)
df_dev = pd.DataFrame(global_dev)

print("\nFirst 5 rows of extracted data:")
print(df_train.head())

# Save to a new CSV file
output_path_train = './train.csv'
df_train.to_csv(output_path_train, index=False)
output_path_dev = './dev.csv'
df_dev.to_csv(output_path_dev, index=False)

print(f"\n\nTrain size: {len(df_train)} rows")
print(f"Dev size: {len(df_dev)} rows\n\n")