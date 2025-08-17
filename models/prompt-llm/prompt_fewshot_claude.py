"""
Few-shot discourse relation classification using Claude Opus 4.0.

Loads a dataset of discourse-unit pairs and builds few-shot prompts 
using either the instance’s language or English-only examples.

Supports:
- Natural argument order (`unordered_arg1`, `unordered_arg2`)
- Relation-directed order (`ordered_arg1`, `ordered_arg2`)

Dynamically samples up to n examples per language to construct prompts.
Saves predictions in a dedicated column and resumes only missing rows.
"""

from litellm import completion
from typing import List, Dict
import pandas as pd
import time

# === Config ===
SEED =42
model = 'anthropic/claude-opus-4-0'
dataset_name = './dev_subset_1.csv'
example_dataset = './balanced_example_dataset.csv'
relation_order = False  # True = relation-directed order | False = natural order
wait_time = 3  # seconds between API calls
num_examples = 4
example_lang = "english-only" # "lang-specific" | "english_only" 

unique_labels = [
    'explanation', 
    'organization', 
    'comment', 
    'elaboration', 
    'conjunction',
    'attribution',
    'concession',
    'frame',
    'reformulation',
    'temporal',
    'condition',
    'causal',
    'contrast',
    'query',
    'purpose',
    'mode',
    'alternation'
    ]

examples_df = pd.read_csv(example_dataset)
dev_data = pd.read_csv(dataset_name)

# === set pred column name ===
if relation_order:
    arg_1_key = "ordered_arg1"
    arg_2_key = "ordered_arg2"
    pred_key = f"few-shot_relation-order_{example_lang}"
else:
    arg_1_key = "unordered_arg1"
    arg_2_key = "unordered_arg2"
    pred_key = f"few-shot_natural-order_{example_lang}"

if pred_key not in dev_data.columns:
    dev_data[pred_key] = None

# === Helper functions ===
def generate_response(msgs: List[Dict]) -> str:
    """Send messages to the model and return the predicted label."""

    response = completion(
        model=model,
        messages=msgs,
        max_tokens=10,
        temperature= 0.0
    )
    return response.choices[0].message.content

def generate_examples(lang, examples_df=examples_df):
    """Sample up to num_examples examples for a given language to build few-shot prompts."""

    filtered_df = examples_df[examples_df['lang']==lang] 
    sampled_df = filtered_df.sample(
         min(num_examples, len(filtered_df)),
        random_state=SEED
    )

    example_text = ""
    for _, row in sampled_df.iterrows():
        example_text += f"Arg1: {row['unordered_arg1']}\n"
        example_text += f"Arg2: {row['unordered_arg2']}\n"
        example_text += f"Label: {row['label_text']}\n"

    return example_text.strip()

def generate_sys_prompt(lang, examples_df=examples_df):
    """Construct the system prompt with task rules and language-specific examples."""

    examples = generate_examples(lang, examples_df)
    system_prompt = f"""
        You are a highly accurate **discourse relation classifier**.  

        ### Task:
        You will be given two text segments (Arg1 and Arg2).  
        Your job is to classify the discourse relation between them.  

        ### Rules:
        1. Choose **exactly one** label from the list below.  
        2. Respond with **only the label text** (no punctuation, explanations, or extra words).  
        3. If you are uncertain, pick the most likely label.  
        4. Do **not** output anything outside the label list.  

        ### Label list:
        {", ".join(unique_labels)}

        ### Examples:
        {examples}

        ### Output format:
        Return **only one label** from the list above.
        """
    return system_prompt

# === Main process ===
print(f"\nProcessing: {dataset_name}\n")
for index, row in dev_data.iterrows():
    if index % 5 == 0 and index > 0:
        dev_data.to_csv(f"{dataset_name}", index=False)

    if dev_data[pred_key].notna().all():
        print("✅ All rows parsed. Exiting loop.")
        break

    if pd.notna(row.get(pred_key, None)):
        continue
    
    if example_lang == "english-only":
        lang = "eng"
    elif example_lang == "lang-specific":
        lang = row["lang"]
    else:
        raise ValueError(f"Unsupported example_lang option: {example_lang}")


    system_prompt = generate_sys_prompt(lang)
    msgs = [
        {"role": "system", "content":system_prompt},
        {"role": "user", "content": f"Arg1: {row[arg_1_key]} \nArg2: {row[arg_2_key]}"}
    ]

    retries = 2
    while retries>0:
        try:
            print(f"{index/len(dev_data)*100:.2f}% ... processing row: {index}")
            print(f"Arg1: {row[arg_1_key]} \nArg2:{row[arg_2_key]}")
            response = generate_response(msgs)
            dev_data.loc[index, pred_key] = response
            print()
            print(f"Correct label: {row['label_text']}")
            print(f"Prediction: {response}")
            print('------------------------------\n')
            time.sleep(wait_time)
            break
        except Exception as e:
            print(f"⚠️ API error on row {index}: {e}")
            retries -= 1
            print(f"\n******issue at row {index}\n___arg1: {row[arg_1_key]}\n___arg2: {row[arg_2_key]}\n")
            time.sleep(wait_time)

dev_data.to_csv(f"{dataset_name}", index=False)