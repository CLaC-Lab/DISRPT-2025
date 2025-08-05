"""
Zero-shot discourse relation classification using Claude Opus 4.0.

Reads a dataset of discourse unit pairs and predicts the relation between Arg1 and Arg2.
Supports two input strategies:
- Natural argument order (`unordered_arg1`, `unordered_arg2`)
- Relation-directed order (`ordered_arg1`, `ordered_arg2`)

Saves predictions under a dynamically chosen zero-shot column based on the selected order.
When restarted, it processes only rows where prediction is missing.
"""

from litellm import completion
from typing import List, Dict
import pandas as pd
import time

model = 'anthropic/claude-opus-4-0'
dataset_name = './dev_subset_1.csv'
relation_order = False  # True = relation-directed order, False = natural order
wait_time = 3  # seconds between API calls

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

dev_data = pd.read_csv(dataset_name)

# Set input field names and prediction column based on argument order strategy
if relation_order:
    arg_1_key = "ordered_arg1"
    arg_2_key = "ordered_arg2"
    pred_key = 'zeroshot_relationOrder'
else:
    arg_1_key = "unordered_arg1"
    arg_2_key = "unordered_arg2"
    pred_key = 'zeroshot_naturalOrder'


if pred_key not in dev_data.columns:
    dev_data[pred_key] = None

# Construct the system prompt for the zero-shot discourse relation classification task
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

        ### Output format:
        Return **only one label** from the list above.
        """


def generate_respose(msgs: List[Dict]) -> str:
    """
    Calls Claude Opus 4.0 to classify a discourse relation given two arguments.
    Returns the predicted label as a plain string.
    """

    response = completion(
        model=model,
        messages=msgs,
        max_tokens=5,
        temperature=0.0
    )
    return response.choices[0].message.content

print(f"Processing dataset: {dataset_name}")
for index, row in dev_data.iterrows():

    if index % 5 == 0 and index > 0:
        dev_data.to_csv(f'{dataset_name}', index=False)

    if dev_data[pred_key].notna().all():
        print("✅ All rows parsed. Exiting loop.")
        break

    if pd.notna(row.get(pred_key, None)):
        continue
    
    msgs = [
        {"role": "system", "content":system_prompt},
        {"role": "user", "content": f"Arg1: {row[arg_1_key]} \nArg2: {row[arg_2_key]}"}
    ]

    retries = 2
    while retries > 0:
        try:
            response = generate_respose(msgs)
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
            time.sleep(5)

dev_data.to_csv(f'{dataset_name}', index=False)
