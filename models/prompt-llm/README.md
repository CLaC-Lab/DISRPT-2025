# Discourse Relation Classification

Two scripts for classifying discourse relations with Claude Opus 4.0:  
- **zero_shot.py**: zero-shot classification of discourse unit pairs.  
- **few_shot.py**: few-shot classification using English-only or language-specific examples.  

Both save predictions back into the dataset CSV.  

## Data Preparation
- **create_balanced_example_dataset.py**: builds a balanced pool of ~1k examples across languages, frameworks, and labels for few-shot prompting.  
- **create_dev_groups.py**: splits the development dataset into stratified groups (equal label distribution) and randomly selects 4 groups (~1k instances each) for prompt engineering experiments, since prompting the full dev set is infeasible.

