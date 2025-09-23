# DISRPT 2025 – Task 3 Participation

This repository contains our system submission for Task 3: Discourse Relation Classification in the DISRPT 2025 Shared Task (part of the CODI Workshop @ EMNLP 2025).

We evaluated strong multilingual BERT-based baselines, explored progressive unfreezing strategies, and experimented with zero-/few-shot prompting using Claude Opus 4.0.  
Finally, we introduced a hierarchical adapter + contrastive learning model that surpassed the baselines with significantly fewer trainable parameters.

**Paper**: [arXiv:2509.16903](https://arxiv.org/abs/2509.16903)  
**Official Task**: [DISRPT 2025](https://sites.google.com/view/disrpt2025/)  
**Dataset**: [DISRPT Shared Task 2025](https://github.com/disrpt/sharedtask2025)

## Structure

```
models/
├── bert-base-models/     # mBERT & XLM-R experiments
│   └── Contains Google Colab notebooks with training logs and results.
├── prompt-llm/           # Prompt-based experiments with Claude (zero-/few-shot)
├── hidac/                # Hierarchical adapter + contrastive learning model (external repo linked)
results/                  # Coming soon: evaluation outputs and summary tables
scripts/                  # Data extraction and preprocessing scripts (extract_data.py)
```

*Work in progress; models are being added incrementally.*

## Authors

- [Nawar Turk](https://www.linkedin.com/in/nawart/)  
- [Daniele Comitogianni](https://www.linkedin.com/in/daniele-comitogianni/)
