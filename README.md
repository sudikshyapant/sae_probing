# SAE Probing

Re-implementation of SAE probing from [Kantamneni et al., 2025](https://arxiv.org/abs/2412.06093). Classifies athletes by sport (football vs. not) using Gemma-2-2B hidden states and GemmaScope SAE latents.

## Run

Open in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sudikshyapant/sae_probing/blob/main/sae_probing.ipynb)

1. Add `HF_TOKEN` to Colab secrets (needs access to `google/gemma-2-2b`)
2. Run the **Google Drive setup cell** first — saves all outputs to `MyDrive/sae_probing/`

## Local

```bash
pip install -r requirements.txt
jupyter notebook sae_probing.ipynb
```
