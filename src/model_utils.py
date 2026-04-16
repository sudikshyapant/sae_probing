"""HuggingFace model loading, last-token activation extraction, and SAE encoding."""

from pathlib import Path

import numpy as np
import torch
from tqdm.notebook import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── LLM ──────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(config: dict):
    """Load Gemma-2-2B (or any causal LM) and its tokenizer.

    Handles Colab HF_TOKEN auth automatically; falls back to local
    ``huggingface-cli login`` credentials otherwise.
    """
    try:
        from google.colab import userdata          # type: ignore
        from huggingface_hub import login          # type: ignore
        login(token=userdata.get("HF_TOKEN"), add_to_git_credential=False)
        print("Authenticated with HuggingFace via Colab secret.")
    except Exception:
        pass

    print(f"Loading tokenizer: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model (this may take a while)…")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        output_hidden_states=True,
        torch_dtype=torch.float16 if config["device"] == "cuda" else torch.float32,
    ).to(config["device"])
    model.eval()
    print("Model loaded.")
    return model, tokenizer


def extract_activations(prompts: list[str], model, tokenizer,
                        layer_idx: int, batch_size: int,
                        device: str) -> np.ndarray:
    """Return last-token hidden states at *layer_idx* for every prompt.

    Shape: (n_prompts, d_model).
    """
    all_acts = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Extracting activations"):
        batch  = prompts[i : i + batch_size]
        tokens = tokenizer(batch, return_tensors="pt",
                           padding=True, truncation=False).to(device)
        with torch.no_grad():
            out = model(**tokens)
        hidden   = out.hidden_states[layer_idx]          # (B, seq_len, d_model)
        seq_lens = tokens["attention_mask"].sum(dim=1) - 1
        last_tok = hidden[torch.arange(len(batch)), seq_lens]  # (B, d_model)
        all_acts.append(last_tok.float().cpu().numpy())
    return np.concatenate(all_acts, axis=0)


def load_and_cache_activations(prompts_dict: dict[str, list[str]],
                               config: dict,
                               cache_path: Path) -> dict[str, np.ndarray]:
    """Load activations from *cache_path* or extract + save them.

    Parameters
    ----------
    prompts_dict : {"train": [...], "val": [...], "test": [...]}
    config       : CONFIG dict (uses model_name, target_layer, batch_size, device)
    cache_path   : path to .pt cache file

    Returns
    -------
    {"train": ndarray, "val": ndarray, "test": ndarray}
    """
    if cache_path.exists():
        print("Loading cached activations…")
        cache = torch.load(cache_path, weights_only=False)
        print(f"Loaded. Train shape: {cache['train'].shape}")
        return cache

    model, tokenizer = load_model_and_tokenizer(config)
    cache = {}
    for split, prompts in prompts_dict.items():
        cache[split] = extract_activations(
            prompts, model, tokenizer,
            config["target_layer"], config["batch_size"], config["device"],
        )
    torch.save(cache, cache_path)
    print(f"Activations saved. Train shape: {cache['train'].shape}")

    del model
    if config["device"] == "cuda":
        torch.cuda.empty_cache()
    return cache


# ── SAE ──────────────────────────────────────────────────────────────────────

def encode_with_sae(activations: np.ndarray, sae,
                    batch_size: int, device: str) -> np.ndarray:
    """Encode dense activations through an SAELens SAE.

    Returns sparse latent matrix of shape (n_prompts, sae_width).
    """
    all_latents = []
    for i in tqdm(range(0, len(activations), batch_size), desc="SAE encoding"):
        batch = torch.tensor(activations[i : i + batch_size],
                             dtype=torch.float32).to(device)
        with torch.no_grad():
            latents = sae.encode(batch)
        all_latents.append(latents.cpu().numpy())
    return np.concatenate(all_latents, axis=0)


def load_and_cache_latents(act_dict: dict[str, np.ndarray],
                           config: dict,
                           cache_path: Path) -> dict[str, np.ndarray]:
    """Load SAE latents from *cache_path* or encode + save them.

    Parameters
    ----------
    act_dict   : {"train": ndarray, "val": ndarray, "test": ndarray}
    config     : CONFIG dict (uses sae_release, sae_id, batch_size, device)
    cache_path : path to .pt cache file

    Returns
    -------
    {"train": ndarray, "val": ndarray, "test": ndarray}
    """
    if cache_path.exists():
        print("Loading cached SAE latents…")
        cache = torch.load(cache_path, weights_only=False)
        print(f"Loaded. Train shape: {cache['train'].shape}")
        return cache

    from sae_lens import SAE  # type: ignore
    print(f"Loading SAE: {config['sae_release']} / {config['sae_id']}")
    sae, cfg_dict, _ = SAE.from_pretrained_with_cfg_and_sparsity(
        release=config["sae_release"],
        sae_id=config["sae_id"],
        device=config["device"],
    )
    sae.eval()
    print(f"SAE loaded. Width: {cfg_dict.get('d_sae', '?')}")

    cache = {
        split: encode_with_sae(acts, sae, config["batch_size"], config["device"])
        for split, acts in act_dict.items()
    }
    torch.save(cache, cache_path)
    print(f"Latents saved. Train shape: {cache['train'].shape}")

    del sae
    if config["device"] == "cuda":
        torch.cuda.empty_cache()
    return cache
