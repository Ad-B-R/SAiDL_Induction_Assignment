import huggingface_hub
from transformer_lens import HookedTransformer
from datasets import load_dataset
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import wandb
import copy

from sae_lens import SAE
device = "cuda" if torch.cuda.is_available() else "cpu"

model = HookedTransformer.from_pretrained("distilgpt2", device=device)
tokenizer = model.tokenizer
model.eval()
hook_point = "blocks.2.hook_resid_pre"

checkpoint_paths = {
    512:  "./sae_checkpoints_saelens/m512/2ezt06ox/90910720",
    1024: "./sae_checkpoints_saelens/m1024/q9f9m5nl/90910720"
}

configurations = [
    (None, None),   # baseline
    (None, 8),      # SAE 8-bit only
    (8, 8),         # joint 8-bit
    (None, 4),      # SAE 4-bit only
    (4, 4),         # joint 4-bit
    (None, 2),      # SAE 2-bit only
    (2, 2),         # joint 2-bit
    (2, None)       # model 2-bit only
]

def quantize_gen(min_, max_, bit, tensor):
    q_max = int(2**bit - 1)
    s = (max_ - min_) / (q_max + 1e-9)
    shift = tensor - min_
    q = torch.round(shift / (s + 1e-9))
    q_clip = torch.clamp(q, 0, q_max)
    return q_clip * s + min_

def quantize_tl_weights(model, bits):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "W_" in name:
                w = param.data
                w_q = quantize_gen(w.min(), w.max(), bits, w)
                param.data.copy_(w_q)

def CKA(X, Y):
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    
    dot_product_similarity = torch.norm(X.t() @ Y)**2
    
    normalization_x = torch.norm(X.t() @ X)
    normalization_y = torch.norm(Y.t() @ Y)
    
    cka_score = dot_product_similarity / (normalization_x * normalization_y + 1e-9)
    return cka_score.item()

def spectral_analysis(Z, Z_hat, k=32):
    Z_c = Z - Z.mean(dim=0, keepdim=True)
    Z_c_hat = Z_hat - Z_hat.mean(dim=0, keepdim=True)

    U, S, Vh = torch.linalg.svd(Z_c, full_matrices=False)
    U_hat, S_hat, Vh_hat = torch.linalg.svd(Z_c_hat, full_matrices=False)

    Uk = Vh[:k].T
    Uk_hat = Vh_hat[:k].T

    cos_thetas = torch.linalg.svdvals(Uk.T @ Uk_hat)
    angles = torch.acos(torch.clamp(cos_thetas, -1, 1)) * (180 / math.pi)

    E = Z - Z_hat
    sds = (torch.norm(E @ Uk)**2 / torch.norm(Z @ Uk)**2).item()

    return {
        "singular_before": S,
        "singular_after":  S_hat,
        "angles": angles,
        "sds": sds
    }

def compute_fisher(Z):
    return (Z ** 2).mean(dim=0)

def get_eval_batches(dataset_stream, num_batches, batch_size=32, seq_len=128):
    batch_texts = []
    token_buffer = []
    batches_yielded = 0
    for example in dataset_stream:
        tokens = tokenizer(example["text"], return_tensors="pt")["input_ids"][0]
        token_buffer.extend(tokens.tolist())
        while len(token_buffer) >= seq_len:
            chunk = token_buffer[:seq_len]
            token_buffer = token_buffer[seq_len:]
            batch_texts.append(torch.tensor(chunk).unsqueeze(0))
            if len(batch_texts) == batch_size:
                yield torch.cat(batch_texts, dim=0).to(device)
                batch_texts = []
                batches_yielded += 1
                if batches_yielded >= num_batches:
                    return

def compute_ppl_with_hook(model, eval_batches, hook_fn):
    total_loss = 0
    with torch.no_grad():
        for batch in eval_batches:
            logits = model.run_with_hooks(
                batch,
                fwd_hooks=[(hook_point, hook_fn)]
            )
            loss = model.loss_fn(logits, batch)
            total_loss += loss.item()
    return math.exp(total_loss / len(eval_batches))

print("Loading dataset...")
dataset = load_dataset("openwebtext", split="train", streaming=True)
skip_dataset = dataset.skip(600_000)

calib_batches = list(get_eval_batches(skip_dataset, num_batches=1))
eval_batches  = list(get_eval_batches(skip_dataset, num_batches=5))
print(f"Loaded {len(calib_batches)} calib batches and {len(eval_batches)} eval batches")

for m_bottleneck, folder_path in checkpoint_paths.items():
    print(f"\n{'='*60}")
    print(f"Loading SAE for m={m_bottleneck}")
    print(f"{'='*60}")
    
    sae_model = SAE.load_from_pretrained(folder_path, device=device)
    sae_model.eval()

    for W_BIT, A_BIT in configurations:
        model_q = copy.deepcopy(model)
        if W_BIT is not None:
            quantize_tl_weights(model_q, bits=W_BIT)
            print(f"\nModel weights quantized to {W_BIT}-bit")
        else:
            print(f"\nModel weights at FP32")
        model_q.eval()

        global_min, global_max = float('inf'), float('-inf')
        with torch.no_grad():
            for batch in calib_batches:
                _, cache = model_q.run_with_cache(batch)
                acts = cache[hook_point]
                sparse_feats = sae_model.encode(acts)
                global_min = min(global_min, sparse_feats.min().item())
                global_max = max(global_max, sparse_feats.max().item())
        print(f"  Calibration | Min: {global_min:.4f}, Max: {global_max:.4f}")

        w_label = f"w{W_BIT}" if W_BIT is not None else "wFP32"
        a_label = f"a{A_BIT}" if A_BIT is not None else "aFP32"
        run_name = f"{w_label}_{a_label}_m{m_bottleneck}"
        
        wandb.init(
            project="sae-quantization-bonus",
            name=run_name,
            config={
                "weight_bits":  W_BIT if W_BIT is not None else 32,
                "act_bits":     A_BIT if A_BIT is not None else 32,
                "m_bottleneck": m_bottleneck,
            },
            reinit=True
        )
        print(f"  Evaluating: {run_name}")

        Z_all, Z_hat_all = [], []
        with torch.no_grad():
            for batch in tqdm(eval_batches, leave=False):
                _, cache = model_q.run_with_cache(batch)
                clean_acts = cache[hook_point]
                B, T, D = clean_acts.shape
                
                Z_batch = sae_model.encode(clean_acts.view(-1, D)).view(B, T, -1)
                
                if A_BIT is not None:
                    Z_hat_batch = quantize_gen(global_min, global_max, A_BIT, Z_batch)
                else:
                    Z_hat_batch = Z_batch
                
                Z_all.append(Z_batch.view(-1, Z_batch.size(-1)))
                Z_hat_all.append(Z_hat_batch.view(-1, Z_batch.size(-1)))

        Z = torch.cat(Z_all)
        Z_hat = torch.cat(Z_hat_all)

        def quant_hook(acts, hook, a_bit=A_BIT, gmin=global_min, gmax=global_max):
            if a_bit is None:
                return acts                          # passthrough baseline
            Z = sae_model.encode(acts)
            Z_hat = quantize_gen(gmin, gmax, a_bit, Z)
            return sae_model.decode(Z_hat)

        ppl = compute_ppl_with_hook(model_q, eval_batches, quant_hook)
        if A_BIT is not None:
            spectral = spectral_analysis(Z, Z_hat, k=32)
            cka = CKA(Z, Z_hat)
            sv_before, sv_after = spectral["singular_before"], spectral["singular_after"]
            angles = spectral["angles"]
            
            energy_before = (sv_before ** 2).sum()
            energy_after  = (sv_after ** 2).sum()
            energy_loss = ((energy_before - energy_after) / (energy_before + 1e-9)).item()
            collapse_ratio = (sv_after < 1e-3).float().mean().item()
            sds = spectral["sds"]
            angle_mean = angles.mean().item()
            angle_max  = angles.max().item()
        else:
            sds, cka = 0.0, 1.0
            angle_mean, angle_max = 0.0, 0.0
            energy_loss, collapse_ratio = 0.0, 0.0

        fisher = compute_fisher(Z)

        def sparsity(x):
            return (x == 0).float().mean().item()

        s_before, s_after = sparsity(Z), sparsity(Z_hat)

        wandb.log({
            "weight_bits":W_BIT if W_BIT is not None else 32,
            "act_bits":A_BIT if A_BIT is not None else 32,
            "Perplexity":ppl,
            "SDS":sds,
            "CKA":cka,
            "angle_mean":angle_mean,
            "angle_max":angle_max,
            "energy_loss":energy_loss,
            "collapse_ratio":collapse_ratio,
            "sparsity_before":s_before,
            "sparsity_after":s_after,
            "sparsity_delta":s_after - s_before,
            "fisher_mean":fisher.mean().item(),
        })

        print(f"Summary ({run_name})")
        print(f"PPL:{ppl:.2f}")
        print(f"SDS:{sds:.4f}")
        print(f"CKA:{cka:.4f}")
        print(f"Angle:{angle_mean:.2f}°")
        wandb.finish()

        del model_q, Z, Z_hat, Z_all, Z_hat_all
        torch.cuda.empty_cache()

print("<---------------------------------------JOINT QUANTIZATION COMPLETE--------------------------------------------->")
