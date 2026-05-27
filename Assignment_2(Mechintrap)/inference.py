from datasets import load_dataset
import huggingface_hub
import sys
import types
from transformer_lens import HookedTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import os
import math
import wandb
import matplotlib.pyplot as plt
import umap
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"
from sae_lens import SAE

device = "cuda" if torch.cuda.is_available() else "cpu"

model = HookedTransformer.from_pretrained("distilgpt2", device=device)
tokenizer = model.tokenizer
model.eval()

checkpoint_paths = {
    512: "./sae_checkpoints_saelens/m512/2ezt06ox/90910720",  
    1024: "./sae_checkpoints_saelens/m1024/q9f9m5nl/90910720" 
}

loaded_saes = {}

# 3. The Loading Loop
for m_bottleneck, folder_path in checkpoint_paths.items():
    print(f"Loading SAELens model for m={m_bottleneck}...")
    
    sae_model = SAE.load_from_pretrained(
        folder_path, 
        device=device
    )
    sae_model.eval()
    loaded_saes[m_bottleneck] = sae_model

    print("\nSuccessfully loaded all SAEs!")
    print(f"m={m_bottleneck} configured d_sae: {loaded_saes[m_bottleneck].cfg.d_sae}")

    dataset = load_dataset("openwebtext", split="train", streaming=True)
    skip_dataset = dataset.skip(600_000) # skip 400k data so that there is no collision between training and testing data

    def get_eval_batches(dataset_stream, num_batches, batch_size=32, seq_len=128):
        batch_texts = []
        token_buffer = []
        batches_yielded = 0

        for example in dataset_stream:
            tokens = tokenizer(example["text"], return_tensors="pt")["input_ids"][0]
            token_buffer.extend(tokens.tolist())

            while len(token_buffer) >= seq_len:
                chunk = token_buffer[:seq_len:1]
                token_buffer = token_buffer[seq_len::1]

                batch_texts.append(torch.tensor(chunk).unsqueeze(0)) 

                if len(batch_texts) == batch_size:
                    yield torch.cat(batch_texts, dim=0).to(device)
                    batch_texts = []
                    batches_yielded += 1
                    if batches_yielded >= num_batches:
                        return # Stop the generator once we hit our eval limit

    def quantize_gen(min, max, bit, tensor):
        q_max = int(2**bit - 1)
        s = (max-min)/(q_max + 1e-9)
        shift = tensor - min
        q = torch.round(shift/(s + 1e-9))
        q_clip = torch.clamp(q, 0, q_max)
        de_quantize = (q_clip*s) + min
        return de_quantize

    def SDS(Z, Z_hat, k):
        Z_c = Z - Z.mean(dim=0, keepdim=True)
        U, S, Vh = torch.linalg.svd(Z_c, full_matrices=False)
        Uk = Vh[:k].T   # [D, k]

        E = Z - Z_hat

        num = torch.norm(E @ Uk)**2
        den = torch.norm(Z @ Uk)**2

        return (num/den).item()

    def CKA(X, Y):
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)
        
        dot_product_similarity = torch.norm(X.t() @ Y)**2
        
        normalization_x = torch.norm(X.t() @ X)
        normalization_y = torch.norm(Y.t() @ Y)
        
        cka_score = dot_product_similarity / (normalization_x * normalization_y + 1e-9)
        return cka_score.item()

    print("\nCalibrating Min/Max ranges on held-out data...")

    calib_batches = list(get_eval_batches(skip_dataset, num_batches=1))
    global_min, global_max = float('inf'), float('-inf')
    feature_min, feature_max = None, None


    with torch.no_grad():
        for batch in calib_batches:

            seq_len = batch.size(1)
            pos = torch.arange(seq_len, device=device).unsqueeze(0)
            
            # Start with Embeddings
            
            _, cache = model.run_with_cache(batch)
            acts = cache["blocks.2.hook_resid_pre"]
            sparse_feats = sae_model.encode(acts) 
            batch_min = sparse_feats.amin(dim=(0,1))   # shape [D]
            batch_max = sparse_feats.amax(dim=(0,1))

            global_min = min(global_min, batch_min.min().item())
            global_max = max(global_max, batch_max.max().item())

            if feature_min is None:
                feature_min = batch_min
                feature_max = batch_max
            else:
                feature_min = torch.minimum(feature_min, batch_min)
                feature_max = torch.maximum(feature_max, batch_max)

    print(f"Calibration Complete | Min: {global_min:.4f}, Max: {global_max:.4f}")
    hook_cache = {}

    def get_intervention_hook(bits, mode="per_tensor"):
        def hook(activations, hook):
            orig_acts = activations
            sparse_feats = sae_model.encode(orig_acts)

            if bits is not None:
                if mode == "per_tensor":
                    damaged_feats = quantize_gen(global_min, global_max, bits, sparse_feats)
                elif mode == "per_feature":
                    damaged_feats = quantize_gen(
                        feature_min.unsqueeze(0).unsqueeze(0),
                        feature_max.unsqueeze(0).unsqueeze(0),
                        bits,
                        sparse_feats
                    )
                reconstructed = sae_model.decode(damaged_feats)
            else:
                damaged_feats = sparse_feats
                reconstructed = orig_acts

            hook_cache['clean_acts'] = orig_acts.detach()
            hook_cache['recon_acts'] = reconstructed.detach()
            hook_cache['Z'] = sparse_feats.detach()
            hook_cache['Z_hat'] = damaged_feats.detach()

            return reconstructed
        return hook
    bit_configs = [None, 8, 6, 4, 2]
    eval_batches = list(get_eval_batches(skip_dataset, num_batches=5))
    modes = ["per_tensor", "per_feature"]

    umap_data = {
        "Baseline": [],
        'per_tensor': {
        "8-bit_per_tensor": [],
        "6-bit_per_tensor": [],
        "4-bit_per_tensor": [],
        "2-bit_per_tensor": []
        },
        'per_feature': {
        "8-bit_per_feature": [],
        "6-bit_per_feature": [],
        "4-bit_per_feature": [],
        "2-bit_per_feature": []
        }
    }

    def normalize(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)

    print("\nEvaluating: Clean model (no SAE intervention)")
    total_loss_clean = 0

    def identity_hook(activations, hook):
        return activations  

    with torch.no_grad():
        for batch in tqdm(eval_batches, leave=False):
            logits = model.run_with_hooks(
                batch,
                fwd_hooks=[("blocks.2.hook_resid_pre", identity_hook)]
            )
            loss = model.loss_fn(logits, batch)
            total_loss_clean += loss.item()

    clean_ppl = math.exp(total_loss_clean / len(eval_batches))
    print(f"Clean model PPL (no SAE): {clean_ppl:.2f}")

    wandb.init(
        project="sae-quantization",
        name=f"clean_model_no_sae_{m_bottleneck}",
        config={
            "model": "distilgpt2",
            "layer": 3,
            "sae_bottleneck": m_bottleneck, 
            "k_percent": 0.10,
            "seq_len": 128,
            "model_batch": 32,
            "mode": "no_sae",                
            "bits": None,                     
        },
        reinit=True
    )
    wandb.log({"clean_model_ppl": clean_ppl})

    for mode in modes:
        print(f"\nMode: {mode}")

        for bits in bit_configs:
            label = f"{bits}-bit" if bits else "Baseline"
            print(f"\nEvaluating: {label}")
            hook_fn = get_intervention_hook(bits, mode)     
            wandb.init(
                project="sae-quantization",
                name = f"{mode}_{bits}_32_batch_{m_bottleneck}",
                config={
                    "model": "distilgpt2",
                    "layer": 3,
                    "sae_bottleneck": f"{m_bottleneck}",
                    "k_percent": 0.10,
                    "seq_len": 128,
                    "model_batch": 32,
                    "mode": f"{mode}",
                    "bits": f"{bits}",
                },
                reinit=True
            )
            total_loss, total_mse, total_sds = 0, 0, 0
            total_sds_128, total_sds_64, total_cka = 0, 0, 0
            Z_all = []
            Z_hat_all = []
            with torch.no_grad():
                idx = 0
                for batch in tqdm(eval_batches, leave=False):
                    logits = model.run_with_hooks(
                        batch,
                        fwd_hooks=[("blocks.2.hook_resid_pre", hook_fn)]
                    )

                    loss = model.loss_fn(logits, batch)
                    total_loss += loss.item()

                    clean_acts = hook_cache['clean_acts']
                    recon_acts = hook_cache['recon_acts']
                    Z_batch = hook_cache['Z']
                    Z_hat_batch = hook_cache['Z_hat']

                    Z = Z_batch.view(-1, Z_batch.size(-1))
                    Z_hat = Z_hat_batch.view(-1, Z_batch.size(-1))
                    Z_all.append(Z)
                    Z_hat_all.append(Z_hat)
                    # 3. Compute Metrics
                    if bits is not None:
                        total_mse += nn.MSELoss()(recon_acts, clean_acts).item()
                        total_sds += SDS(Z=Z, Z_hat=Z_hat, k=32)
                        total_sds_64 += SDS(Z=Z, Z_hat=Z_hat, k=64)
                        total_sds_128 += SDS(Z=Z, Z_hat=Z_hat, k=128)
                        total_cka += CKA(X=recon_acts.view(-1, clean_acts.size(-1)), Y = clean_acts.view(-1, clean_acts.size(-1)))

                    # 4. Save UMAP Data
                    Z_hat_np = Z_hat.detach().cpu().numpy()
                    
                    if bits is None:
                        if mode == "per_tensor": 
                            umap_data["Baseline"].append(Z_hat_np)
                    else: 
                        umap_data[f"{mode}"][f"{bits}-bit_{mode}"].append(Z_hat_np)

                    if idx % 500 == 0:
                        wandb.log({
                            "perplexity": math.exp(total_loss/(idx+1)),
                            "mse": total_mse / (idx+1) if bits is not None else 0,
                            "cka": total_cka / (idx+1) if bits is not None else 0,
                            "sds": total_sds / (idx+1) if bits is not None else 0,
                            "sds_64": total_sds_64 / (idx+1) if bits is not None else 0,
                            "sds_128": total_sds_128 / (idx+1) if bits is not None else 0,
                        })
                    idx += 1
            Z_full = torch.cat(Z_all)
            Z_hat_full = torch.cat(Z_hat_all)

            global_sds = SDS(Z_full, Z_hat_full, k=32)
            global_sds_64 = SDS(Z_full, Z_hat_full, k=64)
            global_sds_128 = SDS(Z_full, Z_hat_full, k=128)
            wandb.log({
                "global_sds_64": global_sds_64,
                "global_sds_128": global_sds_128,
                "global_sds": global_sds
            })
            avg_loss = total_loss / len(eval_batches)
            print(f"Perplexity: {math.exp(avg_loss):.2f}")

            if bits is not None:    
                print(f"MSE: {total_mse / len(eval_batches):.4f}")

            wandb.finish()
    
    print("Running UMAP...")
    for mode in modes:
        reducer = umap.UMAP(metric='euclidean', min_dist=0.1, random_state=67)
        
        baseline = np.concatenate(umap_data["Baseline"], axis=0)
        baseline = baseline.reshape(-1, baseline.shape[-1])[:10000]

        reducer.fit(baseline)
        baseline_2d = reducer.transform(baseline)

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        fig.suptitle(f"UMAP Grid Collapse ({mode})", fontsize=16)

        # These are the titles for our subplots
        ordered_labels = ["Baseline", "8-bit", "6-bit", "4-bit", "2-bit"]
        
        for i, label in enumerate(ordered_labels):
            # 1. Fetch the correct data list based on the label
            if label == "Baseline":
                data_list = umap_data["Baseline"]
            else:
                dict_key = f"{label}_{mode}" 
                if dict_key not in umap_data[mode]: 
                    continue # 
                data_list = umap_data[mode][dict_key]
                
            # 2. Process the data
            data = np.concatenate(data_list, axis=0)
            data = data.reshape(-1, data.shape[-1])[:10000]

            # 3. Transform and Plot
            proj = baseline_2d if label == "Baseline" else reducer.transform(data)

            axes[i].scatter(proj[:, 0], proj[:, 1], s=1, alpha=0.5)
            axes[i].set_title(label)
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(f"umap_fixed_{mode}_{m_bottleneck}.png", dpi=300)
        print(f"Saved: umap_fixed_{mode}.png")