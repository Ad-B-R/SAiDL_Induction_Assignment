import huggingface_hub
from transformer_lens import HookedTransformer
from datasets import load_dataset
import torch
import torch.nn as nn
import random
from tqdm.auto import tqdm
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import wandb
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"


model = HookedTransformer.from_pretrained("distilgpt2")
model = model.to(device)
model.eval()

tokenizer = model.tokenizer
k = 0
m = [512,1024]
for m_bottleneck in m:
    class TopKSAE(nn.Module):
        def __init__(self, input_dim=768, hidden_dim=512, k_percent=0.10):
            super().__init__()
            self.encoder = nn.Linear(input_dim, hidden_dim)
            self.decoder = nn.Linear(hidden_dim, input_dim)
            self.k = int(hidden_dim * k_percent)

        def forward(self, x):
            encoded = self.encoder(x)
            topk_values, topk_indices = torch.topk(encoded, self.k, dim=-1)
            sparse_encoded = torch.zeros_like(encoded)
            sparse_encoded.scatter_(-1, topk_indices, topk_values)
            reconstructed = self.decoder(sparse_encoded)
            return reconstructed, sparse_encoded

    sae_model = TopKSAE(hidden_dim=m_bottleneck).to(device)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_folder = os.path.join(current_script_dir, "weights")
    model_file_name = f"sae_m{m_bottleneck}_final_100k.pt"
    final_weights_path = os.path.join(weights_folder, model_file_name)

    sae_model.load_state_dict(torch.load(final_weights_path, map_location=device))
    sae_model.eval()

    dataset = load_dataset("openwebtext", split="train", streaming=True)
    skip_dataset = dataset.skip(600_000) # skip 400k data so that there is no collision between training and testing data

    hook_point = "blocks.2.hook_resid_pre"

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
                        
    def compute_jacobian_norms(sae_model, acts):
        acts = acts.clone().detach().requires_grad_(True)

        B, T, D = acts.shape
        flat = acts.view(-1, D)

        _, Z = sae_model(flat)  # [N, F]

        grad_outputs = torch.ones_like(Z)

        grads = torch.autograd.grad(
            outputs=Z,
            inputs=acts,
            grad_outputs=grad_outputs,
            retain_graph=False
        )[0]

        # map to feature importance (approx)
        importance = Z.abs().mean(dim=0)

        return importance

    def compute_fisher(Z):
        return (Z ** 2).mean(dim=0)

    def quantize_gen(min, max, bit, tensor):
        q_max = int(2**bit - 1)
        s = (max-min)/(q_max + 1e-9)
        shift = tensor - min
        q = torch.round(shift/(s + 1e-9))
        q_clip = torch.clamp(q, 0, q_max)
        de_quantize = (q_clip*s) + min
        return de_quantize

    def compute_subspace(Z, k=64):
        Zc = Z - Z.mean(dim=0, keepdim=True)
        U, S, Vh = torch.linalg.svd(Zc, full_matrices=False)

        Vk = Vh[:k].T   # [D, k]
        mean = Z.mean(dim=0, keepdim=True)

        return Vk, mean

    def decompose_subspace(Z, Vk, mean):
        Zc = Z - mean

        Z_proj = (Zc @ Vk) @ Vk.T
        Z_res = Zc - Z_proj

        return Z_proj, Z_res
    
    def get_range(x, p=0.999):
        high = torch.quantile(x, p)
        low = torch.quantile(x, 1 - p)
        return low, high
 
    def subspace_quantize(Z, Vk, mean, bits_high=8, bits_low=2,
                          min_val=None, max_val=None):

        sparsity_mask = (Z != 0.0).float()

        # 2. Decompose
        Z_proj, Z_res = decompose_subspace(Z, Vk, mean)

        # 4. Quantize Residual
        proj_min, proj_max = get_range(Z_proj)
        res_min, res_max = get_range(Z_res)
        if bits_high == 16:
            Z_proj_q = Z_proj
        else:
            Z_proj_q = quantize_gen(proj_min, proj_max, bits_high, Z_proj)


        if bits_low==16:
            Z_res_q = Z_res
        else:
            Z_res_q = quantize_gen(res_min, res_max, bits_low, Z_res)

        # 5. Recombine
        Z_hat = Z_proj_q + Z_res_q + mean

        Z_hat = Z_hat * sparsity_mask

        return Z_hat

    def compute_cka(X, Y):
        X = X - X.mean(0, keepdim=True)
        Y = Y - Y.mean(0, keepdim=True)

        K = X @ X.T
        L = Y @ Y.T

        hsic = (K * L).sum()
        norm_x = torch.norm(K)
        norm_y = torch.norm(L)

        return (hsic / (norm_x * norm_y + 1e-9)).item()

    def spectral_analysis(Z, Z_hat, k=32):

        Z_c = Z - Z.mean(dim=0, keepdim=True)
        Z_c_hat = Z_hat - Z_hat.mean(dim=0, keepdim=True)

        U, S, V = torch.svd_lowrank(Z_c, q=k)
        U_hat, S_hat, V_hat = torch.svd_lowrank(Z_c_hat, q=k)

        Uk = V[:, :k]
        Uk_hat = V_hat[:, :k]

        cos_thetas = torch.linalg.svdvals(Uk.T @ Uk_hat)
        angles = torch.acos(torch.clamp(cos_thetas, -1, 1)) * (180 / math.pi)

        E = Z - Z_hat
        sds = (torch.norm(E @ Uk)**2 / torch.norm(Z @ Uk)**2).item()

        return {
            "singular_before": S,
            "singular_after": S_hat,
            "angles": angles,
            "sds": sds
        }

    def compute_calibration_subspace(model, sae_model, calib_batches, device, k=64):
        Z_calib_all = []

        with torch.no_grad():
            for batch in calib_batches:

                seq_len = batch.size(1)
                pos = torch.arange(seq_len, device=device).unsqueeze(0)

                _, cache = model.run_with_cache(batch)
                acts = cache["blocks.2.hook_resid_pre"]
                _, Z = sae_model(acts)

                Z_calib_all.append(Z.view(-1, Z.size(-1)))

        Z_calib = torch.cat(Z_calib_all)

        Vk_global, mean_global = compute_subspace(Z_calib, k=k)

        return Vk_global, mean_global

    print("\nCalibrating Min/Max ranges on held-out data...")

    calib_batches = list(get_eval_batches(skip_dataset, num_batches=1))
    global_min, global_max = float('inf'), float('-inf')
    feature_min, feature_max = None, None

    eval_batches = list(get_eval_batches(skip_dataset, num_batches=5))

    Vk_global, mean_global = compute_calibration_subspace(
        model,
        sae_model,
        calib_batches,
        device,
        k=64
    )

    with torch.no_grad():
        for batch in calib_batches:

            seq_len = batch.size(1)
            pos = torch.arange(seq_len, device=device).unsqueeze(0)
            
            # Start with Embeddings
            
            _, cache = model.run_with_cache(batch)
            acts = cache["blocks.2.hook_resid_pre"]
            _, sparse_feats = sae_model(acts) 
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
    def normalize(x):
        return (x - x.mean()) / (x.std() + 1e-8)    
  
    def compute_importance(k):
        all_Z = []
        acts_samples = []
        max_samples = 5  
        with torch.no_grad():
            for batch in eval_batches:
                _, cache = model.run_with_cache(batch)
                acts = cache["blocks.2.hook_resid_pre"]
                if len(acts_samples) < max_samples:
                    acts_samples.append(acts.detach())

                B, T, D = acts.shape
                flat = acts.view(-1, D)

                _, Z_flat = sae_model(flat)
                Z = Z_flat.view(B, T, -1)

                all_Z.append(Z)

        Z = torch.cat(all_Z, dim=0)

        Z_flat = Z.view(-1, Z.size(-1))
        jacobian_imp = None

        for acts in acts_samples:
            J = compute_jacobian_norms(sae_model, acts)

            if jacobian_imp is None:
                jacobian_imp = J
            else:
                jacobian_imp += J

        jacobian_imp /= len(acts_samples)
        fisher_imp   = compute_fisher(Z_flat)

        F = Z_flat.size(-1)
        k_val = int(F*k)
        importance = normalize(jacobian_imp) + normalize(fisher_imp)
        Z_weighted = Z_flat * importance
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  

        U, S, V = torch.svd(Z_weighted)
        Vk_global = V[:, :k_val]
    
        return Vk_global
    
    Vk_global = compute_importance(k=0.2)
    
    bit_configs = [16, 8, 4, 2, None]

    modes = ["subspace", "per_tensor"]
    umap_data = {
        "Baseline": [],
        'per_tensor': {
        "16-bit_per_tensor": [],
        "8-bit_per_tensor": [],
        "4-bit_per_tensor": [],
        "2-bit_per_tensor": []
        },
        'subspace': {
        "16-bit_subspace": [],
        "8-bit_subspace": [],
        "4-bit_subspace": [],
        "2-bit_subspace": []
        }
    }

    results = {}
    def get_quant_hook(mode, bits,
                    sae_model,
                    global_min, global_max,
                    Vk=None, mean=None):

        def hook_fn(acts, hook):
            # SAE forward
            _, Z = sae_model(acts)

            if bits is None:
                return acts  # baseline

            if mode == "per_tensor":
                Z_flat = Z.view(-1, Z.size(-1))

                Z_hat_flat = subspace_quantize(
                    Z_flat, Vk_global, mean,
                    bits_high=bits,
                    bits_low=bits,
                    min_val=global_min,
                    max_val=global_max
                )

                Z_hat = Z_hat_flat.view_as(Z)

            elif mode == "subspace":
                Z_flat = Z.view(-1, Z.size(-1))

                Z_hat_flat = subspace_quantize(
                    Z_flat, Vk_global, mean,
                    bits_high=16,
                    bits_low=bits,
                    min_val=global_min,
                    max_val=global_max
                )

                Z_hat = Z_hat_flat.view_as(Z)

            else:
                raise ValueError(mode)

            recon = sae_model.decoder(Z_hat)
            return recon

        return hook_fn

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

    for mode in modes:
        print(f"\nMode: {mode}")

        for bits in bit_configs:
            acts_samples = []
            max_samples = 10
            if bits is not None:
                label = f"{bits}-bit" if bits!=16 else "Baseline_16_bit"
            else:
                label = "Standard_Pass_baseline"

            wandb.init(
            project="sae-quantization-pt3",
            name=f"{mode}_{label}_analysis_{m_bottleneck}",
            reinit=True
            )
            print(f"\nEvaluating: {label}")
            Z_all = []
            Z_hat_all = []
            tokens_all = []

            handle = (None)                
            with torch.no_grad():
                idx = 0
                for batch in tqdm(eval_batches, leave=False):

                    _, cache = model.run_with_cache(batch)
                    clean_acts = cache["blocks.2.hook_resid_pre"]
                    B, T, D = clean_acts.shape
                    flat = clean_acts.view(-1, D)

                    _, sparse_feats_flat = sae_model(flat)
                    Z_batch = sparse_feats_flat.view(B, T, -1)
                    if len(acts_samples) < max_samples:
                        acts_samples.append(clean_acts[:1].detach())    
                    
                    if bits is not None:
                        if mode == "subspace":
                        # Flatten for SVD
                            Z_flat = Z_batch.view(-1, Z_batch.size(-1))
                        
                            Z_hat_flat = subspace_quantize(
                                Z_flat, Vk_global, mean_global,
                                bits_high=16,
                                bits_low=bits,
                                min_val=global_min,
                                max_val=global_max
                            )

                            Z_hat_batch = Z_hat_flat.view_as(Z_batch)
                        
                        elif mode == "per_tensor":
                            Z_flat = Z_batch.view(-1, Z_batch.size(-1))
                        
                            Z_hat_flat = subspace_quantize(
                                Z_flat, Vk_global, mean_global,
                                bits_high=bits,
                                bits_low=bits,
                                min_val=global_min,
                                max_val=global_max
                            )

                            Z_hat_batch = Z_hat_flat.view_as(Z_batch)                        
                    else:
                        Z_hat_batch = Z_batch    

                    Z = Z_batch.view(-1, Z_batch.size(-1))
                    Z_hat = Z_hat_batch.view(-1, Z_batch.size(-1))

                    Z_all.append(Z)
                    Z_hat_all.append(Z_hat)
                    tokens_all.append(batch.view(-1))

                    recon_acts = sae_model.decoder(Z_hat_batch)
                    
                    Z_hat_np = (Z_hat.detach().cpu().numpy())
                    if bits is None: umap_data["Baseline"].append(Z_hat_np)
                    else: umap_data[f"{mode}"][f"{bits}-bit_{mode}"].append(Z_hat_np)
            
            jacobian = None

            for acts in acts_samples:
                J = compute_jacobian_norms(sae_model, acts)
                
                if jacobian is None:
                    jacobian = J
                else:
                    jacobian += J

            jacobian = jacobian / len(acts_samples)

            Z = torch.cat(Z_all)
            Z_hat = torch.cat(Z_hat_all)

            tokens = torch.cat(tokens_all)
            
            hook_fn = get_quant_hook(
                mode=mode,
                bits=bits,
                sae_model=sae_model,
                global_min=global_min,
                global_max=global_max,
                Vk=Vk_global,
                mean=mean_global
            )

            ppl = compute_ppl_with_hook(model, eval_batches, hook_fn)

            if bits is not None: print(f"{mode} | {bits}-bit → PPL: {ppl:.2f}")
            else: print(f"{mode} | {bits}-bit → PPL: {ppl:.2f}")
            spectral = spectral_analysis(Z, Z_hat)

            fisher = compute_fisher(Z)

            sv_before = spectral["singular_before"]
            sv_after = spectral["singular_after"]

            # fraction of near-zero singular values
            collapse_ratio = (sv_after < 1e-3).float().mean().item()

            # how much variance is lost
            energy_before = (sv_before**2).sum()
            energy_after = (sv_after**2).sum()
            energy_loss = ((energy_before - energy_after) / (energy_before + 1e-9)).item()
            
            cka = compute_cka(Z, Z_hat)
            
            angles = spectral["angles"]
            mean_angle = angles.mean().item()
            max_angle = angles.max().item()

            def sparsity(x):
                return (x == 0).float().mean().item()

            s_before = sparsity(Z)
            s_after = sparsity(Z_hat)

            # how much sparsity changed
            sparsity_delta = s_after - s_before
                
            wandb.log({
                "bit_high": 16,
                "k_value": k,
                "Perplexity": ppl,
                "jacobian_mean": jacobian.mean().item(),
                "fisher_mean": fisher.mean().item(),
                "collapse_ratio": collapse_ratio,
                "energy_loss": energy_loss,
                "sparsity_before": s_before,
                "sparsity_after": s_after,
                "sparsity_delta": sparsity_delta,
                "angle_mean": mean_angle,
                "angle_max": max_angle,
                "CKA": cka,
                "SDS": spectral["sds"]
            })                

            print(f" Summary ({label})")
            print(f"Representation Damage:")
            print(f"SDS: {spectral['sds']:.4f}")
            print(f"Angle: {spectral['angles'].mean().item():.2f}°")
            
            print(f"\nCausal Alignment (Damage vs Perplexity Impact):")

    print("<------------------- ROBUST QUANTIZATION FINISHED --------------------------->")
    print("<------------------- SUCCESS --------------------------->")  