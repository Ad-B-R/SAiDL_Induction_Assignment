from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import os
import math
import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
model.eval()


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

m_bottleneck = 512 
sae_model = TopKSAE(hidden_dim=m_bottleneck).to(device)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
weights_folder = os.path.join(current_script_dir, "weights")
model_file_name = f"sae_m{m_bottleneck}_final_100k.pt"
final_weights_path = os.path.join(weights_folder, model_file_name)

sae_model.load_state_dict(torch.load(final_weights_path, map_location=device))
sae_model.eval()


dataset = load_dataset("openwebtext", split="train", streaming=True)
skip_dataset = dataset.skip(2_000_000) # skip 2 million data so that there is no collision between training and testing data

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
    q_max = 2**bit - 1
    s = (max-min)/(q_max + 1e-9)
    shift = tensor - min
    q = torch.round(shift/(s + 1e-9))
    q_clip = torch.clamp(q, 0, q_max)
    de_quantize = (q_clip*s) + min
    return de_quantize

print("\nCalibrating Min/Max ranges on held-out data...")
calib_batches = list(get_eval_batches(skip_dataset, num_batches=10))
global_min, global_max = float('inf'), float('-inf')
feature_min, feature_max = None, None
with torch.no_grad():
    for batch in calib_batches:

        acts = model.transformer.h[2](model.transformer.h[1](model.transformer.wte(batch))[0])[0]
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

def get_intervention_hook(bits, mode = "per_tensor"):
    def hook(module, input, output):
        orig_acts = output[0]
        _, sparse_feats = sae_model(orig_acts)
        
        if bits is not None:
            if mode=="per_tensor":    
                damaged_feats = quantize_gen(tensor=sparse_feats, bit=bits, min=global_min, max=global_max)
            elif mode == "per_feature":
                damaged_feats = quantize_gen(
                    sparse_feats, bits,
                    feature_min.unsqueeze(0).unsqueeze(0),
                    feature_max.unsqueeze(0).unsqueeze(0)
                )
            reconstructed = sae_model.decoder(damaged_feats)
        else:
            reconstructed = orig_acts  # baseline
            
        return (reconstructed,) + output[1:]
    return hook

bit_configs = [None, 8, 4, 2] # None = Baseline, 8/4/2 = Quantization Damage
eval_batches = list(get_eval_batches(skip_dataset, num_batches=50)) # Evaluate on ~200k tokens
modes = ["per_tensor", "per_feature"]

for mode in modes:
    print(f"\nMode: {mode}")

    for bits in bit_configs:
        label = f"{bits}-bit" if bits else "Baseline"
        print(f"\nEvaluating: {label}")

        handle = (
            model.transformer.h[2].register_forward_hook(
                get_intervention_hook(bits, mode)
            )
            if bits is not None else None
        )                
        wandb.init(
            project="sae-quantization",
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
        total_loss, total_mse = 0, 0

        with torch.no_grad():
            idx = 0
            for batch in tqdm(eval_batches, leave=False):
                outputs = model(batch, labels=batch)
                total_loss += outputs.loss.item()
                if bits is not None:
                    clean_acts = model.transformer.h[2](
                        model.transformer.h[1](model.transformer.wte(batch))[0])[0]

                    _, sparse_feats = sae_model(clean_acts)

                    if bits > 0:
                        if mode == "per_tensor":
                            sparse_feats = quantize_gen(
                                sparse_feats, bits, global_min, global_max
                            )
                        else:
                            sparse_feats = quantize_gen(
                                sparse_feats, bits,
                                feature_min.unsqueeze(0).unsqueeze(0),
                                feature_max.unsqueeze(0).unsqueeze(0)
                            )

                    recon_acts = sae_model.decoder(sparse_feats)
                    total_mse += nn.MSELoss()(recon_acts, clean_acts).item()
                if idx%50==0:
                    wandb.log({
                    "perplexity": math.exp(total_loss/(idx+1)),
                    "mse": total_mse / (idx+1) if bits is not None else 0
                    })
                idx+=1

        avg_loss = total_loss / len(eval_batches)
        print(f"Perplexity: {math.exp(avg_loss):.2f}")

        if bits is not None:
            print(f"MSE: {total_mse / len(eval_batches):.4f}")

        if handle:
            handle.remove()