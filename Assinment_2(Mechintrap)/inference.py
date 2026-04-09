from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import os

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
held_out_dataset = dataset.skip(2_000_000) # skip 2 million data so that there is no collision between training and testing data

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

def quantize(min, max, bit, tensor):
    pass
