import huggingface_hub
from transformer_lens import HookedTransformer
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
import torch
from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    TopKTrainingSAEConfig,
    LoggingConfig,              
)

cfg_512 = LanguageModelSAERunnerConfig(
    model_name="distilgpt2",
    hook_name="blocks.2.hook_resid_pre",
    dataset_path="Skylion007/openwebtext",
    is_dataset_tokenized=False,
    streaming=True,
    context_size=128,

    training_tokens=int(100e6),
    train_batch_size_tokens=4096,
    lr=1e-4,
    lr_scheduler_name="constant",
    adam_beta1=0.9,
    adam_beta2=0.999,
    n_batches_in_buffer=32,
    store_batch_size_prompts=16,

    logger=LoggingConfig(
        log_to_wandb=True,
        wandb_project="sae-quantization-final",
        wandb_log_frequency=50,
        eval_every_n_wandb_logs=10,
        wandb_run_id="m512_run1",                
    ),
    n_checkpoints=10,
    checkpoint_path="./sae_checkpoints_saelens/m512",
    device="cuda",
    seed=67,

    sae=TopKTrainingSAEConfig(
        d_in=768,
        d_sae=512,
        k=51,                                              
        normalize_activations="expected_average_only_in",   
    )
)

print("=" * 60)
print("Training m=512 SAE")
print("=" * 60)
runner_512 = LanguageModelSAETrainingRunner(cfg_512)
sae_512 = runner_512.run()

cfg_1024 = LanguageModelSAERunnerConfig(
    model_name="distilgpt2",
    hook_name="blocks.2.hook_resid_pre",
    dataset_path="Skylion007/openwebtext",
    is_dataset_tokenized=False,
    streaming=True,
    context_size=128,

    training_tokens=int(100e6),
    train_batch_size_tokens=4096,
    lr=1e-4,
    lr_scheduler_name="constant",
    adam_beta1=0.9,
    adam_beta2=0.999,
    n_batches_in_buffer=32,
    store_batch_size_prompts=16,

    logger=LoggingConfig(
        log_to_wandb=True,
        wandb_project="sae-quantization-final",
        wandb_log_frequency=50,
        eval_every_n_wandb_logs=10,
    ),
    n_checkpoints=10,
    checkpoint_path="./sae_checkpoints_saelens/m1024",
    device="cuda",
    seed=67,

    sae=TopKTrainingSAEConfig(
        d_in=768,
        d_sae=1024,
        k=102, 
        normalize_activations="expected_average_only_in",
    )
)

print("=" * 60)
print("Training m=1024 SAE")
print("=" * 60)
runner_1024 = LanguageModelSAETrainingRunner(cfg_1024)
sae_1024 = runner_1024.run()

print("Training complete. Both SAEs saved.")