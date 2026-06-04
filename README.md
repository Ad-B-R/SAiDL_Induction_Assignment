# SAiDL Induction Assignment

- This report attempts to tackle the Core-ML and Mechanistic Interpretability problem statements from the SAiDL Induction 2026 Assignment.

- All evaluation was performed on an NVIDIA GeForce RTX 3050 6GB Laptop GPU. Therefore, all results pertaining to GPU analysis, throughput, and performance are specific to the aforementioned device.

- GitHub Repository: https://github.com/Ad-B-R/SAiDL_Induction_Assignment

- [Project Report (PDF)](https://github.com/Ad-B-R/SAiDL_Induction_Assignment/blob/main/main.pdf)
- [Assignment 1- Core ML](https://github.com/Ad-B-R/SAiDL_Induction_Assignment/tree/main/Assignment_1(Core_ML))

- [Assignment 2- Mechanistic Interpretability](https://github.com/Ad-B-R/SAiDL_Induction_Assignment/tree/main/Assignment_2(Mechintrap))

# Assignment 1: Core ML
- Core-ML Assignment tackles the empirical study of different attention variants and their performance. 
- For this assignment, Sliding Window, Grouped Query Attention and SoftMax-Free Attention were used alongside the baseline Attention model 
- For Positional Embeddings- Standard sine-cosine absolute encoding, ALiBi, RoPE and RPE positional encodings and their impact on downstream perplexity, throughput etc was measured on the baseline attention model
- For subsection 4- Best performing model (based on perplexity) was incorporated alongside a conformer model, and the impact of a conformer+interleaved/subset convolution model on downstream perplexity and throughput was measured
- A bonus assignment extended the analysis by introducing 4-AFT variants measured against the best performing models from the above experiments 

## Overview

This assignment implements and empirically evaluates a long-context sequence 
model on the WikiText-2 dataset. Starting from a standard Transformer baseline, we 
perform experiments and study the impact on different metrics (such as perplexity, peak gpu memory usage etc) based on different attention mechanisms, positional encoding, conformers and AFT (Bonus)

---

## Dataset and Base Model

- **Dataset:** WikiText-2
- **Task:** Autoregressive language modeling (next-token prediction)
- **Base Model:** Standard Transformer (Attention Is All You Need)
- **Vocabulary Size:** 50257
- **Embedding Dimension (`d_model`):** 256
- **Attention Heads (`h`):** 8
- **Feedforward Dimension (`d_ff`):** 1024
- **Number of Transformer Layers:** 4
- **Dropout:** 0.1

---

## Repository Structure

```text
Assignment_1(Core_ML)/
├── attention/ # Attention Mechanism
│   ├── baseline.py
│   ├── GQA.py
│   ├── Sliding_Window.py
│   └── Softmax.py
│
├── Bonus/ # AFT Mechanism 
│   ├── AFT_conv.py
│   ├── AFT_full.py
│   ├── AFT_local.py
│   ├── AFT_simple.py
│   ├── config.yaml
│   ├── data.py
│   └── run.py # Runs all AFT Mechanisms
│
├── convolution-Attention/ # Conformer Mechanism
│   ├── conformer.py # Conformer
│   ├── conv.py # Convolution layer
│   ├── data_conv.py 
│   ├── model_conv.py # Runs both the Conformer Mechanism
│   └── config.yaml # Config
│
├── positional/ # Positional 
│   ├── alibi.py
│   ├── rope.py
│   ├── tpe.py
│   └── standardpos.py
│
├── conf/ # config files
│   ├── attention/
│   │   ├── GQA.yaml
│   │   ├── Sliding.yaml
│   │   ├── SoftmaxFree.yaml
│   │   └── Standard.yaml
│   ├── positional/
│   │   ├── RoPE.yaml
│   │   ├── RPE.yaml
│   │   └── Standard.yaml
│   └── config.yaml
│
├── data.py # Script to import dataset
├── model_alibi.py # Run ALiBi
├── model_rope.py # Run RoPE
├── model_rpe.py # Run RPE
├── model_scale_rope.py # Scaled RoPE
├── model_std.py # Run Baseline Positional Encoding 
```

# Assignment 2: Mechanistic Interpretability

- This assignment tackles the empirical study of interaction between sparse autoencoders and quantisation, using a frozen distilgpt2 as the base language model with hidden states extracted from layer 3 of 6 on OpenWebText, using SAE-Lens for SAE autoencoder
- For quantisation, per-tensor and per-feature schemes were used across bit-widths of 8, 4 and 2 bits
- Representation damage was identified through spectral analysis (singular values, principle angles, SDS, CKA, energy loss and collapse ratio)
- For Subsection 4, a subspace preserving quantisation scheme was implemented based on the mechanistic findings (specifically Jacobian Norms and Fischer Norms), decomposing active activations into high bit quantisation and a residual quantised at lower precision, then compared against standard quantisation
- A bonus joint quantisation experiment extended the analysis by quantising both the frozen LM weights and SAE bottleneck activations simultaneously, measuring the effect on representation damage and downstream perplexity

## Overview

This assignment empirically studies the interaction between sparse autoencoders (SAEs) 
and quantization applied to the internal representations of a frozen pretrained language 
model. Using distilgpt2 as the base model and OpenWebText as the dataset, we train SAEs 
on layer 3 residual stream activations and systematically study how different quantization 
schemes damage the learned sparse representations.

---

## Dataset and Model

- **Base Model:** distilgpt2 (6-layer GPT-2 distilled, hidden dim 768), frozen throughout
- **Dataset:** OpenWebText (~1% subset, 80-100M tokens for training, held-out split for eval)
- **Hook Point:** `blocks.2.hook_resid_pre` (layer 3 residual stream, pre-attention)
- **Sequence Length:** 128 tokens

---

## Repository Structure

```text
Assignment_2(Mechintrap)/
├── data.py # Train SAE
├── vae.py # Train VAE
├── inference.py # Representation Damage Experiment
├── rank.py # Ranking Neurons Experiment
├── joint_quant.py # Joint Quantisation Experiment
├── robust_quant.py # Robust Quantisation incorporating Subspace Quantisation
├── robust_vae.py # VAE Representation Damage
└── Neurons_ranked/ # Top Neurons Ranked
    ├── ...
```