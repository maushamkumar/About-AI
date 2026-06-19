# Part1 - Core Transformer Architecture 
- 1.1 Positional Embeddings (absolute learned vs. sinusoidal)
- 1.2 Self-attention from first principle (manual computation with tiny example)
- 1.3 Building a single attention head in PyTorch
- 1.4 Multi-head attention (splitting, Concatenation, Projections)
- 1.5 Feed-forward network (MLP layers) - GELU, dimensionality expansion
- 1.6 Residual connections & LayerNorm
- 1.7 Stacking into a full Transformer block


# Part 2 - Train a Tiny LLM 
- 2.1 Byte-level tokenization
- 2.2 Dataset batching & shifiting for next token prediction
- 2.3 Cross entropy loss & label shifting
- 2.4 Training loop from scratch 
- 2.5 Sampling: temperature, top-k, top-p
- 2.6 Evaluating loss on val set 

# Part 3 - Modernizing the Architecture 
- 3.1 RMSNorm (replace LayerNorm, compare gradienta & convergence)
- 3.2 RoPE(Rotary Positional Embedding)
- 3.3 SwiGLU activations in MLP 
- 3.4 KV cache for faster interface
- 3.5 Sliding-window attention & attention sink 
- 3.6 Rolling buffer KV cache for streaming

# Part 4 - Scaling Up 
- 4.1 Switching from byte-level to BPE tokenization 
- 4.2 Gradient accumulation & mixed precision 
- 4.3 Learning rate schedules & warmup 
- 4.4 checkpointing & resuming
- 4.5 Logging & Visualization (TensorBoar / wandb)

# Part 5 - Mixture-of-experts (MoE) 
- 5.1 MoE theory: expert routing, gating networks, and load balancing
- 5.2 Implementing MoE layers in PyTorch 
- 5.3 Training stability and communication overhead in distribution setup. 
- 5.4 Combining MoE with dense layers for hybrid architectures. 

# Part 6 - Supervised Fine Tuning (SFT)
- 6.1 Instruction dataset formatting (prompt + response)
- 6.2 Causal LM loss with masked labels 
- 6.3 Curriculum learning for instruction data 
- 6.4 Evaluating outputs against gold response 

# Part 7 - Reward Modeling 
- 7.1 Preference datasets (Pairwise rankings)
- 7.2 Reward model architecture (shared transformer encoder)
- 7.3 Loss functions: Bradiey-Terry, Margin ranking loss 
- 7.4 Sanity check for reward sharing

# Part 8 - RLHF with PPO 
- 8.1 Policy newtwork: Our base LM (from SFT) with a value head for forward prediction 
- 8.2 Reward signal: Provie by the reward model trained in part 7. 
- 8.3 PPO objective: Balance between maximizing reward and staying close to SFT policy (KL panalty) 
- 8.4 Training loop: sample prompt -> generate completions -> socre with reward model -> optimize policy via PPO 
- 8.5 Logging & stability tricks: reward normalization, KL controlled rollout length, gradient clipping. 