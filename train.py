import torch
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
import tiktoken
import matplotlib.pyplot as plt
import math
import time
import random
from gpt2 import GPT, GPTConfig

# -------- Setup --------

# Training parameters

TRANING_STEPS = 500                   # Steps to train
GRADIENT_ACCUMULATION_STEPS = 24    # Gradient accumulation steps (micro-steps). Total number of forward-backward passes is TRAINING_STEPS * GRADIENT_ACCUMULATION_STEPS.

OPT_LEARNING_RATE = 6e-4            # Optimizer learning rate
OPT_BETAS = (0.9, 0.95)             # Optimizer betas
OPT_EPSILON = 1e-8                  # Optimizer epsilon
OPT_WEIGHT_DECAY = 0.1              # Optimizer weight decay

BATCH_COUNT = 12                    # Training batches
INPUT_PATH = "data/input.txt"       # Dataset path

MAX_LEARNING_RATE = OPT_LEARNING_RATE           # Maximum learning rate for learning rate scheduler
MIN_LEARNING_RATE = OPT_LEARNING_RATE * 0.1     # Minimum learning rate for learning rate scheduler
WARMUP_STEPS = TRANING_STEPS * 0.015            # Warm-up steps for learning rate 

COMPILE_MODEL = True           # Model compilation (torch.compile) only works on Linux.

# Use CUDA if available, otherwise fallback to
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

torch.set_float32_matmul_precision('high') # Setting matmul precision will allow optimization for tensor cores, especially for newer Nvidia GPUs.

# -------- Training --------

class DataLoader:
    """
    Text data loader with batching for GPT-2 training. Uses GPT-2 tokenizer.
    """
    def __init__(self, B, T, file, random_start = True):
        self.B = B  # Batch count  
        self.T = T  # Batch size

        self.random_start = random_start
        self.batches_accumulated = 0

        # Read the input file
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()

        text = text.replace("\n", " ") # Text cleanup
        tokenizer = tiktoken.get_encoding("gpt2") # Get GPT-2 text encoding
        tokens = tokenizer.encode(text) # Encode the input
        self.tokens = torch.tensor(tokens) # Convert tokens to tensor

        self.current_position = 0
        self.total_batches = len(self.tokens) // (B*T)
        print(f"DataLoader initialized with {len(self.tokens)} tokens. 1 Epoch = {self.total_batches} batches")

    def next_batch(self):
        B, T = self.B, self.T

        # Get current batch
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        # Move current position
        self.current_position += B*T
        if self.current_position + (B*T + 1) > len(self.tokens):
            self.current_position = 0

        # Increase accumulated baches
        self.batches_accumulated += 1
        if self.batches_accumulated >= len(self.tokens) // (B*T):
            self.reset_position()

        return x, y
    
    def reset_position(self):
        if self.random_start:
            self.current_position = random.randint(0, self.total_batches-1) * self.B * self.T
        else:
            self.current_position = 0

        self.batches_accumulated = 0


## Initialize model and settings

# Use default GPT-2 parameters
config = GPTConfig(
    block_size=1024,
    vocab_size=50257,
    n_layer=12,
    n_head=12,
    n_embd=768
)
model = GPT(config)

if COMPILE_MODEL:
    model = torch.compile(model) # Only works on Linux with Triton

model = model.to(device)
parameter_count = sum([p.numel() for p in model.parameters()])
print(f"Model initialized with {'{:,}'.format(parameter_count)} parameters")

scaler = GradScaler(device=device)
dataloader = DataLoader(BATCH_COUNT, config.block_size, INPUT_PATH, random_start=True)

# Initialize optimizer with partial weight decay
param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]       # Multi-dimensional tensors ge  t weight decay
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]      # No weight decay for 1D tensors (LayerNorm, Bias)

optim_groups = [
    {'params': decay_params, 'weight_decay': OPT_WEIGHT_DECAY},
    {'params': nodecay_params, 'weight_decay': 0.0}
]

optimizer = torch.optim.AdamW(optim_groups, lr=OPT_LEARNING_RATE, betas=OPT_BETAS, eps=OPT_EPSILON, fused=True)

## Learning rate scheduler
def get_lr(step):
    if step < WARMUP_STEPS: # Warmup
        return MAX_LEARNING_RATE * (step + 1) / WARMUP_STEPS
    if step > TRANING_STEPS: # After max steps, to prevent errors
        return MAX_LEARNING_RATE

    # In between, use cosine decay
    decay_ratio = (step - WARMUP_STEPS) / (TRANING_STEPS - WARMUP_STEPS)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LEARNING_RATE + coeff * (MAX_LEARNING_RATE - MIN_LEARNING_RATE)

## Training loop
print()
print("-------- Training --------")
print(f"Training for {TRANING_STEPS} steps:")
t0 = time.time()
t_prev = time.time()
step_prev = -1
losses = []

for step in range(TRANING_STEPS):
    # Zero gradients
    optimizer.zero_grad()

    loss_accum = 0.0

    # Accumulate gradients over multiple micro-batches
    for micro_step in range(GRADIENT_ACCUMULATION_STEPS):
        # Load data
        x, y = dataloader.next_batch()
        x, y = x.to(device), y.to(device)

        # Forward pass (with loss calculation)
        # Use autocast to use faster half-precision training with minimal impact on quality.
        with torch.autocast(device_type=device, dtype=torch.float16):
            logits, loss = model(x, y)

        # Scale down the loss by accumulation steps
        loss /= GRADIENT_ACCUMULATION_STEPS
        loss_accum += loss.detach()

        # Backpropagation
        scaler.scale(loss).backward()

    # Clip gradients
    scaler.unscale_(optimizer) # Unscale before gradient clipping
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Grad clipping helps reduce effects of big losses

    # Step optimizer with gradient scaling
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    scaler.step(optimizer)
    scaler.update()

    losses.append(loss_accum.item())

    if step % (TRANING_STEPS // 50) == 0 or step == TRANING_STEPS - 1:
        dt = time.time() - t_prev
        t_prev = time.time()
        print(f"Step: {step} ({(step + 1) * GRADIENT_ACCUMULATION_STEPS}) | Loss: {loss_accum.item():.4f} | Lr: {lr:.4e} | Norm: {norm:.4f} | Time: {dt:.4f} sec | {(dataloader.B * dataloader.T * (step - step_prev) * GRADIENT_ACCUMULATION_STEPS / dt):.2f} tok/sec")
        step_prev = step

dt = (time.time() - t0) # In seconds
print(f"Trained for {TRANING_STEPS} steps (total of {TRANING_STEPS * GRADIENT_ACCUMULATION_STEPS} microsteps) in {dt:.2f} seconds.")

## Save the model state dict
model_save_path = "model.pth"
if COMPILE_MODEL:
    torch.save(model._orig_mod.state_dict(), model_save_path)
else:
    torch.save(model.state_dict(), model_save_path)
print(f"Model parameters are saved to {model_save_path}")

print()
print("-------- Statistics --------")
total_tokens = dataloader.B * dataloader.T * TRANING_STEPS * GRADIENT_ACCUMULATION_STEPS
print(f"Total tokens: {total_tokens}")
print(f"Average tokens per second: {total_tokens/dt:.2f}")
print(f"Final loss: {losses[len(losses) - 1]:.4f}")

# Plotting losses
print()
print("-------- Plotting --------")

# Plot the graph
plt.figure(figsize=(10, 6))
plt.plot(losses, label="Training Loss", color="blue", linewidth=1.5)
plt.title("GPT-2 Training Loss over Steps")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()

# Save the figure as a PNG file
plot_path = "loss_plot.png"
plt.tight_layout() # Ensures labels don't get cut off
plt.savefig(plot_path, dpi=300) # dpi=300 ensures high resolution
plt.close()

print(f"Training loss graph saved to {plot_path}")