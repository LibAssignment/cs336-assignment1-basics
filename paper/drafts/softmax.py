# %%
import torch

# 1. Setup the initial tensor (requires gradient)
x = torch.tensor([1.0, 8.0, 3.0, 6.0])
x = torch.tensor([1.0, 80.0, 3.0, 60.0])

# We will use the same input for both tests
x_correct = x.clone().requires_grad_()
x_incorrect = x.clone().requires_grad_()

# ---
## Case 1: Correct (Non-Detached) Implementation
# ---

w = torch.randn(100)
def module(x: torch.Tensor, detach=False) -> torch.Tensor:
  x_max = x.max()
  if detach:
    x_max = x_max.detach()
  x1 = (x - x_max).exp()
  y = x1 / x1.sum()
  loss = (y * w[:len(x)]).mean()
  return loss


# Step 3: Compute a simple "loss" (e.g., the sum of y)
loss_correct = module(x_correct)

# Backward pass
loss_correct.mean().backward()

print("--- Correct (Non-Detached) ---")
print(f"Input x: {x}")
print(f"loss: {loss_correct}")
print(f"Calculated Gradient (x.grad): {x_correct.grad}")
# Expected: [-1.0, 2.0, -1.0]

# ---
## Case 2: Incorrect (Detached) Implementation
# ---

# Step 1: Calculate x_max and DETACH it
loss_incorrect = module(x_incorrect, detach=True)

# Backward pass
loss_incorrect.mean().backward()

print("\n--- Incorrect (Detached) ---")
print(f"Input x: {x}")
print(f"loss: {loss_incorrect}")
print(f"Calculated Gradient (x.grad): {x_incorrect.grad}")
# Expected: [1.0, 1.0, 1.0]

# %%
