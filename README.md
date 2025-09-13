# Adams Optimizer

Adams is a next-generation optimizer that blends the simplicity of element-wise methods with the stability benefits of matrix-aware regularization. It updates neural network parameters in both **1D (per-element)** and **2D (per-matrix)** ways, staying fast and easy to parallelize like Adam/AdamW while improving stability and generalization.

* **Stable:** no loss spikes observed; no gradient clipping required.
* **Fast & scalable:** element-wise updates + one rank-1 spectral decay step per matrix; easily parallelizable.
* **Simple:** no `epsilon` hyperparameter; truly scale-invariant per-parameter update.

## Definition 📝

![Adams pseudocode](./assets/adams_pseudocode.png)

## How Adams Works 🌟

### 1) Bounded, element-wise update (1D)

Small second-moment estimates are a major source of instability and loss spikes in Adam-like methods. Adams replaces the usual preconditioned step with a **bounded** update using `atan2`:

$$
\Delta \theta \propto \text{atan2}\big(\hat m_t,\sqrt{\hat n_t}\big),
$$

which:

* naturally bounds the step size,
* removes the need for the `epsilon` hyperparameter,
* yields true scale invariance of the update.

### 2) Spectral weight decay (2D)

For matrix parameters $W \in \mathbb{R}^{M \times N}$, spectral norm better reflects the scale relevant to activations than the Frobenius norm. Adams therefore applies **decoupled spectral weight decay** (akin to AdamW’s decoupling), replacing the usual $\tfrac{1}{2}\|W\|_F^2$ with the spectral norm $\tfrac{1}{2}\sigma_1^2$:

* We compute a one-step **power iteration** with persistent state (same idea as PyTorch’s `spectral_norm`) to approximate the top singular triplet $(u_1, \sigma_1, v_1)$.
* The decay term is applied as $\sqrt{M}\, u_1\, \sigma_1\, v^\top_1$ (the gradient of $\tfrac{1}{2}\sigma_1^2$, scaled by $\sqrt{M}$ to match the RMS of $W$) per update step.
* This helps control activation scales and mitigates instabilities tied to large spectral norms.

**Efficiency:** the spectral step adds only two GEMV operations per matrix per update, comparable to a handful of extra element-wise ops. In typical FSDP/ZeRO setups the full weight matrix is available during forward/backward, so this integrates cleanly at scale.

## Design Motivation 💡

Recent reports suggest that fully matrix-based optimizers (e.g., Muon) can be hard to implement/parallelize broadly and often show modest end-to-end benefits on large models (~1.1x or less), despite strong stability. Meanwhile, the dominant optimizer Adam is simple and fast but prone to instability and loss spikes.

**Adams** asks: *Can we keep Adam’s speed and simplicity while gaining matrix-level stability?*

## Installation

```bash
pip install adams-torch
```

## Quick Start 📈

You don’t need to manually broadcast parameters or all-reduce gradients—multi-GPU usage matches single-GPU usage. Fully compatible with `torch.compile`.

> FSDP is not supported yet. Contributions welcome.

```python
import os
import torch
import torch.distributed as dist
from adams import Adams_ZeRO  # main optimizer

def init():
    # Initialize distributed training if launched via torchrun/torch.distributed
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)

    # Spectral decay applies to matrix-shaped params.
    # scalar_vector_weight_decay applies standard decoupled L2 to 0D/1D params.
    optimizer = Adams_ZeRO(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.1,                 # spectral decay for matrices
        scalar_vector_weight_decay=0.1,   # L2 for scalars/vectors
        betas=(0.9, 0.95)
    )

    # Sync any internal buffers across ranks if required by your setup.
    optimizer.broadcast_buffers(model.buffers())

    return model, optimizer

@torch.compile  # Optional: works with torch.compile
def train_step(model, optimizer, batch):
    loss = model(batch)        # forward; compute your loss
    loss.backward()            # backward
    optimizer.step()           # no gradient clipping needed
    optimizer.zero_grad(set_to_none=True)
    return loss
```

## Notes ⚠️

Care should be taken as matrix-based optimizers (e.g. Muon).

1. **Non‑matrix parameters.** Disable the matrix‑based part (spectral decay) for parameters that are scalars, vectors, or collections of vectors (e.g. LayerNorm, Embedding, Output Head, etc.) by setting `param.use_spectral_decay = False`. Adams uses a separate decoupled L2 term, controlled by `scalar_vector_weight_decay` (default `0.1`).
2. **Batched matrices.** Parameters that are conceptually multiple matrices concatenated along leading dimensions (e.g., attention QKV projections) should be expressed with shape `(B, M, N)`. Adams treats all dimensions except the last two as batch dimensions.

## Practical Tips ✏️

* **Hyperparameters:** start with AdamW-like settings; the bounded update removes `epsilon`. Adams can handle much larger weight decay to improve generalization, e.g. `1.0`.
* **Stability:** the bounded step and spectral decay together target sources of spikes linked to tiny second moments and large spectral norms.
* **Generalization & adversarial robustness:** spectral regularization is widely observed to improve both, and Adams adopts a lightweight decoupled form.

## License

Apache-2.0
