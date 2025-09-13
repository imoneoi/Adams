from typing import Tuple, Union, Optional, Any, Iterable
import warnings

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, ParamsT
import torch.distributed as dist


class Adams_ZeRO(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        # Collective communication parameters
        master_dtype: torch.dtype = torch.float32,
        process_group: Optional[dist.ProcessGroup] = None,
        # Optimizer parameters
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.1,
        scalar_vector_weight_decay: float = 0.1,
    ):
        # Initialize the AdamW optimizer
        if isinstance(lr, Tensor):
            if lr.numel() != 1:
                raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= scalar_vector_weight_decay:
            raise ValueError(f"Invalid scalar_vector_weight_decay value: {scalar_vector_weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            scalar_vector_weight_decay=scalar_vector_weight_decay,
        )
        super().__init__(params, defaults)
        self.optimizer_config_keys = list(defaults.keys())
        
        # Initialize ZeRO distributed
        self.master_dtype = master_dtype
        self.process_group = process_group

        if dist.is_initialized():
            self.world_size = dist.get_world_size(self.process_group)
            self.rank = dist.get_rank(self.process_group)
        else:
            self.world_size = 1
            self.rank = 0
            
        if self.world_size == 1:
            warnings.warn(
                f"[{__class__.__name__}] world_size is 1, operating in single-process mode.",
                UserWarning
            )
        
        # Initialize state
        self._init_state()

    @torch.no_grad
    def _init_state(self):
        device = None

        for group in self.param_groups:
            for p in group["params"]:
                # Check device
                if device is None:
                    device = p.device
                else:
                    assert device == p.device, f"All parameters must be on the same device, found {device} and {p.device}"

                # Get shard shape
                assert p.is_contiguous(), "All parameters must be contiguous"
                assert p.numel() % self.world_size == 0, f"Size of parameter must be divisible by world size for distributed training, found size {p.numel()} and world size {self.world_size}"
                shard_size = p.numel() // self.world_size
                
                # Scatter params across ranks from rank 0
                if self.world_size > 1:
                    master_param = torch.empty(shard_size, dtype=p.dtype, device=p.device)
                    
                    dist.scatter(master_param, scatter_list=list(p.view(-1).chunk(self.world_size)) if self.rank == 0 else None, src=0, group=self.process_group)
                else:
                    master_param = p.view(-1)
                
                # Cast to master dtype
                master_param = master_param.to(self.master_dtype)
                # Allgather the master parameter to ensure all ranks have the same parameter
                _all_gather_param(p, master_param, self.world_size, self.process_group)

                # Spectral weight decay
                # Only for parameters with at least 2 dimensions (matrices)
                u = sigma = None
                if p.ndim >= 2 and p.shape[-1] != 1 and p.shape[-2] != 1 and getattr(p, "use_spectral_decay", True):
                    u = torch.ones((*p.shape[:-2], p.shape[-2]), dtype=p.dtype, device=p.device)
                    sigma = torch.zeros((*p.shape[:-2], ), dtype=p.dtype, device=p.device)

                # Put inside dict
                self.state[p] = {
                    "master_param": master_param,
                    "exp_avg": torch.zeros_like(master_param),
                    "exp_avg_sq": torch.zeros_like(master_param),

                    "u": u,
                    "sigma": sigma,

                    "step": torch.tensor(0.0, dtype=torch.get_default_dtype()),
                }

    @torch.no_grad
    def __setstate__(self, state: dict[str, Any]):
        """Set the state of the optimizer."""
        super().__setstate__(state)
        # Allgather the master parameter to ensure all ranks have the same parameter
        for group in self.param_groups:
            for p in group["params"]:
                _all_gather_param(p, self.state[p]["master_param"], self.world_size, self.process_group)

    @torch.no_grad
    def broadcast_buffers(self, buffers: Iterable[Tensor]):
        if self.world_size > 1:
            for buffer in buffers:
                dist.broadcast(buffer, src=0, group=self.process_group)

    @torch.no_grad
    def step(self, closure=None):  # pyright: ignore[reportIncompatibleMethodOverride]
        """Perform a single optimization step."""
        assert closure is None, "Closure is not supported"
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError(
                        f"[{__class__.__name__}] Sparse gradients are not supported"
                    )

                _adams_zero_distributed(
                    param=p,
                    grad=p.grad,
                    
                    rank=self.rank,
                    world_size=self.world_size,
                    process_group=self.process_group,
                    
                    **self.state[p],
                    **{k: group[k] for k in self.optimizer_config_keys},
                )


def _all_gather_param(param: Tensor, master_param: Tensor, world_size: int, process_group: Optional[dist.ProcessGroup]) -> None:
    # All-gather
    if world_size > 1:
        dist.all_gather_into_tensor(param.view(-1), master_param.to(param.dtype), group=process_group)
    else:
        param.view(-1).copy_(master_param.to(param.dtype))


def _adams_zero_distributed(
    param: Tensor,
    grad: Tensor,
    
    master_param: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    u: Optional[Tensor],
    sigma: Optional[Tensor],
    step: Tensor,

    lr: Tensor,
    betas: Tuple[float, float],
    weight_decay: float,
    scalar_vector_weight_decay: float,

    rank: int,
    world_size: int,
    process_group: Optional[dist.ProcessGroup],
) -> None:
    # Flatten gradients
    grad = grad.view(-1)
    
    if world_size > 1:
        # Reduce-scatter in master precision (often f32)
        buf = torch.empty_like(master_param)
        dist.reduce_scatter_tensor(buf, grad.to(buf.dtype), op=dist.ReduceOp.SUM, group=process_group)

        grad = buf
        
    # Weight decay update
    if weight_decay != 0:
        if u is not None:
            # Spectral weight decay
            # 1 step of power iter
            v = F.normalize(torch.einsum("...nm,...n->...m", param, u), dim=-1)
            Wv = torch.einsum("...nm,...m->...n", param, v)
            F.normalize(Wv, dim=-1, out=u)  # In-place update u
            
            s = torch.einsum("...n,...n->...", u, Wv)
            best_rank1_approx = torch.einsum("...n, ...m->...nm", s.unsqueeze(-1) * u, v)
            master_param.add_(best_rank1_approx.view(-1).chunk(world_size)[rank], alpha=-(param.size(-2) ** 0.5) * lr * weight_decay)  # pyright: ignore[reportArgumentType]
            
            if sigma is not None:
                sigma.copy_(s)  # For visualization/debugging
        else:
            # Normal weight decay for scalars/vectors/a bunch of vectors (e.g. layernorm, embedding, output head)
            master_param.mul_(1 - lr * scalar_vector_weight_decay)

    # Momentums
    exp_avg.lerp_(grad, 1 - betas[0])
    exp_avg_sq.mul_(betas[1]).addcmul_(grad, grad, value=1 - betas[1])
    
    step += 1
    bias_correction1 = 1 - betas[0] ** step
    bias_correction2 = 1 - betas[1] ** step
    step_size = lr / bias_correction1
    bias_correction2_sqrt = bias_correction2.sqrt()

    denom = exp_avg_sq.sqrt() / bias_correction2_sqrt
    # AdamW-atan2
    master_param.add_(torch.atan2(exp_avg, denom), alpha=-step_size)

    # Final all-gather
    _all_gather_param(param, master_param, world_size, process_group)
