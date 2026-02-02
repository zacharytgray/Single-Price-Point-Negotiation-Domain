"""
HyperLoRA implementation for Janus agent training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Callable


class RhoHyperNet(nn.Module):
    """
    Hypernetwork that maps a scalar rho to a gating vector g of size [rank].
    
    Structure:
      rho [B, 1] -> (Optional Fourier Encode) -> MLP -> Sigmoid
    """
    def __init__(self, rank: int, hidden_dim: int = 64, activation: str = 'sigmoid',
                 use_fourier: bool = False, fourier_freqs: int = 8, include_raw: bool = True):
        super().__init__()
        self.rank = rank
        self.hidden_dim = hidden_dim
        self.use_fourier = use_fourier
        self.include_raw = include_raw
        
        if use_fourier:
            self.input_dim = (2 * fourier_freqs) + (1 if include_raw else 0)
            freqs = (2.0 ** torch.arange(fourier_freqs)) * 3.14159265359
            self.register_buffer("freqs", freqs, persistent=True)
        else:
            self.input_dim = 1
        
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, rank)
        )
        
        if activation == 'sigmoid':
            self.final_act = nn.Sigmoid()
        elif activation == 'tanh':
            self.final_act = nn.Tanh()
        elif activation == 'softplus':
            self.final_act = nn.Softplus()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
    def encode_rho(self, rho: torch.Tensor) -> torch.Tensor:
        """Apply Fourier encoding if enabled."""
        if not self.use_fourier:
            return rho
            
        scaled = rho * self.freqs
        sin_feat = torch.sin(scaled)
        cos_feat = torch.cos(scaled)
        
        features = [sin_feat, cos_feat]
        if self.include_raw:
            features.append(rho)
            
        return torch.cat(features, dim=-1)

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        x = rho.to(self.net[0].weight.dtype)
        
        device = self.net[0].weight.device
        if x.device != device:
            x = x.to(device)
        
        x_encoded = self.encode_rho(x)
        output = self.net(x_encoded)
        return self.final_act(output)


class HyperLoRALinear(nn.Module):
    """
    Wraps a base nn.Linear layer with HyperLoRA adaptation.
    
    W = W_base + (alpha/rank) * ( B @ diag(g(rho)) @ A )
    """
    def __init__(
        self, 
        base_layer: nn.Linear, 
        rank: int, 
        alpha: float, 
        rho_getter: Callable[[], torch.Tensor],
        hyper_hidden: int = 64,
        dropout_p: float = 0.05,
        use_fourier: bool = False,
        fourier_freqs: int = 8,
        include_raw: bool = True
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.rho_getter = rho_getter
        
        self.base_layer.requires_grad_(False)
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        
        self.hyper_net = RhoHyperNet(
            rank, 
            hidden_dim=hyper_hidden,
            use_fourier=use_fourier,
            fourier_freqs=fourier_freqs,
            include_raw=include_raw
        )
        
        self.dropout = nn.Dropout(p=dropout_p)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.lora_A = self.lora_A.to(*args, **kwargs)
        self.lora_B = self.lora_B.to(*args, **kwargs)
        self.hyper_net = self.hyper_net.to(*args, **kwargs)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_base = self.base_layer(x)
        
        rho = self.rho_getter()
        
        if rho is None:
            return y_base
            
        if rho.device != x.device:
            rho = rho.to(x.device)
        
        if self.lora_A.device != x.device:
            self.lora_A.data = self.lora_A.data.to(x.device)
            self.lora_B.data = self.lora_B.data.to(x.device)

        if self.hyper_net.net[0].weight.device != x.device:
            self.hyper_net.to(x.device)

        g = self.hyper_net(rho)
        
        x_lora = x.to(self.lora_A.dtype)
        z = F.linear(x_lora, self.lora_A)
        
        g_casted = g.to(dtype=z.dtype, device=z.device)
        
        if z.dim() == 3:
            z_gated = z * g_casted.unsqueeze(1)
        else:
            z_gated = z * g_casted
            
        delta = F.linear(z_gated, self.lora_B)
        output = y_base + (self.scaling * self.dropout(delta)).to(y_base.dtype)
        
        return output

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}"


def inject_hyperlora(
    model: nn.Module, 
    target_module_names: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    rank: int = 16, 
    alpha: float = 32.0, 
    dropout: float = 0.05, 
    hyper_hidden: int = 64,
    use_fourier: bool = False,
    fourier_freqs: int = 8,
    include_raw: bool = True
):
    """
    Injects HyperLoRALinear layers into the model in-place.
    """
    if not hasattr(model, "current_rho"):
        model.current_rho = None
        
    rho_getter = lambda: model.current_rho
    
    modules_to_replace = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(name.endswith(target) for target in target_module_names):
                modules_to_replace.append(name)
                
    print(f"Found {len(modules_to_replace)} modules to replace with HyperLoRA.")
    
    for name in modules_to_replace:
        parent_name, child_name = name.rsplit('.', 1)
        parent = model.get_submodule(parent_name)
        target_linear = getattr(parent, child_name)
        
        new_module = HyperLoRALinear(
            base_layer=target_linear,
            rank=rank,
            alpha=alpha,
            rho_getter=rho_getter,
            hyper_hidden=hyper_hidden,
            dropout_p=dropout,
            use_fourier=use_fourier,
            fourier_freqs=fourier_freqs,
            include_raw=include_raw
        )
        
        setattr(parent, child_name, new_module)
        
    for n, p in model.named_parameters():
        if "lora_" in n or "hyper_net" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False
            
    return model


def count_trainable_params(model: nn.Module):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
    )
    return trainable_params
