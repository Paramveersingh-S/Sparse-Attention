"""
sparse_attn/vllm/patch_model.py
==================================
Monkey-patch utility: replace all attention layers in a HuggingFace model
with SparseAttentionImpl for inference with sparse attention.

Usage
-----
    from sparse_attn.vllm.patch_model import patch_vllm_model_with_sparse_attention

    model = patch_vllm_model_with_sparse_attention(
        "meta-llama/Meta-Llama-3-8B",
        {"local_window": 1024, "stride": 2048, "global_blocks": 4},
        device="cuda",
    )

    # Use as normal HuggingFace model
    inputs = tokenizer("Hello world", return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=100)

How it works
------------
1. Load the HuggingFace model with from_pretrained().
2. Recursively scan all nn.Module children.
3. Replace any module whose class name contains "Attention" or "SelfAttention"
   with SparseAttentionModule (a thin wrapper around SparseAttentionImpl).
4. The wrapper intercepts forward() calls and routes through our sparse kernels.
"""

from __future__ import annotations

import re
from typing import Dict, Any, Optional

import torch
import torch.nn as nn


class SparseAttentionModule(nn.Module):
    """
    Thin nn.Module wrapper around SparseAttentionImpl.

    Replaces a standard HuggingFace attention module while keeping
    the same forward() signature.

    Parameters
    ----------
    original_module : The original attention module being replaced.
    sparse_impl     : SparseAttentionImpl instance.
    """

    def __init__(self, original_module: nn.Module, sparse_impl):
        super().__init__()
        self._original = original_module
        self._sparse_impl = sparse_impl

        # Preserve weights from original module
        self._q_proj = getattr(original_module, 'q_proj', None)
        self._k_proj = getattr(original_module, 'k_proj', None)
        self._v_proj = getattr(original_module, 'v_proj', None)
        self._o_proj = getattr(original_module, 'o_proj', None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        """
        Forward pass routing through sparse attention.

        For models where QKV projections are separate (e.g., LLaMA-style):
        1. Project hidden_states → Q, K, V
        2. Reshape to [B, H, S, D]
        3. Run sparse attention
        4. Project back to hidden_states
        """
        B, S, hidden_dim = hidden_states.shape

        # Try to use original module's projections if available
        if self._q_proj is not None and self._k_proj is not None:
            # Standard LLaMA/GPT-style: separate QKV projections
            q = self._q_proj(hidden_states)
            k = self._k_proj(hidden_states)
            v = self._v_proj(hidden_states)

            H  = self._sparse_impl.num_heads
            Hk = self._sparse_impl.num_kv_heads
            D  = self._sparse_impl.head_size

            q = q.view(B, S, H,  D).transpose(1, 2)   # [B, H, S, D]
            k = k.view(B, S, Hk, D).transpose(1, 2)   # [B, Hk, S, D]
            v = v.view(B, S, Hk, D).transpose(1, 2)   # [B, Hk, S, D]

            # Handle GQA: expand k, v to match num_heads
            if Hk != H:
                repeat = H // Hk
                k = k.repeat_interleave(repeat, dim=1)
                v = v.repeat_interleave(repeat, dim=1)

            # Sparse attention
            is_decode = (past_key_value is not None and S == 1)
            from sparse_attn.vllm.sparse_attn_backend import MockAttentionMetadata
            meta = MockAttentionMetadata(
                is_prompt=not is_decode,
                seq_lens=[S] * B,
                max_seq_len=S,
            )
            attn_out = self._sparse_impl.forward(q, k, v, attn_metadata=meta)

            # Reshape back: [B, H, S, D] → [B, S, H*D]
            attn_out = attn_out.transpose(1, 2).reshape(B, S, H * D)

            # Output projection
            if self._o_proj is not None:
                attn_out = self._o_proj(attn_out)

            return (attn_out, None, past_key_value)

        else:
            # Fallback: pass through original module
            return self._original.forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )


def patch_vllm_model_with_sparse_attention(
    model_name: str,
    sparse_config: Optional[Dict[str, Any]] = None,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
    max_seq_len: int = 32768,
    trust_remote_code: bool = False,
) -> nn.Module:
    """
    Load a HuggingFace model and replace all attention layers with sparse attention.

    Parameters
    ----------
    model_name       : HuggingFace model identifier (e.g., "meta-llama/Meta-Llama-3-8B").
    sparse_config    : Sparse pattern configuration dict:
                       - local_window_blocks (default 8)
                       - stride_blocks       (default 16)
                       - global_blocks       (default 4)
                       - block_size          (default 64)
                       - chunk_size          (default 512)
    device           : Target device ("cuda" or "cpu").
    torch_dtype      : Model dtype. Default: float16.
    max_seq_len      : Maximum context length. Default: 32768.
    trust_remote_code: Pass to from_pretrained.

    Returns
    -------
    patched_model : The model with all attention layers replaced.

    Example
    -------
    >>> model = patch_vllm_model_with_sparse_attention(
    ...     "meta-llama/Meta-Llama-3-8B",
    ...     {"local_window_blocks": 8, "stride_blocks": 16, "global_blocks": 4},
    ...     device="cuda",
    ... )
    >>> output = model.generate(input_ids, max_new_tokens=100)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoConfig
    except ImportError:
        raise ImportError(
            "transformers is required for model patching. "
            "Install with: pip install transformers"
        )

    from sparse_attn.vllm.sparse_attn_backend import SparseAttentionImpl

    print(f"[sparse_attn] Loading {model_name} ...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=trust_remote_code,
    )

    # Extract model config params
    num_heads    = getattr(config, 'num_attention_heads', 32)
    num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
    head_size    = getattr(config, 'hidden_size', 4096) // num_heads

    # Build sparse impl
    sparse_impl = SparseAttentionImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=head_size ** -0.5,
        num_kv_heads=num_kv_heads,
        max_seq_len=max_seq_len,
        sparse_config=sparse_config,
    )

    # Count patches applied
    num_patched = [0]

    def _patch_module(module: nn.Module, name: str = "root"):
        for child_name, child in list(module.named_children()):
            full_name = f"{name}.{child_name}"
            child_class = child.__class__.__name__

            # Match standard attention class names
            if re.search(r'(Attention|SelfAttention|MultiHeadAttention)', child_class, re.I):
                patched = SparseAttentionModule(child, sparse_impl)
                setattr(module, child_name, patched)
                num_patched[0] += 1
                print(f"  [patch] {full_name}: {child_class} → SparseAttentionModule")
            else:
                _patch_module(child, full_name)

    print(f"[sparse_attn] Patching attention layers ...")
    _patch_module(model)
    print(f"[sparse_attn] Patched {num_patched[0]} attention modules")
    print(f"[sparse_attn] Sparse config: {sparse_config}")
    print(f"[sparse_attn] Pattern sparsity: {sparse_impl._default_pattern.sparsity:.1%}")

    return model
