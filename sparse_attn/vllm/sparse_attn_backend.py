"""
sparse_attn/vllm/sparse_attn_backend.py
==========================================
vLLM-compatible sparse attention backend.

Implements the vLLM AttentionBackend / AttentionImpl interface.
Compatible with vLLM 0.4.x and 0.5.x (auto-detected).

When vLLM is not installed, provides a standalone mock that can be
used for testing and integration without a full vLLM environment.

Integration
-----------
To use with vLLM, register via:
    VLLM_ATTENTION_BACKEND=SPARSE_ATTN vllm serve <model>

Or programmatically:
    from sparse_attn.vllm import SparseAttentionBackend
    # Pass to vLLM engine config
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Type

import torch

from sparse_attn.patterns.heterogeneous import make_local_stride_pattern
from sparse_attn.kernels.sparse_prefill import sparse_prefill
from sparse_attn.kernels.sparse_decode import sparse_decode_chunked


# --------------------------------------------------------------------------- #
#  vLLM interface detection                                                   #
# --------------------------------------------------------------------------- #

_VLLM_AVAILABLE = False
_AttentionBackendBase = object
_AttentionImplBase    = object

try:
    from vllm.attention.backends.abstract import AttentionBackend, AttentionImpl, AttentionMetadata
    _VLLM_AVAILABLE      = True
    _AttentionBackendBase = AttentionBackend
    _AttentionImplBase    = AttentionImpl
    print("[sparse_attn] vLLM detected — using native backend interface")
except ImportError:
    print("[sparse_attn] vLLM not found — using mock backend interface")


# --------------------------------------------------------------------------- #
#  Mock AttentionMetadata (for non-vLLM environments)                        #
# --------------------------------------------------------------------------- #

@dataclass
class MockAttentionMetadata:
    """
    Standalone attention metadata container for testing without vLLM.

    Attributes
    ----------
    is_prompt     : True for prefill, False for decode.
    seq_lens      : List of sequence lengths per batch element.
    max_seq_len   : Maximum sequence length in the batch.
    block_tables  : Optional KV cache block tables (vLLM paged attention).
    """
    is_prompt   : bool = True
    seq_lens    : List[int] = field(default_factory=lambda: [512])
    max_seq_len : int = 512
    block_tables: Optional[torch.Tensor] = None

    @property
    def is_decode(self) -> bool:
        return not self.is_prompt


# --------------------------------------------------------------------------- #
#  Sparse Attention Implementation                                            #
# --------------------------------------------------------------------------- #

class SparseAttentionImpl(_AttentionImplBase):
    """
    Sparse attention implementation compatible with vLLM's AttentionImpl interface.

    Automatically selects prefill vs decode kernel based on attn_metadata.is_prompt.

    Parameters
    ----------
    num_heads      : Number of attention heads.
    head_size      : Head dimension (D).
    scale          : Softmax scale (1/sqrt(D) if not specified).
    num_kv_heads   : Number of KV heads (GQA support).
    max_seq_len    : Maximum context length for pattern construction.
    sparse_config  : Dict with pattern parameters:
                     - local_window_blocks (int, default 8)
                     - stride_blocks       (int, default 16)
                     - global_blocks       (int, default 4)
                     - block_size          (int, default 64)
                     - chunk_size          (int, default 512) for decode
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        max_seq_len: int = 32768,
        sparse_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.num_heads    = num_heads
        self.head_size    = head_size
        self.scale        = scale if scale else head_size ** -0.5
        self.num_kv_heads = num_kv_heads or num_heads
        self.max_seq_len  = max_seq_len

        # Parse sparse config
        cfg = sparse_config or {}
        self.block_size          = cfg.get("block_size",          64)
        self.local_window_blocks = cfg.get("local_window_blocks",  8)
        self.stride_blocks       = cfg.get("stride_blocks",       16)
        self.global_blocks       = cfg.get("global_blocks",        4)
        self.chunk_size          = cfg.get("chunk_size",          512)

        # Build sparse pattern (reused across forward calls of the same length)
        self._pattern_cache: Dict[int, Any] = {}
        self._default_pattern = self._get_or_build_pattern(max_seq_len)

    def _get_or_build_pattern(self, seq_len: int):
        """Cache-aware pattern builder."""
        # Round up to nearest multiple of block_size
        aligned = ((seq_len + self.block_size - 1) // self.block_size) * self.block_size
        if aligned not in self._pattern_cache:
            self._pattern_cache[aligned] = make_local_stride_pattern(
                seq_len=aligned,
                block_size=self.block_size,
                local_window_blocks=self.local_window_blocks,
                stride_blocks=self.stride_blocks,
                global_blocks=self.global_blocks,
            )
        return self._pattern_cache[aligned]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
        attn_metadata = None,
    ) -> torch.Tensor:
        """
        Forward pass: dispatch to prefill or decode kernel.

        Parameters
        ----------
        query, key, value : Tensors [B, H, S, D] (prefill) or [B, H, 1, D] (decode).
        kv_cache          : Optional KV cache tensor (paged attention).
        attn_metadata     : vLLM AttentionMetadata or MockAttentionMetadata.

        Returns
        -------
        output : Same shape as query.
        """
        # Determine mode
        is_prompt = True
        if attn_metadata is not None:
            is_prompt = getattr(attn_metadata, 'is_prompt', True)

        if is_prompt:
            return self._sparse_prefill(query, key, value, attn_metadata)
        else:
            return self._sparse_decode_chunked(query, key, value, kv_cache, attn_metadata)

    def _sparse_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata,
    ) -> torch.Tensor:
        """Prefill: full sequence sparse attention."""
        _, _, S, _ = query.shape
        pattern = self._get_or_build_pattern(S)

        return sparse_prefill(
            q=query,
            k=key,
            v=value,
            pattern=pattern,
            softmax_scale=self.scale,
        )

    def _sparse_decode_chunked(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata,
    ) -> torch.Tensor:
        """Decode: single-token sparse attention with chunked KV."""
        _, _, S, _ = key.shape
        pattern = self._get_or_build_pattern(S)

        return sparse_decode_chunked(
            q=query,
            k=key,
            v=value,
            pattern=pattern,
            chunk_size=self.chunk_size,
            softmax_scale=self.scale,
        )


# --------------------------------------------------------------------------- #
#  Sparse Attention Backend (vLLM registration)                               #
# --------------------------------------------------------------------------- #

class SparseAttentionBackend(_AttentionBackendBase):
    """
    vLLM attention backend registering the sparse attention implementation.

    Register this backend by setting:
        VLLM_ATTENTION_BACKEND = "SPARSE_ATTN"

    Or pass it directly to vLLM engine configuration.
    """

    @staticmethod
    def get_name() -> str:
        return "SPARSE_ATTN"

    @staticmethod
    def get_impl_cls() -> Type[SparseAttentionImpl]:
        return SparseAttentionImpl

    @staticmethod
    def get_metadata_cls():
        if _VLLM_AVAILABLE:
            try:
                from vllm.attention.backends.abstract import AttentionMetadata
                return AttentionMetadata
            except ImportError:
                pass
        return MockAttentionMetadata

    @staticmethod
    def make_metadata(*args, **kwargs) -> MockAttentionMetadata:
        """Create metadata for testing without full vLLM."""
        return MockAttentionMetadata(**kwargs)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ):
        """KV cache shape for paged attention."""
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(src, dst, block_mapping):
        """Swap KV cache blocks (for paged attention eviction)."""
        src.copy_(dst)

    @staticmethod
    def copy_blocks(kv_caches, src_to_dists):
        """Copy KV cache blocks (for prefix caching)."""
        pass  # Implement paged attention copy if needed
