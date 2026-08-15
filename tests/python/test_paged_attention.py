"""
Numerical Accuracy & Integration Tests for Paged Attention CUDA Kernel
======================================================================
"""

import math

import pytest
import torch
import torch.nn.functional as F

try:
    import nexuscache._C as C_ext

    HAS_EXTENSION = True
except ImportError:
    HAS_EXTENSION = False


@pytest.mark.skipif(
    not (HAS_EXTENSION and torch.cuda.is_available()),
    reason="Requires PyTorch GPU support and compiled nexuscache._C extension.",
)
class TestPagedAttentionKernel:

    def test_paged_attention_correctness(self):
        """Validates Paged Attention output against vanilla PyTorch Attention ground truth."""
        torch.manual_seed(42)
        device = "cuda"

        num_seqs = 2
        num_heads = 4
        head_dim = 32
        block_size = 16
        max_blocks_per_seq = 2
        scale = 1.0 / math.sqrt(head_dim)

        seq_lens = [20, 12]  # Seq 0: 20 tokens (2 blocks), Seq 1: 12 tokens (1 block)

        # 1. Initialize Direct 4D Physical KV Tensors [num_blocks, num_heads, block_size, head_dim]
        num_blocks = 16
        key_cache = torch.randn(
            (num_blocks, num_heads, block_size, head_dim),
            dtype=torch.float16,
            device=device,
        ).contiguous()
        val_cache = torch.randn(
            (num_blocks, num_heads, block_size, head_dim),
            dtype=torch.float16,
            device=device,
        ).contiguous()

        # 2. Block table mappings: Seq 0 -> [Block 2, Block 5], Seq 1 -> [Block 1, -1]
        block_tables = [2, 5, 1, -1]

        # 3. Create Decoding Query Tensor [num_seqs, num_heads, head_dim]
        query = torch.randn(
            (num_seqs, num_heads, head_dim), dtype=torch.float16, device=device
        ).contiguous()
        paged_out = torch.zeros_like(query)

        # 4. Run CUDA Kernel
        C_ext.launch_paged_attention(
            out_ptr=paged_out.data_ptr(),
            query_ptr=query.data_ptr(),
            key_cache_ptr=key_cache.data_ptr(),
            value_cache_ptr=val_cache.data_ptr(),
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_num_blocks_per_seq=max_blocks_per_seq,
            num_seqs=num_seqs,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
            scale=scale,
            stream_ptr=torch.cuda.current_stream().cuda_stream,
        )
        torch.cuda.synchronize()

        # 5. Build Reference Attention Ground Truth for Sequence 0
        # Seq 0 uses physical block 2 (16 tokens) and physical block 5 (4 tokens)
        k_b0 = key_cache[2]  # [num_heads, 16, head_dim]
        k_b1 = key_cache[5, :, :4, :]  # [num_heads, 4, head_dim]
        keys = torch.cat([k_b0, k_b1], dim=1)  # [num_heads, 20, head_dim]

        v_b0 = val_cache[2]  # [num_heads, 16, head_dim]
        v_b1 = val_cache[5, :, :4, :]  # [num_heads, 4, head_dim]
        vals = torch.cat([v_b0, v_b1], dim=1)  # [num_heads, 20, head_dim]

        # Standard Attention Computation: Softmax(Q @ K.T * scale) @ V
        q0 = query[0].unsqueeze(1)  # [num_heads, 1, head_dim]
        attn_scores = (
            torch.matmul(q0, keys.transpose(-1, -2)) * scale
        )  # [num_heads, 1, 20]
        attn_weights = F.softmax(attn_scores, dim=-1)
        ref_out0 = torch.matmul(attn_weights, vals).squeeze(1)  # [num_heads, head_dim]

        # 6. Verify Numerical Equivalence
        torch.testing.assert_close(paged_out[0], ref_out0, rtol=1e-2, atol=1e-2)
