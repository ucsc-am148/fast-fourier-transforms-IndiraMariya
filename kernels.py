"""STUDENT FILE: implement the Triton kernels and pipeline drivers.

You implement:
  - Six @triton.jit kernels: f1_kernel, f2_kernel, transpose_kernel,
    f4_kernel_L2, dft_kernel, bailey_scale_kernel.
  - The f1_launch and f2_launch grid-choice wrappers around them.
  - The pipeline drivers: f3_launch, f5_launch, _f6_rec, _f7_rec.
  - f6_factor: the chunk-recipe for F6/F7.

You do NOT implement (left given below):
  - The thin launch wrappers _transpose, _fft_chunk, _scale, _lookup_tw.
    These are mechanical "pick the grid and launch one kernel" helpers.
  - The tuning constants F4_L2_BLOCK_B, DFT_BLOCK_B, SCALE_BLOCK,
    TRANSPOSE_BLOCK.

The signatures below are the ones the harness calls -- your job is to fill
the bodies. When your code passes sanity_check.py, you're done.
"""

import math

import torch
import triton
import triton.language as tl


# Tunings -- GIVEN.
F4_L2_BLOCK_B = 2
DFT_BLOCK_B = 16
SCALE_BLOCK = 32
TRANSPOSE_BLOCK = 32

# =============================================================================
# Device-function helper: complex matmul
# =============================================================================
# Implement this once -- f1_kernel, f4_kernel_L2, and dft_kernel all call it.


@triton.jit
def _cdot(a_re, a_im, b_re, b_im):
    """Complex matmul Y = A @ B as four real tl.dot calls.

    Returns (y_re, y_im) in fp32 (out_dtype=tl.float32). Caller is responsible
    for any fp16 down-cast on store. Works at any matmul shape tl.dot accepts.

    Used by f1_kernel, f4_kernel_L2, and dft_kernel. Don't reimplement the
    four-tl.dot expansion at each call site -- implement once here, call
    everywhere.

    TODO: implement.
    """
    # pass
    return (tl.dot(a_re, b_re, out_dtype=tl.float32) - tl.dot(a_im, b_im, out_dtype=tl.float32),
            tl.dot(a_re, b_im, out_dtype=tl.float32) + tl.dot(a_im, b_re, out_dtype=tl.float32))


# =============================================================================
# Chunk factorization for F6 / F7
# =============================================================================

def f6_factor(N: int) -> list[int]:
    """Factor N = 2^k into FFT chunks.

    Recipe: prefer 256-length chunks (radix-256, handled by f4_kernel_L2), then
    16-length (handled by dft_kernel via the padded radix-16 path), then a
    small leftover in {2, 4, 8} for the remaining bits. chunks[0] is the
    innermost (fastest) input axis. Examples:
        256 -> [256]                4096 -> [256, 16]
        65536 -> [256, 256]         1048576 -> [256, 256, 16]
        64 -> [16, 4]               2 -> [2]
    """
    chunks = []
    while N > 1:
        if N % 256 == 0:
            chunks.append(256)
            N //= 256
        elif N % 16 == 0:
            chunks.append(16)
            N //= 16
        else:
            # leftover must be in {2, 4, 8}
            chunks.append(N)
            N = 1
    return chunks


f7_factor = f6_factor   # F7 reuses F6's chunk recipe


# =============================================================================
# F1: DFT as one dense complex matmul (four tl.dot)
# =============================================================================

@triton.jit
def f1_kernel(
    x_re_ptr, x_im_ptr,    # (B, N) fp16
    W_re_ptr, W_im_ptr,    # (N, N) fp16; W[n, k]
    y_re_ptr, y_im_ptr,    # (B, N) fp32
    B,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Y = X @ W^T as four (BLOCK_M, BLOCK_K) x (BLOCK_K, BLOCK_N) tl.dot calls.

    Y[b, n] = sum_k X[b, k] * W[n, k]. Load W in transposed access
    (W_T[k, n] = W[n, k]) so tl.dot reads it the way it wants.

    Use `_cdot(x_re, x_im, W_T_re, W_T_im)` for the per-block complex matmul;
    accumulate its fp32 output into `acc_re` / `acc_im`.

    Dtype contract (same as F4): loads are fp16, `tl.dot` runs with
    `out_dtype=tl.float32` (handled by `_cdot`), accumulator is fp32, store
    is fp32. Allocations in `f1_alloc` already match this -- x_re/x_im are
    fp16, y_re/y_im are fp32.

    TODO: implement.
    """
    row = tl.program_id(0)
    col = tl.program_id(1)

    row_offs = tl.arange(0,BLOCK_M) + row * BLOCK_M
    col_offs = tl.arange(0,BLOCK_N) + col * BLOCK_N

    acc_re = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_im = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, N, BLOCK_K):
        k_offs = tl.arange(0, BLOCK_K) + k_start

        x_re = tl.load(x_re_ptr + (row_offs[:, None] * N + k_offs[None, :]))
        x_im = tl.load(x_im_ptr + (row_offs[:, None] * N + k_offs[None, :]))
        w_re = tl.load(W_re_ptr + col_offs[:, None] * N + k_offs[None, :])
        w_im = tl.load(W_im_ptr + col_offs[:, None] * N + k_offs[None, :])

        w_re_t = tl.trans(w_re)
        w_im_t = tl.trans(w_im)
        
        re, im = _cdot(x_re, x_im, w_re_t, w_im_t)
        acc_re += re
        acc_im += im

    tl.store(y_re_ptr + row_offs[:,None] * N + col_offs[None,:] , acc_re)
    tl.store(y_im_ptr + row_offs[:,None] * N + col_offs[None,:], acc_im)

def f1_launch(x_re, x_im, W_re, W_im, y_re, y_im):
    """Grid: (cdiv(B, BLOCK_M), cdiv(N, BLOCK_N)). One program tiles a
    (BLOCK_M, BLOCK_N) output square. tl.dot needs all three dims >=16, so B
    should be >= 16.

    TODO:implement.
    """
    B, N = x_re.shape
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16
    grid = (triton.cdiv(B, BLOCK_M), triton.cdiv(N, BLOCK_N))
    f1_kernel[grid](x_re, x_im,W_re,W_im, y_re, y_im, B, N=N ,BLOCK_M=BLOCK_M, BLOCK_K = BLOCK_K, BLOCK_N=BLOCK_N )



# =============================================================================
# F2: radix-2 Cooley-Tukey, single program per signal
# =============================================================================
# F3 reuses this kernel! For F2, only BAILEY_EPILOGUE=False, STRIDED_STORE=False need to be implemented.
#
# Call-site cheatsheet:
#   F2 vanilla:  pid -> one signal in (B, N). Grid: (B,).
#                BAILEY_EPILOGUE=False, STRIDED_STORE=False.
#                OUTER_DIM and N_TOTAL unused (pass 1 / 0).
#                bt_*_ptr: pass tw_*_ptr again (sentinel; never read).
#   F2-A (F3):   pid -> (b, n1). Grid: (B*N1,). FFT length N=N2.
#                BAILEY_EPILOGUE=True, STRIDED_STORE=False.
#                OUTER_DIM=N1 (n1 = pid % N1).
#                bt_*_ptr: real Bailey twiddles shape (N1, N2).
#   F2-B (F3):   pid -> (b, k2). Grid: (B*N2,). FFT length N=N1.
#                BAILEY_EPILOGUE=False, STRIDED_STORE=True.
#                OUTER_DIM=N2, N_TOTAL=N1*N2.
#                bt_*_ptr: sentinel.

@triton.jit
def f2_kernel(
    x_re_ptr, x_im_ptr,        # (B, N) fp32 input
    y_re_ptr, y_im_ptr,        # (B, N) fp32 output (layout depends on STRIDED_STORE)
    tw_re_ptr, tw_im_ptr,      # (N/2,) fp32 radix-2 twiddles
    perm_ptr,                   # (N,) int32 bit-reversal index
    bt_re_ptr, bt_im_ptr,       # (OUTER_DIM, N) fp32 Bailey twiddles (BAILEY_EPILOGUE only)
    OUTER_DIM, N_TOTAL,
    N: tl.constexpr,
    LOG2_N: tl.constexpr,
    BAILEY_EPILOGUE: tl.constexpr,
    STRIDED_STORE: tl.constexpr,
):
    """Radix-2 Cooley-Tukey FFT in registers, with optional Bailey epilogue and
    strided store. log2(N) butterfly stages via tl.gather for partner shuffle.

    TODO: implement.
    """
    pid_b = tl.program_id(0)
    i = tl.arange(0, N)
    offs = pid_b*N + i

    rev = tl.load(perm_ptr + i)
    x_re = tl.load(x_re_ptr + pid_b*N + rev)
    x_im = tl.load(x_im_ptr + pid_b*N + rev)

    for stage in range(LOG2_N):
        partner_idx = i ^ (1 << stage)
        re_p = tl.gather(x_re, partner_idx, 0)
        im_p = tl.gather(x_im, partner_idx, 0)

        tw_idx = (i & ((1 << stage) - 1)) * (N >> (stage + 1))
        tw_re = tl.load(tw_re_ptr + tw_idx)
        tw_im = tl.load(tw_im_ptr + tw_idx)

        mask = (i & (1 << stage)) == 0

        # a is the element where bit s is 0, b is where bit s is 1
        a_re = tl.where(mask, x_re, re_p)
        a_im = tl.where(mask, x_im, im_p)
        b_re = tl.where(mask, re_p, x_re)
        b_im = tl.where(mask, im_p, x_im)

        # w * b
        wb_re = tw_re * b_re - tw_im * b_im
        wb_im = tw_re * b_im + tw_im * b_re

        x_re = tl.where(mask, a_re + wb_re, a_re - wb_re)
        x_im = tl.where(mask, a_im + wb_im, a_im - wb_im)

    if BAILEY_EPILOGUE:
        n1 = pid_b % OUTER_DIM
        bt_re = tl.load(bt_re_ptr + n1 * N + i)
        bt_im = tl.load(bt_im_ptr + n1 * N + i)
        new_re = x_re * bt_re - x_im * bt_im
        new_im = x_re * bt_im + x_im * bt_re
        x_re = new_re
        x_im = new_im

    if STRIDED_STORE:
        k2 = pid_b % OUTER_DIM
        b  = pid_b // OUTER_DIM
        tl.store(y_re_ptr + b * N_TOTAL + i * OUTER_DIM + k2, x_re)
        tl.store(y_im_ptr + b * N_TOTAL + i * OUTER_DIM + k2, x_im)
    else:
        tl.store(y_re_ptr + offs, x_re)
        tl.store(y_im_ptr + offs, x_im)

def f2_launch(x_re, x_im, y_re, y_im, tw_re, tw_im, perm):
    """Grid: (B,). One program per length-N signal. Vanilla mode.

    TODO: implement.
    """
    B, N = x_re.shape  
    LOG_2 = int(math.log2(N))
    grid = (B,)
    f2_kernel[grid](
        x_re, x_im, y_re, y_im, tw_re, tw_im, perm,
        tw_re,tw_im,
        1,0,
        N=N, LOG2_N=LOG_2,
        BAILEY_EPILOGUE=False, STRIDED_STORE=False,
        )


# =============================================================================
# transpose_kernel: (B, R, C) -> (B, C, R), paired re/im
# =============================================================================

@triton.jit
def transpose_kernel(
    x_re_ptr, x_im_ptr,     # (B*R*C,) fp16 or fp32 input
    y_re_ptr, y_im_ptr,     # (B*R*C,) fp16 or fp32 output
    R, C,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Logical (B, R, C) -> (B, C, R) transpose. Grid: (cdiv(R, BLOCK_R),
    cdiv(C, BLOCK_C), B). Each program copies a (BLOCK_R, BLOCK_C) tile.

    TODO: implement.
    """
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_b = tl.program_id(2)

    r_start = pid_r * BLOCK_R
    c_start = pid_c * BLOCK_C

    r_offs = tl.arange(0, BLOCK_R) + r_start
    c_offs = c_start + tl.arange(0,BLOCK_C)

    in_addr = pid_b*(R*C) + r_offs[:, None] * C + c_offs[None, :] #this makes r_offs a column vector and c_offs a row vector, so the broadcasting gives the full (BLOCK_R, BLOCK_C) tile
    out_addr = pid_b*(C*R) + c_offs[None, :]*R + r_offs[:, None]
    
    mask = (r_offs[:, None] < R) & (c_offs[None, :] < C) # mask to ensure the calcualted offsets are within range 
    x_re = tl.load(x_re_ptr + in_addr, mask = mask)   
    x_im = tl.load(x_im_ptr + in_addr, mask = mask)
    
    tl.store(y_re_ptr + out_addr, x_re, mask=mask)
    tl.store(y_im_ptr + out_addr, x_im, mask=mask)

# =============================================================================
# F4: tcFFT radix-16 single-program FFT (N = 256, L = 2)
# =============================================================================
# See the kernel docstring for the tl.permute tuple-literal gotcha.
@triton.jit
def f4_kernel_L2(
    x_re_ptr, x_im_ptr,    # (B, 256) fp16
    y_re_ptr, y_im_ptr,    # (B, 256) or (B//M, 256, M) fp16
    F_re_ptr, F_im_ptr,    # (16, 16) fp16 -- F_16 DFT matrix
    tw_re_ptr, tw_im_ptr,  # (L=2, 16, 16) fp16 stacked stage twiddles
    B, M,
    BLOCK_B: tl.constexpr,
    STAGE_STOP: tl.constexpr,
    STORE_T: tl.constexpr,
):
    """tcFFT length-256 FFT as two stages of (permute + per-stage twiddle +
    length-16 DFT via four tl.dot). fp16 storage, fp32 matmul accumulators.

    `STAGE_STOP` and `M` are both degenerate in vanilla F4 (`STAGE_STOP=L=2`,
    `M=1`). They exist so the same kernel handles two extra uses:
      - `STAGE_STOP=1`: stop after the s=0 stage, for the sanity_check.py
        stage-1 isolation test (no twiddles, no second matmul).
      - `M>1` with `STORE_T=True`: F7's fused FFT-m_0+T3, writing the
        transposed (rows_outer, 256, M) layout the next level expects.

    STORE_T=False (M=1): natural (B, 256) row-major output.
    STORE_T=True  (M>1): transposed (B//M, 256, M) output for F7 fusion.

    Each stage's four-`tl.dot` is one `_cdot` call; cast its fp32 output to
    fp16 before the next stage.

    Dtype contract:
        Loads:           fp16
        Reshape/permute: fp16 (free)
        tl.dot inputs:   fp16, out_dtype=tl.float32  (use _cdot)
        Twiddle mul:     fp32 * fp16 -> fp32
        Inter-stage:     .to(tl.float16) before next iter's reshape
        Store:           fp16
    Forgetting the inter-stage cast doubles register pressure and passes the
    L=2 tolerance, but fails as soon as F6 stacks more stages.

    Triton 3.6 gotcha -- tl.permute requires LITERAL tuples:
        tl.permute(x, (1, 0, 2))                  # works
        perm = (1, 0, 2); tl.permute(x, perm)     # fails
    Inline each stage's permute tuple at the call site; don't store the
    schedule in a loop variable.

    TODO: implement.
    """
    pid = tl.program_id(0)
    batch_offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)  # (BLOCK_B,)
    mask_b = batch_offs < B

    d = tl.arange(0, 16)  # (16,) digit indices

    # load F (16,16) DFT matrix
    F_re = tl.load(F_re_ptr + d[:, None] * 16 + d[None, :])  # (16,16)
    F_im = tl.load(F_im_ptr + d[:, None] * 16 + d[None, :])

    # load input (BLOCK_B, 256) as (BLOCK_B, 16, 16)
    # flat index: batch * 256 + d0 * 16 + d1
    d0 = tl.arange(0, 16)  # high digit
    d1 = tl.arange(0, 16)  # low digit

    # load full (BLOCK_B, 16, 16) tile
    # addr[b, i, j] = batch_offs[b]*256 + i*16 + j
    addr = (batch_offs[:, None, None] * 256
            + d0[None, :, None] * 16
            + d1[None, None, :])  # (BLOCK_B, 16, 16)

    tile_re = tl.load(x_re_ptr + addr, mask=mask_b[:, None, None]).to(tl.float16)
    tile_im = tl.load(x_im_ptr + addr, mask=mask_b[:, None, None]).to(tl.float16)

    # bring d0 to last axis so _cdot transforms it
    tile_re = tl.permute(tile_re, (0, 2, 1))
    tile_im = tl.permute(tile_im, (0, 2, 1))
    # ── stage 0 ──────────────────────────────────────────────────────────────
    # permute: (BLOCK_B, 16, 16) -> (BLOCK_B, 16, 16)  (no-op for s=0)
    # no twiddle at s=0
    # DFT along axis 1: reshape to (BLOCK_B*16, 16), cdot with F, reshape back

    t0_re = tl.reshape(tile_re, (BLOCK_B * 16, 16))
    t0_im = tl.reshape(tile_im, (BLOCK_B * 16, 16))

    r0_re, r0_im = _cdot(t0_re, t0_im, F_re, F_im)

    # cast back to fp16 for next stage
    r0_re = r0_re.to(tl.float16)
    r0_im = r0_im.to(tl.float16)

    if STAGE_STOP == 1:
        # early exit for sanity_check stage-1 isolation test
        out_re = tl.reshape(r0_re, (BLOCK_B, 16, 16))
        out_im = tl.reshape(r0_im, (BLOCK_B, 16, 16))
        out_addr = (batch_offs[:, None, None] * 256
                    + d0[None, :, None] * 16
                    + d1[None, None, :])
        
        out_re = tl.permute(out_re, (0, 2, 1))
        out_im = tl.permute(out_im, (0, 2, 1))
        tl.store(y_re_ptr + out_addr, out_re, mask=mask_b[:, None, None])
        tl.store(y_im_ptr + out_addr, out_im, mask=mask_b[:, None, None])
        return

    # ── stage 1 ──────────────────────────────────────────────────────────────
    stage1_re = tl.reshape(r0_re, (BLOCK_B, 16, 16))  # (BLOCK_B, d1, e1)
    stage1_im = tl.reshape(r0_im, (BLOCK_B, 16, 16))

    tw_re = tl.load(tw_re_ptr + 16 * 16 + d[:, None] * 16 + d[None, :])
    tw_im = tl.load(tw_im_ptr + 16 * 16 + d[:, None] * 16 + d[None, :])

    tw_re_3d = tl.broadcast_to(tw_re[None, :, :], (BLOCK_B, 16, 16))
    tw_im_3d = tl.broadcast_to(tw_im[None, :, :], (BLOCK_B, 16, 16))

    # fp32 twiddle multiply — stay in fp32, don't cast down yet
    s1_re = stage1_re.to(tl.float32)
    s1_im = stage1_im.to(tl.float32)
    tw_re_f = tw_re_3d.to(tl.float32)
    tw_im_f = tw_im_3d.to(tl.float32)

    tw_out_re = s1_re * tw_re_f - s1_im * tw_im_f   # fp32, no .to(fp16) here
    tw_out_im = s1_re * tw_im_f + s1_im * tw_re_f   # fp32

    # permute (BLOCK_B, d1, e1) -> (BLOCK_B, e1, d1), still fp32
    tw_out_re = tl.permute(tw_out_re, (0, 2, 1))
    tw_out_im = tl.permute(tw_out_im, (0, 2, 1))

    t1_re = tl.reshape(tw_out_re, (BLOCK_B * 16, 16))
    t1_im = tl.reshape(tw_out_im, (BLOCK_B * 16, 16))

    # cast to fp16 only at the _cdot boundary (tl.dot requires fp16 inputs)
    r1_re, r1_im = _cdot(t1_re.to(tl.float16), t1_im.to(tl.float16), F_re, F_im)
    r1_re = r1_re.to(tl.float16)
    r1_im = r1_im.to(tl.float16)

    out_re = tl.permute(tl.reshape(r1_re, (BLOCK_B, 16, 16)), (0, 2, 1))
    out_im = tl.permute(tl.reshape(r1_im, (BLOCK_B, 16, 16)), (0, 2, 1))

    if STORE_T:
        b_outer = batch_offs // M
        m_idx   = batch_offs % M
        out_addr = (b_outer[:, None, None] * 256 * M
                    + (d0[None, :, None] * 16 + d1[None, None, :]) * M
                    + m_idx[:, None, None])
        tl.store(y_re_ptr + out_addr, out_re, mask=mask_b[:, None, None])
        tl.store(y_im_ptr + out_addr, out_im, mask=mask_b[:, None, None])
    else:
        out_addr = (batch_offs[:, None, None] * 256
                    + d0[None, :, None] * 16
                    + d1[None, None, :])
        tl.store(y_re_ptr + out_addr, out_re, mask=mask_b[:, None, None])
        tl.store(y_im_ptr + out_addr, out_im, mask=mask_b[:, None, None])

# =============================================================================
# dft_kernel: padded length-R DFT for the small chunks (R in {2, 4, 8, 16})
# =============================================================================

@triton.jit
def dft_kernel(
    x_re_ptr, x_im_ptr,     # (rows, R) fp16
    y_re_ptr, y_im_ptr,     # (rows, R) or (rows//M, R, M) fp16
    M_re_ptr, M_im_ptr,     # (16, 16) fp16 padded-R DFT matrix
    rows, M,
    R: tl.constexpr,
    BLOCK_B: tl.constexpr,
    STORE_T: tl.constexpr,
):
    """Padded length-R DFT via a (16, 16) tl.dot. STORE_T toggles natural
    vs transposed output (same pattern as f4_kernel_L2).

    One `_cdot(x_re, x_im, MT_re, MT_im)` call replaces the four `tl.dot`
    expansions; cast its fp32 result to fp16 on store.

    TODO: implement.
    """
    pid = tl.program_id(0)
    row_offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)  # (BLOCK_B,)
    mask_r = row_offs < rows

    r = tl.arange(0, 16)   # always 16 for the padded matmul

    # Load padded (16, 16) DFT matrix — we use it transposed: MT[k, n]
    # _cdot(x, MT) computes x @ MT which is (BLOCK_B, 16) @ (16, 16) = (BLOCK_B, 16)
    # giving Y[b, k] = sum_n x[b, n] * M[k, n]  -- correct DFT formula
    MT_re = tl.load(M_re_ptr + r[:, None] * 16 + r[None, :])  # (16, 16)
    MT_im = tl.load(M_im_ptr + r[:, None] * 16 + r[None, :])

    # Load input (BLOCK_B, R) padded to (BLOCK_B, 16) with zeros
    col_offs = tl.arange(0, 16)   # (16,)
    in_mask = mask_r[:, None] & (col_offs[None, :] < R)

    x_re = tl.load(x_re_ptr + row_offs[:, None] * R + col_offs[None, :],
                   mask=in_mask, other=0.0)   # (BLOCK_B, 16)
    x_im = tl.load(x_im_ptr + row_offs[:, None] * R + col_offs[None, :],
                   mask=in_mask, other=0.0)

    # DFT via padded matmul: (BLOCK_B, 16) @ (16, 16) -> (BLOCK_B, 16)
    y_re, y_im = _cdot(x_re, x_im, MT_re, MT_im)

    # Cast fp32 result to fp16, keep only first R output columns
    y_re = y_re.to(tl.float16)
    y_im = y_im.to(tl.float16)

    out_cols = tl.arange(0, 16)
    out_mask = mask_r[:, None] & (out_cols[None, :] < R)

    if STORE_T:
        # Transposed output (rows//M, R, M): each row maps to (row//M, :, row%M)
        row_outer = row_offs // M   # (BLOCK_B,)
        m_idx     = row_offs % M    # (BLOCK_B,)
        out_addr = (row_outer[:, None] * R * M
                    + out_cols[None, :] * M
                    + m_idx[:, None])
    else:
        # Natural output (rows, R)
        out_addr = row_offs[:, None] * R + out_cols[None, :]

    tl.store(y_re_ptr + out_addr, y_re, mask=out_mask)
    tl.store(y_im_ptr + out_addr, y_im, mask=out_mask)


# =============================================================================
# bailey_scale_kernel: elementwise w_N^{n1 kM} multiply with optional fused T2
# =============================================================================

@triton.jit
def bailey_scale_kernel(
    x_re_ptr, x_im_ptr,     # (rows*m0*M,) fp16 input (logical (rows, m0, M))
    y_re_ptr, y_im_ptr,     # (rows*m0*M,) fp16 output ((rows, m0, M) or (rows, M, m0))
    tw_re_ptr, tw_im_ptr,   # (m0, M) fp16
    m0, M,
    BLOCK_M0: tl.constexpr,
    BLOCK_M: tl.constexpr,
    STORE_T: tl.constexpr,
):
    """Elementwise complex multiply by bt[n1, kM] over the (rows, m0, M) view.
    fp32 arithmetic, fp16 result. STORE_T=True fuses with a transpose to
    produce (rows, M, m0).

    Grid: (cdiv(m0, BLOCK_M0), cdiv(M, BLOCK_M), rows).

    TODO: implement.
    """
    pid_m0  = tl.program_id(0)   # which m0 block
    pid_M   = tl.program_id(1)   # which M block
    pid_row = tl.program_id(2)   # which row

    m0_offs = pid_m0 * BLOCK_M0 + tl.arange(0, BLOCK_M0)   # (BLOCK_M0,)
    M_offs  = pid_M  * BLOCK_M  + tl.arange(0, BLOCK_M)    # (BLOCK_M,)

    mask = (m0_offs[:, None] < m0) & (M_offs[None, :] < M)  # (BLOCK_M0, BLOCK_M)

    # Input: logical (rows, m0, M), row-major
    # flat index = row * m0 * M  +  m0_idx * M  +  M_idx
    in_addr = (pid_row * m0 * M
               + m0_offs[:, None] * M
               + M_offs[None, :])   # (BLOCK_M0, BLOCK_M)

    x_re = tl.load(x_re_ptr + in_addr, mask=mask).to(tl.float32)
    x_im = tl.load(x_im_ptr + in_addr, mask=mask).to(tl.float32)

    # Twiddle: (m0, M), index by [m0_idx, M_idx]
    tw_addr = m0_offs[:, None] * M + M_offs[None, :]
    tw_re = tl.load(tw_re_ptr + tw_addr, mask=mask).to(tl.float32)
    tw_im = tl.load(tw_im_ptr + tw_addr, mask=mask).to(tl.float32)

    # Complex multiply
    out_re = (x_re * tw_re - x_im * tw_im).to(tl.float16)
    out_im = (x_re * tw_im + x_im * tw_re).to(tl.float16)

    if STORE_T:
        # Transpose m0 <-> M: output layout (rows, M, m0)
        # flat index = row * M * m0  +  M_idx * m0  +  m0_idx
        out_addr = (pid_row * M * m0
                    + M_offs[None, :] * m0
                    + m0_offs[:, None])   # (BLOCK_M0, BLOCK_M)
    else:
        # Natural layout (rows, m0, M) — same as input
        out_addr = in_addr

    tl.store(y_re_ptr + out_addr, out_re, mask=mask)
    tl.store(y_im_ptr + out_addr, out_im, mask=mask)


# =============================================================================
# Thin launch wrappers -- GIVEN, do not edit
# =============================================================================

def _transpose(in_re, in_im, out_re, out_im, B, R, C):
    """Logical (B, R, C) -> (B, C, R) transpose, paired re/im."""
    grid = (triton.cdiv(R, TRANSPOSE_BLOCK), triton.cdiv(C, TRANSPOSE_BLOCK), B)
    transpose_kernel[grid](
        in_re, in_im, out_re, out_im, R, C,
        BLOCK_R=TRANSPOSE_BLOCK, BLOCK_C=TRANSPOSE_BLOCK,
    )


def _fft_chunk(in_re, in_im, out_re, out_im, rows, m, plan, M=1, store_t=False):
    """Length-m FFT over `rows` contiguous (rows, m) signals.

    M / store_t control the output layout:
      store_t=False, M=1: natural (rows, m) row-major (F6 leaf path)
      store_t=True,  M>1: transposed (rows//M, m, M) (F7 fused FFT-m0+T3)
    """
    if m == 256:
        f4_plan = plan['f4_plan']
        f4_kernel_L2[(triton.cdiv(rows, F4_L2_BLOCK_B),)](
            in_re.view(rows, 256), in_im.view(rows, 256),
            out_re.view(rows, 256), out_im.view(rows, 256),
            f4_plan['F_re'], f4_plan['F_im'],
            f4_plan['tw_re'], f4_plan['tw_im'],
            rows, M,
            BLOCK_B=F4_L2_BLOCK_B, STAGE_STOP=f4_plan['L'], STORE_T=store_t,
            num_warps=4, num_stages=1,
        )
    else:
        M_re, M_im = plan['dft_mats'][m]
        dft_kernel[(triton.cdiv(rows, DFT_BLOCK_B),)](
            in_re.view(rows, m), in_im.view(rows, m),
            out_re.view(rows, m), out_im.view(rows, m),
            M_re, M_im, rows, M,
            R=m, BLOCK_B=DFT_BLOCK_B, STORE_T=store_t,
        )


def _scale(in_re, in_im, out_re, out_im, rows, m0, M, twr, twi, store_t=False):
    """Bailey scale over logical (rows, m0, M)."""
    grid = (triton.cdiv(m0, SCALE_BLOCK), triton.cdiv(M, SCALE_BLOCK), rows)
    bailey_scale_kernel[grid](
        in_re, in_im, out_re, out_im, twr, twi,
        m0, M, BLOCK_M0=SCALE_BLOCK, BLOCK_M=SCALE_BLOCK, STORE_T=store_t,
    )


def _lookup_tw(plan, m0, M, N_i):
    """Find the precomputed Bailey twiddle table for (m0, M, N_i) in plan['tw']."""
    for (a, b, n, tr, ti) in plan['tw']:
        if a == m0 and b == M and n == N_i:
            return tr, ti
    raise KeyError(f"no twiddle table for (m0={m0}, M={M}, N={N_i})")


# =============================================================================
# F3 pipeline: 4-step Bailey six-step (T1 -> F2-A -> T2 -> F2-B)
# =============================================================================

def f3_launch(in_re, in_im, out_re, out_im, mid_re, mid_im, plan, B):
    """Run the 4-step F3 pipeline. Buffer ping-pong: in -> mid -> out -> mid
    -> out. The Bailey twiddle fuses into F2-A (BAILEY_EPILOGUE=True), and
    the would-be T3 is absorbed by F2-B (STRIDED_STORE=True).

    Steps:
      1. T1 (transpose): x[b, n2, n1] -> A[b, n1, n2]
      2. F2-A:           length-N2 FFT over (B*N1) signals with Bailey epilogue
      3. T2 (transpose): Z[b, n1, k2] -> Z'[b, k2, n1]
      4. F2-B:           length-N1 FFT over (B*N2) signals with strided store

    TODO: implement.
    """

    N1     = plan['N1']
    N2     = plan['N2']
    N      = plan['N']
    LOG2_N1 = plan['LOG2_N1']
    LOG2_N2 = plan['LOG2_N2']

    perm1  = plan['perm_n1']
    perm2  = plan['perm_n2']
    tw_re1 = plan['tw_re_n1']
    tw_im1 = plan['tw_im_n1']
    tw_re2 = plan['tw_re_n2']
    tw_im2 = plan['tw_im_n2']

    bt_re  = plan['bt_re']
    bt_im  = plan['bt_im']

    # Step 1 — T1: (B, N2, N1) -> (B, N1, N2)
    _transpose(in_re, in_im, mid_re, mid_im, B, N2, N1)
    # _transpose(in_re, in_im, mid_re, mid_im, B, N1, N2)
    
    # Step 2 — F2-A: length-N2 FFT over (B*N1) signals, Bailey epilogue
    grid_a = (B * N1,)
    f2_kernel[grid_a](
        mid_re, mid_im, out_re, out_im,
        tw_re2, tw_im2, perm2,
        bt_re, bt_im,
        N1, 0,
        N=N2, LOG2_N=LOG2_N2,
        BAILEY_EPILOGUE=True, STRIDED_STORE=False,
        num_warps=4,
    )

    # Step 3 — T2: (B, N1, N2) -> (B, N2, N1)
    _transpose(out_re, out_im, mid_re, mid_im, B, N1, N2)

    # Step 4 — F2-B: length-N1 FFT over (B*N2) signals, strided store
    grid_b = (B * N2,)
    f2_kernel[grid_b](
        mid_re, mid_im, out_re, out_im,
        tw_re1, tw_im1, perm1,
        tw_re1, tw_im1,   # sentinel — never read
        N2, N,
        N=N1, LOG2_N=LOG2_N1,
        BAILEY_EPILOGUE=False, STRIDED_STORE=True,
        num_warps=4,
    )


# =============================================================================
# F5 pipeline: 6-step Bailey at N1=N2=256 with F4 as inner FFT
# =============================================================================

def f5_launch(in_re, in_im, b0_re, b0_im, b1_re, b1_im, b2_re, b2_im, plan, B):
    """Run the 6-step F5 pipeline at N = 65536 = 256 * 256.

    Buffer ping-pong: in -> b0 -> b1 -> b0 -> b1 -> b2 -> b0 (final).
    The Bailey twiddle is NOT fused into F4 (F4 stays unmodified), so this is
    6 launches; F7 generalizes the fusion idea recursively.

    Steps:
      1. T1:    x[b, n2, n1] -> A[b, n1, n2]
      2. FFT-A: length-256 FFT along last axis -> Y[b, n1, k2]
      3. Scale: Z[b, n1, k2] = Y[b, n1, k2] * bt[n1, k2]
      4. T2:    Z[b, n1, k2] -> Z'[b, k2, n1]
      5. FFT-B: length-256 FFT along last axis -> V[b, k2, k1]
      6. T3:    V[b, k2, k1] -> X[b, k1, k2]   (final in b0)

    TODO: implement.
    """
    N   = plan['N']
    N1  = plan['N1']
    N2  = plan['N2']
    bt_re = plan['bt_re']
    bt_im = plan['bt_im']

    # Step 1 — T1: (B, N2, N1) -> (B, N1, N2)
    _transpose(in_re, in_im, b0_re, b0_im, B, N2, N1)

    # Step 2 — FFT-A: length-N2 FFT over (B*N1) signals
    _fft_chunk(b0_re, b0_im, b1_re, b1_im, B * N1, 256, plan)

    # Step 3 — Scale: Z[b, n1, k2] = Y[b, n1, k2] * bt[n1, k2]
    _scale(b1_re, b1_im, b0_re, b0_im, B, N1, N2, bt_re, bt_im)

    # Step 4 — T2: (B, N1, N2) -> (B, N2, N1)
    _transpose(b0_re, b0_im, b1_re, b1_im, B, N1, N2)

    # Step 5 — FFT-B: length-N1 FFT over (B*N2) signals
    _fft_chunk(b1_re, b1_im, b2_re, b2_im, B * N2, 256, plan)

    # Step 6 — T3: (B, N2, N1) -> (B, N1, N2)  (final in b0)
    _transpose(b2_re, b2_im, b0_re, b0_im, B, N2, N1)


# =============================================================================
# F6 / F7 recursion
# =============================================================================
# Per level i with chunks = [m_0, m_1, ..., m_{p-1}], M = prod(chunks[1:]):
#   T1 :       (rows, M, m_0) -> (rows, m_0, M)
#   recurse:   length-M FFT over (rows*m_0, M)
#   Scale :    y *= w_{N_i}^{n_1 k_M}            (n_1 = the m_0 digit)
#   T2 :       (rows, m_0, M) -> (rows, M, m_0)
#   FFT-m_0 :  length-m_0 FFT over (rows*M, m_0)
#   T3 :       (rows, M, m_0) -> (rows, m_0, M)   [F6 only; F7 fuses]

def _f6_rec(cur_re, cur_im, rows, chunks, plan, cyc):
    """Recursive 2-factor Bailey split. Leaf (len(chunks)==1) is one
    _fft_chunk call; non-leaf is the 6-step pipeline above.

    Returns the (re, im) cycler-managed buffers holding the (rows, prod(chunks))
    FFT result.

    TODO: implement.
    """
    m0 = chunks[0]
    N_i = 1
    for c in chunks:
        N_i *= c

    # Leaf: single FFT chunk, no Bailey split needed
    if len(chunks) == 1:
        nxt_re, nxt_im = cyc.next()
        _fft_chunk(cur_re, cur_im, nxt_re, nxt_im, rows, m0, plan)
        return nxt_re, nxt_im

    M = N_i // m0   # product of chunks[1:]

    # Step 1 — T1: (rows, M, m0) -> (rows, m0, M)
    t1_re, t1_im = cyc.next()
    _transpose(cur_re, cur_im, t1_re, t1_im, rows, M, m0)

    # Step 2 — recurse: length-M FFT over (rows*m0, M)
    rec_re, rec_im = _f6_rec(t1_re, t1_im, rows * m0, chunks[1:], plan, cyc)

    # Step 3 — Scale: multiply by w_{N_i}^{n1 * kM}
    twr, twi = _lookup_tw(plan, m0, M, N_i)
    sc_re, sc_im = cyc.next()
    _scale(rec_re, rec_im, sc_re, sc_im, rows, m0, M, twr, twi)

    # Step 4 — T2: (rows, m0, M) -> (rows, M, m0)
    t2_re, t2_im = cyc.next()
    _transpose(sc_re, sc_im, t2_re, t2_im, rows, m0, M)

    # Step 5 — FFT-m0: length-m0 FFT over (rows*M, m0)
    fft_re, fft_im = cyc.next()
    _fft_chunk(t2_re, t2_im, fft_re, fft_im, rows * M, m0, plan)

    # Step 6 — T3: (rows, M, m0) -> (rows, m0, M)
    t3_re, t3_im = cyc.next()
    _transpose(fft_re, fft_im, t3_re, t3_im, rows, M, m0)

    return t3_re, t3_im



def _f7_rec(cur_re, cur_im, rows, chunks, plan, cyc):
    """Same recursion as _f6_rec but with Scale+T2 fused (store_t=True on
    bailey_scale_kernel) and FFT-m_0+T3 fused (store_t=True, M=M on the inner
    FFT kernel). Output should be bitwise-equal to _f6_rec.

    TODO: implement.
    """
    m0 = chunks[0]
    N_i = 1
    for c in chunks:
        N_i *= c

    # Leaf: same as F6
    if len(chunks) == 1:
        nxt_re, nxt_im = cyc.next()
        _fft_chunk(cur_re, cur_im, nxt_re, nxt_im, rows, m0, plan)
        return nxt_re, nxt_im

    M = N_i // m0

    # Step 1 — T1: (rows, M, m0) -> (rows, m0, M)
    t1_re, t1_im = cyc.next()
    _transpose(cur_re, cur_im, t1_re, t1_im, rows, M, m0)

    # Step 2 — recurse: length-M FFT over (rows*m0, M)
    rec_re, rec_im = _f7_rec(t1_re, t1_im, rows * m0, chunks[1:], plan, cyc)

    # Step 3+4 — Scale+T2 fused: (rows, m0, M) -> (rows, M, m0)
    twr, twi = _lookup_tw(plan, m0, M, N_i)
    sc_re, sc_im = cyc.next()
    _scale(rec_re, rec_im, sc_re, sc_im, rows, m0, M, twr, twi, store_t=True)

    # Step 5+6 — FFT-m0+T3 fused: (rows, M, m0) -> (rows, m0, M)
    fft_re, fft_im = cyc.next()
    _fft_chunk(sc_re, sc_im, fft_re, fft_im, rows * M, m0, plan, M=M, store_t=True)

    return fft_re, fft_im
