"""PyTorch throughput baseline for the MoE trainer in `gpu/model/src/bin/train.rs`.

An external reference for the Rust trainer's tokens/s and MFU: the same 4.39B
parameter model, the same tokens off the same shard, the same measurement
window, and the same MFU denominator, written the way a careful PyTorch
practitioner would write it.

The architecture is transcribed from the CPU reference in `crates/nn/` (which
`gpu/model/src/lib.rs` mirrors on the device), so every structural choice below
has a counterpart there: bias-free linears, pre-norm blocks with RMSNorm,
interleaved-pair RoPE at theta 10000, causal attention, and a top-2-of-8 MoE
whose per-expert capacity drops the tokens that overflow it.

Run it through `modal run modal_app.py::pytorch_baseline`.
"""

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

T = 2_048
VOCAB = 50_257
# The lm-head's output is padded to a tensor-core-friendly width, exactly as
# `GpuBf16Head<D, VP>` pads it; the loss still only sees the first VOCAB
# columns, which is what the fused classifier kernel does with its VOCAB/VP
# pair. Leaving the head at 50257 would cost cuBLAS its aligned epilogue.
VP = 50_432
D = 3_072
H = 24
HD = 128
FF = 4_096
E = 8
K = 2
L = 12
B200_BF16_PEAK_FLOPS = 2.25e15

RMS_EPS = 1e-5
ROPE_THETA = 10_000.0
AUX_BASE_COEFFICIENT = 1e-2
AUX_DECAY_HORIZON = 10_000.0
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8


def training_flops_per_token() -> float:
    """`train.rs`'s formula, transcribed unchanged.

    The MFU denominator has to be identical for the comparison to mean
    anything, so this counts what the Rust counts -- and in particular does not
    consult any of PyTorch's own FLOP counters.
    """
    linear_parameters = D * VP + L * (4 * D * D + D * E + 3 * K * D * FF)
    linear_flops = 6 * linear_parameters
    attention_flops = 12 * L * T * H * HD
    return float(linear_flops + attention_flops)


def aux_coefficient(step: int) -> float:
    """`AuxLossSchedule::coefficient`: linear decay to zero at the horizon."""
    return AUX_BASE_COEFFICIENT * max(1.0 - step / AUX_DECAY_HORIZON, 0.0)


def uniform_(tensor: torch.Tensor, fan_in: int) -> torch.Tensor:
    """`CpuTensor::uniform(seed).scale(1/sqrt(fan_in))`: U[-1, 1) over fan-in."""
    with torch.no_grad():
        return tensor.uniform_(-1.0, 1.0).mul_(1.0 / math.sqrt(fan_in))


class RmsNorm(nn.Module):
    """`nn::RmsNorm`: `x * w / sqrt(mean(x^2) + eps)`, weight initialized to one.

    The weight is an fp32 parameter cast to the activation dtype on the way in,
    so the fused kernel keeps bf16 storage while accumulating the row statistic
    in fp32 -- the split `GpuRmsNorm` uses.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (D,), self.weight.to(x.dtype), RMS_EPS)


def rope_tables() -> tuple[torch.Tensor, torch.Tensor]:
    """`ops::rope_table`: one (cos, sin) per (position, pair), fp32.

    Shaped `[1, T, 1, HD/2]` to broadcast over batch and head.
    """
    pair = torch.arange(HD // 2, dtype=torch.float32)
    position = torch.arange(T, dtype=torch.float32)
    angle = position[:, None] * ROPE_THETA ** (-2.0 * pair[None, :] / HD)
    return angle.cos()[None, :, None, :], angle.sin()[None, :, None, :]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent channel pairs, in fp32, per head.

    `nn::Rope` pairs channels `(2p, 2p+1)` rather than halves, so the reshape
    below splits the head dimension into `HD/2` pairs and not two blocks.
    """
    pairs = x.float().unflatten(-1, (HD // 2, 2))
    even, odd = pairs.unbind(-1)
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2).to(x.dtype)


class Attention(nn.Module):
    """Pre-norm causal multi-head attention with a fused QKV projection.

    The fused projection matches `GpuGroupedLinear<D, 3, D>`, whose weight is
    laid out `[D, 3, D]` -- q, then k, then v along the output axis.
    """

    def __init__(self) -> None:
        super().__init__()
        self.norm = RmsNorm()
        self.qkv_proj = nn.Linear(D, 3 * D, bias=False)
        self.o_proj = nn.Linear(D, D, bias=False)
        uniform_(self.qkv_proj.weight, D)
        uniform_(self.o_proj.weight, D)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = x.shape
        q, k, v = self.qkv_proj(self.norm(x)).chunk(3, dim=-1)
        q = apply_rope(q.unflatten(-1, (H, HD)), cos, sin).transpose(1, 2)
        k = apply_rope(k.unflatten(-1, (H, HD)), cos, sin).transpose(1, 2)
        v = v.unflatten(-1, (H, HD)).transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(attended.transpose(1, 2).reshape(batch, sequence, D))


class MoeFfn(nn.Module):
    """`nn::MoeFfn`: top-K softmax routing into E capacity-limited SwiGLU experts.

    Capacity is fixed at `N * K / E` (a capacity factor of one), and a token
    whose expert bin is already full is dropped -- it contributes nothing to the
    output and receives no gradient, which is what `moe_bin_assign` does once a
    bin's running count reaches C. Slots are handed out in flattened
    `(token, rank)` order, so the assignment here is the one the kernel
    computes.

    Every buffer below has a shape fixed by `capacity`, never by how many tokens
    happened to be kept. That keeps the expert GEMMs dense (and the graph free
    of data-dependent shapes) at the cost of computing padding rows -- the same
    trade the Rust makes, and the reason routing balance does not move the
    measured throughput.
    """

    def __init__(self, capacity: int) -> None:
        super().__init__()
        self.capacity = capacity
        self.norm = RmsNorm()
        # The router stays fp32 on both sides (SPEC decision #22), so it is a
        # bare parameter applied outside autocast rather than an nn.Linear.
        self.router = nn.Parameter(torch.empty(D, E))
        self.gate_up = nn.Parameter(torch.empty(E, D, 2 * FF))
        self.down = nn.Parameter(torch.empty(E, FF, D))
        uniform_(self.router, D)
        uniform_(self.gate_up, D)
        uniform_(self.down, FF)

    def route(self, x: torch.Tensor):
        with torch.autocast("cuda", enabled=False):
            probabilities = (x.float() @ self.router).softmax(-1)
        top_probabilities, experts = probabilities.topk(K, dim=-1)
        gates = (top_probabilities / top_probabilities.sum(-1, keepdim=True)).flatten()
        experts = experts.flatten()

        # Position of each (token, rank) pair within its expert's arrival order,
        # which is its capacity slot as long as it is below C.
        occupancy = F.one_hot(experts, E).to(torch.int32)
        slots = ((occupancy.cumsum(0) - 1) * occupancy).sum(-1)
        kept = (slots < self.capacity).to(gates.dtype)
        # Overflowed pairs are parked on one extra scratch row no expert reads,
        # which is how a fixed-shape scatter says "dropped". A zero gate then
        # keeps them out of both the output and the gradient.
        destinations = torch.where(
            slots < self.capacity, experts * self.capacity + slots, E * self.capacity
        )
        return probabilities, gates * kept, destinations, occupancy.sum(0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, sequence, _ = x.shape
        tokens = batch * sequence
        normalized = self.norm(x).reshape(tokens, D)
        probabilities, gates, destinations, assignments = self.route(normalized)

        sources = torch.arange(tokens, device=x.device).repeat_interleave(K)
        dispatched = normalized.index_select(0, sources)
        binned = normalized.new_zeros(E * self.capacity + 1, D)
        binned = binned.index_copy(0, destinations, dispatched)
        binned = binned[: E * self.capacity].view(E, self.capacity, D)

        gate, up = torch.bmm(binned, self.gate_up).chunk(2, dim=-1)
        expert_output = torch.bmm(F.silu(gate) * up, self.down).reshape(E * self.capacity, D)

        combined = expert_output.index_select(0, destinations.clamp(max=E * self.capacity - 1))
        combined = combined * gates.unsqueeze(-1).to(combined.dtype)
        output = combined.new_zeros(tokens, D).index_add(0, sources, combined)

        # Load balancing: E * sum_e (assignment fraction) * (mean probability).
        # The fraction comes from integer counts and carries no gradient, which
        # is also all the CPU reference routes through it.
        fractions = assignments.float() / (tokens * K)
        aux_loss = E * (fractions * probabilities.mean(0)).sum()
        return output.view(batch, sequence, D), aux_loss


class Block(nn.Module):
    def __init__(self, capacity: int) -> None:
        super().__init__()
        self.attention = Attention()
        self.ffn = MoeFfn(capacity)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x = x + self.attention(x, cos, sin)
        ffn_output, aux_loss = self.ffn(x)
        return x + ffn_output, aux_loss


class MoeDense(nn.Module):
    """`GpuDense`: embedding, L MoE blocks, final norm, padded lm-head."""

    def __init__(self, capacity: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, D)
        self.blocks = nn.ModuleList(Block(capacity) for _ in range(L))
        self.final_norm = RmsNorm()
        self.lm_head = nn.Linear(D, VP, bias=False)
        uniform_(self.embedding.weight, D)
        uniform_(self.lm_head.weight, D)
        with torch.no_grad():
            self.lm_head.weight[VOCAB:].zero_()
        cos, sin = rope_tables()
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, coefficient: torch.Tensor):
        # Autocast leaves an embedding lookup in the parameter dtype, but the
        # Rust carries a bf16 residual stream; one cast here buys that back
        # instead of running every residual add and norm in fp32.
        x = self.embedding(inputs).to(torch.bfloat16)
        aux_loss = torch.zeros((), dtype=torch.float32, device=x.device)
        for block in self.blocks:
            x, block_aux = block(x, self.cos, self.sin)
            aux_loss = aux_loss + block_aux
        logits = self.lm_head(self.final_norm(x))
        # The padded columns are not classes: the fused classifier is handed
        # VOCAB and VP separately and normalizes over VOCAB alone.
        loss = F.cross_entropy(logits.view(-1, VP)[:, :VOCAB], targets.reshape(-1))
        return loss + coefficient * aux_loss


class Shard:
    """The `TOK1` reader from `crates/data/src/shard.rs`, and `Batches`.

    Batches are contiguous `B * T` windows of the token stream with targets
    shifted one token later, so the PyTorch loop sees the same tokens in the
    same order the Rust trainer does.
    """

    HEADER_BYTES = 24
    MAGIC = int.from_bytes(b"TOK1", "little")

    def __init__(self, path: str) -> None:
        header = np.fromfile(path, dtype=np.uint32, count=6)
        if header[0] != self.MAGIC or header[1] != 1 or header[2] != 1:
            raise SystemExit(f"{path} is not a version-1 u16 TOK1 shard")
        count = int(header[4]) | (int(header[5]) << 32)
        self.tokens = np.memmap(
            path, dtype=np.uint16, mode="r", offset=self.HEADER_BYTES, shape=(count,)
        )

    def batch(self, index: int, batch_size: int, device: torch.device):
        span = batch_size * T
        epoch = (len(self.tokens) - 1) // span
        if epoch == 0:
            raise SystemExit(f"shard is too short for a [{batch_size}, {T}] batch")
        start = (index % epoch) * span
        window = torch.from_numpy(np.asarray(self.tokens[start : start + span + 1], dtype=np.int64))
        window = window.pin_memory().to(device, non_blocking=True)
        return window[:-1].view(batch_size, T), window[1:].view(batch_size, T)


class RandomShard:
    """Uniform tokens, for a throughput run without the volume attached.

    Capacity-limited experts compute `E * C` rows whatever the routing looks
    like, so this measures the same work; it is still not the default, because
    only the real shard also gives a loss curve worth reading.
    """

    def __init__(self, seed: int = 0) -> None:
        self.generator = torch.Generator().manual_seed(seed)

    def batch(self, index: int, batch_size: int, device: torch.device):
        window = torch.randint(
            0, VOCAB, (batch_size * T + 1,), generator=self.generator, dtype=torch.int64
        )
        window = window.pin_memory().to(device, non_blocking=True)
        return window[:-1].view(batch_size, T), window[1:].view(batch_size, T)


def parameter_report(model: MoeDense) -> dict[str, int]:
    """The Rust model's parameter count, term by term, against this one's."""
    counts = {
        "embedding": VOCAB * D,
        "lm_head": D * VP,
        "final_norm": D,
        "blocks": L * (2 * D + 4 * D * D + D * E + E * (2 * D * FF + FF * D)),
    }
    counts["expected_total"] = sum(counts.values())
    counts["actual_total"] = sum(p.numel() for p in model.parameters())
    return counts


def attention_backends() -> list[SDPBackend]:
    """Flash if this device has it, cuDNN otherwise, and say which ran."""
    probe = torch.empty(1, H, 256, HD, device="cuda", dtype=torch.bfloat16)
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            F.scaled_dot_product_attention(probe, probe, probe, is_causal=True)
    except RuntimeError as error:
        print(f"attention backend: cudnn (flash unavailable: {error})", flush=True)
        return [SDPBackend.CUDNN_ATTENTION]
    print("attention backend: flash", flush=True)
    return [SDPBackend.FLASH_ATTENTION]


def cudagraph_report() -> dict:
    """Did inductor record CUDA graphs, or quietly decline to?

    `mode="reduce-overhead"` is CUDA graphs, and a comparison against a trainer
    that has none is only honest if the label is checked rather than assumed --
    inductor silently falls back to plain launches when a graph is unsafe.
    """
    report: dict = {"recorded_cuda_graphs": None}
    try:
        from torch._inductor import cudagraph_trees

        manager = cudagraph_trees.get_container(0).tree_manager
        report["recorded_cuda_graphs"] = manager is not None and manager.new_graph_id().id > 0
    except Exception as error:  # a private API; its absence is not a failure
        report["recorded_cuda_graphs"] = f"unavailable: {error}"
    skips = torch._dynamo.utils.counters.get("cudagraph_skips", {})
    report["cudagraph_skips"] = dict(skips)
    return report


def measure(
    model: nn.Module,
    parameters,
    data,
    batch_size: int,
    steps: int,
    warmup: int,
    device: torch.device,
    backends: list[SDPBackend],
) -> dict:
    optimizer = torch.optim.AdamW(
        parameters,
        lr=LEARNING_RATE,
        betas=BETAS,
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
        fused=True,
    )
    # A device scalar, so the decaying coefficient never becomes a guard that
    # torch.compile would recompile against.
    coefficient = torch.zeros((), device=device)
    losses: list[torch.Tensor] = []

    def step(index: int) -> None:
        inputs, targets = data.batch(index, batch_size, device)
        coefficient.fill_(aux_coefficient(index))
        with torch.autocast("cuda", dtype=torch.bfloat16), sdpa_kernel(backends):
            loss = model(inputs, targets, coefficient)
        loss.backward()
        optimizer.step()
        # Zeroing after the step mirrors the Rust, where each AdamW write-back
        # clears the gradient it consumed.
        optimizer.zero_grad(set_to_none=True)
        # Kept on the device: `.item()` inside the timed window would measure
        # the readback too, and the Rust logs outside it.
        losses.append(loss.detach().clone())

    compile_seconds = 0.0
    for index in range(warmup):
        began = time.perf_counter()
        step(index)
        torch.cuda.synchronize()
        if index == 0:
            compile_seconds = time.perf_counter() - began

    start = None
    for index in range(warmup, warmup + steps):
        step(index)
        # The first measured step is discarded, exactly as `train.rs` starts its
        # timer only once step one has landed.
        if index == warmup:
            torch.cuda.synchronize()
            start = time.perf_counter()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    measured_steps = steps - 1
    tokens_per_second = measured_steps * batch_size * T / elapsed
    return {
        "batch": batch_size,
        "tokens_per_second": tokens_per_second,
        "mfu": tokens_per_second * training_flops_per_token() / B200_BF16_PEAK_FLOPS,
        "measured_steps": measured_steps,
        "elapsed": elapsed,
        "flops_per_token": training_flops_per_token(),
        "compile_seconds": compile_seconds,
        "warmup_steps": warmup,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "losses": [loss.item() for loss in losses],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--shard", default="/data/wiki-val-00000.tok")
    parser.add_argument("--random-tokens", action="store_true")
    parser.add_argument("--compile", default="", help="eager when empty, else a torch.compile mode")
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(42)
    print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}", flush=True)

    model = MoeDense(args.batch * T * K // E).to(device)
    counts = parameter_report(model)
    print(f"parameters {json.dumps(counts)}", flush=True)
    if counts["actual_total"] != counts["expected_total"]:
        raise SystemExit("PyTorch parameter count does not match the Rust model")

    tier = f"compile:{args.compile}" if args.compile else "eager"
    compiled = torch.compile(model, mode=args.compile) if args.compile else model

    data = RandomShard() if args.random_tokens else Shard(args.shard)
    result = measure(
        compiled,
        model.parameters(),
        data,
        args.batch,
        args.steps,
        args.warmup,
        device,
        attention_backends(),
    )
    result["tier"] = tier
    result["torch"] = torch.__version__
    result.update(cudagraph_report())
    if not all(math.isfinite(loss) for loss in result["losses"]):
        raise SystemExit(f"non-finite loss: {result['losses']}")
    print(f"losses {[round(loss, 4) for loss in result['losses']]}", flush=True)
    print(
        f"throughput={result['tokens_per_second']:.1f} tokens/s "
        f"mfu={100 * result['mfu']:.2f}% tier={tier} batch={args.batch} "
        f"measured_steps={result['measured_steps']} elapsed={result['elapsed']:.3f}s "
        f"compile={result['compile_seconds']:.1f}s peak_memory={result['peak_memory_gib']:.1f}GiB "
        f"cuda_graphs={result['recorded_cuda_graphs']}",
        flush=True,
    )
    print(f"RESULT {json.dumps(result)}", flush=True)


if __name__ == "__main__":
    main()
