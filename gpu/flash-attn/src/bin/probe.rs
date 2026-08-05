//! Where a flash backward key-tile visit's time goes — the phase budget the
//! flash-backward front opens on.
//!
//! Run on B200 with `./run.sh flash-attn probe`. Prints, per shape and per
//! kernel, the shipped kernel's wall time, the instrumented kernel's wall time
//! (so the stopwatch's own cost is on the record), and the per-visit tick
//! budget from `tcgen05::phase_probe`.
//!
//! The two roles are reported separately because they are the same 128 threads
//! at different times: the leader's column is what it does *before* joining the
//! register pass, and the warpgroup's column is what all four warps do. A visit
//! costs the warpgroup column plus whatever of the leader's the other warps
//! wait out at the block barrier.

use bench_util::{time_gpu_iters, uniform_vec};
use cuda_core::{CudaContext, DeviceBuffer};

#[path = "../host.rs"]
mod host;
#[path = "../tcgen05.rs"]
#[allow(dead_code)]
mod tcgen05_device;

use tcgen05_device as tcgen05;

use host::{
    FLASH_HD, Tcgen05Flash, create_flash_head_tma_map, flash_backward_kv_config,
    flash_backward_q_config,
};
use tcgen05::phase_probe as probe;

const LOG2_E: f32 = std::f32::consts::LOG2_E;

/// `(batches, sequence_length, heads)`. The bench shape, the §13.9 training
/// shape, and a short one — three depths, so a per-item fixed cost and a
/// per-visit slope can be told apart.
const SHAPES: [(usize, usize, usize); 3] = [(32, 1024, 24), (12, 2048, 24), (32, 512, 24)];

fn f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let round = 0x7fffu32 + ((bits >> 16) & 1);
    (bits.wrapping_add(round) >> 16) as u16
}

/// `flash.rs`'s CPU mirror of `stage_attention_heads_bf16`.
fn stage_heads(input: &[f32], b: usize, t: usize, h: usize, scale: f32) -> Vec<u32> {
    let mut staged = vec![0u32; b * h * t * FLASH_HD / 2];
    for plane in 0..b * h {
        let (batch, head) = (plane / h, plane % h);
        for token in 0..t {
            for pair in 0..FLASH_HD / 2 {
                let base = ((batch * t + token) * h + head) * FLASH_HD + pair * 2;
                let low = f32_to_bf16_rne(input[base] * scale) as u32;
                let high = f32_to_bf16_rne(input[base + 1] * scale) as u32;
                staged[(plane * t + token) * FLASH_HD / 2 + pair] = low | (high << 16);
            }
        }
    }
    staged
}

/// One CTA's counter block, averaged over the CTAs that ran an item.
#[derive(Default, Clone, Copy)]
struct Budget {
    visits: f64,
    items: f64,
    feed: f64,
    recycle: f64,
    issue: f64,
    scored: f64,
    grad: f64,
    stat: f64,
    pass: f64,
    sync: f64,
    idle: f64,
    epi: f64,
    span: f64,
    tma: f64,
    smma: f64,
    gmma: f64,
    tread: f64,
    sstore: f64,
    gap: f64,
    fill: f64,
    drainwait: f64,
}

impl Budget {
    fn mean(counters: &[u64]) -> Self {
        let mut sum = Self::default();
        let mut ctas = 0.0f64;
        for block in counters.chunks_exact(probe::COUNTERS) {
            if block[probe::ITEMS] == 0 {
                continue;
            }
            ctas += 1.0;
            sum.visits += block[probe::VISITS] as f64;
            sum.items += block[probe::ITEMS] as f64;
            sum.feed += block[probe::FEED] as f64;
            sum.recycle += block[probe::RECYCLE] as f64;
            sum.issue += block[probe::ISSUE] as f64;
            sum.scored += block[probe::SCORED] as f64;
            sum.grad += block[probe::GRAD] as f64;
            sum.stat += block[probe::STAT] as f64;
            sum.pass += block[probe::PASS] as f64;
            sum.sync += block[probe::SYNC] as f64;
            sum.idle += block[probe::IDLE] as f64;
            sum.epi += block[probe::EPI] as f64;
            sum.span += block[probe::SPAN] as f64;
            sum.tma += block[probe::TMA] as f64;
            sum.smma += block[probe::SMMA] as f64;
            sum.gmma += block[probe::GMMA] as f64;
            sum.tread += block[probe::TREAD] as f64;
            sum.sstore += block[probe::SSTORE] as f64;
            sum.gap += block[probe::GAP] as f64;
            sum.fill += block[probe::FILL] as f64;
            sum.drainwait += block[probe::DRAINWAIT] as f64;
        }
        let ctas = ctas.max(1.0);
        Self {
            visits: sum.visits / ctas,
            items: sum.items / ctas,
            feed: sum.feed / ctas,
            recycle: sum.recycle / ctas,
            issue: sum.issue / ctas,
            scored: sum.scored / ctas,
            grad: sum.grad / ctas,
            stat: sum.stat / ctas,
            pass: sum.pass / ctas,
            sync: sum.sync / ctas,
            idle: sum.idle / ctas,
            epi: sum.epi / ctas,
            span: sum.span / ctas,
            tma: sum.tma / ctas,
            smma: sum.smma / ctas,
            gmma: sum.gmma / ctas,
            tread: sum.tread / ctas,
            sstore: sum.sstore / ctas,
            gap: sum.gap / ctas,
            fill: sum.fill / ctas,
            drainwait: sum.drainwait / ctas,
        }
    }

    /// The issue warp's whole tile, its wait at the shared barrier included.
    fn issuer(&self) -> f64 {
        self.feed + self.recycle + self.issue + self.idle
    }

    /// The pass warps' whole tile, their wait at the same barrier included.
    /// The two roles meet there every tile, so these two should agree — and
    /// which of `IDLE` and `SYNC` carries the slack says which role is the
    /// critical path.
    fn warpgroup(&self) -> f64 {
        self.scored + self.grad + self.stat + self.pass + self.sync
    }

    /// The item's own span, less what the per-visit columns account for — the
    /// per-item fixed cost the depth sweep is for.
    fn residual(&self) -> f64 {
        let per_item = self.visits / self.items.max(1.0);
        self.span - self.epi - per_item * self.warpgroup()
    }

    fn report(&self, name: &str, shipped_ms: f64, probe_ms: f64, visits_total: f64) {
        let per_visit = self.visits / self.items.max(1.0);
        println!("  {name}");
        println!(
            "    items/CTA {:.1}, visits/item {:.2}, probe {probe_ms:.3} ms vs shipped \
             {shipped_ms:.3} ms ({:+.1}%)",
            self.items,
            per_visit,
            (probe_ms / shipped_ms - 1.0) * 100.0,
        );
        // Ticks are the SM's own clock; the shipped kernel's wall time over the
        // visits one SM makes converts them, so ns are quoted against the
        // SHIPPED kernel rather than the perturbed one. The implied clock is
        // the sanity check: a plausible GHz says the two roles between them
        // account for the visit, and a low one says something outside both
        // columns does.
        let ns_per_visit = shipped_ms * 1.0e6 / (visits_total / 148.0);
        let ghz = self.warpgroup() / ns_per_visit;
        println!(
            "    pass warps {:>7.0} ticks/visit   issue warp {:>7.0}   span/item {:>8.0}   \
             residual/item {:>7.0}",
            self.warpgroup(),
            self.issuer(),
            self.span,
            self.residual(),
        );
        let share = |ticks: f64| ticks / self.warpgroup().max(1.0) * 100.0;
        println!(
            "      warpgroup: SCORED {:>6.0} ({:>4.1}%)  GRAD {:>6.0} ({:>4.1}%)  \
             STAT {:>6.0} ({:>4.1}%)  PASS {:>6.0} ({:>4.1}%)  SYNC {:>6.0} ({:>4.1}%)",
            self.scored,
            share(self.scored),
            self.grad,
            share(self.grad),
            self.stat,
            share(self.stat),
            self.pass,
            share(self.pass),
            self.sync,
            share(self.sync),
        );
        println!(
            "      issue warp: FEED  {:>6.0}          RECYCLE {:>6.0}         ISSUE {:>6.0}   \
             IDLE {:>6.0}\n      per item:   EPI   {:>6.0}",
            self.feed, self.recycle, self.issue, self.idle, self.epi,
        );
        // The two splits, printed only where the probe fills them (kernel A).
        // `TMA + SMMA + GMMA` partitions `ISSUE` and `TREAD + ARITH + SSTORE`
        // partitions `PASS`, so each line is a decomposition and not a sample.
        if self.smma > 0.0 {
            let arith = self.pass - self.tread - self.sstore;
            println!(
                "      ISSUE split: TMA   {:>6.0}          SMMA    {:>6.0}         GMMA  \
                 {:>6.0}\n      PASS split:  TREAD {:>6.0}          ARITH   {:>6.0}         \
                 STORE {:>6.0}\n      per item:   FILL  {:>6.0}          GAP     {:>6.0}         \
                 EPI = wait {:>4.0} + drain {:>4.0}",
                self.tma,
                self.smma,
                self.gmma,
                self.tread,
                arith,
                self.sstore,
                self.fill,
                self.gap,
                self.drainwait,
                self.epi - self.drainwait,
            );
        }
        println!(
            "      shipped ns/visit {ns_per_visit:.0} at 1 CTA/SM over 148 SMs \
             (implied SM clock {ghz:.2} GHz)",
        );
    }
}

/// Key-tile visits both kernels make over a whole shape: `nt*(nt+1)/2` causal
/// 64-tile pairs per plane, which each kernel walks at 128-row granularity, so
/// `2 * sum(2i+2)` — the same count for A and B.
fn visits(b: usize, t: usize, h: usize) -> f64 {
    let blocks = t / 128;
    let per_plane: usize = (0..blocks).map(|i| 2 * i + 2).sum();
    (per_plane * b * h) as f64
}

/// The cadence probe's variants, in the order the kernel writes them: the
/// three widths of the score MMA's own walk, then kernel A's gradient MMA both
/// ways. `units` is the variant's work in `M128_N64_K16` instructions, so the
/// last column is directly comparable across widths.
/// `(name, instructions a round, M128_N64_K16-equivalents a round)`.
const CADENCE: [(&str, f64, f64); 5] = [
    ("mma_abt  M128_N64   K=128", 8.0, 8.0),
    ("mma_abt  M128_N128  K=128", 8.0, 16.0),
    ("mma_abt  M128_N256  K=128", 8.0, 32.0),
    ("mma_ab   M128_N64x2 K=64 (dQ, shipped)", 8.0, 8.0),
    ("mma_ab   M128_N128  K=64 (dQ, one band)", 4.0, 8.0),
];

fn report_cadence(counters: &[u64], rounds: u32) {
    println!("mma cadence — one warp, one CTA, {rounds} rounds a variant\n");
    println!(
        "  {:<41} {:>7} {:>10} {:>10} {:>10}",
        "variant", "instrs", "issue/ins", "retire/ins", "ticks/unit"
    );
    for (variant, (name, per_round, units)) in CADENCE.iter().enumerate() {
        let block = &counters[variant * probe::CADENCE_SLOTS..];
        let (issued, retired, latency, instructions) = (
            block[0] as f64,
            block[1] as f64,
            block[2] as f64,
            block[3] as f64,
        );
        if instructions == 0.0 {
            continue;
        }
        println!(
            "  {name:<41} {instructions:>7.0} {:>10.1} {:>10.1} {:>10.1}   \
             (drained chain {latency:.0} ticks)",
            issued / instructions,
            retired / instructions,
            retired / (instructions / per_round * units),
        );
    }
    println!();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let stream = &stream;
    let flash = Tcgen05Flash::load(&ctx)?;
    let sm_count = flash.sm_count();
    println!("flash backward phase budget, {sm_count} SMs\n");

    const ROUNDS: u32 = 256;
    let mut cadence = DeviceBuffer::<u64>::zeroed(stream, probe::CADENCE_COUNTERS)?;
    unsafe { flash.mma_cadence(stream, ROUNDS, &mut cadence)? };
    unsafe { flash.mma_cadence(stream, ROUNDS, &mut cadence)? };
    report_cadence(&cadence.to_host_vec(stream)?, ROUNDS);

    for (b, t, h) in SHAPES {
        let d = h * FLASH_HD;
        let n = b * t;
        let q_scale = LOG2_E / (FLASH_HD as f32).sqrt();
        let q = DeviceBuffer::from_host(
            stream,
            &stage_heads(&uniform_vec(n * d, 81), b, t, h, q_scale),
        )?;
        let k =
            DeviceBuffer::from_host(stream, &stage_heads(&uniform_vec(n * d, 82), b, t, h, 1.0))?;
        let v =
            DeviceBuffer::from_host(stream, &stage_heads(&uniform_vec(n * d, 83), b, t, h, 1.0))?;
        let dy =
            DeviceBuffer::from_host(stream, &stage_heads(&uniform_vec(n * d, 84), b, t, h, 1.0))?;
        let q_tma = unsafe { create_flash_head_tma_map(stream, &q, t, b * h)? };
        let k_tma = unsafe { create_flash_head_tma_map(stream, &k, t, b * h)? };
        let v_tma = unsafe { create_flash_head_tma_map(stream, &v, t, b * h)? };
        let dy_tma = unsafe { create_flash_head_tma_map(stream, &dy, t, b * h)? };
        let lse = DeviceBuffer::from_host(stream, &vec![0.0f32; n * h])?;
        let dot = DeviceBuffer::from_host(stream, &vec![0.0f32; n * h])?;
        let mut dq = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
        let mut dk = DeviceBuffer::<f32>::zeroed(stream, n * d)?;
        let mut dv = DeviceBuffer::<f32>::zeroed(stream, n * d)?;

        let q_config = flash_backward_q_config(b, t, h, sm_count);
        let kv_config = flash_backward_kv_config(b, t, h, sm_count);
        let ctas = q_config.grid_dim.0 as usize;
        let mut clocks = DeviceBuffer::<u64>::zeroed(stream, probe::COUNTERS * ctas)?;
        let total_visits = visits(b, t, h);

        println!("[{b},{t},{h},{FLASH_HD}] — {ctas} CTAs, {total_visits:.0} tile visits");

        let shipped_q = time_gpu_iters(stream, 3, 20, || {
            unsafe {
                flash.backward_q(
                    stream,
                    q_config,
                    q_tma.as_ptr(),
                    k_tma.as_ptr(),
                    v_tma.as_ptr(),
                    dy_tma.as_ptr(),
                    &lse,
                    &dot,
                    t as u32,
                    h as u32,
                    b as u32,
                    &mut dq,
                )?;
            }
            Ok(())
        })?;
        let probe_q = time_gpu_iters(stream, 3, 20, || {
            unsafe {
                flash.backward_q_probe(
                    stream,
                    q_config,
                    q_tma.as_ptr(),
                    k_tma.as_ptr(),
                    v_tma.as_ptr(),
                    dy_tma.as_ptr(),
                    &lse,
                    &dot,
                    t as u32,
                    h as u32,
                    b as u32,
                    &mut dq,
                    &mut clocks,
                )?;
            }
            Ok(())
        })?;
        Budget::mean(&clocks.to_host_vec(stream)?).report(
            "backward q (query-parallel, dQ)",
            shipped_q,
            probe_q,
            total_visits,
        );

        let mut clocks = DeviceBuffer::<u64>::zeroed(stream, probe::COUNTERS * ctas)?;
        let shipped_kv = time_gpu_iters(stream, 3, 20, || {
            unsafe {
                flash.backward_kv(
                    stream,
                    kv_config,
                    q_tma.as_ptr(),
                    k_tma.as_ptr(),
                    v_tma.as_ptr(),
                    dy_tma.as_ptr(),
                    &lse,
                    &dot,
                    t as u32,
                    h as u32,
                    b as u32,
                    &mut dk,
                    &mut dv,
                )?;
            }
            Ok(())
        })?;
        let probe_kv = time_gpu_iters(stream, 3, 20, || {
            unsafe {
                flash.backward_kv_probe(
                    stream,
                    kv_config,
                    q_tma.as_ptr(),
                    k_tma.as_ptr(),
                    v_tma.as_ptr(),
                    dy_tma.as_ptr(),
                    &lse,
                    &dot,
                    t as u32,
                    h as u32,
                    b as u32,
                    &mut dk,
                    &mut dv,
                    &mut clocks,
                )?;
            }
            Ok(())
        })?;
        Budget::mean(&clocks.to_host_vec(stream)?).report(
            "backward kv (key-parallel, dK/dV)",
            shipped_kv,
            probe_kv,
            total_visits,
        );
        println!();
    }

    Ok(())
}
