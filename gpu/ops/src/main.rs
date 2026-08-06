//! CPU/GPU parity checks for the reference Dense kernels.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use nn::{
    CausalAttention, Embedding, ExpertFfn, Linear, Module, MoeFfn, RmsNorm, Rope,
    SoftmaxCrossEntropy, SoftmaxCrossEntropyInput, SwiGlu,
};
use tensor_core::{Rank1, Rank2, Rank3};
use tensor_cpu::CpuTensor;

// `cargo oxide` collects kernels from the selected binary target, not from a
// separately compiled library dependency. Reuse the canonical library source
// as a module so this binary's embedded artifact contains the kernels.
#[path = "lib.rs"]
mod device;
use device::{
    CLASSIFIER_THREADS, LOSS_TAIL_THREADS, MOE_ASSIGN_THREADS, MOE_AUX_TERMS_THREADS,
    MOE_DROPPED_SLOT, MOE_SCATTER_DY_THREADS, MOE_ZERO_BINS_BLOCKS, MOE_ZERO_BINS_THREADS,
    NORM_THREADS, NORM_TILE_BLOCK_ROWS, NORM_TILE_CHUNK, NORM_TILE_THREADS,
    NORM_WEIGHT_ROWS_PER_BLOCK, ROUTER_GEMM_BM, ROUTER_GEMM_BN, ROUTER_GEMM_THREADS,
    ROUTER_INPUT_BN, ROUTER_INPUT_THREADS, ROUTER_INPUT_TOKENS, ROUTER_WGRAD_BM,
    ROUTER_WGRAD_SPLITS, ROUTER_WGRAD_THREADS, SWIGLU_TILE_BLOCK_ROWS, SWIGLU_TILE_CHUNK,
    SWIGLU_TILE_THREADS, kernels, rope_table,
};
use tensor_core::bf16;

/// Launch for the dead-slot zeroing checks, clamped the way production clamps
/// it. The checks pass one block per expert — deliberately fewer than the
/// capacity — so what they cover is a block striding its expert's dead tail
/// rather than the one-block-per-slot special case.
fn zero_dead_bins_config<const E: usize>(blocks_per_expert: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (
            blocks_per_expert.min(MOE_ZERO_BINS_BLOCKS) as u32,
            E as u32,
            1,
        ),
        block_dim: (MOE_ZERO_BINS_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn assert_close(name: &str, actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let tolerance = atol + rtol * e.abs();
        assert!(
            (a - e).abs() <= tolerance,
            "{name} mismatch at {i}: gpu={a}, cpu={e}, tolerance={tolerance}"
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    eprintln!("[ops] checking rms_norm");
    check_rms_norm(&stream, &module)?;
    eprintln!("[ops] checking rms_norm tiles");
    check_rms_norm_tile(&stream, &module)?;
    eprintln!("[ops] checking swiglu");
    check_swiglu(&stream, &module)?;
    eprintln!("[ops] checking swiglu tiles");
    check_swiglu_tile(&stream, &module)?;
    eprintln!("[ops] checking swiglu interleaved");
    check_swiglu_interleaved(&stream, &module)?;
    eprintln!("[ops] checking embedding");
    check_embedding(&stream, &module)?;
    eprintln!("[ops] checking cross_entropy");
    check_cross_entropy(&stream, &module)?;
    eprintln!("[ops] checking classifier_bf16");
    check_classifier_bf16(&stream, &module)?;
    eprintln!("[ops] checking rope");
    check_rope(&stream, &module)?;
    eprintln!("[ops] checking attention");
    check_attention(&stream, &module)?;
    eprintln!("[ops] checking group_split_join");
    check_group_split_join(&stream, &module)?;
    eprintln!("[ops] checking moe_routing");
    check_moe_routing(&stream, &module)?;
    eprintln!("[ops] checking loss tail");
    check_loss_tail(&stream, &module)?;

    println!("✓ ops forward/backward parity checks passed");
    Ok(())
}

#[allow(unused_unsafe)]
fn check_moe_routing(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    // SAFETY: every launch in this check derives its buffer sizes and launch
    // geometry from the N/D/FF/E/K/C constants below.
    unsafe {
        const N: usize = 6;
        const D: usize = 4;
        const FF: usize = 5;
        const E: usize = 3;
        const K: usize = 2;
        const C: usize = 2;
        const AUX: f32 = 0.17;

        let x = CpuTensor::<f32, Rank2<N, D>>::from_slice(&[
            1.0, 0.1, 0.2, 0.3, //
            0.9, 0.3, 0.1, 0.2, //
            1.1, 0.2, 0.4, 0.1, //
            0.8, 0.4, 0.2, 0.5, //
            1.2, 0.2, 0.3, 0.4, //
            0.7, 0.5, 0.1, 0.3,
        ]);
        let router_weight = CpuTensor::<f32, Rank2<D, E>>::from_slice(&[
            1.0, 0.7, -1.0, //
            0.2, 0.4, -0.5, //
            0.1, -0.2, -0.5, //
            0.3, 0.1, -0.5,
        ]);
        let experts = std::array::from_fn(|expert| ExpertFfn::initialized(300 + 3 * expert as u64));
        let mut cpu =
            MoeFfn::<N, D, FF, E, K, C>::new(Linear::new(router_weight.clone()), experts, AUX);
        let (cpu_output, cpu_ctx) = cpu.forward(x.clone());
        assert!(
            cpu_ctx.routing.slots.iter().any(Option::is_none),
            "MoE parity shape must force capacity drops"
        );
        assert!(
            cpu_ctx.routing.accepted_counts.contains(&0),
            "MoE parity shape must leave an expert underfull"
        );

        let cpu_logits = x.matmul(&router_weight);
        let cpu_probabilities = cpu_logits.softmax_rows();
        let expected_selected: Vec<u32> = cpu_ctx
            .routing
            .selected_experts
            .iter()
            .map(|&expert| expert as u32)
            .collect();
        let expected_slots: Vec<u32> = cpu_ctx
            .routing
            .slots
            .iter()
            .map(|slot| slot.map_or(MOE_DROPPED_SLOT, |slot| slot as u32))
            .collect();
        let expected_counts: Vec<u32> = cpu_ctx
            .routing
            .assignment_counts
            .iter()
            .map(|&count| count as u32)
            .collect();
        let mut expert_outputs = Vec::with_capacity(E * C * D);
        for expert in &cpu_ctx.expert_outputs {
            expert_outputs.extend_from_slice(expert.as_slice());
        }
        let mut expected_expert_input = vec![0.0f32; E * C * D];
        for token in 0..N {
            for rank in 0..K {
                let pair = token * K + rank;
                let slot = expected_slots[pair];
                if slot == MOE_DROPPED_SLOT {
                    continue;
                }
                let expert = expected_selected[pair] as usize;
                let output = (expert * C + slot as usize) * D;
                expected_expert_input[output..output + D]
                    .copy_from_slice(&x.as_slice()[token * D..(token + 1) * D]);
            }
        }

        let x_dev = DeviceBuffer::from_host(stream, x.as_slice())?;
        let weight_dev = DeviceBuffer::from_host(stream, router_weight.as_slice())?;
        let mut logits_dev = DeviceBuffer::<f32>::zeroed(stream, N * E)?;
        let mut probabilities_dev = DeviceBuffer::<f32>::zeroed(stream, N * E)?;
        let mut selected_dev = DeviceBuffer::<u32>::zeroed(stream, N * K)?;
        let mut gates_dev = DeviceBuffer::<f32>::zeroed(stream, N * K)?;
        let mut slots_dev = DeviceBuffer::<u32>::zeroed(stream, N * K)?;
        let mut counts_dev = DeviceBuffer::<u32>::zeroed(stream, E)?;
        let mut serial_slots_dev = DeviceBuffer::<u32>::zeroed(stream, N * K)?;
        let mut serial_counts_dev = DeviceBuffer::<u32>::zeroed(stream, E)?;
        let mut expert_input_dev = DeviceBuffer::<f32>::zeroed(stream, E * C * D)?;
        let expert_output_dev = DeviceBuffer::from_host(stream, &expert_outputs)?;
        let mut output_dev = DeviceBuffer::<f32>::zeroed(stream, N * D)?;

        unsafe {
            module.router_logits(
                stream,
                LaunchConfig {
                    grid_dim: (
                        (E as u32).div_ceil(ROUTER_GEMM_BN as u32),
                        (N as u32).div_ceil(ROUTER_GEMM_BM as u32),
                        1,
                    ),
                    block_dim: (ROUTER_GEMM_THREADS as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                &x_dev,
                &weight_dev,
                D as u32,
                E as u32,
                &mut logits_dev,
            )?;
        }
        unsafe {
            module.router_softmax_topk(
                stream,
                LaunchConfig::for_num_elems(N as u32),
                &logits_dev,
                E as u32,
                K as u32,
                &mut probabilities_dev,
                &mut selected_dev,
                &mut gates_dev,
            )?;
            module.moe_bin_assign(
                stream,
                LaunchConfig {
                    grid_dim: (E as u32, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                &selected_dev,
                N as u32,
                E as u32,
                K as u32,
                C as u32,
                &mut serial_slots_dev,
                &mut serial_counts_dev,
            )?;
            module.moe_bin_assign_parallel(
                stream,
                LaunchConfig {
                    grid_dim: (E as u32, 1, 1),
                    block_dim: (MOE_ASSIGN_THREADS as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                &selected_dev,
                N as u32,
                E as u32,
                K as u32,
                C as u32,
                &mut slots_dev,
                &mut counts_dev,
            )?;
            module.moe_scatter(
                stream,
                LaunchConfig::for_num_elems((N * K * D) as u32),
                &x_dev,
                &selected_dev,
                &slots_dev,
                D as u32,
                K as u32,
                C as u32,
                &mut expert_input_dev,
            )?;
        }
        module.moe_gather_combine(
            stream,
            LaunchConfig::for_num_elems((N * D) as u32),
            &expert_output_dev,
            &selected_dev,
            &gates_dev,
            &slots_dev,
            D as u32,
            K as u32,
            C as u32,
            &mut output_dev,
        )?;

        assert_close(
            "MoE router logits",
            &logits_dev.to_host_vec(stream)?,
            cpu_logits.as_slice(),
            1e-6,
            1e-6,
        );
        assert_close(
            "MoE router probabilities",
            &probabilities_dev.to_host_vec(stream)?,
            cpu_probabilities.as_slice(),
            1e-6,
            1e-6,
        );
        assert_eq!(selected_dev.to_host_vec(stream)?, expected_selected);
        assert_eq!(
            slots_dev.to_host_vec(stream)?,
            serial_slots_dev.to_host_vec(stream)?,
            "parallel MoE assignment slots must match the serial GPU oracle"
        );
        assert_eq!(
            counts_dev.to_host_vec(stream)?,
            serial_counts_dev.to_host_vec(stream)?,
            "parallel MoE assignment counts must match the serial GPU oracle"
        );
        assert_close(
            "MoE gate weights",
            &gates_dev.to_host_vec(stream)?,
            &cpu_ctx.routing.gate_weights,
            1e-6,
            1e-6,
        );
        assert_eq!(slots_dev.to_host_vec(stream)?, expected_slots);
        assert_eq!(counts_dev.to_host_vec(stream)?, expected_counts);
        assert_eq!(
            expert_input_dev.to_host_vec(stream)?,
            expected_expert_input,
            "MoE scatter must preserve accepted rows and zero-fill unused slots"
        );
        let mut roundtrip_dev = DeviceBuffer::<f32>::zeroed(stream, N * D)?;
        module.moe_gather_combine(
            stream,
            LaunchConfig::for_num_elems((N * D) as u32),
            &expert_input_dev,
            &selected_dev,
            &gates_dev,
            &slots_dev,
            D as u32,
            K as u32,
            C as u32,
            &mut roundtrip_dev,
        )?;
        let mut expected_roundtrip = vec![0.0f32; N * D];
        for token in 0..N {
            for rank in 0..K {
                let pair = token * K + rank;
                if expected_slots[pair] != MOE_DROPPED_SLOT {
                    for column in 0..D {
                        expected_roundtrip[token * D + column] +=
                            cpu_ctx.routing.gate_weights[pair] * x.as_slice()[token * D + column];
                    }
                }
            }
        }
        assert_close(
            "MoE scatter/gather round trip",
            &roundtrip_dev.to_host_vec(stream)?,
            &expected_roundtrip,
            1e-6,
            1e-6,
        );
        assert_close(
            "MoE surviving-token round trip",
            &expected_roundtrip[..2 * D],
            &x.as_slice()[..2 * D],
            1e-6,
            1e-6,
        );
        assert_eq!(&expected_roundtrip[2 * D..], &[0.0; (N - 2) * D]);
        assert_close(
            "MoE gather/combine",
            &output_dev.to_host_vec(stream)?,
            cpu_output.as_slice(),
            1e-6,
            1e-6,
        );

        let dy = CpuTensor::<f32, Rank2<N, D>>::uniform(400);
        let dy_dev = DeviceBuffer::from_host(stream, dy.as_slice())?;
        // Poisoned, not zeroed: the dead-slot pass plus the scatter must between
        // them rewrite every bin, so a surviving poison value fails the compare.
        let poison: Vec<f32> = (0..E * C * D).map(|index| index as f32 + 1.0).collect();
        let mut expert_output_gradient_dev = DeviceBuffer::from_host(stream, &poison)?;
        let mut gate_gradients_dev = DeviceBuffer::<f32>::zeroed(stream, N * K)?;
        unsafe {
            module.moe_zero_dead_bins(
                stream,
                zero_dead_bins_config::<E>(1),
                &counts_dev,
                D as u32,
                C as u32,
                &mut expert_output_gradient_dev,
            )?;
            module.moe_scatter_dy(
                stream,
                LaunchConfig {
                    grid_dim: ((N * K) as u32, 1, 1),
                    block_dim: (MOE_SCATTER_DY_THREADS as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                &expert_output_dev,
                &dy_dev,
                &selected_dev,
                &gates_dev,
                &slots_dev,
                D as u32,
                K as u32,
                C as u32,
                &mut expert_output_gradient_dev,
                &mut gate_gradients_dev,
            )?;
        }
        let mut expected_expert_output_gradient = vec![0.0f32; E * C * D];
        let mut expected_gate_gradients = vec![0.0f32; N * K];
        for token in 0..N {
            for rank in 0..K {
                let pair = token * K + rank;
                let slot = expected_slots[pair];
                if slot == MOE_DROPPED_SLOT {
                    continue;
                }
                let expert = expected_selected[pair] as usize;
                let bin_base = (expert * C + slot as usize) * D;
                let token_base = token * D;
                for column in 0..D {
                    expected_expert_output_gradient[bin_base + column] =
                        cpu_ctx.routing.gate_weights[pair] * dy.as_slice()[token_base + column];
                    expected_gate_gradients[pair] +=
                        expert_outputs[bin_base + column] * dy.as_slice()[token_base + column];
                }
            }
        }
        assert_close(
            "MoE expert output gradient scatter",
            &expert_output_gradient_dev.to_host_vec(stream)?,
            &expected_expert_output_gradient,
            1e-6,
            1e-6,
        );
        assert_close(
            "MoE gate gradients",
            &gate_gradients_dev.to_host_vec(stream)?,
            &expected_gate_gradients,
            1e-6,
            1e-6,
        );

        let expert_input_gradient: Vec<f32> = (0..E * C * D)
            .map(|index| index as f32 * 0.03125 - 0.5)
            .collect();
        let expert_input_gradient_dev = DeviceBuffer::from_host(stream, &expert_input_gradient)?;
        let mut expert_dx_dev = DeviceBuffer::<f32>::zeroed(stream, N * D)?;
        module.moe_gather_dx(
            stream,
            LaunchConfig::for_num_elems((N * D) as u32),
            &expert_input_gradient_dev,
            &selected_dev,
            &slots_dev,
            D as u32,
            K as u32,
            C as u32,
            &mut expert_dx_dev,
        )?;
        let mut expected_expert_dx = vec![0.0f32; N * D];
        for token in 0..N {
            for rank in 0..K {
                let pair = token * K + rank;
                let slot = expected_slots[pair];
                if slot == MOE_DROPPED_SLOT {
                    continue;
                }
                let expert = expected_selected[pair] as usize;
                let bin_base = (expert * C + slot as usize) * D;
                for column in 0..D {
                    expected_expert_dx[token * D + column] +=
                        expert_input_gradient[bin_base + column];
                }
            }
        }
        assert_eq!(
            expert_dx_dev.to_host_vec(stream)?,
            expected_expert_dx,
            "MoE gather dx must sum surviving top-k paths and skip drops"
        );

        let mut dlogits_dev = DeviceBuffer::<f32>::zeroed(stream, N * E)?;
        let mut router_dx_dev = DeviceBuffer::<f32>::zeroed(stream, N * D)?;
        let mut router_dweight_dev = DeviceBuffer::<f32>::zeroed(stream, D * E)?;
        let mut router_dweight_partials_dev =
            DeviceBuffer::<f32>::zeroed(stream, ROUTER_WGRAD_SPLITS * E * D)?;
        let mut serial_router_dweight_dev = DeviceBuffer::<f32>::zeroed(stream, D * E)?;
        unsafe {
            module.router_backward(
                stream,
                LaunchConfig::for_num_elems(N as u32),
                &probabilities_dev,
                &selected_dev,
                &gates_dev,
                &gate_gradients_dev,
                &counts_dev,
                N as u32,
                E as u32,
                K as u32,
                AUX,
                &mut dlogits_dev,
            )?;
        }
        unsafe {
            module.router_backward_input(
                stream,
                LaunchConfig {
                    grid_dim: (
                        D.div_ceil(ROUTER_INPUT_BN) as u32,
                        N.div_ceil(ROUTER_INPUT_TOKENS) as u32,
                        1,
                    ),
                    block_dim: (ROUTER_INPUT_THREADS as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                &dlogits_dev,
                &weight_dev,
                E as u32,
                &mut router_dx_dev,
            )?;
        }
        module.router_backward_weight(
            stream,
            LaunchConfig::for_num_elems((D * E) as u32),
            &x_dev,
            &dlogits_dev,
            N as u32,
            E as u32,
            &mut serial_router_dweight_dev,
        )?;
        let router_wgrad_config = LaunchConfig {
            grid_dim: (
                D.div_ceil(ROUTER_WGRAD_BM) as u32,
                ROUTER_WGRAD_SPLITS as u32,
                1,
            ),
            block_dim: (ROUTER_WGRAD_THREADS as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            module.router_backward_weight_split(
                stream,
                router_wgrad_config,
                &x_dev,
                &dlogits_dev,
                N as u32,
                E as u32,
                D as u32,
                &mut router_dweight_partials_dev,
            )?;
            module.router_backward_weight_merge(
                stream,
                LaunchConfig::for_num_elems((D * E) as u32),
                &router_dweight_partials_dev,
                E as u32,
                &mut router_dweight_dev,
            )?;
        }
        assert_close(
            "split MoE router weight gradient vs serial GPU oracle",
            &router_dweight_dev.to_host_vec(stream)?,
            &serial_router_dweight_dev.to_host_vec(stream)?,
            2e-6,
            2e-6,
        );
        // The split reduction owes its determinism to a fixed order, not to a
        // fixed schedule: relaunching must reproduce the gradient bit for bit.
        let mut repeat_dweight_dev = DeviceBuffer::<f32>::zeroed(stream, D * E)?;
        unsafe {
            module.router_backward_weight_split(
                stream,
                router_wgrad_config,
                &x_dev,
                &dlogits_dev,
                N as u32,
                E as u32,
                D as u32,
                &mut router_dweight_partials_dev,
            )?;
            module.router_backward_weight_merge(
                stream,
                LaunchConfig::for_num_elems((D * E) as u32),
                &router_dweight_partials_dev,
                E as u32,
                &mut repeat_dweight_dev,
            )?;
        }
        assert_eq!(
            repeat_dweight_dev.to_host_vec(stream)?,
            router_dweight_dev.to_host_vec(stream)?,
            "split MoE router weight gradient must be bit-identical across launches"
        );
        cpu.backward(cpu_ctx, dy);
        assert_close(
            "MoE router weight gradient including aux",
            &router_dweight_dev.to_host_vec(stream)?,
            cpu.router.dw.as_slice(),
            2e-6,
            2e-6,
        );
        let dlogits = CpuTensor::<f32, Rank2<N, E>>::from_slice(&dlogits_dev.to_host_vec(stream)?);
        let expected_router_dx = dlogits.matmul_nt(&router_weight);
        assert_close(
            "MoE router input gradient",
            &router_dx_dev.to_host_vec(stream)?,
            expected_router_dx.as_slice(),
            2e-6,
            2e-6,
        );

        // Fused gathers: the residual (forward) and router-dx (backward) adds
        // folded in, in both the scalar and 16-byte-quad walks (D = 4 divides
        // by four, so the quad kernels cover every lane).
        let residual = CpuTensor::<f32, Rank2<N, D>>::uniform(401);
        let residual_dev = DeviceBuffer::from_host(stream, residual.as_slice())?;
        let expected_combine_add: Vec<f32> = residual
            .as_slice()
            .iter()
            .zip(cpu_output.as_slice())
            .map(|(&base, &combined)| base + combined)
            .collect();
        let mut combine_add_dev = DeviceBuffer::<f32>::zeroed(stream, N * D)?;
        module.moe_gather_combine_add(
            stream,
            LaunchConfig::for_num_elems((N * D) as u32),
            &expert_output_dev,
            &selected_dev,
            &gates_dev,
            &slots_dev,
            &residual_dev,
            D as u32,
            K as u32,
            C as u32,
            &mut combine_add_dev,
        )?;
        assert_close(
            "MoE fused gather/combine + residual",
            &combine_add_dev.to_host_vec(stream)?,
            &expected_combine_add,
            1e-6,
            1e-6,
        );
        let mut combine_add_quad_dev = DeviceBuffer::<f32>::zeroed(stream, N * D)?;
        unsafe {
            module.moe_gather_combine_add_quad(
                stream,
                LaunchConfig::for_num_elems((N * D / 4) as u32),
                &expert_output_dev,
                &selected_dev,
                &gates_dev,
                &slots_dev,
                &residual_dev,
                D as u32,
                K as u32,
                C as u32,
                &mut combine_add_quad_dev,
            )?;
        }
        assert_eq!(
            combine_add_quad_dev.to_host_vec(stream)?,
            combine_add_dev.to_host_vec(stream)?,
            "quad fused gather/combine must match the scalar kernel bit for bit"
        );

        // `dy` stands in for the router input gradient the fused kernel adds.
        let router_dx_stand_in = dy_dev.to_host_vec(stream)?;
        let expected_dx_add: Vec<f32> = expected_expert_dx
            .iter()
            .zip(&router_dx_stand_in)
            .map(|(&gathered, &router)| gathered + router)
            .collect();
        let mut dx_add_dev = DeviceBuffer::<f32>::zeroed(stream, N * D)?;
        module.moe_gather_dx_add(
            stream,
            LaunchConfig::for_num_elems((N * D) as u32),
            &expert_input_gradient_dev,
            &selected_dev,
            &slots_dev,
            &dy_dev,
            D as u32,
            K as u32,
            C as u32,
            &mut dx_add_dev,
        )?;
        assert_close(
            "MoE fused gather dx + router dx",
            &dx_add_dev.to_host_vec(stream)?,
            &expected_dx_add,
            1e-6,
            1e-6,
        );
        let mut dx_add_quad_dev = DeviceBuffer::<f32>::zeroed(stream, N * D)?;
        unsafe {
            module.moe_gather_dx_add_quad(
                stream,
                LaunchConfig::for_num_elems((N * D / 4) as u32),
                &expert_input_gradient_dev,
                &selected_dev,
                &slots_dev,
                &dy_dev,
                D as u32,
                K as u32,
                C as u32,
                &mut dx_add_quad_dev,
            )?;
        }
        assert_eq!(
            dx_add_quad_dev.to_host_vec(stream)?,
            dx_add_dev.to_host_vec(stream)?,
            "quad fused gather dx must match the scalar kernel bit for bit"
        );

        // The forward dead-bin pass composed with the scatter must between
        // them rewrite every input bin (the forward no longer pre-fills the
        // whole panel): poisoned storage that survives fails the compare.
        let input_poison: Vec<f32> = (0..E * C * D).map(|index| -(index as f32) - 2.0).collect();
        let mut poisoned_bins_dev = DeviceBuffer::from_host(stream, &input_poison)?;
        unsafe {
            module.moe_zero_dead_bins(
                stream,
                zero_dead_bins_config::<E>(1),
                &counts_dev,
                D as u32,
                C as u32,
                &mut poisoned_bins_dev,
            )?;
            module.moe_scatter(
                stream,
                LaunchConfig::for_num_elems((N * K * D) as u32),
                &x_dev,
                &selected_dev,
                &slots_dev,
                D as u32,
                K as u32,
                C as u32,
                &mut poisoned_bins_dev,
            )?;
        }
        assert_eq!(
            poisoned_bins_dev.to_host_vec(stream)?,
            expected_expert_input,
            "dead-bin zeroing plus scatter must rewrite every input bin"
        );

        // Same composition over the packed panel, through both the pair and
        // quad scatter walks.
        let expected_packed_input: Vec<u32> = expected_expert_input
            .chunks(2)
            .map(|pair| {
                bf16::from_f32(pair[0]).to_bits() as u32
                    | ((bf16::from_f32(pair[1]).to_bits() as u32) << 16)
            })
            .collect();
        let word_poison = vec![0xdead_beefu32; E * C * D / 2];
        let mut packed_pair_dev = DeviceBuffer::from_host(stream, &word_poison)?;
        let mut packed_quad_dev = DeviceBuffer::from_host(stream, &word_poison)?;
        unsafe {
            module.moe_zero_dead_bins_bf16(
                stream,
                zero_dead_bins_config::<E>(1),
                &counts_dev,
                D as u32,
                C as u32,
                &mut packed_pair_dev,
            )?;
            module.moe_scatter_bf16(
                stream,
                LaunchConfig::for_num_elems((N * K * D / 2) as u32),
                &x_dev,
                &selected_dev,
                &slots_dev,
                D as u32,
                K as u32,
                C as u32,
                &mut packed_pair_dev,
            )?;
            module.moe_zero_dead_bins_bf16(
                stream,
                zero_dead_bins_config::<E>(1),
                &counts_dev,
                D as u32,
                C as u32,
                &mut packed_quad_dev,
            )?;
            module.moe_scatter_bf16_quad(
                stream,
                LaunchConfig::for_num_elems((N * K * D / 4) as u32),
                &x_dev,
                &selected_dev,
                &slots_dev,
                D as u32,
                K as u32,
                C as u32,
                &mut packed_quad_dev,
            )?;
        }
        assert_eq!(
            packed_pair_dev.to_host_vec(stream)?,
            expected_packed_input,
            "dead-bin zeroing plus packed scatter must rewrite every input bin"
        );
        assert_eq!(
            packed_quad_dev.to_host_vec(stream)?,
            expected_packed_input,
            "quad packed scatter must match the pair walk bit for bit"
        );

        check_moe_tie_routing(stream, module)?;
        check_moe_scatter_dy_rows(stream, module)?;
        Ok(())
    }
}

/// Exercises the backward scatter row walks at a `D` the float4 path takes and
/// one it cannot, over a routing with a dropped pair, a partly dead expert, and
/// an entirely unassigned expert.
fn check_moe_scatter_dy_rows(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    check_moe_scatter_dy_case::<8>(stream, module)?;
    check_moe_scatter_dy_case::<5>(stream, module)
}

fn check_moe_scatter_dy_case<const D: usize>(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    const N: usize = 4;
    const E: usize = 3;
    const K: usize = 1;
    const C: usize = 2;

    let selected = [0u32, 0, 0, 1];
    let gates: Vec<f32> = (0..N * K).map(|pair| 0.25 + pair as f32 * 0.5).collect();
    let expert_output: Vec<f32> = (0..E * C * D)
        .map(|index| index as f32 * 0.125 - 1.0)
        .collect();
    let dy: Vec<f32> = (0..N * D)
        .map(|index| 1.0 - index as f32 * 0.0625)
        .collect();
    // Poisoned, not zeroed: the dead-slot pass plus the scatter must between
    // them rewrite every bin, so a surviving poison value fails the compare.
    let poison: Vec<f32> = (0..E * C * D).map(|index| index as f32 + 1.0).collect();

    let selected_dev = DeviceBuffer::from_host(stream, &selected)?;
    let gates_dev = DeviceBuffer::from_host(stream, &gates)?;
    let expert_output_dev = DeviceBuffer::from_host(stream, &expert_output)?;
    let dy_dev = DeviceBuffer::from_host(stream, &dy)?;
    let mut slots_dev = DeviceBuffer::<u32>::zeroed(stream, N * K)?;
    let mut counts_dev = DeviceBuffer::<u32>::zeroed(stream, E)?;
    let mut gradient_dev = DeviceBuffer::from_host(stream, &poison)?;
    let mut gate_gradients_dev = DeviceBuffer::<f32>::zeroed(stream, N * K)?;

    unsafe {
        module.moe_bin_assign_parallel(
            stream,
            LaunchConfig {
                grid_dim: (E as u32, 1, 1),
                block_dim: (MOE_ASSIGN_THREADS as u32, 1, 1),
                shared_mem_bytes: 0,
            },
            &selected_dev,
            N as u32,
            E as u32,
            K as u32,
            C as u32,
            &mut slots_dev,
            &mut counts_dev,
        )?;
        module.moe_zero_dead_bins(
            stream,
            zero_dead_bins_config::<E>(1),
            &counts_dev,
            D as u32,
            C as u32,
            &mut gradient_dev,
        )?;
        module.moe_scatter_dy(
            stream,
            LaunchConfig {
                grid_dim: ((N * K) as u32, 1, 1),
                block_dim: (MOE_SCATTER_DY_THREADS as u32, 1, 1),
                shared_mem_bytes: 0,
            },
            &expert_output_dev,
            &dy_dev,
            &selected_dev,
            &gates_dev,
            &slots_dev,
            D as u32,
            K as u32,
            C as u32,
            &mut gradient_dev,
            &mut gate_gradients_dev,
        )?;
    }

    let slots = slots_dev.to_host_vec(stream)?;
    assert_eq!(
        slots,
        [0, 1, MOE_DROPPED_SLOT, 0],
        "MoE scatter row shape must drop one pair and leave expert 2 empty"
    );
    assert_eq!(counts_dev.to_host_vec(stream)?, [3, 1, 0]);

    let mut expected_gradient = vec![0.0f32; E * C * D];
    let mut expected_gate_gradients = vec![0.0f32; N * K];
    for pair in 0..N * K {
        if slots[pair] == MOE_DROPPED_SLOT {
            continue;
        }
        let bin_base = (selected[pair] as usize * C + slots[pair] as usize) * D;
        let token_base = (pair / K) * D;
        for column in 0..D {
            expected_gradient[bin_base + column] = gates[pair] * dy[token_base + column];
            expected_gate_gradients[pair] +=
                expert_output[bin_base + column] * dy[token_base + column];
        }
    }
    assert_close(
        "MoE dead-slot zeroing and scatter cover every bin",
        &gradient_dev.to_host_vec(stream)?,
        &expected_gradient,
        1e-6,
        1e-6,
    );
    assert_close(
        "MoE gate gradients over strided rows",
        &gate_gradients_dev.to_host_vec(stream)?,
        &expected_gate_gradients,
        1e-6,
        1e-6,
    );

    // The packed twin, where both the saved output it dots against and the bin
    // gradient it writes are bf16. Only an even `D` reaches it: a packed bin
    // row of odd width would straddle words, which is why the kernel bails on
    // one, and why a packed bin panel only exists at tcgen05-aligned shapes.
    if D.is_multiple_of(2) {
        let pack = |values: &[f32]| -> Vec<u32> {
            values
                .chunks(2)
                .map(|pair| {
                    bf16::from_f32(pair[0]).to_bits() as u32
                        | ((bf16::from_f32(pair[1]).to_bits() as u32) << 16)
                })
                .collect()
        };
        let expert_output_words = DeviceBuffer::from_host(stream, &pack(&expert_output))?;
        let mut gradient_words = DeviceBuffer::from_host(stream, &pack(&poison))?;
        let mut packed_gate_gradients_dev = DeviceBuffer::<f32>::zeroed(stream, N * K)?;
        unsafe {
            module.moe_zero_dead_bins_bf16(
                stream,
                zero_dead_bins_config::<E>(1),
                &counts_dev,
                D as u32,
                C as u32,
                &mut gradient_words,
            )?;
            module.moe_scatter_dy_packed(
                stream,
                LaunchConfig {
                    grid_dim: ((N * K) as u32, 1, 1),
                    block_dim: (MOE_SCATTER_DY_THREADS as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                &expert_output_words,
                &dy_dev,
                &selected_dev,
                &gates_dev,
                &slots_dev,
                D as u32,
                K as u32,
                C as u32,
                &mut gradient_words,
                &mut packed_gate_gradients_dev,
            )?;
        }
        assert_eq!(
            gradient_words.to_host_vec(stream)?,
            pack(&expected_gradient),
            "packed scatter must be the wide bin gradient rounded to bf16"
        );
        // Every operand here is exact in bf16, so the dot product reads the
        // same values the wide kernel did and differs only by summation order.
        assert_close(
            "MoE gate gradients off a packed saved output",
            &packed_gate_gradients_dev.to_host_vec(stream)?,
            &expected_gate_gradients,
            1e-6,
            1e-6,
        );
    }
    Ok(())
}

fn check_moe_tie_routing(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    const N: usize = 3;
    const E: usize = 3;
    const K: usize = 2;
    const C: usize = 2;
    let logits_dev = DeviceBuffer::from_host(stream, &[0.0f32; N * E])?;
    let mut probabilities_dev = DeviceBuffer::<f32>::zeroed(stream, N * E)?;
    let mut selected_dev = DeviceBuffer::<u32>::zeroed(stream, N * K)?;
    let mut gates_dev = DeviceBuffer::<f32>::zeroed(stream, N * K)?;
    let mut slots_dev = DeviceBuffer::<u32>::zeroed(stream, N * K)?;
    let mut counts_dev = DeviceBuffer::<u32>::zeroed(stream, E)?;
    unsafe {
        module.router_softmax_topk(
            stream,
            LaunchConfig::for_num_elems(N as u32),
            &logits_dev,
            E as u32,
            K as u32,
            &mut probabilities_dev,
            &mut selected_dev,
            &mut gates_dev,
        )?;
        module.moe_bin_assign_parallel(
            stream,
            LaunchConfig {
                grid_dim: (E as u32, 1, 1),
                block_dim: (MOE_ASSIGN_THREADS as u32, 1, 1),
                shared_mem_bytes: 0,
            },
            &selected_dev,
            N as u32,
            E as u32,
            K as u32,
            C as u32,
            &mut slots_dev,
            &mut counts_dev,
        )?;
    }
    assert_eq!(
        selected_dev.to_host_vec(stream)?,
        [0, 1, 0, 1, 0, 1],
        "MoE top-k ties must select lower expert indices"
    );
    assert_eq!(
        slots_dev.to_host_vec(stream)?,
        [0, 0, 1, 1, MOE_DROPPED_SLOT, MOE_DROPPED_SLOT],
        "MoE tie shape must preserve token-order capacity assignment"
    );
    assert_close(
        "MoE tie gate weights",
        &gates_dev.to_host_vec(stream)?,
        &[0.5; N * K],
        0.0,
        0.0,
    );
    Ok(())
}

fn check_rope(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    // SAFETY: all device buffers and launches use the same N/T/D/H/HD shape.
    unsafe {
        const N: usize = 10;
        const T: usize = 5;
        const D: usize = 12;
        const H: usize = 3;
        const HD: usize = 4;
        let x = CpuTensor::<f32, Rank2<N, D>>::uniform(10);
        let dy = CpuTensor::<f32, Rank2<N, D>>::uniform(11);
        let mut cpu = Rope::<N, T, D, H, HD>;
        let (cpu_y, ()) = cpu.forward(x.clone());
        let cpu_dx = cpu.backward((), dy.clone());
        let x_dev = DeviceBuffer::from_host(stream, x.as_slice())?;
        let dy_dev = DeviceBuffer::from_host(stream, dy.as_slice())?;
        let table = DeviceBuffer::from_host(stream, &rope_table(T, HD))?;
        let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, N * D)?;
        let mut dx_dev = DeviceBuffer::<f32>::zeroed(stream, N * D)?;

        module.rope_forward(
            stream,
            LaunchConfig::for_num_elems((N * D / 2) as u32),
            &x_dev,
            &table,
            T as u32,
            H as u32,
            HD as u32,
            &mut y_dev,
        )?;
        module.rope_backward(
            stream,
            LaunchConfig::for_num_elems((N * D / 2) as u32),
            &dy_dev,
            &table,
            T as u32,
            H as u32,
            HD as u32,
            &mut dx_dev,
        )?;
        assert_close(
            "rope y",
            &y_dev.to_host_vec(stream)?,
            cpu_y.as_slice(),
            2e-6,
            2e-6,
        );
        assert_close(
            "rope dx",
            &dx_dev.to_host_vec(stream)?,
            cpu_dx.as_slice(),
            2e-6,
            2e-6,
        );

        // The fused backward join is a substitution for `rope_backward` twice
        // followed by `join_group3`, so the oracle it owes parity to is that
        // composition — and it owes it exactly, not to a tolerance.
        let dk = CpuTensor::<f32, Rank2<N, D>>::uniform(12);
        let dv = CpuTensor::<f32, Rank2<N, D>>::uniform(13);
        let dk_dev = DeviceBuffer::from_host(stream, dk.as_slice())?;
        let dv_dev = DeviceBuffer::from_host(stream, dv.as_slice())?;
        let mut dk_rotated = DeviceBuffer::<f32>::zeroed(stream, N * D)?;
        let mut composed = DeviceBuffer::<f32>::zeroed(stream, N * 3 * D)?;
        let mut fused = DeviceBuffer::<f32>::zeroed(stream, N * 3 * D)?;
        let pairs = LaunchConfig::for_num_elems((N * D / 2) as u32);
        module.rope_backward(
            stream,
            pairs,
            &dk_dev,
            &table,
            T as u32,
            H as u32,
            HD as u32,
            &mut dk_rotated,
        )?;
        module.join_group3(
            stream,
            LaunchConfig::for_num_elems((N * D) as u32),
            &dx_dev,
            &dk_rotated,
            &dv_dev,
            D as u32,
            &mut composed,
        )?;
        module.join_group3_rope(
            stream, pairs, &dy_dev, &dk_dev, &dv_dev, &table, T as u32, H as u32, HD as u32,
            &mut fused,
        )?;
        let composed_host = composed.to_host_vec(stream)?;
        assert_close(
            "join_group3_rope vs rope_backward + join_group3",
            &fused.to_host_vec(stream)?,
            &composed_host,
            0.0,
            0.0,
        );

        // The packed join substitutes for that composition followed by the
        // `convert_f32_to_bf16_pairs` the qkv backward used to run over the
        // whole panel, so its oracle is the composition rounded to nearest
        // even and packed low half first. There is one rounding here and there
        // was one before — both backward GEMMs read the single quantized
        // buffer — so the bytes are equal, not merely close.
        let mut packed = DeviceBuffer::<u32>::zeroed(stream, N * 3 * D / 2)?;
        module.join_group3_rope_bf16(
            stream,
            pairs,
            &dy_dev,
            &dk_dev,
            &dv_dev,
            &table,
            T as u32,
            H as u32,
            HD as u32,
            &mut packed,
        )?;
        let quantized: Vec<u32> = composed_host
            .chunks(2)
            .map(|couple| {
                bf16::from_f32(couple[0]).to_bits() as u32
                    | ((bf16::from_f32(couple[1]).to_bits() as u32) << 16)
            })
            .collect();
        assert_eq!(
            packed.to_host_vec(stream)?,
            quantized,
            "join_group3_rope_bf16 vs quantized rope_backward + join_group3"
        );
        Ok(())
    }
}

fn check_attention(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    // SAFETY: all device buffers and launches use the same N/T/D/H/HD shape.
    unsafe {
        const N: usize = 10;
        const T: usize = 5;
        const D: usize = 12;
        const H: usize = 3;
        const HD: usize = 4;
        let q = CpuTensor::<f32, Rank2<N, D>>::uniform(12);
        let k = CpuTensor::<f32, Rank2<N, D>>::uniform(13);
        let v = CpuTensor::<f32, Rank2<N, D>>::uniform(14);
        let dy = CpuTensor::<f32, Rank2<N, D>>::uniform(15);
        let mut cpu = CausalAttention::<N, T, D, H, HD>;
        let (cpu_y, cpu_ctx) = cpu.forward((q.clone(), k.clone(), v.clone()));
        let (cpu_dq, cpu_dk, cpu_dv) = cpu.backward(cpu_ctx, dy.clone());

        let q_dev = DeviceBuffer::from_host(stream, q.as_slice())?;
        let k_dev = DeviceBuffer::from_host(stream, k.as_slice())?;
        let v_dev = DeviceBuffer::from_host(stream, v.as_slice())?;
        let dy_dev = DeviceBuffer::from_host(stream, dy.as_slice())?;
        let mut p_dev = DeviceBuffer::<f32>::zeroed(stream, N * H * T)?;
        let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, N * D)?;
        let mut dq_dev = DeviceBuffer::<f32>::zeroed(stream, N * D)?;
        let mut dk_dev = DeviceBuffer::<f32>::zeroed(stream, N * D)?;
        let mut dv_dev = DeviceBuffer::<f32>::zeroed(stream, N * D)?;
        module.attention_probabilities(
            stream,
            LaunchConfig::for_num_elems((N * H * T) as u32),
            &q_dev,
            &k_dev,
            T as u32,
            H as u32,
            HD as u32,
            &mut p_dev,
        )?;
        module.attention_output(
            stream,
            LaunchConfig::for_num_elems((N * D) as u32),
            &p_dev,
            &v_dev,
            T as u32,
            H as u32,
            HD as u32,
            &mut y_dev,
        )?;
        module.attention_backward_q(
            stream,
            LaunchConfig::for_num_elems((N * D) as u32),
            &q_dev,
            &k_dev,
            &v_dev,
            &p_dev,
            &dy_dev,
            T as u32,
            H as u32,
            HD as u32,
            &mut dq_dev,
        )?;
        module.attention_backward_k(
            stream,
            LaunchConfig::for_num_elems((N * D) as u32),
            &q_dev,
            &v_dev,
            &p_dev,
            &dy_dev,
            T as u32,
            H as u32,
            HD as u32,
            &mut dk_dev,
        )?;
        module.attention_backward_v(
            stream,
            LaunchConfig::for_num_elems((N * D) as u32),
            &p_dev,
            &dy_dev,
            T as u32,
            H as u32,
            HD as u32,
            &mut dv_dev,
        )?;

        assert_close(
            "attention y",
            &y_dev.to_host_vec(stream)?,
            cpu_y.as_slice(),
            3e-5,
            3e-5,
        );
        assert_close(
            "attention dq",
            &dq_dev.to_host_vec(stream)?,
            cpu_dq.as_slice(),
            5e-5,
            5e-5,
        );
        assert_close(
            "attention dk",
            &dk_dev.to_host_vec(stream)?,
            cpu_dk.as_slice(),
            5e-5,
            5e-5,
        );
        assert_close(
            "attention dv",
            &dv_dev.to_host_vec(stream)?,
            cpu_dv.as_slice(),
            5e-5,
            5e-5,
        );
        let probabilities = p_dev.to_host_vec(stream)?;
        let probabilities = CpuTensor::<f32, Rank3<N, H, T>>::from_slice(&probabilities);
        for row in 0..N {
            for head in 0..H {
                let start = (row * H + head) * T;
                let sum: f32 = probabilities.as_slice()[start..start + T].iter().sum();
                assert!((sum - 1.0).abs() < 1e-5);
            }
        }
        Ok(())
    }
}

#[allow(unused_unsafe)]
fn check_rms_norm(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    // SAFETY: all device buffers and launches use the same N x D shape.
    unsafe {
        // Cross both optimized kernels' 256-wide row/column tile boundaries.
        const N: usize = 259;
        const D: usize = 261;
        let x = CpuTensor::<f32, Rank2<N, D>>::uniform(1);
        let weight = CpuTensor::<f32, Rank1<D>>::uniform(2).map(|v| v + 1.25);
        let dy = CpuTensor::<f32, Rank2<N, D>>::uniform(3);
        let mut cpu = RmsNorm::<N, D>::new(weight.clone(), 1e-5);
        let (cpu_y, cpu_ctx) = cpu.forward(x.clone());
        let cpu_dx = cpu.backward(cpu_ctx, dy.clone());

        let x_dev = DeviceBuffer::from_host(stream, x.as_slice())?;
        let weight_dev = DeviceBuffer::from_host(stream, weight.as_slice())?;
        let dy_dev = DeviceBuffer::from_host(stream, dy.as_slice())?;
        let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, N * D)?;
        let mut dx_dev = DeviceBuffer::<f32>::zeroed(stream, N * D)?;
        let mut dw_dev = DeviceBuffer::<f32>::zeroed(stream, D)?;

        module.rms_norm_forward(
            stream,
            LaunchConfig::for_num_elems((N * D) as u32),
            &x_dev,
            &weight_dev,
            1e-5,
            D as u32,
            &mut y_dev,
        )?;
        module.rms_norm_backward_x(
            stream,
            LaunchConfig::for_num_elems((N * D) as u32),
            &x_dev,
            &weight_dev,
            &dy_dev,
            1e-5,
            D as u32,
            &mut dx_dev,
        )?;
        module.rms_norm_backward_weight(
            stream,
            LaunchConfig::for_num_elems(D as u32),
            &x_dev,
            &dy_dev,
            1e-5,
            N as u32,
            D as u32,
            &mut dw_dev,
        )?;

        assert_close(
            "rmsnorm y",
            &y_dev.to_host_vec(stream)?,
            cpu_y.as_slice(),
            2e-5,
            2e-5,
        );
        assert_close(
            "rmsnorm dx",
            &dx_dev.to_host_vec(stream)?,
            cpu_dx.as_slice(),
            3e-5,
            3e-5,
        );
        assert_close(
            "rmsnorm dw",
            &dw_dev.to_host_vec(stream)?,
            cpu.dw.as_slice(),
            3e-5,
            3e-5,
        );

        // Optimized model path against the naive oracle above.
        let mut inv_dev = DeviceBuffer::<f32>::zeroed(stream, N)?;
        let mut inv_fast_dev = DeviceBuffer::<f32>::zeroed(stream, N)?;
        let mut y_fast_dev = DeviceBuffer::<f32>::zeroed(stream, N * D)?;
        let mut dx_fast_dev = DeviceBuffer::<f32>::zeroed(stream, N * D)?;
        let mut dw_fast_dev = DeviceBuffer::<f32>::zeroed(stream, D)?;
        module.rms_norm_row_inv(
            stream,
            LaunchConfig {
                grid_dim: (N as u32, 1, 1),
                block_dim: (NORM_THREADS as u32, 1, 1),
                shared_mem_bytes: 0,
            },
            &x_dev,
            1e-5,
            D as u32,
            &mut inv_dev,
        )?;
        module.rms_norm_forward_fast(
            stream,
            LaunchConfig {
                grid_dim: (N as u32, 1, 1),
                block_dim: (NORM_THREADS as u32, 1, 1),
                shared_mem_bytes: 0,
            },
            &x_dev,
            &weight_dev,
            1e-5,
            D as u32,
            &mut y_fast_dev,
        )?;
        module.rms_norm_backward_x_fast(
            stream,
            LaunchConfig {
                grid_dim: (N as u32, 1, 1),
                block_dim: (NORM_THREADS as u32, 1, 1),
                shared_mem_bytes: 0,
            },
            &x_dev,
            &weight_dev,
            &dy_dev,
            1e-5,
            D as u32,
            &mut dx_fast_dev,
            &mut inv_fast_dev,
        )?;
        unsafe {
            module.rms_norm_backward_weight_fast(
                stream,
                LaunchConfig {
                    grid_dim: (
                        D.div_ceil(NORM_THREADS) as u32,
                        N.div_ceil(NORM_WEIGHT_ROWS_PER_BLOCK) as u32,
                        1,
                    ),
                    block_dim: (NORM_THREADS as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                &x_dev,
                &dy_dev,
                &inv_fast_dev,
                N as u32,
                D as u32,
                &mut dw_fast_dev,
            )?;
        }
        assert_close(
            "rmsnorm y fast vs naive",
            &y_fast_dev.to_host_vec(stream)?,
            &y_dev.to_host_vec(stream)?,
            2e-5,
            2e-5,
        );
        assert_close(
            "rmsnorm dx fast vs naive",
            &dx_fast_dev.to_host_vec(stream)?,
            &dx_dev.to_host_vec(stream)?,
            3e-5,
            3e-5,
        );
        assert_close(
            "rmsnorm inv fused vs standalone",
            &inv_fast_dev.to_host_vec(stream)?,
            &inv_dev.to_host_vec(stream)?,
            1e-6,
            1e-6,
        );
        assert_close(
            "rmsnorm dw fast vs naive",
            &dw_fast_dev.to_host_vec(stream)?,
            &dw_dev.to_host_vec(stream)?,
            5e-6,
            2e-5,
        );
        Ok(())
    }
}

/// The tile RMSNorm forward against the same CPU oracle the shipped one uses.
///
/// A separate shape from [`check_rms_norm`]'s deliberately unaligned one: the
/// tile kernel takes its divisibility from the launcher and does not bounds-
/// check, so the case that exercises it is an aligned one. `1024` rows cross
/// several blocks at every `NORM_TILE_WARPS` a sweep might set, and `192`
/// columns divide by every `NORM_TILE_CHUNK` up to 64.
fn check_rms_norm_tile(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    // SAFETY: every buffer and launch below uses the same N x D shape, and N is
    // a multiple of NORM_TILE_BLOCK_ROWS and D of NORM_TILE_CHUNK.
    unsafe {
        const N: usize = 1024;
        const D: usize = 192;
        assert_eq!(N % NORM_TILE_BLOCK_ROWS, 0);
        assert_eq!(D % NORM_TILE_CHUNK, 0);

        let x = CpuTensor::<f32, Rank2<N, D>>::uniform(1);
        let weight = CpuTensor::<f32, Rank1<D>>::uniform(2).map(|v| v + 1.25);
        let cpu = RmsNorm::<N, D>::new(weight.clone(), 1e-5);
        let (cpu_y, _) = cpu.forward(x.clone());

        let x_dev = DeviceBuffer::from_host(stream, x.as_slice())?;
        let weight_dev = DeviceBuffer::from_host(stream, weight.as_slice())?;
        let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, N * D)?;

        module.rms_norm_forward_tile(
            stream,
            LaunchConfig {
                grid_dim: ((N / NORM_TILE_BLOCK_ROWS) as u32, 1, 1),
                block_dim: (NORM_TILE_THREADS as u32, 1, 1),
                shared_mem_bytes: 0,
            },
            &x_dev,
            &weight_dev,
            1e-5,
            D as u32,
            &mut y_dev,
        )?;
        assert_close(
            "rmsnorm tile y",
            &y_dev.to_host_vec(stream)?,
            cpu_y.as_slice(),
            2e-5,
            2e-5,
        );
        Ok(())
    }
}

/// Every tile SwiGLU kernel against the flat one it would replace.
///
/// The flat kernels are already gated against the CPU module by
/// [`check_swiglu`]; what is left to prove here is that the tile walk covers
/// the same rectangle, which a wrong row band or column stride would break
/// loudly.
fn check_swiglu_tile(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    // SAFETY: every buffer is ROWS x COLUMNS, ROWS is a multiple of
    // SWIGLU_TILE_BLOCK_ROWS and COLUMNS of SWIGLU_TILE_CHUNK, and both
    // launches cover exactly that rectangle.
    unsafe {
        const ROWS: usize = 1024;
        const COLUMNS: usize = 192;
        const LEN: usize = ROWS * COLUMNS;
        assert_eq!(ROWS % SWIGLU_TILE_BLOCK_ROWS, 0);
        assert_eq!(COLUMNS % SWIGLU_TILE_CHUNK, 0);

        let gate = CpuTensor::<f32, Rank2<ROWS, COLUMNS>>::uniform(20);
        let up = CpuTensor::<f32, Rank2<ROWS, COLUMNS>>::uniform(21);
        let dy = CpuTensor::<f32, Rank2<ROWS, COLUMNS>>::uniform(22);
        let gate_dev = DeviceBuffer::from_host(stream, gate.as_slice())?;
        let up_dev = DeviceBuffer::from_host(stream, up.as_slice())?;
        let dy_dev = DeviceBuffer::from_host(stream, dy.as_slice())?;

        let flat = LaunchConfig::for_num_elems(LEN as u32);
        let tiles = LaunchConfig {
            grid_dim: ((ROWS / SWIGLU_TILE_BLOCK_ROWS) as u32, 1, 1),
            block_dim: (SWIGLU_TILE_THREADS as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        // Looser than the flat kernels' own gate against the CPU: kittens'
        // `exp` is not libdevice's, and every kernel here is a sigmoid.
        let (atol, rtol) = (1e-6, 1e-4);

        let mut shipped = DeviceBuffer::<f32>::zeroed(stream, LEN)?;
        let mut tiled = DeviceBuffer::<f32>::zeroed(stream, LEN)?;
        module.swiglu_forward(stream, flat, &gate_dev, &up_dev, &mut shipped)?;
        module.swiglu_forward_tile(
            stream,
            tiles,
            &gate_dev,
            &up_dev,
            COLUMNS as u32,
            &mut tiled,
        )?;
        assert_close(
            "swiglu tile vs flat",
            &tiled.to_host_vec(stream)?,
            &shipped.to_host_vec(stream)?,
            atol,
            rtol,
        );

        module.swiglu_backward_gate(stream, flat, &gate_dev, &up_dev, &dy_dev, &mut shipped)?;
        module.swiglu_backward_gate_tile(
            stream,
            tiles,
            &gate_dev,
            &up_dev,
            &dy_dev,
            COLUMNS as u32,
            &mut tiled,
        )?;
        assert_close(
            "swiglu dgate tile vs flat",
            &tiled.to_host_vec(stream)?,
            &shipped.to_host_vec(stream)?,
            atol,
            rtol,
        );

        module.swiglu_backward_up(stream, flat, &gate_dev, &dy_dev, &mut shipped)?;
        module.swiglu_backward_up_tile(
            stream,
            tiles,
            &gate_dev,
            &dy_dev,
            COLUMNS as u32,
            &mut tiled,
        )?;
        assert_close(
            "swiglu dup tile vs flat",
            &tiled.to_host_vec(stream)?,
            &shipped.to_host_vec(stream)?,
            atol,
            rtol,
        );
        Ok(())
    }
}

fn check_classifier_bf16(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    // Odd real vocabulary exercises the packed tail; the second case covers a
    // multi-iteration lane stride.
    check_classifier_bf16_case::<5, 13, 16>(stream, module)?;
    check_classifier_bf16_case::<3, 517, 520>(stream, module)
}

fn check_classifier_bf16_case<const N: usize, const C: usize, const CP: usize>(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    // SAFETY: CP is validated as an even padded width, and every buffer and
    // launch is allocated from N, C, and CP.
    unsafe {
        assert!(CP % 2 == 0 && CP >= C);
        let logits = CpuTensor::<f32, Rank2<N, C>>::uniform(16).scale(5.0);
        // Round to bf16 once; the f32 oracle then sees the exact values the bf16
        // kernels decode, so the only differences are lane order and the bf16
        // rounding of the written gradients.
        let rounded: Vec<f32> = logits
            .as_slice()
            .iter()
            .map(|&value| bf16::from_f32(value).to_f32())
            .collect();
        let mut packed = vec![0u32; N * CP / 2];
        for row in 0..N {
            for col in 0..C {
                let bits = bf16::from_f32(rounded[row * C + col]).to_bits() as u32;
                packed[(row * CP + col) / 2] |= bits << (16 * (col % 2));
            }
        }
        let targets_usize: [usize; N] = std::array::from_fn(|row| (row * 101 + C - 1) % C);
        let targets = targets_usize.map(|v| v as u32);
        let targets_dev = DeviceBuffer::from_host(stream, &targets)?;
        let classifier_config = LaunchConfig {
            grid_dim: (N as u32, 1, 1),
            block_dim: (CLASSIFIER_THREADS as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // f32 fused oracle on the rounded values.
        let rounded_dev = DeviceBuffer::from_host(stream, &rounded)?;
        let mut oracle_losses = DeviceBuffer::<f32>::zeroed(stream, N)?;
        let mut oracle_dlogits = DeviceBuffer::from_host(stream, &rounded)?;
        module.fused_classifier_forward(
            stream,
            classifier_config,
            &rounded_dev,
            &targets_dev,
            N as u32,
            C as u32,
            &mut oracle_losses,
        )?;
        module.fused_classifier_backward_in_place(
            stream,
            classifier_config,
            &targets_dev,
            1.0,
            N as u32,
            C as u32,
            &mut oracle_dlogits,
        )?;

        let packed_dev = DeviceBuffer::from_host(stream, &packed)?;
        let mut losses = DeviceBuffer::<f32>::zeroed(stream, N)?;
        let mut dlogits = DeviceBuffer::from_host(stream, &packed)?;
        module.fused_classifier_forward_bf16(
            stream,
            classifier_config,
            &packed_dev,
            &targets_dev,
            N as u32,
            C as u32,
            CP as u32,
            &mut losses,
        )?;
        module.fused_classifier_backward_in_place_bf16(
            stream,
            classifier_config,
            &targets_dev,
            1.0,
            N as u32,
            C as u32,
            CP as u32,
            &mut dlogits,
        )?;

        assert_close(
            "bf16 classifier losses vs f32 fused",
            &losses.to_host_vec(stream)?,
            &oracle_losses.to_host_vec(stream)?,
            5e-5,
            2e-5,
        );
        let dlogits = dlogits.to_host_vec(stream)?;
        let oracle = oracle_dlogits.to_host_vec(stream)?;
        for row in 0..N {
            for col in 0..CP {
                let word = dlogits[(row * CP + col) / 2];
                let bits = (word >> (16 * (col % 2))) as u16;
                if col < C {
                    let actual = bf16::from_bits(bits).to_f32();
                    let expected = oracle[row * C + col];
                    let tolerance = 1e-6 + 4e-3 * expected.abs();
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "bf16 classifier dlogits mismatch at [{row},{col}]: \
                         gpu={actual}, oracle={expected}, tolerance={tolerance}"
                    );
                } else {
                    assert_eq!(bits, 0, "padded dlogits column [{row},{col}] is not zero");
                }
            }
        }
        Ok(())
    }
}

fn check_swiglu(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    // SAFETY: each input and output has LEN elements and each launch covers LEN.
    unsafe {
        const LEN: usize = 33;
        let gate = CpuTensor::<f32, Rank2<3, 11>>::uniform(4);
        let up = CpuTensor::<f32, Rank2<3, 11>>::uniform(5);
        let dy = CpuTensor::<f32, Rank2<3, 11>>::uniform(6);
        let mut cpu = SwiGlu::<3, 11>;
        let (cpu_y, cpu_ctx) = cpu.forward((gate.clone(), up.clone()));
        let (cpu_dgate, cpu_dup) = cpu.backward(cpu_ctx, dy.clone());

        let gate_dev = DeviceBuffer::from_host(stream, gate.as_slice())?;
        let up_dev = DeviceBuffer::from_host(stream, up.as_slice())?;
        let dy_dev = DeviceBuffer::from_host(stream, dy.as_slice())?;
        let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, LEN)?;
        let mut dgate_dev = DeviceBuffer::<f32>::zeroed(stream, LEN)?;
        let mut dup_dev = DeviceBuffer::<f32>::zeroed(stream, LEN)?;
        module.swiglu_forward(
            stream,
            LaunchConfig::for_num_elems(LEN as u32),
            &gate_dev,
            &up_dev,
            &mut y_dev,
        )?;
        module.swiglu_backward_gate(
            stream,
            LaunchConfig::for_num_elems(LEN as u32),
            &gate_dev,
            &up_dev,
            &dy_dev,
            &mut dgate_dev,
        )?;
        module.swiglu_backward_up(
            stream,
            LaunchConfig::for_num_elems(LEN as u32),
            &gate_dev,
            &dy_dev,
            &mut dup_dev,
        )?;

        assert_close(
            "swiglu y",
            &y_dev.to_host_vec(stream)?,
            cpu_y.as_slice(),
            1e-6,
            1e-5,
        );
        assert_close(
            "swiglu dgate",
            &dgate_dev.to_host_vec(stream)?,
            cpu_dgate.as_slice(),
            2e-6,
            1e-5,
        );
        assert_close(
            "swiglu dup",
            &dup_dev.to_host_vec(stream)?,
            cpu_dup.as_slice(),
            2e-6,
            1e-5,
        );
        Ok(())
    }
}

/// The fused interleaved-layout SwiGLU kernels against the flat split-layout
/// ones: identical math on identical values, with gate and up read out of one
/// `[rows, 2, ff]` panel and — in backward — both gradient halves written
/// back interleaved, replacing the split and join passes. The packed-bf16
/// forms must reproduce the flat fp32 results rounded to bf16 exactly.
fn check_swiglu_interleaved(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    // FF = 5 exercises the fp32 oracle arm alone; FF = 8 is quad-aligned so
    // the packed-bf16 arms run too.
    check_swiglu_interleaved_case::<3, 5>(stream, module)?;
    check_swiglu_interleaved_case::<4, 8>(stream, module)
}

#[allow(unused_unsafe)]
fn check_swiglu_interleaved_case<const ROWS: usize, const FF: usize>(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    // SAFETY: every launch covers exactly the element count its output dtype
    // packs the `ROWS x FF` (or interleaved `ROWS x 2FF`) rectangle into, and
    // the bf16 launches only run at an FF divisible by four.
    unsafe {
        // The packed panel holds gate and up rounded to bf16, so the flat fp32
        // reference reads the same rounded values the packed kernels unpack
        // (as `check_embedding` does for the bf16 master) — which keeps every
        // comparison below exact rather than tolerant.
        let gate = CpuTensor::<f32, Rank2<ROWS, FF>>::uniform(14)
            .to_bf16()
            .to_f32();
        let up = CpuTensor::<f32, Rank2<ROWS, FF>>::uniform(15)
            .to_bf16()
            .to_f32();
        let dy = CpuTensor::<f32, Rank2<ROWS, FF>>::uniform(16);
        let mut interleaved = vec![0.0f32; ROWS * 2 * FF];
        for row in 0..ROWS {
            for column in 0..FF {
                interleaved[row * 2 * FF + column] = gate.as_slice()[row * FF + column];
                interleaved[row * 2 * FF + FF + column] = up.as_slice()[row * FF + column];
            }
        }

        let flat = LaunchConfig::for_num_elems((ROWS * FF) as u32);
        let gate_dev = DeviceBuffer::from_host(stream, gate.as_slice())?;
        let up_dev = DeviceBuffer::from_host(stream, up.as_slice())?;
        let dy_dev = DeviceBuffer::from_host(stream, dy.as_slice())?;
        let mut y_ref_dev = DeviceBuffer::<f32>::zeroed(stream, ROWS * FF)?;
        let mut dgate_ref_dev = DeviceBuffer::<f32>::zeroed(stream, ROWS * FF)?;
        let mut dup_ref_dev = DeviceBuffer::<f32>::zeroed(stream, ROWS * FF)?;
        module.swiglu_forward(stream, flat, &gate_dev, &up_dev, &mut y_ref_dev)?;
        module.swiglu_backward_gate(
            stream,
            flat,
            &gate_dev,
            &up_dev,
            &dy_dev,
            &mut dgate_ref_dev,
        )?;
        module.swiglu_backward_up(stream, flat, &gate_dev, &dy_dev, &mut dup_ref_dev)?;
        let y_ref = y_ref_dev.to_host_vec(stream)?;
        let dgate_ref = dgate_ref_dev.to_host_vec(stream)?;
        let dup_ref = dup_ref_dev.to_host_vec(stream)?;

        let gate_up_dev = DeviceBuffer::from_host(stream, &interleaved)?;
        let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, ROWS * FF)?;
        module.swiglu_forward_interleaved(stream, flat, &gate_up_dev, FF as u32, &mut y_dev)?;
        assert_eq!(
            y_dev.to_host_vec(stream)?,
            y_ref,
            "interleaved swiglu forward must match the flat kernel bit for bit"
        );

        let mut d_gate_up_dev = DeviceBuffer::<f32>::zeroed(stream, ROWS * 2 * FF)?;
        module.swiglu_backward_interleaved(
            stream,
            flat,
            &gate_up_dev,
            &dy_dev,
            FF as u32,
            &mut d_gate_up_dev,
        )?;
        let mut expected_d_interleaved = vec![0.0f32; ROWS * 2 * FF];
        for row in 0..ROWS {
            for column in 0..FF {
                expected_d_interleaved[row * 2 * FF + column] = dgate_ref[row * FF + column];
                expected_d_interleaved[row * 2 * FF + FF + column] = dup_ref[row * FF + column];
            }
        }
        assert_eq!(
            d_gate_up_dev.to_host_vec(stream)?,
            expected_d_interleaved,
            "fused interleaved swiglu backward must match the flat kernels bit for bit"
        );

        if FF.is_multiple_of(4) {
            let pack = |values: &[f32]| -> Vec<u32> {
                values
                    .chunks(2)
                    .map(|pair| {
                        bf16::from_f32(pair[0]).to_bits() as u32
                            | ((bf16::from_f32(pair[1]).to_bits() as u32) << 16)
                    })
                    .collect()
            };
            // The packed arms read the panel packed too: it is the gate/up
            // GEMM's own epilogue target now, never an fp32 buffer.
            let gate_up_words = DeviceBuffer::from_host(stream, &pack(&interleaved))?;
            let mut y_words = DeviceBuffer::<u32>::zeroed(stream, ROWS * FF / 2)?;
            module.swiglu_forward_interleaved_packed(
                stream,
                LaunchConfig::for_num_elems((ROWS * FF / 4) as u32),
                &gate_up_words,
                FF as u32,
                &mut y_words,
            )?;
            assert_eq!(
                y_words.to_host_vec(stream)?,
                pack(&y_ref),
                "packed interleaved swiglu forward must be the flat result rounded to bf16"
            );

            let mut d_words = DeviceBuffer::<u32>::zeroed(stream, ROWS * FF)?;
            module.swiglu_backward_interleaved_packed(
                stream,
                LaunchConfig::for_num_elems((ROWS * FF / 4) as u32),
                &gate_up_words,
                &dy_dev,
                FF as u32,
                &mut d_words,
            )?;
            assert_eq!(
                d_words.to_host_vec(stream)?,
                pack(&expected_d_interleaved),
                "packed interleaved swiglu backward must be the flat result rounded to bf16"
            );
        }
        Ok(())
    }
}

#[allow(unused_unsafe)]
fn check_embedding(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    // SAFETY: token, embedding, and gradient buffers are allocated from the
    // same N/V/D constants used by every launch.
    unsafe {
        const N: usize = 6;
        const V: usize = 9;
        // Even, because the bf16 embedding master is stored as packed pairs.
        const D: usize = 6;
        let tokens_usize = [2, 7, 2, 0, 7, 4];
        let tokens = tokens_usize.map(|v| v as u32);
        // The master is bf16 (#57): the reference reads the same rounded values the
        // lookup kernel unpacks, so the forward stays an exact comparison.
        let weight = CpuTensor::<f32, Rank2<V, D>>::uniform(7).to_bf16().to_f32();
        let dy = CpuTensor::<f32, Rank2<N, D>>::uniform(8);
        let mut cpu = Embedding::<N, V, D>::new(weight.clone());
        let (cpu_y, cpu_ctx) = cpu.forward(tokens_usize);
        cpu.backward(cpu_ctx, dy.clone());

        let packed_weight: Vec<u32> = weight
            .as_slice()
            .chunks_exact(2)
            .map(|pair| {
                bf16::from_f32(pair[0]).to_bits() as u32
                    | ((bf16::from_f32(pair[1]).to_bits() as u32) << 16)
            })
            .collect();
        let weight_dev = DeviceBuffer::from_host(stream, &packed_weight)?;
        let tokens_dev = DeviceBuffer::from_host(stream, &tokens)?;
        let dy_dev = DeviceBuffer::from_host(stream, dy.as_slice())?;
        let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, N * D)?;
        let mut dw_dev = DeviceBuffer::<f32>::zeroed(stream, V * D)?;
        let mut dw_scatter_dev = DeviceBuffer::<f32>::zeroed(stream, V * D)?;
        module.embedding_forward(
            stream,
            LaunchConfig::for_num_elems((N * D) as u32),
            &weight_dev,
            &tokens_dev,
            D as u32,
            &mut y_dev,
        )?;
        module.embedding_backward(
            stream,
            LaunchConfig::for_num_elems((V * D) as u32),
            &tokens_dev,
            &dy_dev,
            N as u32,
            D as u32,
            &mut dw_dev,
        )?;
        unsafe {
            module.embedding_backward_scatter(
                stream,
                LaunchConfig::for_num_elems((N * D) as u32),
                &tokens_dev,
                &dy_dev,
                D as u32,
                &mut dw_scatter_dev,
            )?;
        }

        assert_close(
            "embedding y",
            &y_dev.to_host_vec(stream)?,
            cpu_y.as_slice(),
            0.0,
            0.0,
        );
        assert_close(
            "embedding dw",
            &dw_dev.to_host_vec(stream)?,
            cpu.dw.as_slice(),
            1e-6,
            1e-6,
        );
        assert_close(
            "embedding dw scatter vs naive",
            &dw_scatter_dev.to_host_vec(stream)?,
            &dw_dev.to_host_vec(stream)?,
            1e-6,
            1e-6,
        );
        Ok(())
    }
}

/// Round-trips grouped tensors through split and join at a shape that does
/// not divide the 256-thread launch rounding, so block-excess threads are
/// exercised in both kernels.
fn check_group_split_join(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    const ROWS: usize = 7;
    const WIDTH: usize = 13;
    let packed3 = CpuTensor::<f32, Rank2<ROWS, 39>>::uniform(11);
    let packed2 = CpuTensor::<f32, Rank2<ROWS, 26>>::uniform(12);
    let part = |packed: &[f32], groups: usize, group: usize| -> Vec<f32> {
        (0..ROWS * WIDTH)
            .map(|i| packed[(i / WIDTH * groups + group) * WIDTH + i % WIDTH])
            .collect()
    };

    let packed3_dev = DeviceBuffer::from_host(stream, packed3.as_slice())?;
    let mut first = DeviceBuffer::<f32>::zeroed(stream, ROWS * WIDTH)?;
    let mut second = DeviceBuffer::<f32>::zeroed(stream, ROWS * WIDTH)?;
    let mut third = DeviceBuffer::<f32>::zeroed(stream, ROWS * WIDTH)?;
    let elems = LaunchConfig::for_num_elems((ROWS * WIDTH) as u32);
    // SAFETY: the packed input and three outputs match ROWS x WIDTH.
    unsafe {
        module.split_group3(
            stream,
            elems,
            &packed3_dev,
            WIDTH as u32,
            &mut first,
            &mut second,
            &mut third,
        )
    }?;
    for (name, buffer, group) in [
        ("split_group3 first", &first, 0),
        ("split_group3 second", &second, 1),
        ("split_group3 third", &third, 2),
    ] {
        assert_close(
            name,
            &buffer.to_host_vec(stream)?,
            &part(packed3.as_slice(), 3, group),
            0.0,
            0.0,
        );
    }
    let mut joined3 = DeviceBuffer::<f32>::zeroed(stream, ROWS * 3 * WIDTH)?;
    // SAFETY: the three parts are disjoint [ROWS, WIDTH] tensors and the
    // output holds exactly ROWS * 3 * WIDTH elements.
    unsafe {
        module.join_group3(
            stream,
            elems,
            &first,
            &second,
            &third,
            WIDTH as u32,
            &mut joined3,
        )?;
    }
    assert_close(
        "join_group3",
        &joined3.to_host_vec(stream)?,
        packed3.as_slice(),
        0.0,
        0.0,
    );

    let packed2_dev = DeviceBuffer::from_host(stream, packed2.as_slice())?;
    // SAFETY: the packed input and two outputs match ROWS x WIDTH.
    unsafe {
        module.split_group2(
            stream,
            elems,
            &packed2_dev,
            WIDTH as u32,
            &mut first,
            &mut second,
        )
    }?;
    for (name, buffer, group) in [
        ("split_group2 first", &first, 0),
        ("split_group2 second", &second, 1),
    ] {
        assert_close(
            name,
            &buffer.to_host_vec(stream)?,
            &part(packed2.as_slice(), 2, group),
            0.0,
            0.0,
        );
    }
    let mut joined2 = DeviceBuffer::<f32>::zeroed(stream, ROWS * 2 * WIDTH)?;
    // SAFETY: both parts are disjoint [ROWS, WIDTH] tensors and the output
    // holds exactly ROWS * 2 * WIDTH elements.
    unsafe {
        module.join_group2(stream, elems, &first, &second, WIDTH as u32, &mut joined2)?;
    }
    assert_close(
        "join_group2",
        &joined2.to_host_vec(stream)?,
        packed2.as_slice(),
        0.0,
        0.0,
    );
    Ok(())
}

fn check_cross_entropy(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    check_cross_entropy_case::<5, 13>(stream, module)?;
    check_cross_entropy_case::<5, 517>(stream, module)
}

fn check_cross_entropy_case<const N: usize, const C: usize>(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    // SAFETY: logits, probabilities, gradients, targets, and launch geometry
    // all use the same N x C dimensions.
    unsafe {
        let logits = CpuTensor::<f32, Rank2<N, C>>::uniform(9).scale(5.0);
        let targets_usize = std::array::from_fn(|row| (row * 101 + C - 1) % C);
        let targets = targets_usize.map(|v| v as u32);
        let mut cpu = SoftmaxCrossEntropy::<N, C>;
        let (cpu_loss, cpu_ctx) = cpu.forward(SoftmaxCrossEntropyInput {
            logits: logits.clone(),
            targets: targets_usize,
        });
        let cpu_dx = cpu.backward(cpu_ctx, CpuTensor::from_slice(&[1.0])).logits;

        let logits_dev = DeviceBuffer::from_host(stream, logits.as_slice())?;
        let targets_dev = DeviceBuffer::from_host(stream, &targets)?;
        let mut probabilities_dev = DeviceBuffer::<f32>::zeroed(stream, N * C)?;
        let mut losses_dev = DeviceBuffer::<f32>::zeroed(stream, N)?;
        let mut dlogits_dev = DeviceBuffer::<f32>::zeroed(stream, N * C)?;
        let mut fused_losses_dev = DeviceBuffer::<f32>::zeroed(stream, N)?;
        let mut fused_dlogits_dev = DeviceBuffer::from_host(stream, logits.as_slice())?;
        module.softmax_forward(
            stream,
            LaunchConfig::for_num_elems((N * C) as u32),
            &logits_dev,
            C as u32,
            &mut probabilities_dev,
        )?;
        module.cross_entropy_loss(
            stream,
            LaunchConfig::for_num_elems(N as u32),
            &logits_dev,
            &targets_dev,
            N as u32,
            C as u32,
            &mut losses_dev,
        )?;
        module.softmax_cross_entropy_backward(
            stream,
            LaunchConfig::for_num_elems((N * C) as u32),
            &probabilities_dev,
            &targets_dev,
            1.0,
            N as u32,
            C as u32,
            &mut dlogits_dev,
        )?;
        let classifier_config = LaunchConfig {
            grid_dim: (N as u32, 1, 1),
            block_dim: (CLASSIFIER_THREADS as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        module.fused_classifier_forward(
            stream,
            classifier_config,
            &logits_dev,
            &targets_dev,
            N as u32,
            C as u32,
            &mut fused_losses_dev,
        )?;
        module.fused_classifier_backward_in_place(
            stream,
            classifier_config,
            &targets_dev,
            1.0,
            N as u32,
            C as u32,
            &mut fused_dlogits_dev,
        )?;

        let losses = losses_dev.to_host_vec(stream)?;
        let fused_losses = fused_losses_dev.to_host_vec(stream)?;
        assert_close(
            "fused classifier losses vs naive",
            &fused_losses,
            &losses,
            5e-5,
            2e-5,
        );
        let gpu_loss = fused_losses.iter().sum::<f32>() / N as f32;
        assert_close(
            "cross entropy loss",
            &[gpu_loss],
            cpu_loss.as_slice(),
            2e-5,
            2e-5,
        );
        assert_close(
            "fused classifier dx vs naive",
            &fused_dlogits_dev.to_host_vec(stream)?,
            &dlogits_dev.to_host_vec(stream)?,
            5e-6,
            2e-5,
        );
        assert_close(
            "fused classifier dx vs CPU",
            &fused_dlogits_dev.to_host_vec(stream)?,
            cpu_dx.as_slice(),
            5e-6,
            2e-5,
        );
        Ok(())
    }
}

/// The loss tail: every layer's auxiliary terms, then the one reduction that
/// turns the per-token losses and those terms into the scalar training loss.
///
/// The model parity gate compares the same scalar against the CPU reference,
/// but at a tolerance wide enough to hide the auxiliary term entirely: a
/// load-balancing loss that vanished, or that landed in another layer's row,
/// would pass it. This check is tight enough to see either.
fn check_loss_tail(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    const L: usize = 3;
    const E: usize = 4;
    const K: usize = 2;
    const N: usize = 517;
    const COEFFICIENT: f32 = 0.02;

    // SAFETY: every buffer is allocated from L, E, K and N, and each launch
    // covers exactly the shape its arguments describe.
    unsafe {
        let probabilities: [CpuTensor<f32, Rank2<N, E>>; L] =
            std::array::from_fn(|layer| CpuTensor::uniform(7 + layer as u64));
        // Distinct per layer, so a term stored in the wrong row shows up.
        let counts: [[u32; E]; L] = std::array::from_fn(|layer| {
            std::array::from_fn(|expert| (((layer * E + expert) * 37) % (N * K / E)) as u32)
        });
        let losses = CpuTensor::<f32, Rank1<N>>::uniform(11);

        // Held for the whole sequence: the launches below are stream-ordered
        // and the frees a scoped buffer would do are not.
        let probabilities_dev = probabilities
            .iter()
            .map(|layer| DeviceBuffer::from_host(stream, layer.as_slice()))
            .collect::<Result<Vec<_>, _>>()?;
        let counts_dev = counts
            .iter()
            .map(|layer| DeviceBuffer::from_host(stream, layer))
            .collect::<Result<Vec<_>, _>>()?;
        let losses_dev = DeviceBuffer::from_host(stream, losses.as_slice())?;
        let mut aux_terms = DeviceBuffer::<f32>::zeroed(stream, L * E)?;
        let mut loss = DeviceBuffer::<f32>::zeroed(stream, 1)?;

        for layer in 0..L {
            module.moe_aux_terms(
                stream,
                LaunchConfig {
                    grid_dim: (E as u32, 1, 1),
                    block_dim: (MOE_AUX_TERMS_THREADS as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                &probabilities_dev[layer],
                &counts_dev[layer],
                N as u32,
                E as u32,
                K as u32,
                layer as u32,
                &mut aux_terms,
            )?;
        }
        module.loss_mean_with_aux(
            stream,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (LOSS_TAIL_THREADS as u32, 1, 1),
                shared_mem_bytes: 0,
            },
            &losses_dev,
            N as u32,
            &aux_terms,
            COEFFICIENT,
            &mut loss,
        )?;

        let mut expected_terms = Vec::with_capacity(L * E);
        for layer in 0..L {
            for expert in 0..E {
                let mean = (0..N)
                    .map(|token| probabilities[layer].as_slice()[token * E + expert] as f64)
                    .sum::<f64>()
                    / N as f64;
                let assignment_fraction = counts[layer][expert] as f64 / (N * K) as f64;
                expected_terms.push((E as f64 * assignment_fraction * mean) as f32);
            }
        }
        let expected_loss = (losses.as_slice().iter().map(|&v| v as f64).sum::<f64>() / N as f64
            + COEFFICIENT as f64 * expected_terms.iter().map(|&v| v as f64).sum::<f64>())
            as f32;

        assert_close(
            "moe auxiliary terms",
            &aux_terms.to_host_vec(stream)?,
            &expected_terms,
            1e-7,
            1e-5,
        );
        assert_close(
            "loss mean with auxiliary",
            &loss.to_host_vec(stream)?,
            &[expected_loss],
            1e-6,
            1e-5,
        );
        Ok(())
    }
}
