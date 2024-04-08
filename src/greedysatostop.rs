use gatenative::cpu_build_exec::*;
//use gatenative::opencl_build_exec::*;
use gatenative::*;
use gatesim::*;

use clap::Parser;
// use opencl3::command_queue::CommandQueue;
// use opencl3::context::Context;
// use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
// use opencl3::kernel::{ExecuteKernel, Kernel};
// use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
// use opencl3::program::Program;
// use opencl3::types::{cl_ulong, CL_BLOCKING};

use rayon::prelude::*;

use std::fs;
use std::ops::Range;
use std::str::FromStr;
use std::sync::atomic::{self, AtomicU32};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

#[derive(Clone, Parser, Debug)]
#[command(version, about, long_about = None)]
struct CommandArgs {
    circuit: String,
    unknowns: usize,
    #[arg(short = 'v', long)]
    verify: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Solution {
    start: u64,
    end: u64,
}

#[derive(Clone, Copy, Debug)]
enum FinalResult {
    Solution(Solution),
    NoSolution,
}

struct AtomicU32Array<'a> {
    original: Vec<u32>,
    atomic: &'a [AtomicU32],
}

impl From<Vec<u32>> for AtomicU32Array<'_> {
    fn from(mut t: Vec<u32>) -> Self {
        let atomic = unsafe {
            &*std::ptr::slice_from_raw_parts(
                t.as_mut_slice().as_mut_ptr().cast::<AtomicU32>(),
                t.len(),
            )
        };
        Self {
            original: t,
            atomic,
        }
    }
}

impl<'a> AtomicU32Array<'a> {
    #[inline]
    fn get(&self, i: usize) -> &'a AtomicU32 {
        &self.atomic[i]
    }

    #[inline]
    fn as_slice(&self) -> &'a [AtomicU32] {
        &self.atomic
    }

    #[inline]
    fn len(&self) -> usize {
        self.original.len()
    }
}

//
// join nexts
//

fn join_nexts(input_len: usize, nexts: Arc<AtomicU32Array>) {
    let cpu_num = rayon::current_num_threads();
    let chunk_num = std::cmp::min(std::cmp::max(cpu_num * 8, 64), nexts.len() >> 6);
    let chunk_len = nexts.len() / chunk_num;
    let stop_mask = 1u32 << input_len;
    let next_mask = stop_mask - 1u32;
    nexts
        .as_slice()
        .chunks(chunk_len)
        .par_bridge()
        .for_each(|chunk| {
            for cell in chunk {
                std::sync::atomic::fence(atomic::Ordering::SeqCst);
                let old_value = cell.load(atomic::Ordering::SeqCst);
                let old_next = old_value & next_mask;
                std::sync::atomic::fence(atomic::Ordering::SeqCst);
                if (old_value & stop_mask) == 0 {
                    cell.store(
                        nexts.get(old_next as usize).load(atomic::Ordering::SeqCst),
                        atomic::Ordering::SeqCst,
                    );
                }
                std::sync::atomic::fence(atomic::Ordering::SeqCst);
            }
        });
}

fn join_nexts_exact_u32(nexts: Arc<AtomicU32Array>) {
    let input_len = 32;
    let nexts_len = 1usize << input_len;
    let cpu_num = rayon::current_num_threads();
    let chunk_num = std::cmp::min(std::cmp::max(cpu_num * 8, 64), nexts_len >> 6);
    let chunk_len = nexts_len / chunk_num;
    let chunk_len = (chunk_len + 31) & !31usize;
    nexts.as_slice()[0..nexts_len]
        .chunks(chunk_len)
        .zip(nexts.as_slice()[nexts_len..].chunks(chunk_len >> 5))
        .par_bridge()
        .for_each(|(chunk, stop_chunk)| {
            for (i, cell) in chunk.iter().enumerate() {
                std::sync::atomic::fence(atomic::Ordering::SeqCst);
                let old_next = cell.load(atomic::Ordering::SeqCst);
                let old_stop =
                    ((stop_chunk[i >> 5].load(atomic::Ordering::SeqCst) >> (i & 31)) & 1) != 0;
                std::sync::atomic::fence(atomic::Ordering::SeqCst);
                if !old_stop {
                    cell.store(
                        nexts.get(old_next as usize).load(atomic::Ordering::SeqCst),
                        atomic::Ordering::SeqCst,
                    );
                    stop_chunk[i >> 5].fetch_or(
                        ((nexts.as_slice()[nexts_len + ((old_next >> 5) as usize)]
                            .load(atomic::Ordering::SeqCst)
                            >> (old_next & 31))
                            & 1)
                            << (i & 31),
                        atomic::Ordering::SeqCst,
                    );
                }
                std::sync::atomic::fence(atomic::Ordering::SeqCst);
            }
        });
}

fn check_stop(input_len: usize, nexts: Arc<AtomicU32Array>) -> bool {
    let cpu_num = rayon::current_num_threads();
    let chunk_num = std::cmp::min(std::cmp::max(cpu_num * 8, 64), nexts.len() >> 6);
    let chunk_len = nexts.len() / chunk_num;
    let stop_mask = 1u32 << input_len;
    let stop = Arc::new(AtomicU32::new(0));
    nexts
        .as_slice()
        .chunks(chunk_len)
        .par_bridge()
        .for_each(|chunk| {
            if chunk
                .iter()
                .any(|cell| (cell.load(atomic::Ordering::SeqCst) & stop_mask) != 0)
            {
                stop.fetch_or(1, atomic::Ordering::SeqCst);
            }
        });
    stop.load(atomic::Ordering::SeqCst) != 0
}

fn find_solution(
    input_len: usize,
    unknowns: usize,
    nexts: Arc<AtomicU32Array>,
) -> Option<Solution> {
    let unknown_state_num = 1 << unknowns;
    let cpu_num = rayon::current_num_threads();
    let chunk_num = std::cmp::min(std::cmp::max(cpu_num * 8, 64), unknown_state_num >> 6);
    let chunk_len = unknown_state_num / chunk_num;
    let stop_mask = 1u32 << input_len;
    let next_mask = stop_mask - 1;
    let unknowns_mult = 1 << (input_len - unknowns);
    let result = Mutex::new(None);
    nexts
        .as_slice()
        .chunks(unknowns_mult * chunk_len)
        .enumerate()
        .par_bridge()
        .for_each(|(ch_idx, chunk)| {
            for (i, v) in chunk.chunks(unknowns_mult).enumerate() {
                let value = v[0].load(atomic::Ordering::SeqCst);
                if (value & stop_mask) != 0 {
                    let mut r = result.lock().unwrap();
                    *r = Some(Solution {
                        start: ((i + (ch_idx * chunk_len)) as u64) << (input_len - unknowns),
                        end: (value & next_mask) as u64,
                    });
                }
            }
        });
    result.into_inner().unwrap()
}

//
// main solver code
//

fn gen_output_transform_def(postfix: &str, range: Range<usize>) -> String {
    let args = range.clone().map(|i| format!("o{}", i)).collect::<Vec<_>>();
    format!(
        "#define OUTPUT_TRANSFORM_{}(O) OUTPUT_TRANSFORM_B{}(O,{})\n",
        postfix,
        range.end - range.start,
        args.join(",")
    )
}

fn gen_output_transform_code(output_len: usize) -> String {
    gen_output_transform_def("FIRST_32", 0..std::cmp::min(32, output_len))
}

const AGGR_OUTPUT_CPU_CODE: &str = r##"{
#if OUTPUT_NUM <= 32
    uint32_t* output_u = ((uint32_t*)output) + idx *
        ((OUTPUT_NUM + 31) >> 5) * TYPE_LEN;
    OUTPUT_TRANSFORM_FIRST_32(output_u);
#else // end of OUTPUT_NUM <= 32
#  if OUTPUT_NUM == 33
    uint32_t* output_u = ((uint32_t*)output) + idx *
        ((OUTPUT_NUM + 31) >> 5) * TYPE_LEN;
    OUTPUT_TRANSFORM_FIRST_32(output_u);
    uint32_t* output_stop = ((uint32_t*)output) + (1ULL << (OUTPUT_NUM - 1)) +
            ((TYPE_LEN >> 5) * idx);
    GET_U32_ALL(output_stop, o32);
#  else
#  error "Unsupported!"
#  endif
#endif
}"##;

fn do_solve_with_cpu_builder(circuit: Circuit<usize>, cmd_args: &CommandArgs) -> FinalResult {
    let input_len = circuit.input_len();
    let output_len = input_len + 1;
    let words_per_elem = (output_len + 31) >> 5;
    let output_buf_len = if input_len < 32 {
        words_per_elem * (1 << input_len)
    } else if input_len == 32 {
        (1 << input_len) + (1 << (input_len - 5))
    } else {
        panic!("Unsupported");
    };
    let (output, start) = {
        let mut builder = CPUBuilder::new_parallel(None, Some(2048));
        builder.transform_helpers();
        builder.user_defs(&format!("#define OUTPUT_NUM ({})\n", output_len));
        builder.user_defs(&gen_output_transform_code(output_len));
        builder.add_with_config(
            "formula",
            circuit,
            CodeConfig::new()
                .elem_inputs(Some(&(0..input_len).collect::<Vec<usize>>()))
                .aggr_output_code(Some(AGGR_OUTPUT_CPU_CODE))
                .aggr_output_len(Some(output_buf_len))
                .dont_clear_outputs(true),
        );
        let mut execs = builder.build().unwrap();
        let start = SystemTime::now();
        let input = execs[0].new_data(16);
        println!("Calculate first nexts");
        (execs[0].execute(&input, 0).unwrap().release(), start)
    };
    let nexts = Arc::new(AtomicU32Array::from(output));
    let mut final_result = FinalResult::NoSolution;
    if check_stop(input_len, nexts.clone()) {
        for i in 0..input_len {
            println!("Joining nexts: Stage: {} / {}", i, input_len);
            join_nexts(input_len, nexts.clone());
            if let Some(sol) = find_solution(input_len, cmd_args.unknowns, nexts.clone()) {
                final_result = FinalResult::Solution(sol);
                break;
            }
        }
    }
    let time = start.elapsed().unwrap();
    println!("Time: {}", time.as_secs_f64());
    final_result
}

fn do_solve(circuit: Circuit<usize>, cmd_args: CommandArgs) {
    let input_len = circuit.input_len();
    assert!(input_len < 32);
    let result = do_solve_with_cpu_builder(circuit.clone(), &cmd_args);
    if let FinalResult::Solution(sol) = result {
        println!("Solution: {:?}", sol);
        if cmd_args.verify {
            let mut state = sol.start;
            loop {
                let state_vec = (0..input_len)
                    .map(|b| ((state >> b) & 1) != 0)
                    .collect::<Vec<_>>();
                let output_vec = circuit.eval(state_vec.clone());
                state = output_vec[0..input_len]
                    .iter()
                    .take(input_len)
                    .enumerate()
                    .fold(0u64, |a, (i, x)| a | (u64::from(*x) << i));
                if output_vec[input_len] {
                    break;
                }
            }
            println!("Verified: end={}", state);
            if sol.end != state {
                println!("INCORRECT");
            }
        }
    } else {
        println!("Unsatisfiable!");
    }
}

fn main() {
    // for x in get_all_devices(CL_DEVICE_TYPE_GPU).unwrap() {
    //     println!("OpenCLDevice: {:?}", x);
    // }
    let cmd_args = CommandArgs::parse();
    let circuit_str = fs::read_to_string(cmd_args.circuit.clone()).unwrap();
    let circuit = Circuit::<usize>::from_str(&circuit_str).unwrap();
    let input_len = circuit.input_len();
    assert_eq!(input_len + 1, circuit.outputs().len());
    assert!(cmd_args.unknowns <= input_len);
    do_solve(circuit, cmd_args);
}
