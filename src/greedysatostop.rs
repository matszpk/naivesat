use gatenative::cpu_build_exec::*;
use gatenative::opencl_build_exec::*;
use gatenative::*;
use gatesim::*;

use clap::Parser;
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{cl_ulong, CL_BLOCKING};

use rayon::prelude::*;

use std::cell::UnsafeCell;
use std::fs;
use std::ops::Range;
use std::str::FromStr;
use std::sync::atomic::{self, AtomicU32, AtomicU64};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

// recommended setup for 32-bit problem for 16GB main memory and 8GB GPU RAM:
// partitions=8, main_partition_mult=2.

#[derive(Clone, Parser, Debug)]
#[command(version, about, long_about = None)]
struct CommandArgs {
    circuit: String,
    unknowns: usize,
    #[arg(short = 'C', long)]
    opencl: Option<usize>,
    #[arg(short = 'P', long, default_value_t = 1)]
    partitions: usize,
    #[arg(
        short = 'Q',
        long,
        default_value_t = 2,
        help = "How many partitions contains main partitions"
    )]
    main_partition_mult: usize,
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
    uint32_t* output_u = ((uint32_t*)output) + idx *
        ((OUTPUT_NUM + 31) >> 5) * TYPE_LEN;
#if OUTPUT_NUM <= 32
    OUTPUT_TRANSFORM_FIRST_32(output_u);
#else
#error "Unsupported!"
#endif
}"##;

fn do_solve_with_cpu_builder<'a>(
    mut builder: CPUBuilder<'a>,
    circuit: Circuit<usize>,
    unknowns: usize,
    elem_inputs: usize,
    unknown_fill_bits: usize,
    cmd_args: &CommandArgs,
) -> Option<FinalResult> {
    let input_len = circuit.input_len();
    let output_len = input_len + 1;
    let arg_steps = 1u128 << (input_len - elem_inputs);
    builder.transform_helpers();
    builder.user_defs(&format!("#define OUTPUT_NUM ({})\n", output_len));
    builder.user_defs(&gen_output_transform_code(output_len));
    let words_per_elem = (output_len + 31) >> 5;
    builder.add_with_config(
        "formula",
        circuit,
        CodeConfig::new()
            .elem_inputs(Some(&(0..input_len).collect::<Vec<usize>>()))
            .aggr_output_code(Some(AGGR_OUTPUT_CPU_CODE))
            .aggr_output_len(Some(words_per_elem * (1 << input_len)))
            .dont_clear_outputs(true),
    );
    let mut execs = builder.build().unwrap();
    let input = execs[0].new_data(16);
    let output = execs[0].execute(&input, 0).unwrap();
    None
}

fn do_solve(circuit: Circuit<usize>, cmd_args: CommandArgs) {
    let partitions = std::cmp::max(1, cmd_args.partitions);
    let main_partition_mult = std::cmp::max(1, cmd_args.main_partition_mult);
    let input_len = circuit.input_len();
    assert!(input_len < 32);
    let result = Some(FinalResult::NoSolution);
    let result = result.unwrap();
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
    for x in get_all_devices(CL_DEVICE_TYPE_GPU).unwrap() {
        println!("OpenCLDevice: {:?}", x);
    }
    let cmd_args = CommandArgs::parse();
    let circuit_str = fs::read_to_string(cmd_args.circuit.clone()).unwrap();
    let circuit = Circuit::<usize>::from_str(&circuit_str).unwrap();
    let input_len = circuit.input_len();
    assert_eq!(input_len + 1, circuit.outputs().len());
    assert!(cmd_args.unknowns < input_len);
    do_solve(circuit, cmd_args);
}
