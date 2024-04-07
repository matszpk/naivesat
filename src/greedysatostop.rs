use gatenative::cpu_build_exec::*;
use gatenative::mapper::*;
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

#[derive(Clone, Parser, Debug)]
#[command(version, about, long_about = None)]
struct CommandArgs {
    circuit: String,
    unknowns: usize,
    #[arg(short = 'C', long)]
    opencl: Option<usize>,
    #[arg(short = 'P', long, default_value_t = 1)]
    partitions: usize,
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

fn do_solve(circuit: Circuit<usize>, cmd_args: CommandArgs) {
    let partitions = std::cmp::max(1, cmd_args.partitions);
    let input_len = circuit.input_len();
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
