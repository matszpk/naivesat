use gatenative::cpu_build_exec::*;
use gatenative::mapper::*;
use gatenative::opencl_build_exec::*;
use gatenative::parseq_mapper::*;
use gatenative::*;
use gatesim::*;

use clap::Parser;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};

use std::fs;
use std::str::FromStr;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct CommandArgs {
    circuit: String,
    unknowns: usize,
    simple: bool,
    #[arg(short = 'e', long, default_value_t = 24)]
    elem_inputs: usize,
    #[arg(short = 'G', long)]
    opencl_group_len: Option<usize>,
}

fn do_solve(circuit: Circuit<usize>, unknowns: usize, cmd_args: CommandArgs) {}

fn simple_solve(circuit: Circuit<usize>, unknowns: usize) {
    let input_len = circuit.input_len();
    let total_comb_num = 1u128 << unknowns;
    let total_step_num = 1u128 << input_len;
    let mut solution = None;
    'a: for v in 0..total_comb_num {
        let mut state = std::iter::repeat(false)
            .take(input_len - unknowns)
            .chain((0..unknowns).map(|b| (v >> b) & 1 != 0))
            .collect::<Vec<_>>();
        for _ in 0..total_step_num {
            let next_state = circuit.eval(state.clone());
            if *next_state.last().unwrap() {
                solution = Some((
                    v,
                    next_state
                        .into_iter()
                        .take(input_len)
                        .enumerate()
                        .fold(0u128, |a, (i, x)| a | (u128::from(x) << i)),
                ));
                break 'a;
            }
            state.copy_from_slice(&next_state[0..input_len]);
        }
    }
    if let Some((u, end)) = solution {
        println!("Stopped at {1:00$} {3:02$}", unknowns, u, input_len, end);
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
    if cmd_args.simple {
        simple_solve(circuit, cmd_args.unknowns);
    } else {
        do_solve(circuit, cmd_args.unknowns, cmd_args);
    }
}
