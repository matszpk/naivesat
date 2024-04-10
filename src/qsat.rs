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
use std::time::SystemTime;

// IDEA HOW TO WRITE REDUCTION:
// level 1: reduce single execution to one bit of result
//       for OpenCL: use work group to reduce to some level.
//                   and use extra kernel to reduce more and reduce at CPU.
//                   or (alternatively) use reduction at CPU.
//         or just make all reduction for execution in OpenCL kernel:
//           by multi level global reduction: global1, global2, ... final by adding
//           additional bit: filled entry.
// level 2: reduce in argument bits:
//       write special object that reduces and remove reduced data and handle
//       final reduction.
//

#[derive(Clone, Copy, Debug)]
enum ExecType {
    CPU,
    OpenCL(usize),
    CPUAndOpenCL,
    CPUAndOpenCLD,
    CPUAndOpenCL1(usize),
    CPUAndOpenCL1D(usize),
}

impl FromStr for ExecType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        if s == "cpu" {
            Ok(ExecType::CPU)
        } else if s.starts_with("opencl:") {
            Ok(ExecType::OpenCL(
                usize::from_str(&s[7..]).map_err(|e| e.to_string())?,
            ))
        } else if s == "cpu_and_opencl" {
            Ok(ExecType::CPUAndOpenCL)
        } else if s == "cpu_and_opencl_d" {
            Ok(ExecType::CPUAndOpenCLD)
        } else if s.starts_with("cpu_and_opencl_1:") {
            Ok(ExecType::CPUAndOpenCL1(
                usize::from_str(&s[17..]).map_err(|e| e.to_string())?,
            ))
        } else if s.starts_with("cpu_and_opencl_1d:") {
            Ok(ExecType::CPUAndOpenCL1D(
                usize::from_str(&s[18..]).map_err(|e| e.to_string())?,
            ))
        } else {
            Err("Unknown exec type".to_string())
        }
    }
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct CommandArgs {
    circuit: String,
    #[arg(short = 'e', long, default_value_t = 24)]
    elem_inputs: usize,
    #[arg(short = 't', long)]
    exec_type: ExecType,
    #[arg(short = 'G', long)]
    opencl_group_len: Option<usize>,
}

fn main() {
    println!(
        "OpenCL devices: {:?}",
        get_all_devices(CL_DEVICE_TYPE_GPU).unwrap_or(vec![])
    );
    let cmd_args = CommandArgs::parse();
    let circuit_str = fs::read_to_string(cmd_args.circuit.clone()).unwrap();
    let qcircuit = QuantCircuit::<usize>::from_str(&circuit_str).unwrap();
    // do_command(circuit, cmd_args);
}
