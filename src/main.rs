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
}

fn do_command(circuit: Circuit<usize>, cmd_args: CommandArgs, exec_type: ExecType) {
    //let (circuit, input_map) = gen_bench_circuit_type(ctype, n);
    //println!("Circuit length: {}", circuit.len());
    match exec_type {
        ExecType::CPU => {
            println!("Execute in CPU");
            let builder = ParBasicMapperBuilder::new(CPUBuilder::new(None));
            //do_command_with_par_mapper(cmd, builder, n, ctype, circuit, input_map);
        }
        ExecType::OpenCL(didx) => {
            println!("Execute in OpenCL device={}", didx);
            let device = Device::new(
                *get_all_devices(CL_DEVICE_TYPE_GPU)
                    .unwrap()
                    .get(didx)
                    .unwrap(),
            );
            let builder = BasicMapperBuilder::new(OpenCLBuilder::new(&device, None));
            //do_command_with_mapper(cmd, builder, n, ctype, circuit, input_map);
        }
        ExecType::CPUAndOpenCL
        | ExecType::CPUAndOpenCLD
        | ExecType::CPUAndOpenCL1(_)
        | ExecType::CPUAndOpenCL1D(_) => {
            let par_builder = CPUBuilder::new(None);
            let seq_builders = if let ExecType::CPUAndOpenCL1(didx) = exec_type {
                println!("Execute in CPUAndOpenCL1");
                get_all_devices(CL_DEVICE_TYPE_GPU).unwrap()[didx..=didx]
                    .into_iter()
                    .map(|dev_id| {
                        let device = Device::new(dev_id.clone());
                        OpenCLBuilder::new(&device, None)
                    })
                    .collect::<Vec<_>>()
            } else if let ExecType::CPUAndOpenCL1D(didx) = exec_type {
                println!("Execute in CPUAndOpenCL1D");
                get_all_devices(CL_DEVICE_TYPE_GPU).unwrap()[didx..=didx]
                    .into_iter()
                    .map(|dev_id| {
                        let device = Device::new(dev_id.clone());
                        [
                            OpenCLBuilder::new(&device, None),
                            OpenCLBuilder::new(&device, None),
                        ]
                    })
                    .flatten()
                    .collect::<Vec<_>>()
            } else if matches!(exec_type, ExecType::CPUAndOpenCL) {
                println!("Execute in CPUAndOpenCL");
                get_all_devices(CL_DEVICE_TYPE_GPU)
                    .unwrap()
                    .into_iter()
                    .map(|dev_id| {
                        let device = Device::new(dev_id);
                        OpenCLBuilder::new(&device, None)
                    })
                    .collect::<Vec<_>>()
            } else {
                println!("Execute in CPUAndOpenCLD");
                get_all_devices(CL_DEVICE_TYPE_GPU)
                    .unwrap()
                    .into_iter()
                    .map(|dev_id| {
                        let device = Device::new(dev_id);
                        [
                            OpenCLBuilder::new(&device, None),
                            OpenCLBuilder::new(&device, None),
                        ]
                    })
                    .flatten()
                    .collect::<Vec<_>>()
            };
            let builder = ParSeqMapperBuilder::new(par_builder, seq_builders);
            println!("Do execute");
            //let (exec, input) = do_exec_bench0_parseq_0(builder, circuit, input_map);
            //do_exec_bench0_parseq_1(exec, &input);
        }
    }
}

fn main() {
    let cmd_args = CommandArgs::parse();
    let circuit =
        Circuit::<usize>::from_str(&fs::read_to_string(cmd_args.circuit).unwrap()).unwrap();
    println!("Hello, world!: {:?}", cmd_args.exec_type);
}
