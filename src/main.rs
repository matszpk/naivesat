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
use std::time::{Duration, SystemTime};

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

const AGGR_OUTPUT_CPU_CODE: &str = r##"{
    uint32_t i = 0;
    uint32_t* out = (uint32_t*)output;
    for (i = 0; i < (TYPE_LEN >> 5); i++) {
        uint32_t v;
        GET_U32(v, o0, i);
        if ((v != 0) && (__sync_fetch_and_or(out, v) == 0)) {
            out[1] = idx;
            out[2] = i;
            out[3] = v;
        }
    }
}"##;

const AGGR_OUTPUT_OPENCL_CODE: &str = r##"{
    uint i = 0;
    global uint* out = (global uint*)output;
    for (i = 0; i < (TYPE_LEN >> 5); i++) {
        uint v;
        GET_U32(v, o0, i);
        if ((v != 0) && (atomic_or(out, v) == 0)) {
            out[1] = idx;
            out[2] = i;
            out[3] = v;
        }
    }
}"##;

fn do_command_with_par_mapper<'a>(
    mut mapper: ParBasicMapperBuilder<
        'a,
        CPUDataReader<'a>,
        CPUDataWriter<'a>,
        CPUDataHolder,
        CPUExecutor,
        CPUBuilder<'a>,
    >,
    circuit: Circuit<usize>,
    elem_inputs: usize,
) -> Option<u128> {
    let input_len = circuit.input_len();
    let arg_steps = 1u128 << (input_len - elem_inputs);
    mapper.add_with_config(
        "formula",
        circuit,
        CodeConfig::new()
            .elem_inputs(Some(&(0..elem_inputs).collect::<Vec<usize>>()))
            .arg_inputs(Some(&(elem_inputs..input_len).collect::<Vec<usize>>()))
            .aggr_output_code(Some(AGGR_OUTPUT_CPU_CODE))
            .aggr_output_len(Some(4)),
    );
    let type_len = mapper.type_len();
    let mut execs = mapper.build().unwrap();
    let input = execs[0].new_data(16);
    let start = SystemTime::now();
    let result = execs[0]
        .execute_direct(
            &input,
            None,
            |_, output, arg| {
                println!("Step: {} / {}", arg, arg_steps);
                if output[0] != 0 {
                    let elem_idx =
                        output[3].trailing_zeros() | (output[2] << 5) | (output[1] * type_len);
                    Some((elem_idx as u128) | ((arg as u128) << elem_inputs))
                } else {
                    None
                }
            },
            |a, b| {
                if a.is_some() {
                    a
                } else {
                    b
                }
            },
            |a| a.is_some(),
        )
        .unwrap();
    let time = start.elapsed().unwrap();
    println!("Time: {}", time.as_secs_f64());
    result
}

fn do_command_with_opencl_mapper<'a>(
    mut mapper: BasicMapperBuilder<
        'a,
        OpenCLDataReader<'a>,
        OpenCLDataWriter<'a>,
        OpenCLDataHolder,
        OpenCLExecutor,
        OpenCLBuilder<'a>,
    >,
    circuit: Circuit<usize>,
    elem_inputs: usize,
) -> Option<u128> {
    let input_len = circuit.input_len();
    let arg_steps = 1u128 << (input_len - elem_inputs);
    mapper.add_with_config(
        "formula",
        circuit,
        CodeConfig::new()
            .elem_inputs(Some(&(0..elem_inputs).collect::<Vec<usize>>()))
            .arg_inputs(Some(&(elem_inputs..input_len).collect::<Vec<usize>>()))
            .aggr_output_code(Some(AGGR_OUTPUT_OPENCL_CODE))
            .aggr_output_len(Some(4)),
    );
    let type_len = mapper.type_len();
    let mut execs = mapper.build().unwrap();
    let input = execs[0].new_data(16);
    let start = SystemTime::now();
    let result = execs[0]
        .execute_direct(
            &input,
            None,
            |result, _, output, arg| {
                println!("Step: {} / {}", arg, arg_steps);
                if result.is_some() {
                    result
                } else if output[0] != 0 {
                    let elem_idx =
                        output[3].trailing_zeros() | (output[2] << 5) | (output[1] * type_len);
                    Some((elem_idx as u128) | ((arg as u128) << elem_inputs))
                } else {
                    None
                }
            },
            |a| a.is_some(),
        )
        .unwrap();
    let time = start.elapsed().unwrap();
    println!("Time: {}", time.as_secs_f64());
    result
}

fn do_command_with_parseq_mapper<'a>(
    mut mapper: ParSeqMapperBuilder<
        'a,
        CPUDataReader<'a>,
        CPUDataWriter<'a>,
        CPUDataHolder,
        CPUExecutor,
        CPUBuilder<'a>,
        OpenCLDataReader<'a>,
        OpenCLDataWriter<'a>,
        OpenCLDataHolder,
        OpenCLExecutor,
        OpenCLBuilder<'a>,
    >,
    circuit: Circuit<usize>,
    elem_inputs: usize,
) -> Option<u128> {
    let input_len = circuit.input_len();
    let arg_steps = 1u128 << (input_len - elem_inputs);
    mapper.add_with_config(
        "formula",
        circuit,
        &(elem_inputs..input_len).collect::<Vec<usize>>(),
        Some(&(0..elem_inputs).collect::<Vec<usize>>()),
        |sel| match sel {
            ParSeqSelection::Par => ParSeqDynamicConfig::new()
                .aggr_output_code(Some(AGGR_OUTPUT_CPU_CODE))
                .aggr_output_len(Some(4)),
            ParSeqSelection::Seq(_) => ParSeqDynamicConfig::new()
                .aggr_output_code(Some(AGGR_OUTPUT_OPENCL_CODE))
                .aggr_output_len(Some(4)),
        },
    );
    let cpu_type_len = mapper.type_len(ParSeqSelection::Par);
    let opencl_type_lens = (0..mapper.seq_builder_num())
        .map(|i| mapper.type_len(ParSeqSelection::Seq(i)))
        .collect::<Vec<_>>();
    let mut execs = mapper.build().unwrap();
    let input = execs[0].new_data(16);
    let start = SystemTime::now();
    let result = execs[0]
        .execute_direct(
            &input,
            None,
            |sel, _, output, arg| {
                println!("Step: {} / {}", arg, arg_steps);
                let type_len = match sel {
                    ParSeqSelection::Par => cpu_type_len,
                    ParSeqSelection::Seq(i) => opencl_type_lens[i],
                };
                if output[0] != 0 {
                    let elem_idx =
                        output[3].trailing_zeros() | (output[2] << 5) | (output[1] * type_len);
                    Some((elem_idx as u128) | ((arg as u128) << elem_inputs))
                } else {
                    None
                }
            },
            |a, b| {
                if a.is_some() {
                    a
                } else {
                    b
                }
            },
            |a| a.is_some(),
        )
        .unwrap();
    let time = start.elapsed().unwrap();
    println!("Time: {}", time.as_secs_f64());
    result
}

fn do_command(circuit: Circuit<usize>, cmd_args: CommandArgs) {
    let input_len = circuit.input_len();
    let result = if input_len >= 10 {
        let elem_inputs = if cmd_args.elem_inputs >= input_len {
            input_len - 1
        } else {
            cmd_args.elem_inputs
        };
        assert!(elem_inputs > 0 && elem_inputs <= 37);
        assert!(input_len - elem_inputs > 0 && input_len - elem_inputs <= 64);
        assert_eq!(circuit.outputs().len(), 1);
        println!("Elem inputs: {}", elem_inputs);
        let opencl_config = OpenCLBuilderConfig {
            optimize_negs: true,
            group_len: cmd_args.opencl_group_len,
            group_vec: false,
        };
        let exec_type = cmd_args.exec_type;
        match exec_type {
            ExecType::CPU => {
                println!("Execute in CPU");
                let builder = ParBasicMapperBuilder::new(CPUBuilder::new(None));
                do_command_with_par_mapper(builder, circuit.clone(), elem_inputs)
            }
            ExecType::OpenCL(didx) => {
                println!("Execute in OpenCL device={}", didx);
                let device = Device::new(
                    *get_all_devices(CL_DEVICE_TYPE_GPU)
                        .unwrap()
                        .get(didx)
                        .unwrap(),
                );
                let builder = BasicMapperBuilder::new(OpenCLBuilder::new(
                    &device,
                    Some(opencl_config.clone()),
                ));
                do_command_with_opencl_mapper(builder, circuit.clone(), elem_inputs)
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
                            OpenCLBuilder::new(&device, Some(opencl_config.clone()))
                        })
                        .collect::<Vec<_>>()
                } else if let ExecType::CPUAndOpenCL1D(didx) = exec_type {
                    println!("Execute in CPUAndOpenCL1D");
                    get_all_devices(CL_DEVICE_TYPE_GPU).unwrap()[didx..=didx]
                        .into_iter()
                        .map(|dev_id| {
                            let device = Device::new(dev_id.clone());
                            [
                                OpenCLBuilder::new(&device, Some(opencl_config.clone())),
                                OpenCLBuilder::new(&device, Some(opencl_config.clone())),
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
                            OpenCLBuilder::new(&device, Some(opencl_config.clone()))
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
                                OpenCLBuilder::new(&device, Some(opencl_config.clone())),
                                OpenCLBuilder::new(&device, Some(opencl_config.clone())),
                            ]
                        })
                        .flatten()
                        .collect::<Vec<_>>()
                };
                let builder = ParSeqMapperBuilder::new(par_builder, seq_builders);
                println!("Do execute");
                do_command_with_parseq_mapper(builder, circuit.clone(), elem_inputs)
            }
        }
    } else {
        let mut result = None;
        for v in 0..1 << input_len {
            if circuit.eval((0..input_len).map(|b| (v >> b) & 1 != 0))[0] {
                result = Some(v);
                break;
            }
        }
        result
    };

    if let Some(result) = result {
        if !circuit.eval((0..input_len).map(|b| (result >> b) & 1 != 0))[0] {
            println!("INCORRECT!! {1:00$b}", input_len, result);
        } else {
            println!("Found Input: {1:00$b}", input_len, result);
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
    do_command(circuit, cmd_args);
}
