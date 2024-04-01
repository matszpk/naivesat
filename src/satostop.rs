use gatenative::cpu_build_exec::*;
use gatenative::mapper::*;
use gatenative::opencl_build_exec::*;
use gatenative::parseq_mapper::*;
use gatenative::*;
use gatesim::*;

use clap::Parser;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use rayon::prelude::*;

use std::fs;
use std::ops::Range;
use std::str::FromStr;
use std::time::{Duration, SystemTime};

// HashMap entry structure
// current: State - current state
// next: State - next state
// path_len: State - length of path between current and next node
// pred - number of predecessors for current node
// state - entry state. Values
//   * unused, empty
//   * used and not resolved
//   * stopped at next
//   * looped
//   * flag that applied to other states: during allocation
//     (can't be replaced by other thread).

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
    unknowns: usize,
    simple: bool,
    #[arg(short = 'e', long, default_value_t = 24)]
    elem_inputs: usize,
    #[arg(short = 't', long)]
    exec_type: ExecType,
    #[arg(short = 'G', long)]
    opencl_group_len: Option<usize>,
}

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
    let mut defs = gen_output_transform_def("FIRST_32", 0..std::cmp::min(32, output_len));
    if output_len > 32 {
        defs.push_str(&gen_output_transform_def("SECOND_32", 32..output_len));
    }
    defs
}

// INFO: only higher bits are important because they impacts on hashmap entry index.
#[inline]
fn hash_function_64(bits: usize, value: u64) -> usize {
    let mask = u64::try_from((1u128 << bits) - 1).unwrap();
    let half_bits = bits >> 1;
    let temp = value * 9615409803190489167u64;
    let temp = (temp << half_bits) | (temp >> (bits - half_bits));
    let hash = (value * 6171710485021949031u64) ^ temp ^ 0xb89d2ecda078ca1f;
    usize::try_from(hash & mask).unwrap()
}

const HASH_FUNC_OPENCL_DEF: &str = r##"
#define HASH_FN(H,V) {
    const uint bits = OUTPUT_NUM - 1;
    const ulong mask = (1ULL << bits) - 1ULL;
    const uint half_bits = bits >> 1;
    const ulong temp = ((V) * 9615409803190489167ULL);
    (H) = (((V) * 6171710485021949031ULL) ^
        ((temp << half_bits) | (temp >> (bits - half_bits))) ^
        0xb89d2ecda078ca1fULL) & mask;
}
"##;

const AGGR_OUTPUT_CPU_CODE: &str = r##"{
    uint32_t* output_u = ((uint32_t*)output) + idx *
        ((OUTPUT_NUM + 31) >> 5) * TYPE_LEN;
#if OUTPUT_NUM <= 32
    OUTPUT_TRANSFORM_FIRST_32(output_u);
#else
    uint32_t i;
    uint32_t temp[((OUTPUT_NUM + 31) >> 5) * TYPE_LEN];
    OUTPUT_TRANSFORM_FIRST_32(temp);
    OUTPUT_TRANSFORM_SECOND_32(temp + 32 * (TYPE_LEN >> 5));
    for (i = 0; i < TYPE_LEN; i++) {
        output_u[i*2] = temp[i];
        output_u[i*2 + 1] = temp[i + TYPE_LEN];
    }
#endif
}"##;

#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct HashEntry {
    current: u64,
    next: u64,
    steps: u64,
    predecessors: u32,
    state: u32,
}

const HASH_STATE_UNUSED: u32 = 0;
const HASH_STATE_USED: u32 = 1;
const HASH_STATE_STOPPED: u32 = 2;
const HASH_STATE_LOOPED: u32 = 3;
const HASH_STATE_RESERVED_BY_OTHER_FLAG: u32 = 4;

// TODO: add setting looped entry from number of steps.
// 2**state_bits or more steps - loop, check stop before check loop.
fn join_to_hashmap_cpu(
    output_len: usize,
    arg_bit_place: usize,
    arg: u64,
    outputs: &[u32],
    hashmap: &mut [HashEntry],
) {
    let cpu_num = rayon::current_num_threads();
    let chunk_num = std::cmp::min(std::cmp::max(cpu_num * 8, 64), hashmap.len() >> 6);
    let chunk_len = hashmap.len() / chunk_num;
    let arg_start = arg << arg_bit_place;
    let arg_end = arg_start + (1u64 << arg_bit_place);
    // word_per_elem - elem length in outputs in words (can be 1 or 2).
    let word_per_elem = (output_len + 31) >> 5;
    let state_mask = (1u64 << (output_len - 1)) - 1;
    hashmap
        .chunks_mut(chunk_len)
        .enumerate()
        .par_bridge()
        .for_each(|(chunk_id, hashchunk)| {
            for (i, he) in hashchunk.iter_mut().enumerate() {
                if he.state == HASH_STATE_USED && arg_start <= he.next && he.next < arg_end {
                    // update hash entry next field.
                    let output_entry_start =
                        word_per_elem * usize::try_from(he.next - arg_start).unwrap();
                    let output = if word_per_elem == 2 {
                        (outputs[output_entry_start] as u64)
                            | ((outputs[output_entry_start + 1] as u64) << 32)
                    } else {
                        outputs[output_entry_start] as u64
                    };
                    let old_next = he.next;
                    he.next = output & state_mask;
                    he.state = if ((output >> (output_len - 1)) & 1) != 0 {
                        HASH_STATE_STOPPED
                    } else if he.next == he.current || he.next == old_next {
                        HASH_STATE_LOOPED
                    } else {
                        HASH_STATE_USED
                    };
                    he.steps += 1;
                }
            }
        });
}

fn do_solve_with_cpu_mapper<'a>(
    mut mapper: CPUBasicMapperBuilder<'a>,
    circuit: Circuit<usize>,
    unknowns: usize,
    elem_inputs: usize,
) {
    let input_len = circuit.input_len();
    let output_len = input_len + 1;
    let arg_steps = 1u128 << (input_len - elem_inputs);
    mapper.transform_helpers();
    mapper.user_defs(&format!("#define OUTPUT_NUM ({})\n", output_len));
    mapper.user_defs(&gen_output_transform_code(output_len));
    let word_per_elem = (output_len + 31) >> 5;
    mapper.add_with_config(
        "formula",
        circuit,
        CodeConfig::new()
            .elem_inputs(Some(&(0..elem_inputs).collect::<Vec<usize>>()))
            .arg_inputs(Some(&(elem_inputs..input_len).collect::<Vec<usize>>()))
            .aggr_output_code(Some(AGGR_OUTPUT_CPU_CODE))
            .aggr_output_len(Some(word_per_elem * (1 << elem_inputs)))
            .dont_clear_outputs(true),
    );
    let type_len = mapper.type_len();
    let mut execs = mapper.build().unwrap();
    let input = execs[0].new_data(16);
    let start = SystemTime::now();
    execs[0]
        .execute(
            &input,
            (),
            |result, _, output, arg| {
                println!("Step: {} / {}", arg, arg_steps);
            },
            |_| false,
        )
        .unwrap();
    let time = start.elapsed().unwrap();
    println!("Time: {}", time.as_secs_f64());
}

const AGGR_OUTPUT_OPENCL_CODE: &str = r##"{
    global uint* output_u = ((global uint*)output) + idx *
        ((OUTPUT_NUM + 31) >> 5) * TYPE_LEN;
#if OUTPUT_NUM <= 32
    OUTPUT_TRANSFORM_FIRST_32(output_u);
#else
    uint i;
    uint temp[((OUTPUT_NUM + 31) >> 5) * TYPE_LEN];
    OUTPUT_TRANSFORM_FIRST_32(temp);
    OUTPUT_TRANSFORM_SECOND_32(temp + 32 * (TYPE_LEN >> 5));
    for (i = 0; i < TYPE_LEN; i++) {
        output_u[i*2] = temp[i];
        output_u[i*2 + 1] = temp[i + TYPE_LEN];
    }
#endif
}"##;

fn do_solve_with_opencl_mapper<'a>(
    mut mapper: OpenCLBasicMapperBuilder<'a>,
    circuit: Circuit<usize>,
    unknowns: usize,
    elem_inputs: usize,
) {
    let input_len = circuit.input_len();
    let output_len = input_len + 1;
    let arg_steps = 1u128 << (input_len - elem_inputs);
    mapper.transform_helpers();
    mapper.user_defs(&format!("#define OUTPUT_NUM ({})\n", output_len));
    mapper.user_defs(&gen_output_transform_code(output_len));
    let word_per_elem = (output_len + 31) >> 5;
    mapper.add_with_config(
        "formula",
        circuit,
        CodeConfig::new()
            .elem_inputs(Some(&(0..elem_inputs).collect::<Vec<usize>>()))
            .arg_inputs(Some(&(elem_inputs..input_len).collect::<Vec<usize>>()))
            .aggr_output_code(Some(AGGR_OUTPUT_OPENCL_CODE))
            .aggr_output_len(Some(word_per_elem * (1 << elem_inputs)))
            .dont_clear_outputs(true),
    );
    let type_len = mapper.type_len();
    let mut execs = mapper.build().unwrap();
    let input = execs[0].new_data(16);
    let start = SystemTime::now();
    execs[0]
        .execute(
            &input,
            (),
            |result, _, output, arg| {
                println!("Step: {} / {}", arg, arg_steps);
            },
            |_| false,
        )
        .unwrap();
    let time = start.elapsed().unwrap();
    println!("Time: {}", time.as_secs_f64());
}

fn do_solve(circuit: Circuit<usize>, unknowns: usize, cmd_args: CommandArgs) {
    let input_len = circuit.input_len();
    let result = if input_len >= 10 {
        let elem_inputs = if cmd_args.elem_inputs >= input_len {
            input_len - 1
        } else {
            cmd_args.elem_inputs
        };
        assert!(elem_inputs > 0 && elem_inputs <= 37);
        assert!(input_len - elem_inputs > 0 && input_len - elem_inputs <= 64);
        assert_eq!(circuit.outputs().len(), input_len + 1);
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
                let builder = BasicMapperBuilder::new(CPUBuilder::new_parallel(None, Some(4096)));
                do_solve_with_cpu_mapper(builder, circuit.clone(), unknowns, elem_inputs)
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
                do_solve_with_opencl_mapper(builder, circuit.clone(), unknowns, elem_inputs)
            }
            ExecType::CPUAndOpenCL
            | ExecType::CPUAndOpenCLD
            | ExecType::CPUAndOpenCL1(_)
            | ExecType::CPUAndOpenCL1D(_) => {
                panic!("Unsupported!");
            }
        }
    } else {
        panic!("Unsupported!");
    };
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggr_output_code_cpu() {
        for output_len in [24, 32, 33, 44] {
            let word_per_elem = (output_len + 31) >> 5;
            let circuit =
                Circuit::new(output_len, [], (0..output_len).map(|i| (i, false))).unwrap();
            let mut builder = CPUBuilder::new(None);
            builder.transform_helpers();
            builder.user_defs(&format!("#define OUTPUT_NUM ({})\n", output_len));
            builder.user_defs(&gen_output_transform_code(output_len));
            builder.add_with_config(
                "formula",
                circuit.clone(),
                CodeConfig::new()
                    .elem_inputs(Some(&(0..20).collect::<Vec<usize>>()))
                    .arg_inputs(Some(&(20..output_len).collect::<Vec<usize>>()))
                    .aggr_output_code(Some(AGGR_OUTPUT_CPU_CODE))
                    .aggr_output_len(Some(word_per_elem * (1 << 20))),
            );
            builder.add_with_config(
                "formula2",
                circuit,
                CodeConfig::new()
                    .elem_inputs(Some(&(output_len - 20..output_len).collect::<Vec<usize>>()))
                    .arg_inputs(Some(&(0..output_len - 20).collect::<Vec<usize>>()))
                    .aggr_output_code(Some(AGGR_OUTPUT_CPU_CODE))
                    .aggr_output_len(Some(word_per_elem * (1 << 20))),
            );
            let mut execs = builder.build().unwrap();
            let arg_mask = (1u64 << (output_len - 20)) - 1;
            let arg = 138317356571 & arg_mask;
            println!("Arg: {}", arg);
            let input = execs[0].new_data(16);
            let output = execs[0].execute(&input, arg).unwrap().release();
            assert_eq!(output.len(), word_per_elem * (1 << 20));
            for i in 0..1 << 20 {
                let out = if word_per_elem == 2 {
                    (output[2 * i] as u64) | ((output[2 * i + 1] as u64) << 32)
                } else {
                    output[i] as u64
                };
                assert_eq!((i as u64) | (arg << 20), out, "{}: {}", output_len, i);
            }

            let input = execs[1].new_data(16);
            let output = execs[1].execute(&input, arg).unwrap().release();
            assert_eq!(output.len(), word_per_elem * (1 << 20));
            for i in 0..1 << 20 {
                let out = if word_per_elem == 2 {
                    (output[2 * i] as u64) | ((output[2 * i + 1] as u64) << 32)
                } else {
                    output[i] as u64
                };
                assert_eq!(
                    ((i as u64) << (output_len - 20)) | arg,
                    out,
                    "{}: {}",
                    output_len,
                    i
                );
            }
        }
    }

    #[test]
    fn test_aggr_output_code_opencl() {
        let device = Device::new(*get_all_devices(CL_DEVICE_TYPE_GPU).unwrap().get(0).unwrap());
        for output_len in [24, 32, 33, 44] {
            let word_per_elem = (output_len + 31) >> 5;
            let circuit =
                Circuit::new(output_len, [], (0..output_len).map(|i| (i, false))).unwrap();
            let mut builder = OpenCLBuilder::new(&device, None);
            builder.transform_helpers();
            builder.user_defs(&format!("#define OUTPUT_NUM ({})\n", output_len));
            builder.user_defs(&gen_output_transform_code(output_len));
            builder.add_with_config(
                "formula",
                circuit.clone(),
                CodeConfig::new()
                    .elem_inputs(Some(&(0..20).collect::<Vec<usize>>()))
                    .arg_inputs(Some(&(20..output_len).collect::<Vec<usize>>()))
                    .aggr_output_code(Some(AGGR_OUTPUT_OPENCL_CODE))
                    .aggr_output_len(Some(word_per_elem * (1 << 20))),
            );
            builder.add_with_config(
                "formula2",
                circuit,
                CodeConfig::new()
                    .elem_inputs(Some(&(output_len - 20..output_len).collect::<Vec<usize>>()))
                    .arg_inputs(Some(&(0..output_len - 20).collect::<Vec<usize>>()))
                    .aggr_output_code(Some(AGGR_OUTPUT_OPENCL_CODE))
                    .aggr_output_len(Some(word_per_elem * (1 << 20))),
            );
            let mut execs = builder.build().unwrap();
            let arg_mask = (1u64 << (output_len - 20)) - 1;
            let arg = 138317356571 & arg_mask;
            println!("Arg: {}", arg);
            let input = execs[0].new_data(16);
            let output = execs[0].execute(&input, arg).unwrap().release();
            assert_eq!(output.len(), word_per_elem * (1 << 20));
            for i in 0..1 << 20 {
                let out = if word_per_elem == 2 {
                    (output[2 * i] as u64) | ((output[2 * i + 1] as u64) << 32)
                } else {
                    output[i] as u64
                };
                assert_eq!((i as u64) | (arg << 20), out, "{}: {}", output_len, i);
            }

            let input = execs[1].new_data(16);
            let output = execs[1].execute(&input, arg).unwrap().release();
            assert_eq!(output.len(), word_per_elem * (1 << 20));
            for i in 0..1 << 20 {
                let out = if word_per_elem == 2 {
                    (output[2 * i] as u64) | ((output[2 * i + 1] as u64) << 32)
                } else {
                    output[i] as u64
                };
                assert_eq!(
                    ((i as u64) << (output_len - 20)) | arg,
                    out,
                    "{}: {}",
                    output_len,
                    i
                );
            }
        }
    }

    #[test]
    fn test_join_to_hashmap_cpu() {
        // 24-bit
        let output_len = 24 + 1;
        let arg_bit_place = 16;
        let arg: u64 = 173;
        let outputs = {
            let mut outputs = vec![0u32; 1 << 16];
            outputs[652] = 0xfa214;
            outputs[5911] = 0x2a01d7 | (1 << 24);
            outputs[23416] = 0xdda0a1;
            outputs[34071] = 0x0451e8;
            outputs[44158] = 0x55df8a;
            outputs[49774] = ((arg as u32) << arg_bit_place) | 49774;
            outputs[53029] = 0xdda02 | (1 << 24);
            outputs
        };
        let mut hashmap = {
            let mut hashmap = vec![
                HashEntry {
                    current: 0,
                    next: 0,
                    steps: 0,
                    predecessors: 0,
                    state: HASH_STATE_UNUSED,
                };
                1 << 14
            ];
            hashmap[4901] = HashEntry {
                current: 0x895911,
                next: (arg << arg_bit_place) | 652,
                steps: 441,
                predecessors: 2,
                state: HASH_STATE_USED,
            };
            // to stop
            hashmap[9487] = HashEntry {
                current: 0x7490c3,
                next: (arg << arg_bit_place) | 5911,
                steps: 8741,
                predecessors: 2,
                state: HASH_STATE_USED,
            };
            hashmap[10771] = HashEntry {
                current: 0xea061d,
                next: (arg << arg_bit_place) | 34071,
                steps: 72,
                predecessors: 3,
                state: HASH_STATE_STOPPED,
            };
            hashmap[2971] = HashEntry {
                current: (arg << arg_bit_place) | 23416,
                next: (arg << arg_bit_place) | 23416,
                steps: 826,
                predecessors: 8,
                state: HASH_STATE_LOOPED,
            };
            // to loop
            hashmap[3957] = HashEntry {
                current: 0x55df8a,
                next: (arg << arg_bit_place) | 44158,
                steps: 211,
                predecessors: 5,
                state: HASH_STATE_USED,
            };
            // to loop 2
            hashmap[15995] = HashEntry {
                current: 0x70392d,
                next: (arg << arg_bit_place) | 49774,
                steps: 211,
                predecessors: 10,
                state: HASH_STATE_USED,
            };
            // stop not loop
            hashmap[12061] = HashEntry {
                current: 0xdda02,
                next: (arg << arg_bit_place) | 53029,
                steps: 10950696030321,
                predecessors: 12,
                state: HASH_STATE_USED,
            };
            // used and belog arg range
            hashmap[7955] = HashEntry {
                current: 0xd0c9a0,
                next: ((arg + 2) << arg_bit_place) | 6961,
                steps: 731,
                predecessors: 14,
                state: HASH_STATE_USED,
            };
            // used and belog arg range
            hashmap[921] = HashEntry {
                current: 0x1afcdc,
                next: ((arg - 2) << arg_bit_place) | 6961,
                steps: 947,
                predecessors: 7,
                state: HASH_STATE_USED,
            };
            hashmap
        };
        join_to_hashmap_cpu(output_len, arg_bit_place, arg, &outputs, &mut hashmap);
        let expected_hashmap = {
            let mut hashmap = vec![
                HashEntry {
                    current: 0,
                    next: 0,
                    steps: 0,
                    predecessors: 0,
                    state: HASH_STATE_UNUSED,
                };
                1 << 14
            ];
            hashmap[4901] = HashEntry {
                current: 0x895911,
                next: 0xfa214,
                steps: 442,
                predecessors: 2,
                state: HASH_STATE_USED,
            };
            // to stop
            hashmap[9487] = HashEntry {
                current: 0x7490c3,
                next: 0x2a01d7,
                steps: 8742,
                predecessors: 2,
                state: HASH_STATE_STOPPED,
            };
            hashmap[10771] = HashEntry {
                current: 0xea061d,
                next: (arg << arg_bit_place) | 34071,
                steps: 72,
                predecessors: 3,
                state: HASH_STATE_STOPPED,
            };
            hashmap[2971] = HashEntry {
                current: (arg << arg_bit_place) | 23416,
                next: (arg << arg_bit_place) | 23416,
                steps: 826,
                predecessors: 8,
                state: HASH_STATE_LOOPED,
            };
            // to loop
            hashmap[3957] = HashEntry {
                current: 0x55df8a,
                next: 0x55df8a,
                steps: 212,
                predecessors: 5,
                state: HASH_STATE_LOOPED,
            };
            // to loop 2
            hashmap[15995] = HashEntry {
                current: 0x70392d,
                next: (arg << arg_bit_place) | 49774,
                steps: 212,
                predecessors: 10,
                state: HASH_STATE_LOOPED,
            };
            // stop not loop
            hashmap[12061] = HashEntry {
                current: 0xdda02,
                next: 0xdda02,
                steps: 10950696030322,
                predecessors: 12,
                state: HASH_STATE_STOPPED,
            };
            // used and belog arg range
            hashmap[7955] = HashEntry {
                current: 0xd0c9a0,
                next: ((arg + 2) << arg_bit_place) | 6961,
                steps: 731,
                predecessors: 14,
                state: HASH_STATE_USED,
            };
            // used and belog arg range
            hashmap[921] = HashEntry {
                current: 0x1afcdc,
                next: ((arg - 2) << arg_bit_place) | 6961,
                steps: 947,
                predecessors: 7,
                state: HASH_STATE_USED,
            };
            hashmap
        };
        for (i, he) in hashmap.into_iter().enumerate() {
            assert_eq!(expected_hashmap[i], he, "{}: {}", 24, i);
        }

        // 32-bit
        let output_len = 32 + 1;
        let arg_bit_place = 24;
        let arg: u64 = 119;
        let outputs = {
            let mut outputs = vec![0u32; 1 << (24 + 1)];
            outputs[2 * 61232] = 0xaafa214;
            outputs[2 * 61232 + 1] = 0;
            outputs[2 * 594167] = 0x062a01d7;
            outputs[2 * 594167 + 1] = 1;
            outputs[2 * 2461601] = 0x13dda0a1;
            outputs[2 * 2461601 + 1] = 0;
            outputs[2 * 3161365] = 0xd60451e8;
            outputs[2 * 3161365 + 1] = 0;
            outputs[2 * 4509138] = 0xd155df8a;
            outputs[2 * 4509138 + 1] = 0;
            outputs[2 * 5167006] = ((arg as u32) << 24) | 5167006;
            outputs[2 * 5167006 + 1] = 0;
            outputs[2 * 5972782] = 0x210689a1;
            outputs[2 * 5972782 + 1] = 1;
            outputs
        };
        let mut hashmap = {
            let mut hashmap = vec![
                HashEntry {
                    current: 0,
                    next: 0,
                    steps: 0,
                    predecessors: 0,
                    state: HASH_STATE_UNUSED,
                };
                1 << 14
            ];
            hashmap[4901] = HashEntry {
                current: 0x12895911,
                next: (arg << arg_bit_place) | 61232,
                steps: 481,
                predecessors: 2,
                state: HASH_STATE_USED,
            };
            // to stop
            hashmap[9487] = HashEntry {
                current: 0xd17490c3,
                next: (arg << arg_bit_place) | 594167,
                steps: 6505940239021677,
                predecessors: 2,
                state: HASH_STATE_USED,
            };
            hashmap[10771] = HashEntry {
                current: 0x50ea061d,
                next: (arg << arg_bit_place) | 3161365,
                steps: 762,
                predecessors: 3,
                state: HASH_STATE_STOPPED,
            };
            hashmap[2971] = HashEntry {
                current: (arg << arg_bit_place) | 2461601,
                next: (arg << arg_bit_place) | 2461601,
                steps: 1826,
                predecessors: 8,
                state: HASH_STATE_LOOPED,
            };
            // to loop
            hashmap[3957] = HashEntry {
                current: 0xd155df8a,
                next: (arg << arg_bit_place) | 4509138,
                steps: 2711,
                predecessors: 5,
                state: HASH_STATE_USED,
            };
            // to loop 2
            hashmap[15995] = HashEntry {
                current: 0x703921d,
                next: (arg << arg_bit_place) | 5167006,
                steps: 2181,
                predecessors: 10,
                state: HASH_STATE_USED,
            };
            // stop not loop
            hashmap[12061] = HashEntry {
                current: 0x210689a1,
                next: (arg << arg_bit_place) | 5972782,
                steps: 44195,
                predecessors: 12,
                state: HASH_STATE_USED,
            };
            // used and belog arg range
            hashmap[7955] = HashEntry {
                current: 0x3cd0c9a0,
                next: ((arg + 2) << arg_bit_place) | 6961,
                steps: 76631,
                predecessors: 14,
                state: HASH_STATE_USED,
            };
            // used and belog arg range
            hashmap[921] = HashEntry {
                current: 0x10fafcdc,
                next: ((arg - 2) << arg_bit_place) | 6961,
                steps: 94137,
                predecessors: 7,
                state: HASH_STATE_USED,
            };
            hashmap
        };
        join_to_hashmap_cpu(output_len, arg_bit_place, arg, &outputs, &mut hashmap);
        let expected_hashmap = {
            let mut hashmap = vec![
                HashEntry {
                    current: 0,
                    next: 0,
                    steps: 0,
                    predecessors: 0,
                    state: HASH_STATE_UNUSED,
                };
                1 << 14
            ];
            hashmap[4901] = HashEntry {
                current: 0x12895911,
                next: 0xaafa214,
                steps: 482,
                predecessors: 2,
                state: HASH_STATE_USED,
            };
            // to stop
            hashmap[9487] = HashEntry {
                current: 0xd17490c3,
                next: 0x062a01d7,
                steps: 6505940239021678,
                predecessors: 2,
                state: HASH_STATE_STOPPED,
            };
            hashmap[10771] = HashEntry {
                current: 0x50ea061d,
                next: (arg << arg_bit_place) | 3161365,
                steps: 762,
                predecessors: 3,
                state: HASH_STATE_STOPPED,
            };
            hashmap[2971] = HashEntry {
                current: (arg << arg_bit_place) | 2461601,
                next: (arg << arg_bit_place) | 2461601,
                steps: 1826,
                predecessors: 8,
                state: HASH_STATE_LOOPED,
            };
            // to loop
            hashmap[3957] = HashEntry {
                current: 0xd155df8a,
                next: 0xd155df8a,
                steps: 2712,
                predecessors: 5,
                state: HASH_STATE_LOOPED,
            };
            // to loop 2
            hashmap[15995] = HashEntry {
                current: 0x703921d,
                next: (arg << arg_bit_place) | 5167006,
                steps: 2182,
                predecessors: 10,
                state: HASH_STATE_LOOPED,
            };
            // stop not loop
            hashmap[12061] = HashEntry {
                current: 0x210689a1,
                next: 0x210689a1,
                steps: 44196,
                predecessors: 12,
                state: HASH_STATE_STOPPED,
            };
            // used and belog arg range
            hashmap[7955] = HashEntry {
                current: 0x3cd0c9a0,
                next: ((arg + 2) << arg_bit_place) | 6961,
                steps: 76631,
                predecessors: 14,
                state: HASH_STATE_USED,
            };
            // used and belog arg range
            hashmap[921] = HashEntry {
                current: 0x10fafcdc,
                next: ((arg - 2) << arg_bit_place) | 6961,
                steps: 94137,
                predecessors: 7,
                state: HASH_STATE_USED,
            };
            hashmap
        };
        for (i, he) in hashmap.into_iter().enumerate() {
            assert_eq!(expected_hashmap[i], he, "{}: {}", 32, i);
        }

        // 40-bit
        let output_len = 40 + 1;
        let arg_bit_place = 24;
        let arg: u64 = 43051;
        let outputs = {
            let mut outputs = vec![0u32; 1 << (24 + 1)];
            outputs[2 * 61232] = 0xaafa214;
            outputs[2 * 61232 + 1] = 131;
            outputs[2 * 594167] = 0x062a01d7;
            outputs[2 * 594167 + 1] = 21 | (1 << 8);
            outputs[2 * 2461601] = 0x13dda0a1;
            outputs[2 * 2461601 + 1] = 79;
            outputs[2 * 3161365] = 0xd60451e8;
            outputs[2 * 3161365 + 1] = 186;
            outputs[2 * 4509138] = 0xd155df8a;
            outputs[2 * 4509138 + 1] = 231;
            outputs[2 * 5167006] = (((arg & 0xff) as u32) << 24) | 5167006;
            outputs[2 * 5167006 + 1] = (arg >> 8) as u32;
            outputs[2 * 5972782] = 0x210689a1;
            outputs[2 * 5972782 + 1] = 206 | (1 << 8);
            outputs
        };
        let mut hashmap = {
            let mut hashmap = vec![
                HashEntry {
                    current: 0,
                    next: 0,
                    steps: 0,
                    predecessors: 0,
                    state: HASH_STATE_UNUSED,
                };
                1 << 14
            ];
            hashmap[4901] = HashEntry {
                current: 0xa112895911,
                next: (arg << arg_bit_place) | 61232,
                steps: 481,
                predecessors: 2,
                state: HASH_STATE_USED,
            };
            // to stop
            hashmap[9487] = HashEntry {
                current: 0xcd17490c3,
                next: (arg << arg_bit_place) | 594167,
                steps: 6505940239021677,
                predecessors: 2,
                state: HASH_STATE_USED,
            };
            hashmap[10771] = HashEntry {
                current: 0x1350ea061d,
                next: (arg << arg_bit_place) | 3161365,
                steps: 762,
                predecessors: 3,
                state: HASH_STATE_STOPPED,
            };
            hashmap[2971] = HashEntry {
                current: (arg << arg_bit_place) | 2461601,
                next: (arg << arg_bit_place) | 2461601,
                steps: 1826,
                predecessors: 8,
                state: HASH_STATE_LOOPED,
            };
            // to loop
            hashmap[3957] = HashEntry {
                current: 0xd155df8a | (231 << 32),
                next: (arg << arg_bit_place) | 4509138,
                steps: 2711,
                predecessors: 5,
                state: HASH_STATE_USED,
            };
            // to loop 2
            hashmap[15995] = HashEntry {
                current: 0x9e703921d,
                next: (arg << arg_bit_place) | 5167006,
                steps: 2181,
                predecessors: 10,
                state: HASH_STATE_USED,
            };
            // stop not loop
            hashmap[12061] = HashEntry {
                current: 0x21fb0689a1,
                next: (arg << arg_bit_place) | 5972782,
                steps: 44195,
                predecessors: 12,
                state: HASH_STATE_USED,
            };
            // used and belog arg range
            hashmap[7955] = HashEntry {
                current: 0x3c2dd0c9a0,
                next: ((arg + 2) << arg_bit_place) | 6961,
                steps: 76631,
                predecessors: 14,
                state: HASH_STATE_USED,
            };
            // used and belog arg range
            hashmap[921] = HashEntry {
                current: 0x1012fafcdc,
                next: ((arg - 2) << arg_bit_place) | 6961,
                steps: 94137,
                predecessors: 7,
                state: HASH_STATE_USED,
            };
            hashmap
        };
        join_to_hashmap_cpu(output_len, arg_bit_place, arg, &outputs, &mut hashmap);
        let expected_hashmap = {
            let mut hashmap = vec![
                HashEntry {
                    current: 0,
                    next: 0,
                    steps: 0,
                    predecessors: 0,
                    state: HASH_STATE_UNUSED,
                };
                1 << 14
            ];
            hashmap[4901] = HashEntry {
                current: 0xa112895911,
                next: 0xaafa214 | (131 << 32),
                steps: 482,
                predecessors: 2,
                state: HASH_STATE_USED,
            };
            // to stop
            hashmap[9487] = HashEntry {
                current: 0xcd17490c3,
                next: 0x062a01d7 | (21 << 32),
                steps: 6505940239021678,
                predecessors: 2,
                state: HASH_STATE_STOPPED,
            };
            hashmap[10771] = HashEntry {
                current: 0x1350ea061d,
                next: (arg << arg_bit_place) | 3161365,
                steps: 762,
                predecessors: 3,
                state: HASH_STATE_STOPPED,
            };
            hashmap[2971] = HashEntry {
                current: (arg << arg_bit_place) | 2461601,
                next: (arg << arg_bit_place) | 2461601,
                steps: 1826,
                predecessors: 8,
                state: HASH_STATE_LOOPED,
            };
            // to loop
            hashmap[3957] = HashEntry {
                current: 0xd155df8a | (231 << 32),
                next: 0xd155df8a | (231 << 32),
                steps: 2712,
                predecessors: 5,
                state: HASH_STATE_LOOPED,
            };
            // to loop 2
            hashmap[15995] = HashEntry {
                current: 0x9e703921d,
                next: (arg << arg_bit_place) | 5167006,
                steps: 2182,
                predecessors: 10,
                state: HASH_STATE_LOOPED,
            };
            // stop not loop
            hashmap[12061] = HashEntry {
                current: 0x21fb0689a1,
                next: 0x210689a1 | (206 << 32),
                steps: 44196,
                predecessors: 12,
                state: HASH_STATE_STOPPED,
            };
            // used and belog arg range
            hashmap[7955] = HashEntry {
                current: 0x3c2dd0c9a0,
                next: ((arg + 2) << arg_bit_place) | 6961,
                steps: 76631,
                predecessors: 14,
                state: HASH_STATE_USED,
            };
            // used and belog arg range
            hashmap[921] = HashEntry {
                current: 0x1012fafcdc,
                next: ((arg - 2) << arg_bit_place) | 6961,
                steps: 94137,
                predecessors: 7,
                state: HASH_STATE_USED,
            };
            hashmap
        };
        for (i, he) in hashmap.into_iter().enumerate() {
            assert_eq!(expected_hashmap[i], he, "{}: {}", 40, i);
        }
    }
}

fn main() {
    for i in 0..1000000 {
        println!(
            "Hashfunction: {:016x} = {:016x}",
            i,
            hash_function_64(48, i)
        )
    }
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
