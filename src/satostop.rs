use gatenative::cpu_build_exec::*;
use gatenative::mapper::*;
use gatenative::opencl_build_exec::*;
use gatenative::parseq_mapper::*;
use gatenative::*;
use gatesim::*;

use clap::Parser;
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::error_codes::ClError;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{cl_mem, cl_mem_flags, cl_uint, cl_ulong, CL_BLOCKING};

use rayon::prelude::*;

use std::cell::UnsafeCell;
use std::fs;
use std::ops::Range;
use std::str::FromStr;
use std::sync::atomic::{self, AtomicU32, AtomicU64};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

// TODO: Add handling partial handling of outputs while joining

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
ulong hash_function_64(ulong value) {
    const uint bits = STATE_LEN;
    const ulong mask = (1UL << STATE_LEN) - 1UL;
    const uint half_bits = STATE_LEN >> 1;
    const ulong temp = (value * 9615409803190489167UL);
    return ((value * 6171710485021949031UL) ^
        ((temp << half_bits) | (temp >> (STATE_LEN - half_bits))) ^
        0xb89d2ecda078ca1fUL) & mask;
}
"##;

const HASH_ENTRY_OPENCL_DEF: &str = r##"
typedef struct _HashEntry {
    ulong current;
    ulong next;
    ulong steps;
    uint predecessors;
    uint state;
} HashEntry;

#define HASH_STATE_UNUSED (0)
#define HASH_STATE_USED (1)
#define HASH_STATE_STOPPED (2)
#define HASH_STATE_LOOPED (3)
#define HASH_STATE_RESERVED_BY_OTHER_FLAG (4)
"##;

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

// TODO: add checking solution to join_to_hashmap and join_hashmap_itself and to add....

//
// join_to_hashmap - join outputs to hashmap entries
//

fn join_to_hashmap_cpu(
    state_len: usize,
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
    // words_per_elem - elem length in outputs in words (can be 1 or 2).
    let words_per_elem = (state_len + 1 + 31) >> 5;
    let state_mask = (1u64 << state_len) - 1;
    hashmap
        .chunks_mut(chunk_len)
        .par_bridge()
        .for_each(|hashchunk| {
            for he in hashchunk {
                if he.state == HASH_STATE_USED && arg_start <= he.next && he.next < arg_end {
                    // update hash entry next field.
                    let output_entry_start =
                        words_per_elem * usize::try_from(he.next - arg_start).unwrap();
                    // get output state
                    let output = if words_per_elem == 2 {
                        (outputs[output_entry_start] as u64)
                            | ((outputs[output_entry_start + 1] as u64) << 32)
                    } else {
                        outputs[output_entry_start] as u64
                    };
                    let old_next = he.next;
                    he.next = output & state_mask;
                    he.state = if ((output >> state_len) & 1) != 0 {
                        // if stop enabled from output
                        HASH_STATE_STOPPED
                    } else if state_mask <= he.steps || he.next == he.current || he.next == old_next
                    {
                        // if step number is bigger than max step number or
                        // previous next equal to new next or current in hash entry equal to
                        // new next then it is loop
                        HASH_STATE_LOOPED
                    } else {
                        HASH_STATE_USED
                    };
                    he.steps += 1;
                }
            }
        });
}

const JOIN_TO_HASHMAP_OPENCL_CODE: &str = r##"
kernel void join_to_hashmap(ulong arg, const global uint* outputs, global HashEntry* hashmap) {
    const size_t idx = get_global_id(0);
    if (idx >= HASHMAP_LEN)
        return;
    const ulong arg_start = arg << ARG_BIT_PLACE;
    const ulong arg_end = arg_start + (1UL << ARG_BIT_PLACE);
    global HashEntry* he = hashmap + idx;
    if (he->state == HASH_STATE_USED && arg_start <= he->next && he->next < arg_end) {
        const ulong state_mask = (1UL << STATE_LEN) - 1UL;
#if WORDS_PER_ELEM == 2
        const size_t output_entry_start = (he->next - arg_start) << 1;
        const ulong output = ((ulong)outputs[output_entry_start]) |
            (((ulong)outputs[output_entry_start + 1]) << 32);
#else
        const ulong output = outputs[he->next - arg_start];
#endif
        const ulong old_next = he->next;
        he->next = output & state_mask;
        if (((output >> STATE_LEN) & 1) != 0) {
            he->state = HASH_STATE_STOPPED;
        } else if (state_mask <= he->steps || he->next == he->current || he->next == old_next) {
            he->state = HASH_STATE_LOOPED;
        } else {
            he->state = HASH_STATE_USED;
        }
        he->steps += 1;
    }
}
"##;

struct OpenCLJoinToHashMap {
    hashmap_len: usize,
    cmd_queue: Arc<CommandQueue>,
    group_len: usize,
    kernel: Kernel,
}

impl OpenCLJoinToHashMap {
    fn new(
        state_len: usize,
        arg_bit_place: usize,
        hashmap_len: usize,
        context: Arc<Context>,
        cmd_queue: Arc<CommandQueue>,
    ) -> Self {
        let device = Device::new(context.devices()[0]);
        let group_len = usize::try_from(device.max_work_group_size().unwrap()).unwrap();
        let words_per_elem = (state_len + 1 + 31) >> 5;
        let defs = format!(
            "-DSTATE_LEN=({}) -DARG_BIT_PLACE=({}) -DWORDS_PER_ELEM=({}) -DHASHMAP_LEN=({})",
            state_len, arg_bit_place, words_per_elem, hashmap_len,
        );
        let source = HASH_ENTRY_OPENCL_DEF.to_string() + JOIN_TO_HASHMAP_OPENCL_CODE;
        let program = Program::create_and_build_from_source(&context, &source, &defs).unwrap();
        OpenCLJoinToHashMap {
            hashmap_len,
            cmd_queue,
            group_len,
            kernel: Kernel::create(&program, "join_to_hashmap").unwrap(),
        }
    }

    fn execute(&self, arg: u64, outputs: &Buffer<u32>, hashmap: &mut Buffer<HashEntry>) {
        let cl_arg = cl_ulong::try_from(arg).unwrap();
        unsafe {
            ExecuteKernel::new(&self.kernel)
                .set_arg(&cl_arg)
                .set_arg(outputs)
                .set_arg(hashmap)
                .set_local_work_size(self.group_len)
                .set_global_work_size(
                    ((self.hashmap_len + self.group_len - 1) / self.group_len) * self.group_len,
                )
                .enqueue_nd_range(&self.cmd_queue)
                .unwrap();
            self.cmd_queue.finish().unwrap();
        }
    }
}

//
// join_hashmap_itself - join hash entries with other hash entries in hashmap
//

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Solution {
    start: u64,
    end: u64,
    steps: u64,
}

#[inline]
fn resolve_unknowns(
    state_len: usize,
    unknown_bits: usize,
    unknown_fill_bits: usize,
    current: u64,
    next: u64,
    steps: u64,
    entry_state: u32,
    unknown_fills: Arc<Vec<AtomicU32>>,
    resolved_unknowns: Arc<AtomicU64>,
    solution: &Mutex<Option<Solution>>,
) {
    // unknown fill mapping to state:
    //     [unknown_fill_entry_idx][unknown_fill_value][00000000000....]
    // only for unknown paths: state bits: 0..(state_len-unknown_bits) = 0b000...000
    if (current & ((1u64 << (state_len - unknown_bits)) - 1)) == 0
        && (entry_state == HASH_STATE_LOOPED || entry_state == HASH_STATE_STOPPED)
    {
        if entry_state == HASH_STATE_STOPPED {
            // just set solution
            let mut sol = solution.lock().unwrap();
            if sol.is_none() {
                *sol = Some(Solution {
                    start: current,
                    end: next,
                    steps,
                });
            }
        }
        let unknown_fill_idx =
            usize::try_from(current >> (state_len - unknown_bits + unknown_fill_bits)).unwrap();
        let unknown_fill_mask = (1u32 << unknown_fill_bits) - 1;
        let unknown_fill_value =
            u32::try_from((current >> (state_len - unknown_bits)) & (unknown_fill_mask as u64))
                .unwrap();
        // if match to unknown fill field then increase this field
        if unknown_fills[unknown_fill_idx]
            .compare_exchange(
                unknown_fill_value,
                unknown_fill_value + 1,
                atomic::Ordering::SeqCst,
                atomic::Ordering::SeqCst,
            )
            .is_ok()
            && unknown_fill_value == unknown_fill_mask
        {
            // increase resolved unknowns if it last unknown in this unknown fill
            resolved_unknowns.fetch_add(1, atomic::Ordering::SeqCst);
        }
    }
}

fn create_vec_of_atomic_u32(len: usize) -> Arc<Vec<AtomicU32>> {
    Arc::new(
        std::iter::repeat_with(|| AtomicU32::new(0))
            .take(len)
            .collect::<Vec<_>>(),
    )
}

fn join_hashmap_itself_and_check_solution_cpu(
    state_len: usize,
    preds_update: Arc<Vec<AtomicU32>>,
    in_hashmap: &[HashEntry],
    out_hashmap: &mut [HashEntry],
    unknown_bits: usize,
    unknown_fill_bits: usize,
    unknown_fills: Arc<Vec<AtomicU32>>,
    resolved_unknowns: Arc<AtomicU64>,
    solution: &Mutex<Option<Solution>>,
) {
    assert_eq!(in_hashmap.len(), out_hashmap.len());
    assert_eq!(in_hashmap.len().count_ones(), 1);
    let hashlen_bits = usize::BITS - in_hashmap.len().leading_zeros() - 1;
    let hashentry_shift = state_len - hashlen_bits as usize;
    let cpu_num = rayon::current_num_threads();
    let chunk_num = std::cmp::min(std::cmp::max(cpu_num * 8, 64), in_hashmap.len() >> 6);
    let chunk_len = in_hashmap.len() / chunk_num;
    let state_mask = (1u64 << (state_len - 1)) - 1;
    // zeroing predecessors updates
    preds_update
        .chunks(chunk_len)
        .par_bridge()
        .for_each(|chunk| {
            for v in chunk {
                v.store(0, atomic::Ordering::SeqCst);
            }
        });
    // main routine
    in_hashmap
        .chunks(chunk_len)
        .zip(out_hashmap.chunks_mut(chunk_len))
        .par_bridge()
        .for_each(|(in_hashchunk, out_hashchunk)| {
            for (inhe, outhe) in in_hashchunk.iter().zip(out_hashchunk.iter_mut()) {
                if inhe.state == HASH_STATE_USED {
                    let next_hash = hash_function_64(state_len, inhe.next);
                    let nexthe = &in_hashmap[next_hash >> hashentry_shift];
                    if nexthe.state != HASH_STATE_UNUSED && nexthe.current == inhe.next {
                        // if next found in hashmap entry
                        outhe.current = inhe.current;
                        let old_next = inhe.next;
                        outhe.next = nexthe.next;
                        let (res, ov) = inhe.steps.overflowing_add(nexthe.steps);
                        outhe.steps = res;
                        outhe.state = nexthe.state;
                        if outhe.state == HASH_STATE_USED {
                            if inhe.current == nexthe.next || ov || res > state_mask {
                                // if overflow of steps or steps greater than max step number.
                                // then loop
                                outhe.state = HASH_STATE_LOOPED;
                            }
                        }
                        outhe.predecessors = inhe.predecessors;
                        // update predecessors update for output hashmap
                        preds_update[next_hash >> hashentry_shift]
                            .fetch_add(1, atomic::Ordering::SeqCst);
                    } else {
                        *outhe = *inhe;
                    }
                } else {
                    *outhe = *inhe;
                }
                resolve_unknowns(
                    state_len,
                    unknown_bits,
                    unknown_fill_bits,
                    outhe.current,
                    outhe.next,
                    outhe.steps,
                    outhe.state,
                    unknown_fills.clone(),
                    resolved_unknowns.clone(),
                    solution,
                );
            }
        });
    // finally add predecessors updates to output hashmap
    out_hashmap
        .chunks_mut(chunk_len)
        .zip(preds_update.chunks(chunk_len))
        .par_bridge()
        .for_each(|(hchunk, pchunk)| {
            for (he, pred_update) in hchunk.iter_mut().zip(pchunk.iter()) {
                he.predecessors += pred_update.load(atomic::Ordering::SeqCst);
            }
        });
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct SolutionAndResUnknowns {
    resolved_unknowns: u64,
    sol_start: u64,
    sol_end: u64,
    sol_steps: u64,
    sol_defined: u32,
}

const RESOLVE_UNKNOWNS_OPENCL_CODE: &str = r##"
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

typedef struct _SolutionAndResUnknowns {
    ulong resolved_unknowns;
    ulong sol_start;
    ulong sol_end;
    ulong sol_steps;
    uint sol_defined;
} SolutionAndResUnknowns;

// state_len: usize,
// unknown_bits: usize,
// unknown_fill_bits: usize,
void resolve_unknowns(
    ulong current,
    ulong next,
    ulong steps,
    uint entry_state,
    global uint* unknown_fills,
    global SolutionAndResUnknowns* sol_and_res_unk
) {
    // unknown fill mapping to state:
    //     [unknown_fill_entry_idx][unknown_fill_value][00000000000....]
    // only for unknown paths: state bits: 0..(state_len-unknown_bits) = 0b000...000
    if ((current & ((1UL << (STATE_LEN - UNKNOWN_BITS)) - 1UL)) == 0
        && (entry_state == HASH_STATE_LOOPED || entry_state == HASH_STATE_STOPPED))
    {
        if (entry_state == HASH_STATE_STOPPED) {
            // just set solution
            if (atomic_or(&(sol_and_res_unk->sol_defined), 1) == 0) {
                sol_and_res_unk->sol_start = current;
                sol_and_res_unk->sol_end = next;
                sol_and_res_unk->sol_steps = steps;
            }
        }
        const size_t unknown_fill_idx = current >> (STATE_LEN - UNKNOWN_BITS + UNKNOWN_FILL_BITS);
        const uint unknown_fill_mask = (1 << UNKNOWN_FILL_BITS) - 1;
        const uint unknown_fill_value =
            (current >> (STATE_LEN - UNKNOWN_BITS)) & unknown_fill_mask;
        // if match to unknown fill field then increase this field
        if ((atomic_cmpxchg(unknown_fills + unknown_fill_idx, unknown_fill_value,
                            unknown_fill_value + 1) == unknown_fill_value)
            && (unknown_fill_value == unknown_fill_mask)) {
            // increase resolved unknowns if it last unknown in this unknown fill
            atom_inc(&(sol_and_res_unk->resolved_unknowns));
        }
    }
}
"##;

const JOIN_HASHMAP_ITSELF_AND_CHECK_SOLUTION_OPENCL_CODE: &str = r##"
kernel void join_hashmap_itself_zero_pred(global HashEntry* out_hashmap) {
    const size_t idx = get_global_id(0);
    if (idx >= HASHMAP_LEN)
        return;
    out_hashmap[idx].predecessors = 0;
}

kernel void join_hashmap_itself_and_check_solution(const global HashEntry* in_hashmap,
        global HashEntry* out_hashmap, global uint* unknown_fills,
        global SolutionAndResUnknowns* sol_and_res_unk) {
    const size_t idx = get_global_id(0);
    uint do_copy = 1;
    if (idx >= HASHMAP_LEN)
        return;
    const global HashEntry* inhe = in_hashmap + idx;
    global HashEntry* outhe = out_hashmap + idx;
    const size_t hashentry_shift = STATE_LEN - HASHMAP_LEN_BITS;
    const ulong state_mask = (1UL << STATE_LEN) - 1UL;
    if (inhe->state == HASH_STATE_USED) {
        const ulong next_hash = hash_function_64(inhe->next);
        const size_t next_idx = (next_hash >> hashentry_shift);
        const global HashEntry* nexthe = in_hashmap + next_idx;
        if (nexthe->state != HASH_STATE_UNUSED && nexthe->current == inhe->next) {
            // if next found in hashmap entry
            const ulong next_next = nexthe->next;
            const ulong current = inhe->current;
            const ulong insteps = inhe->steps;
            const ulong steps_sum = insteps + nexthe->steps;
            outhe->current = current;
            outhe->next = next_next;
            outhe->steps = steps_sum;
            outhe->state = nexthe->state;
            if (outhe->state == HASH_STATE_USED) {
                if (current == next_next || steps_sum < insteps || steps_sum > state_mask) {
                    // if overflow of steps or steps greater than max step number.
                    // then loop
                    outhe->state = HASH_STATE_LOOPED;
                }
            }
            atomic_inc(&(out_hashmap[next_idx].predecessors));
            do_copy = 0;
        }
    }
    if (do_copy) {
        outhe->current = inhe->current;
        outhe->next = inhe->next;
        outhe->steps = inhe->steps;
        outhe->state = inhe->state;
    }
    atomic_add(&outhe->predecessors, inhe->predecessors);
    resolve_unknowns( outhe->current, outhe->next, outhe->steps, outhe->state,
        unknown_fills, sol_and_res_unk);
}
"##;

struct OpenCLJoinHashMapItselfAndCheckSolution {
    hashmap_len: usize,
    cmd_queue: Arc<CommandQueue>,
    group_len: usize,
    kernel_zero: Kernel,
    kernel: Kernel,
}

impl OpenCLJoinHashMapItselfAndCheckSolution {
    fn new(
        state_len: usize,
        hashmap_len: usize,
        unknown_bits: usize,
        unknown_fill_bits: usize,
        context: Arc<Context>,
        cmd_queue: Arc<CommandQueue>,
    ) -> Self {
        let device = Device::new(context.devices()[0]);
        let group_len = usize::try_from(device.max_work_group_size().unwrap()).unwrap();
        let defs = format!(
            concat!(
                "-DSTATE_LEN=({}) -DHASHMAP_LEN=({}) -DHASHMAP_LEN_BITS=({}) ",
                "-DUNKNOWN_BITS=({}) -DUNKNOWN_FILL_BITS=({})"
            ),
            state_len,
            hashmap_len,
            usize::BITS - hashmap_len.leading_zeros() - 1,
            unknown_bits,
            unknown_fill_bits,
        );
        let source = HASH_ENTRY_OPENCL_DEF.to_string()
            + RESOLVE_UNKNOWNS_OPENCL_CODE
            + HASH_FUNC_OPENCL_DEF
            + JOIN_HASHMAP_ITSELF_AND_CHECK_SOLUTION_OPENCL_CODE;
        let program = Program::create_and_build_from_source(&context, &source, &defs).unwrap();
        OpenCLJoinHashMapItselfAndCheckSolution {
            hashmap_len,
            cmd_queue,
            group_len,
            kernel_zero: Kernel::create(&program, "join_hashmap_itself_zero_pred").unwrap(),
            kernel: Kernel::create(&program, "join_hashmap_itself_and_check_solution").unwrap(),
        }
    }

    fn execute_reset_predecessors(&self, out_hashmap: &mut Buffer<HashEntry>) {
        unsafe {
            ExecuteKernel::new(&self.kernel_zero)
                .set_arg(out_hashmap)
                .set_local_work_size(self.group_len)
                .set_global_work_size(
                    ((self.hashmap_len + self.group_len - 1) / self.group_len) * self.group_len,
                )
                .enqueue_nd_range(&self.cmd_queue)
                .unwrap();
            self.cmd_queue.finish().unwrap();
        }
    }

    fn execute(
        &self,
        in_hashmap: &Buffer<HashEntry>,
        out_hashmap: &mut Buffer<HashEntry>,
        unknown_fills: &mut Buffer<u32>,
        sol_and_res_unk: &mut Buffer<SolutionAndResUnknowns>,
    ) {
        unsafe {
            ExecuteKernel::new(&self.kernel_zero)
                .set_arg(out_hashmap)
                .set_local_work_size(self.group_len)
                .set_global_work_size(
                    ((self.hashmap_len + self.group_len - 1) / self.group_len) * self.group_len,
                )
                .enqueue_nd_range(&self.cmd_queue)
                .unwrap();
            ExecuteKernel::new(&self.kernel)
                .set_arg(in_hashmap)
                .set_arg(out_hashmap)
                .set_arg(unknown_fills)
                .set_arg(sol_and_res_unk)
                .set_local_work_size(self.group_len)
                .set_global_work_size(
                    ((self.hashmap_len + self.group_len - 1) / self.group_len) * self.group_len,
                )
                .enqueue_nd_range(&self.cmd_queue)
                .unwrap();
            self.cmd_queue.finish().unwrap();
        }
    }
}

//
// add_to_hashmap - add outputs to hashmap as new entries
//

// special workaround for implement synchronization for HashMap.
// implement unsafe slice with get_mut method.
struct UnsafeSlice<'a, T> {
    slice: &'a [UnsafeCell<T>],
}

unsafe impl<'a, T: Send + Sync> Send for UnsafeSlice<'a, T> {}
unsafe impl<'a, T: Send + Sync> Sync for UnsafeSlice<'a, T> {}

impl<'a, T> UnsafeSlice<'a, T> {
    fn new(slice: &'a mut [T]) -> Self {
        let ptr = slice as *mut [T] as *const [UnsafeCell<T>];
        Self {
            slice: unsafe { &*ptr },
        }
    }

    unsafe fn get_mut(&self, i: usize) -> &mut T {
        let ptr = self.slice[i].get();
        ptr.as_mut().unwrap()
    }
}

fn add_to_hashmap_and_check_solution_cpu(
    state_len: usize,
    arg_bit_place: usize,
    arg: u64,
    outputs: &[u32],
    hashmap: &mut [HashEntry],
    unknown_bits: usize,
    unknown_fill_bits: usize,
    unknown_fills: Arc<Vec<AtomicU32>>,
    resolved_unknowns: Arc<AtomicU64>,
    solution: &Mutex<Option<Solution>>,
    max_predecessors: u32,
    test: bool,
) {
    assert_eq!(hashmap.len().count_ones(), 1);
    let cpu_num = rayon::current_num_threads();
    // words_per_elem - elem length in outputs in words (can be 1 or 2).
    let words_per_elem = (state_len + 1 + 31) >> 5;
    let elem_num = outputs.len() / words_per_elem;
    let chunk_num = std::cmp::min(std::cmp::max(cpu_num * 8, 64), elem_num >> 6);
    let chunk_len = elem_num / chunk_num;
    let arg_start = arg << arg_bit_place;
    let arg_end = arg_start + (1u64 << arg_bit_place);
    let state_mask = (1u64 << state_len) - 1;
    let hashlen_bits = usize::BITS - hashmap.len().leading_zeros() - 1;
    let hashentry_shift = state_len - hashlen_bits as usize;
    let shared_hashmap = UnsafeSlice::new(hashmap);
    let unknown_fill_mask = (1u64 << unknown_fill_bits) - 1;
    outputs
        .chunks(chunk_len * words_per_elem)
        .enumerate()
        .par_bridge()
        .for_each(|(ch_idx, chunk)| {
            let istart = chunk_len * ch_idx;
            for ie in 0..chunk.len() / words_per_elem {
                let i = istart + ie;
                let output = if words_per_elem == 2 {
                    (outputs[2 * i] as u64) | ((outputs[2 * i + 1] as u64) << 32)
                } else {
                    outputs[i] as u64
                };
                let current = (i as u64) + arg_start;
                let next = output & state_mask;
                let state = if (output >> state_len) & 1 != 0 {
                    HASH_STATE_STOPPED
                } else if next == current {
                    HASH_STATE_LOOPED
                } else {
                    HASH_STATE_USED
                };
                if test && next == 0 {
                    // special testcase for testing!!!
                    continue;
                }
                resolve_unknowns(
                    state_len,
                    unknown_bits,
                    unknown_fill_bits,
                    current,
                    next,
                    1,
                    state,
                    unknown_fills.clone(),
                    resolved_unknowns.clone(),
                    solution,
                );
                let cur_hash = hash_function_64(state_len, current);

                let current_unknown_fill_idx =
                    usize::try_from(current >> (state_len - unknown_bits + unknown_fill_bits))
                        .unwrap();
                let current_unknown_fill_value =
                    u32::try_from((current >> (state_len - unknown_bits)) & unknown_fill_mask)
                        .unwrap();
                let current_currently_solved =
                    (current & ((1u64 << (state_len - unknown_bits)) - 1)) == 0
                        && unknown_fills[current_unknown_fill_idx].load(atomic::Ordering::SeqCst)
                            == current_unknown_fill_value;

                // update hash map entry - use unsafe code implement
                // atomic synchronized updating mechanism
                unsafe {
                    let curhe = shared_hashmap.get_mut(cur_hash >> hashentry_shift);
                    let mut try_again = true;
                    // try again until if current is currently solved and
                    // old current is not solved.
                    while try_again {
                        let curhe_state_atomic = AtomicU32::from_ptr(&mut curhe.state as *mut u32);
                        // if previous entry have:
                        // if not currently solved unknown (state).
                        // if predecessors is less and state is not have
                        // HASH_STATE_RESERVED_BY_OTHER_FLAG
                        // update to HASH_STATE_RESERVED_BY_OTHER_FLAG and retrieve old value.
                        let old_state = curhe_state_atomic
                            .fetch_or(HASH_STATE_RESERVED_BY_OTHER_FLAG, atomic::Ordering::SeqCst);
                        std::sync::atomic::fence(atomic::Ordering::SeqCst);

                        let old_current_currently_solved = if old_state != HASH_STATE_UNUSED
                            && (old_state & HASH_STATE_RESERVED_BY_OTHER_FLAG) == 0
                        {
                            let old_current_unknown_fill_idx = usize::try_from(
                                curhe.current >> (state_len - unknown_bits + unknown_fill_bits),
                            )
                            .unwrap();
                            let old_current_unknown_fill_value = u32::try_from(
                                (curhe.current >> (state_len - unknown_bits)) & unknown_fill_mask,
                            )
                            .unwrap();
                            (curhe.current & ((1u64 << (state_len - unknown_bits)) - 1)) == 0
                                && unknown_fills[old_current_unknown_fill_idx]
                                    .load(atomic::Ordering::SeqCst)
                                    == old_current_unknown_fill_value
                        } else {
                            false
                        };

                        if ((current_currently_solved && !old_current_currently_solved)
                            || (!old_current_currently_solved
                                && curhe.predecessors <= max_predecessors))
                            && (old_state & HASH_STATE_RESERVED_BY_OTHER_FLAG) == 0
                        {
                            // do update
                            curhe.current = current;
                            curhe.next = next;
                            curhe.steps = 1;
                            curhe.predecessors = 0;
                            std::sync::atomic::fence(atomic::Ordering::SeqCst);
                            // update state
                            curhe_state_atomic.store(state, atomic::Ordering::SeqCst);
                            try_again = false;
                        } else {
                            try_again = (old_state & HASH_STATE_RESERVED_BY_OTHER_FLAG) != 0
                                || (current_currently_solved && !old_current_currently_solved);
                            std::sync::atomic::fence(atomic::Ordering::SeqCst);
                            curhe_state_atomic.store(old_state, atomic::Ordering::SeqCst);
                        }
                    }
                }
            }
        });
}

//
// main solver code
//

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
    let words_per_elem = (output_len + 31) >> 5;
    mapper.add_with_config(
        "formula",
        circuit,
        CodeConfig::new()
            .elem_inputs(Some(&(0..elem_inputs).collect::<Vec<usize>>()))
            .arg_inputs(Some(&(elem_inputs..input_len).collect::<Vec<usize>>()))
            .aggr_output_code(Some(AGGR_OUTPUT_CPU_CODE))
            .aggr_output_len(Some(words_per_elem * (1 << elem_inputs)))
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
    let words_per_elem = (output_len + 31) >> 5;
    mapper.add_with_config(
        "formula",
        circuit,
        CodeConfig::new()
            .elem_inputs(Some(&(0..elem_inputs).collect::<Vec<usize>>()))
            .arg_inputs(Some(&(elem_inputs..input_len).collect::<Vec<usize>>()))
            .aggr_output_code(Some(AGGR_OUTPUT_OPENCL_CODE))
            .aggr_output_len(Some(words_per_elem * (1 << elem_inputs)))
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
            let words_per_elem = (output_len + 31) >> 5;
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
                    .aggr_output_len(Some(words_per_elem * (1 << 20))),
            );
            builder.add_with_config(
                "formula2",
                circuit,
                CodeConfig::new()
                    .elem_inputs(Some(&(output_len - 20..output_len).collect::<Vec<usize>>()))
                    .arg_inputs(Some(&(0..output_len - 20).collect::<Vec<usize>>()))
                    .aggr_output_code(Some(AGGR_OUTPUT_CPU_CODE))
                    .aggr_output_len(Some(words_per_elem * (1 << 20))),
            );
            let mut execs = builder.build().unwrap();
            let arg_mask = (1u64 << (output_len - 20)) - 1;
            let arg = 138317356571 & arg_mask;
            println!("Arg: {}", arg);
            let input = execs[0].new_data(16);
            let output = execs[0].execute(&input, arg).unwrap().release();
            assert_eq!(output.len(), words_per_elem * (1 << 20));
            for i in 0..1 << 20 {
                let out = if words_per_elem == 2 {
                    (output[2 * i] as u64) | ((output[2 * i + 1] as u64) << 32)
                } else {
                    output[i] as u64
                };
                assert_eq!((i as u64) | (arg << 20), out, "{}: {}", output_len, i);
            }

            let input = execs[1].new_data(16);
            let output = execs[1].execute(&input, arg).unwrap().release();
            assert_eq!(output.len(), words_per_elem * (1 << 20));
            for i in 0..1 << 20 {
                let out = if words_per_elem == 2 {
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
            let words_per_elem = (output_len + 31) >> 5;
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
                    .aggr_output_len(Some(words_per_elem * (1 << 20))),
            );
            builder.add_with_config(
                "formula2",
                circuit,
                CodeConfig::new()
                    .elem_inputs(Some(&(output_len - 20..output_len).collect::<Vec<usize>>()))
                    .arg_inputs(Some(&(0..output_len - 20).collect::<Vec<usize>>()))
                    .aggr_output_code(Some(AGGR_OUTPUT_OPENCL_CODE))
                    .aggr_output_len(Some(words_per_elem * (1 << 20))),
            );
            let mut execs = builder.build().unwrap();
            let arg_mask = (1u64 << (output_len - 20)) - 1;
            let arg = 138317356571 & arg_mask;
            println!("Arg: {}", arg);
            let input = execs[0].new_data(16);
            let output = execs[0].execute(&input, arg).unwrap().release();
            assert_eq!(output.len(), words_per_elem * (1 << 20));
            for i in 0..1 << 20 {
                let out = if words_per_elem == 2 {
                    (output[2 * i] as u64) | ((output[2 * i + 1] as u64) << 32)
                } else {
                    output[i] as u64
                };
                assert_eq!((i as u64) | (arg << 20), out, "{}: {}", output_len, i);
            }

            let input = execs[1].new_data(16);
            let output = execs[1].execute(&input, arg).unwrap().release();
            assert_eq!(output.len(), words_per_elem * (1 << 20));
            for i in 0..1 << 20 {
                let out = if words_per_elem == 2 {
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

    fn join_to_hashmap_cpu_testcase_data_1(
    ) -> (usize, usize, u64, Vec<u32>, Vec<HashEntry>, Vec<HashEntry>) {
        let state_len = 24;
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
            outputs[59021] = 0x11aa22;
            outputs[59045] = 0x77da1b | (1 << 24);
            outputs
        };
        let hashmap = {
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
            // to loop 3
            hashmap[14072] = HashEntry {
                current: 0x2589fa,
                next: (arg << arg_bit_place) | 59021,
                steps: (1 << state_len) - 1,
                predecessors: 5,
                state: HASH_STATE_USED,
            };
            // stop not loop
            hashmap[12061] = HashEntry {
                current: 0xdda02,
                next: (arg << arg_bit_place) | 53029,
                steps: 108,
                predecessors: 12,
                state: HASH_STATE_USED,
            };
            // stop not loop 2
            hashmap[14456] = HashEntry {
                current: 0xfa2ca5,
                next: (arg << arg_bit_place) | 59045,
                steps: (1 << state_len) - 1,
                predecessors: 5,
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
            // to loop 3
            hashmap[14072] = HashEntry {
                current: 0x2589fa,
                next: 0x11aa22,
                steps: 1 << state_len,
                predecessors: 5,
                state: HASH_STATE_LOOPED,
            };
            // stop not loop
            hashmap[12061] = HashEntry {
                current: 0xdda02,
                next: 0xdda02,
                steps: 109,
                predecessors: 12,
                state: HASH_STATE_STOPPED,
            };
            // stop not loop 2
            hashmap[14456] = HashEntry {
                current: 0xfa2ca5,
                next: 0x77da1b,
                steps: 1 << state_len,
                predecessors: 5,
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
        (
            state_len,
            arg_bit_place,
            arg,
            outputs,
            hashmap,
            expected_hashmap,
        )
    }

    fn join_to_hashmap_cpu_testcase_data_2(
    ) -> (usize, usize, u64, Vec<u32>, Vec<HashEntry>, Vec<HashEntry>) {
        let state_len = 32;
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
            outputs[2 * 5902199] = 0x11aa2233;
            outputs[2 * 5902199 + 1] = 0;
            outputs[2 * 5904531] = 0x77da1b1c;
            outputs[2 * 5904531 + 1] = 1;
            outputs
        };
        let hashmap = {
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
                steps: 649,
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
            // to loop 3
            hashmap[14072] = HashEntry {
                current: 0x2589fa11,
                next: (arg << arg_bit_place) | 5902199,
                steps: (1 << state_len) - 1,
                predecessors: 5,
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
            // stop not loop 2
            hashmap[14456] = HashEntry {
                current: 0xfa2ca5d4,
                next: (arg << arg_bit_place) | 5904531,
                steps: (1 << state_len) - 1,
                predecessors: 5,
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
                steps: 650,
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
            // to loop 3
            hashmap[14072] = HashEntry {
                current: 0x2589fa11,
                next: 0x11aa2233,
                steps: 1 << state_len,
                predecessors: 5,
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
            // stop not loop 2
            hashmap[14456] = HashEntry {
                current: 0xfa2ca5d4,
                next: 0x77da1b1c,
                steps: 1 << state_len,
                predecessors: 5,
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
        (
            state_len,
            arg_bit_place,
            arg,
            outputs,
            hashmap,
            expected_hashmap,
        )
    }

    fn join_to_hashmap_cpu_testcase_data_3(
    ) -> (usize, usize, u64, Vec<u32>, Vec<HashEntry>, Vec<HashEntry>) {
        let state_len = 40;
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
            outputs[2 * 5902199] = 0x11aa2233;
            outputs[2 * 5902199 + 1] = 93;
            outputs[2 * 5904531] = 0x77da1b1c;
            outputs[2 * 5904531 + 1] = 142 | (1 << 8);
            outputs
        };
        let hashmap = {
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
                steps: 649,
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
            // to loop 3
            hashmap[14072] = HashEntry {
                current: 0x2589fa114a,
                next: (arg << arg_bit_place) | 5902199,
                steps: (1 << state_len) - 1,
                predecessors: 5,
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
            // stop not loop 2
            hashmap[14456] = HashEntry {
                current: 0xfa218ca5d4,
                next: (arg << arg_bit_place) | 5904531,
                steps: (1 << state_len) - 1,
                predecessors: 5,
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
                steps: 650,
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
            // to loop 3
            hashmap[14072] = HashEntry {
                current: 0x2589fa114a,
                next: 0x11aa2233 | (93 << 32),
                steps: 1 << state_len,
                predecessors: 5,
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
            // stop not loop 2
            hashmap[14456] = HashEntry {
                current: 0xfa218ca5d4,
                next: 0x77da1b1c | (142 << 32),
                steps: 1 << state_len,
                predecessors: 5,
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
        (
            state_len,
            arg_bit_place,
            arg,
            outputs,
            hashmap,
            expected_hashmap,
        )
    }

    #[test]
    fn test_join_to_hashmap_cpu() {
        // 24-bit
        let (state_len, arg_bit_place, arg, outputs, mut hashmap, expected_hashmap) =
            join_to_hashmap_cpu_testcase_data_1();
        join_to_hashmap_cpu(state_len, arg_bit_place, arg, &outputs, &mut hashmap);
        for (i, he) in hashmap.into_iter().enumerate() {
            assert_eq!(expected_hashmap[i], he, "{}: {}", 24, i);
        }

        // 32-bit
        let (state_len, arg_bit_place, arg, outputs, mut hashmap, expected_hashmap) =
            join_to_hashmap_cpu_testcase_data_2();
        join_to_hashmap_cpu(state_len, arg_bit_place, arg, &outputs, &mut hashmap);
        for (i, he) in hashmap.into_iter().enumerate() {
            assert_eq!(expected_hashmap[i], he, "{}: {}", 32, i);
        }

        // 40-bit
        let (state_len, arg_bit_place, arg, outputs, mut hashmap, expected_hashmap) =
            join_to_hashmap_cpu_testcase_data_3();
        join_to_hashmap_cpu(state_len, arg_bit_place, arg, &outputs, &mut hashmap);
        for (i, he) in hashmap.into_iter().enumerate() {
            assert_eq!(expected_hashmap[i], he, "{}: {}", 40, i);
        }
    }

    fn join_to_hashmap_opencl_buffers(
        context: Arc<Context>,
        cmd_queue: Arc<CommandQueue>,
        outputs: &[u32],
        hashmap: &[HashEntry],
    ) -> (Buffer<u32>, Buffer<HashEntry>, usize) {
        unsafe {
            let mut outputs_buffer = Buffer::create(
                &context,
                CL_MEM_READ_WRITE,
                outputs.len(),
                std::ptr::null_mut(),
            )
            .unwrap();
            let mut hashmap_buffer = Buffer::create(
                &context,
                CL_MEM_READ_WRITE,
                hashmap.len(),
                std::ptr::null_mut(),
            )
            .unwrap();
            cmd_queue
                .enqueue_write_buffer(&mut outputs_buffer, CL_BLOCKING, 0, outputs, &[])
                .unwrap();
            cmd_queue
                .enqueue_write_buffer(&mut hashmap_buffer, CL_BLOCKING, 0, hashmap, &[])
                .unwrap();
            (outputs_buffer, hashmap_buffer, hashmap.len())
        }
    }

    #[test]
    fn test_join_to_hashmap_opencl() {
        let device = Device::new(*get_all_devices(CL_DEVICE_TYPE_GPU).unwrap().get(0).unwrap());
        let context = Arc::new(Context::from_device(&device).unwrap());
        #[allow(deprecated)]
        let cmd_queue =
            Arc::new(unsafe { CommandQueue::create(&context, device.id(), 0).unwrap() });
        // 24-bit
        let (state_len, arg_bit_place, arg, outputs, hashmap, expected_hashmap) =
            join_to_hashmap_cpu_testcase_data_1();
        let (outputs_buffer, mut hashmap_buffer, hashmap_len) =
            join_to_hashmap_opencl_buffers(context.clone(), cmd_queue.clone(), &outputs, &hashmap);
        let exec = OpenCLJoinToHashMap::new(
            state_len,
            arg_bit_place,
            hashmap_len,
            context.clone(),
            cmd_queue.clone(),
        );
        exec.execute(arg, &outputs_buffer, &mut hashmap_buffer);
        let mut out_hashmap = vec![HashEntry::default(); hashmap.len()];
        unsafe {
            cmd_queue
                .enqueue_read_buffer(&hashmap_buffer, CL_BLOCKING, 0, &mut out_hashmap, &[])
                .unwrap();
        }
        for (i, he) in out_hashmap.into_iter().enumerate() {
            assert_eq!(expected_hashmap[i], he, "{}: {}", 24, i);
        }

        // 32-bit
        let (output_len, arg_bit_place, arg, outputs, hashmap, expected_hashmap) =
            join_to_hashmap_cpu_testcase_data_2();
        let (outputs_buffer, mut hashmap_buffer, hashmap_len) =
            join_to_hashmap_opencl_buffers(context.clone(), cmd_queue.clone(), &outputs, &hashmap);
        let exec = OpenCLJoinToHashMap::new(
            output_len,
            arg_bit_place,
            hashmap_len,
            context.clone(),
            cmd_queue.clone(),
        );
        exec.execute(arg, &outputs_buffer, &mut hashmap_buffer);
        let mut out_hashmap = vec![HashEntry::default(); hashmap.len()];
        unsafe {
            cmd_queue
                .enqueue_read_buffer(&hashmap_buffer, CL_BLOCKING, 0, &mut out_hashmap, &[])
                .unwrap();
        }
        for (i, he) in out_hashmap.into_iter().enumerate() {
            assert_eq!(expected_hashmap[i], he, "{}: {}", 32, i);
        }

        // 40-bit
        let (output_len, arg_bit_place, arg, outputs, hashmap, expected_hashmap) =
            join_to_hashmap_cpu_testcase_data_3();
        let (outputs_buffer, mut hashmap_buffer, hashmap_len) =
            join_to_hashmap_opencl_buffers(context.clone(), cmd_queue.clone(), &outputs, &hashmap);
        let exec = OpenCLJoinToHashMap::new(
            output_len,
            arg_bit_place,
            hashmap_len,
            context.clone(),
            cmd_queue.clone(),
        );
        exec.execute(arg, &outputs_buffer, &mut hashmap_buffer);
        let mut out_hashmap = vec![HashEntry::default(); hashmap.len()];
        unsafe {
            cmd_queue
                .enqueue_read_buffer(&hashmap_buffer, CL_BLOCKING, 0, &mut out_hashmap, &[])
                .unwrap();
        }
        for (i, he) in out_hashmap.into_iter().enumerate() {
            assert_eq!(expected_hashmap[i], he, "{}: {}", 40, i);
        }
    }

    fn hashmap_insert(state_len: usize, hbits: usize, hashmap: &mut [HashEntry], e: HashEntry) {
        assert_eq!(e.current >> state_len, 0);
        assert_eq!(e.next >> state_len, 0);
        let idx = hash_function_64(state_len, e.current) >> (state_len - hbits);
        assert!(
            hashmap[idx].state == HASH_STATE_UNUSED,
            "{} {} {}: {} {}",
            state_len,
            hbits,
            idx,
            e.current,
            hashmap[idx].state
        );
        hashmap[idx] = e;
    }

    struct JoinHashMapItselfAndCheckSolutionData {
        state_len: usize,
        hbits: usize,
        hashmap: Vec<HashEntry>,
        expected_hashmap: Vec<HashEntry>,
        unknown_bits: usize,
        unknown_fill_bits: usize,
        unknown_fills: Arc<Vec<AtomicU32>>,
        resolved_unknowns: Arc<AtomicU64>,
        solution: Mutex<Option<Solution>>,
        expected_unknown_fills: Arc<Vec<AtomicU32>>,
        expected_resolved_unknowns: u64,
        expected_solution: Option<Solution>,
    }

    fn join_hashmap_itself_and_check_solution_data() -> JoinHashMapItselfAndCheckSolutionData {
        let state_len = 44;
        let hbits = 15;
        let unknown_bits = 20;
        let unknown_fill_bits = 4;
        let hashmap = {
            let mut hashmap = vec![HashEntry::default(); 1 << hbits];
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0xda02489490d,
                    next: 0xe6730bc0114, // stopped
                    steps: 3416,
                    state: HASH_STATE_USED,
                    predecessors: 7,
                },
            );
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0x50146d0acba,
                    next: 0xc2a9588207b,
                    steps: 771,
                    state: HASH_STATE_USED,
                    predecessors: 16,
                },
            );
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0x6bc3592a5d3,
                    next: 0x94491894941,
                    steps: 9940295402168,
                    state: HASH_STATE_USED,
                    predecessors: 16,
                },
            );
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0x7e05689bca1,
                    next: 0x6bc3592a5d3,
                    steps: 11058562066515,
                    state: HASH_STATE_USED,
                    predecessors: 18,
                },
            );
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0xed0a90551bc,
                    next: 0xd0a9551b05d,
                    steps: 76711,
                    state: HASH_STATE_USED,
                    predecessors: 27,
                },
            );
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0x14dc0a0452e,
                    next: 0xa52065bc0a0,
                    steps: 6829,
                    state: HASH_STATE_USED,
                    predecessors: 24,
                },
            );
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0xe6730bc0114,
                    next: 0x0123494411a,
                    steps: 2415,
                    state: HASH_STATE_STOPPED,
                    predecessors: 19,
                },
            );
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0xc2a9588207b,
                    next: 0x293939391,
                    steps: 2831,
                    state: HASH_STATE_STOPPED,
                    predecessors: 5,
                },
            );
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0xa52065bc0a0,
                    next: 0xabcdefaaa,
                    steps: 826611,
                    state: HASH_STATE_LOOPED,
                    predecessors: 22,
                },
            );
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0xd0a9551b05d,
                    next: 0x55690a11a,
                    steps: 1928921,
                    state: HASH_STATE_LOOPED,
                    predecessors: 14,
                },
            );
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0x26d0a46141c,
                    next: 0xc562019a014,
                    steps: 76792,
                    state: HASH_STATE_USED,
                    predecessors: 4,
                },
            );
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0xc562019a014,
                    next: 0x93102144323,
                    steps: 385491,
                    state: HASH_STATE_USED,
                    predecessors: 12,
                },
            );
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0x4a0589502a1,
                    next: 0x70489302935,
                    steps: 556111,
                    state: HASH_STATE_USED,
                    predecessors: 11,
                },
            );
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0x6d0a942961a, // this same hidx but not same current
                    next: 0x1043aabbcc7,
                    steps: 88211,
                    state: HASH_STATE_USED,
                    predecessors: 16,
                },
            );
            // loop resolving 2
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0x3c0da054677,
                    next: 0xdca03afa1fa,
                    steps: 23891,
                    state: HASH_STATE_USED,
                    predecessors: 26,
                },
            );
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0xdca03afa1fa,
                    next: 0x3c0da054677,
                    steps: 35421,
                    state: HASH_STATE_USED,
                    predecessors: 31,
                },
            );
            // resolved 1
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: ((49201 << 4) | 15) << (state_len - unknown_bits),
                    next: 0x3c0d3441167,
                    steps: 35421,
                    state: HASH_STATE_LOOPED,
                    predecessors: 31,
                },
            );
            // resolved 2
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: ((25901 << 4) | 15) << (state_len - unknown_bits),
                    next: 0xccd14859211,
                    steps: 585691,
                    state: HASH_STATE_USED,
                    predecessors: 10,
                },
            );
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0xccd14859211,
                    next: 0x14859211,
                    steps: 18901,
                    state: HASH_STATE_LOOPED,
                    predecessors: 17,
                },
            );
            // resolved 3
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: ((3058 << 4) | 12) << (state_len - unknown_bits),
                    next: 0xdaba053a14,
                    steps: 11055,
                    state: HASH_STATE_LOOPED,
                    predecessors: 3,
                },
            );
            // unresolved 3 - because doesn't match unknown
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: (((3058 << 4) | 12) << (state_len - unknown_bits)) | 21,
                    next: 0xdaba0831,
                    steps: 7073,
                    state: HASH_STATE_LOOPED,
                    predecessors: 3,
                },
            );
            // unresolved 3 - because doesn't match unknown
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: (((60710 << 4) | 15) << (state_len - unknown_bits)),
                    next: 0xdabaddaa,
                    steps: 77721,
                    state: HASH_STATE_USED,
                    predecessors: 3,
                },
            );
            // solution
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: (((20791 << 4) | 5) << (state_len - unknown_bits)),
                    next: 0xd0a95051905,
                    steps: 7611,
                    state: HASH_STATE_USED,
                    predecessors: 8,
                },
            );
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0xd0a95051905,
                    next: 0x44066aa0bc1,
                    steps: 1765,
                    state: HASH_STATE_STOPPED,
                    predecessors: 9,
                },
            );
            // for i in 0..10000000 {
            //     let state = 0x6d0a9405157 + i;
            //     let hidx = hash_function_64(state_len, state) >> (state_len - hbits);
            //     if i % 1000 == 0 {
            //         println!("Hidx: {} {}", i, hidx);
            //     }
            //     if hidx == 11900 {
            //         println!("State: 0x{:016x}", state);
            //         break;
            //     }
            // }
            // println!(
            //     "Hashfunc: {}",
            //     hash_function_64(state_len, 0x70489302935) >> (state_len - hbits)
            // );
            hashmap
        };
        let unknown_fills = create_vec_of_atomic_u32(1 << (unknown_bits - unknown_fill_bits));
        unknown_fills[117].store(16, atomic::Ordering::SeqCst);
        unknown_fills[6692].store(16, atomic::Ordering::SeqCst);
        unknown_fills[17982].store(16, atomic::Ordering::SeqCst);
        unknown_fills[52069].store(16, atomic::Ordering::SeqCst);
        unknown_fills[3058].store(12, atomic::Ordering::SeqCst);
        unknown_fills[49201].store(15, atomic::Ordering::SeqCst);
        unknown_fills[25901].store(15, atomic::Ordering::SeqCst);
        unknown_fills[60710].store(11, atomic::Ordering::SeqCst);
        unknown_fills[20791].store(5, atomic::Ordering::SeqCst);
        let resolved_unknowns = Arc::new(AtomicU64::new(
            unknown_fills
                .iter()
                .filter(|v| (v.load(atomic::Ordering::SeqCst) >> unknown_fill_bits) != 0)
                .count() as u64,
        ));
        let solution = Mutex::new(None);
        let expected_hashmap = {
            let mut hashmap = vec![HashEntry::default(); 1 << hbits];
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0xda02489490d,
                    next: 0x0123494411a, // stopped
                    steps: 3416 + 2415,
                    state: HASH_STATE_STOPPED,
                    predecessors: 7,
                },
            );
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0x50146d0acba,
                    next: 0x293939391,
                    steps: 771 + 2831,
                    state: HASH_STATE_STOPPED,
                    predecessors: 16,
                },
            );
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0x6bc3592a5d3,
                    next: 0x94491894941,
                    steps: 9940295402168,
                    state: HASH_STATE_USED,
                    predecessors: 17,
                },
            );
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0x7e05689bca1,
                    next: 0x94491894941,
                    steps: 11058562066515 + 9940295402168,
                    state: HASH_STATE_LOOPED,
                    predecessors: 18,
                },
            );
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0xed0a90551bc,
                    next: 0x55690a11a,
                    steps: 76711 + 1928921,
                    state: HASH_STATE_LOOPED,
                    predecessors: 27,
                },
            );
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0x14dc0a0452e,
                    next: 0xabcdefaaa,
                    steps: 6829 + 826611,
                    state: HASH_STATE_LOOPED,
                    predecessors: 24,
                },
            );
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0xe6730bc0114,
                    next: 0x0123494411a,
                    steps: 2415,
                    state: HASH_STATE_STOPPED,
                    predecessors: 20,
                },
            );
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0xc2a9588207b,
                    next: 0x293939391,
                    steps: 2831,
                    state: HASH_STATE_STOPPED,
                    predecessors: 6,
                },
            );
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0xa52065bc0a0,
                    next: 0xabcdefaaa,
                    steps: 826611,
                    state: HASH_STATE_LOOPED,
                    predecessors: 23,
                },
            );
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0xd0a9551b05d,
                    next: 0x55690a11a,
                    steps: 1928921,
                    state: HASH_STATE_LOOPED,
                    predecessors: 15,
                },
            );
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0x26d0a46141c,
                    next: 0x93102144323,
                    steps: 76792 + 385491,
                    state: HASH_STATE_USED,
                    predecessors: 4,
                },
            );
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0xc562019a014,
                    next: 0x93102144323,
                    steps: 385491,
                    state: HASH_STATE_USED,
                    predecessors: 13,
                },
            );
            // not joined
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0x4a0589502a1,
                    next: 0x70489302935,
                    steps: 556111,
                    state: HASH_STATE_USED,
                    predecessors: 11,
                },
            );
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0x6d0a942961a, // this same hidx but not same current
                    next: 0x1043aabbcc7,
                    steps: 88211,
                    state: HASH_STATE_USED,
                    predecessors: 16,
                },
            );
            // loop resolving 2
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0x3c0da054677,
                    next: 0x3c0da054677,
                    steps: 35421 + 23891,
                    state: HASH_STATE_LOOPED,
                    predecessors: 27,
                },
            );
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0xdca03afa1fa,
                    next: 0xdca03afa1fa,
                    steps: 35421 + 23891,
                    state: HASH_STATE_LOOPED,
                    predecessors: 32,
                },
            );
            // resolved 1
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: ((49201 << 4) | 15) << (state_len - unknown_bits),
                    next: 0x3c0d3441167,
                    steps: 35421,
                    state: HASH_STATE_LOOPED,
                    predecessors: 31,
                },
            );
            // resolved 2
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: ((25901 << 4) | 15) << (state_len - unknown_bits),
                    next: 0x14859211,
                    steps: 585691 + 18901,
                    state: HASH_STATE_LOOPED,
                    predecessors: 10,
                },
            );
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0xccd14859211,
                    next: 0x14859211,
                    steps: 18901,
                    state: HASH_STATE_LOOPED,
                    predecessors: 18,
                },
            );
            // resolved 3
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: ((3058 << 4) | 12) << (state_len - unknown_bits),
                    next: 0xdaba053a14,
                    steps: 11055,
                    state: HASH_STATE_LOOPED,
                    predecessors: 3,
                },
            );
            // unresolved 3 - because doesn't match unknown
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: (((3058 << 4) | 12) << (state_len - unknown_bits)) | 21,
                    next: 0xdaba0831,
                    steps: 7073,
                    state: HASH_STATE_LOOPED,
                    predecessors: 3,
                },
            );
            // unresolved 3 - because doesn't match unknown
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: (((60710 << 4) | 15) << (state_len - unknown_bits)),
                    next: 0xdabaddaa,
                    steps: 77721,
                    state: HASH_STATE_USED,
                    predecessors: 3,
                },
            );
            // solution
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: (((20791 << 4) | 5) << (state_len - unknown_bits)),
                    next: 0x44066aa0bc1,
                    steps: 7611 + 1765,
                    state: HASH_STATE_STOPPED,
                    predecessors: 8,
                },
            );
            hashmap_insert(
                state_len,
                hbits,
                &mut hashmap,
                HashEntry {
                    current: 0xd0a95051905,
                    next: 0x44066aa0bc1,
                    steps: 1765,
                    state: HASH_STATE_STOPPED,
                    predecessors: 10,
                },
            );
            hashmap
        };
        let expected_unknown_fills =
            create_vec_of_atomic_u32(1 << (unknown_bits - unknown_fill_bits));
        for i in 0..1 << 16 {
            expected_unknown_fills[i].store(
                unknown_fills[i].load(atomic::Ordering::SeqCst),
                atomic::Ordering::SeqCst,
            );
        }
        expected_unknown_fills[49201].store(16, atomic::Ordering::SeqCst);
        expected_unknown_fills[25901].store(16, atomic::Ordering::SeqCst);
        expected_unknown_fills[3058].store(13, atomic::Ordering::SeqCst);
        expected_unknown_fills[20791].store(6, atomic::Ordering::SeqCst);
        let expected_resolved_unknowns = expected_unknown_fills
            .iter()
            .filter(|v| (v.load(atomic::Ordering::SeqCst) >> unknown_fill_bits) != 0)
            .count() as u64;
        let expected_solution = Some(Solution {
            start: (((20791 << 4) | 5) << (state_len - unknown_bits)),
            end: 0x44066aa0bc1,
            steps: 7611 + 1765,
        });
        JoinHashMapItselfAndCheckSolutionData {
            state_len,
            hbits,
            hashmap,
            expected_hashmap,
            unknown_bits,
            unknown_fill_bits,
            unknown_fills,
            resolved_unknowns,
            solution,
            expected_unknown_fills,
            expected_resolved_unknowns,
            expected_solution,
        }
    }

    #[test]
    fn test_join_hashmap_itself_and_check_solution_cpu() {
        let JoinHashMapItselfAndCheckSolutionData {
            state_len,
            hbits,
            hashmap,
            expected_hashmap,
            unknown_bits,
            unknown_fill_bits,
            unknown_fills,
            resolved_unknowns,
            solution,
            expected_unknown_fills,
            expected_resolved_unknowns,
            expected_solution,
        } = join_hashmap_itself_and_check_solution_data();
        let preds_update = create_vec_of_atomic_u32(hashmap.len());
        let mut out_hashmap = vec![
            HashEntry {
                current: 1115774,
                next: 494022,
                steps: 450589390239,
                state: 8459021,
                predecessors: 489481,
            };
            hashmap.len()
        ];
        join_hashmap_itself_and_check_solution_cpu(
            state_len,
            preds_update.clone(),
            &hashmap,
            &mut out_hashmap,
            unknown_bits,
            unknown_fill_bits,
            unknown_fills.clone(),
            resolved_unknowns.clone(),
            &solution,
        );
        for (i, he) in out_hashmap.into_iter().enumerate() {
            assert_eq!(expected_hashmap[i], he, "{}", i);
        }
        for (i, uf) in unknown_fills.iter().enumerate() {
            assert_eq!(
                expected_unknown_fills[i].load(atomic::Ordering::SeqCst),
                uf.load(atomic::Ordering::SeqCst),
                "{}",
                i
            );
        }
        assert_eq!(
            expected_resolved_unknowns,
            resolved_unknowns.load(atomic::Ordering::SeqCst)
        );
        assert_eq!(expected_solution, *solution.lock().unwrap());
    }

    #[test]
    fn test_join_hashmap_itself_and_check_solution_opencl() {
        let device = Device::new(*get_all_devices(CL_DEVICE_TYPE_GPU).unwrap().get(0).unwrap());
        let context = Arc::new(Context::from_device(&device).unwrap());
        #[allow(deprecated)]
        let cmd_queue =
            Arc::new(unsafe { CommandQueue::create(&context, device.id(), 0).unwrap() });
        let JoinHashMapItselfAndCheckSolutionData {
            state_len,
            hbits,
            hashmap,
            expected_hashmap,
            unknown_bits,
            unknown_fill_bits,
            unknown_fills,
            resolved_unknowns,
            solution,
            expected_unknown_fills,
            expected_resolved_unknowns,
            expected_solution,
        } = join_hashmap_itself_and_check_solution_data();
        let mut in_hashmap_buffer = unsafe {
            Buffer::<HashEntry>::create(
                &context,
                CL_MEM_READ_WRITE,
                hashmap.len(),
                std::ptr::null_mut(),
            )
            .unwrap()
        };
        let mut out_hashmap_buffer = unsafe {
            Buffer::<HashEntry>::create(
                &context,
                CL_MEM_READ_WRITE,
                hashmap.len(),
                std::ptr::null_mut(),
            )
            .unwrap()
        };
        let mut unknown_fills_buffer = unsafe {
            Buffer::<u32>::create(
                &context,
                CL_MEM_READ_WRITE,
                unknown_fills.len(),
                std::ptr::null_mut(),
            )
            .unwrap()
        };
        let mut sol_and_res_unk_buffer = unsafe {
            Buffer::<SolutionAndResUnknowns>::create(
                &context,
                CL_MEM_READ_WRITE,
                1,
                std::ptr::null_mut(),
            )
            .unwrap()
        };
        unsafe {
            cmd_queue
                .enqueue_fill_buffer(
                    &mut out_hashmap_buffer,
                    &[HashEntry {
                        current: 1115774,
                        next: 494022,
                        steps: 450589390239,
                        state: 8459021,
                        predecessors: 489481,
                    }],
                    0,
                    std::mem::size_of::<HashEntry>() * hashmap.len(),
                    &[],
                )
                .unwrap();
            cmd_queue
                .enqueue_write_buffer(&mut in_hashmap_buffer, CL_BLOCKING, 0, &hashmap, &[])
                .unwrap();
            let unknown_fills_data = unknown_fills
                .iter()
                .map(|v| v.load(atomic::Ordering::SeqCst))
                .collect::<Vec<_>>();
            cmd_queue
                .enqueue_write_buffer(
                    &mut unknown_fills_buffer,
                    CL_BLOCKING,
                    0,
                    &unknown_fills_data,
                    &[],
                )
                .unwrap();
            let sol_and_res_unk = solution
                .lock()
                .unwrap()
                .map(|sol| SolutionAndResUnknowns {
                    resolved_unknowns: resolved_unknowns.load(atomic::Ordering::SeqCst),
                    sol_start: sol.start,
                    sol_end: sol.end,
                    sol_steps: sol.steps,
                    sol_defined: 1,
                })
                .unwrap_or(SolutionAndResUnknowns {
                    resolved_unknowns: resolved_unknowns.load(atomic::Ordering::SeqCst),
                    sol_start: 0,
                    sol_end: 0,
                    sol_steps: 0,
                    sol_defined: 0,
                });
            cmd_queue
                .enqueue_write_buffer(
                    &mut sol_and_res_unk_buffer,
                    CL_BLOCKING,
                    0,
                    &[sol_and_res_unk],
                    &[],
                )
                .unwrap();
        }
        cmd_queue.finish().unwrap();
        let join_itself = OpenCLJoinHashMapItselfAndCheckSolution::new(
            state_len,
            hashmap.len(),
            unknown_bits,
            unknown_fill_bits,
            context.clone(),
            cmd_queue.clone(),
        );
        join_itself.execute(
            &in_hashmap_buffer,
            &mut out_hashmap_buffer,
            &mut unknown_fills_buffer,
            &mut sol_and_res_unk_buffer,
        );
        let mut out_hashmap = vec![HashEntry::default(); hashmap.len()];
        let mut out_unknown_fills = vec![0; unknown_fills.len()];
        let mut out_sol_and_res_unk = [SolutionAndResUnknowns::default()];
        unsafe {
            cmd_queue
                .enqueue_read_buffer(&out_hashmap_buffer, CL_BLOCKING, 0, &mut out_hashmap, &[])
                .unwrap();
            cmd_queue
                .enqueue_read_buffer(
                    &unknown_fills_buffer,
                    CL_BLOCKING,
                    0,
                    &mut out_unknown_fills,
                    &[],
                )
                .unwrap();
            cmd_queue
                .enqueue_read_buffer(
                    &sol_and_res_unk_buffer,
                    CL_BLOCKING,
                    0,
                    &mut out_sol_and_res_unk,
                    &[],
                )
                .unwrap();
        }
        for (i, he) in out_hashmap.into_iter().enumerate() {
            assert_eq!(expected_hashmap[i], he, "{}", i);
        }
        for (i, uf) in out_unknown_fills.iter().enumerate() {
            assert_eq!(
                expected_unknown_fills[i].load(atomic::Ordering::SeqCst),
                *uf,
                "{}",
                i
            );
        }
        let out_sol_and_res_unk = out_sol_and_res_unk[0];
        //println!("sol_and_res: {:?}", out_sol_and_res_unk);
        assert_eq!(
            expected_resolved_unknowns,
            out_sol_and_res_unk.resolved_unknowns
        );
        assert_eq!(
            expected_solution.is_some(),
            out_sol_and_res_unk.sol_defined != 0
        );
        if let Some(sol) = expected_solution {
            assert_eq!(
                sol,
                Solution {
                    start: out_sol_and_res_unk.sol_start,
                    end: out_sol_and_res_unk.sol_end,
                    steps: out_sol_and_res_unk.sol_steps,
                }
            );
        }
    }

    struct AddToHashMapAndCheckSolutionData {
        state_len: usize,
        arg_bit_place: usize,
        arg: u64,
        outputs: Vec<u32>,
        hashmap: Vec<HashEntry>,
        expected_hashmap: Vec<HashEntry>,
        unknown_bits: usize,
        unknown_fill_bits: usize,
        unknown_fills: Arc<Vec<AtomicU32>>,
        resolved_unknowns: Arc<AtomicU64>,
        solution: Mutex<Option<Solution>>,
        expected_unknown_fills: Arc<Vec<AtomicU32>>,
        expected_resolved_unknowns: u64,
        expected_solution: Option<Solution>,
    }

    fn add_to_hashmap_and_check_solution_data_1() -> AddToHashMapAndCheckSolutionData {
        let state_len = 24;
        let arg_bit_place = 16;
        let arg = 142;
        let unknown_bits = 14;
        let unknown_fill_bits = 4;
        let hbits = 15;
        let mut outputs = vec![0u32; 1 << arg_bit_place];
        // let mut vh = vec![false; 1 << hbits];
        // for i in 0..1 << arg_bit_place {
        // for i in [4321, 7934, 14723, 15720, ((1 << 4) + 8) << 10, 18056, 21953, 32994,
        //         ((2 << 4) + 6) << 10, ((3 << 4) + 10) << 10, 40691] {
        //     outputs[i] = (i as u32)*9;
        //     let idx = hash_function_64(state_len,
        //         (i as u64) | ((arg as u64) << arg_bit_place)) >> (state_len - hbits);
        //     if vh[idx] {
        //         println!("We have conflicts! {}", i);
        //     }
        //     vh[idx] = true;
        // }
        outputs[4321] = 0x23d5b2;
        outputs[7934] = 7934 | (arg << arg_bit_place);
        outputs[14723] = 0xda61c4 | (1 << state_len);
        outputs[15720] = 15720 | (arg << arg_bit_place) | (1 << state_len);
        outputs[((1 << 4) + 8) << 10] = 0x1ac7d3 | (1 << state_len); // solution
        outputs[18056] = 0xd09f42; // not filled
        outputs[21953] = 0xd493a;
        outputs[32994] = 0xffaa4c;
        outputs[((2 << 4) + 6) << 10] = (((2 << 4) + 6) << 10) | (arg << arg_bit_place);
        outputs[((3 << 4) + 10) << 10] = 0x140ca2; // not filled
        outputs[40691] = 0x1956e3 | (1 << state_len);
        outputs[58727] = 0x50c1a3; // not filled
        let mut hashmap = vec![HashEntry::default(); 1 << hbits];
        // let output_idx = ((3 << 4) + 10) << 10;
        // let needed_hidx = hash_function_64(state_len, output_idx | ((arg as u64) << arg_bit_place))
        //     >> (state_len - hbits);
        // println!(
        //     "Hash for {}: {} {}",
        //     output_idx | ((arg as u64) << arg_bit_place),
        //     needed_hidx,
        //     state_len - hbits
        // );
        // println!("Hash2: {}", hash_function_64(state_len, 0x357c00) >> (state_len - hbits));
        // for i in 0..16000000 {
        //     let state = 0x321a1 + i;
        //     let hidx = hash_function_64(state_len, state) >> (state_len - hbits);
        //     if i % 1000 == 0 {
        //         println!("Hidx: {} {}", i, hidx);
        //     }
        //     if hidx == needed_hidx && (state & ((1u64 << (state_len - unknown_bits)) - 1)) == 0 {
        //         println!("State: 0x{:016x}", state);
        //         break;
        //     }
        // }
        hashmap_insert(
            state_len,
            hbits,
            &mut hashmap,
            HashEntry {
                // this same hash index as: 18056 | ((arg as u64) << arg_bit_place): conflict
                current: 0x39f5b,
                next: 0x1a0bc1,
                steps: 15,
                state: HASH_STATE_USED,
                predecessors: 10,
            },
        );
        hashmap_insert(
            state_len,
            hbits,
            &mut hashmap,
            HashEntry {
                // this same hash index as: 21953 | ((arg as u64) << arg_bit_place): conflict
                current: 0x3693c,
                next: 0x24b0a5,
                steps: 77,
                state: HASH_STATE_USED,
                predecessors: 0,
            },
        );
        hashmap_insert(
            state_len,
            hbits,
            &mut hashmap,
            HashEntry {
                // this same hash index as:
                // (((1 << 4) + 8) << 10) | ((arg as u64) << arg_bit_place): conflict
                current: 0x3d2c00,
                next: 0x481da,
                steps: 2700,
                state: HASH_STATE_USED,
                predecessors: 67,
            },
        );
        hashmap_insert(
            state_len,
            hbits,
            &mut hashmap,
            HashEntry {
                // this same hash index as:
                // (((2 << 4) + 6) << 10) | ((arg as u64) << arg_bit_place): conflict
                current: 0x34fe7,
                next: 0x481da,
                steps: 1612,
                state: HASH_STATE_USED,
                predecessors: 0,
            },
        );
        hashmap_insert(
            state_len,
            hbits,
            &mut hashmap,
            HashEntry {
                // this same hash index as:
                // (((3 << 4) + 6) << 10) | ((arg as u64) << arg_bit_place): conflict
                current: 0x357c00,
                next: 0xda10da,
                steps: 66581,
                state: HASH_STATE_USED,
                predecessors: 0,
            },
        );
        hashmap_insert(
            state_len,
            hbits,
            &mut hashmap,
            HashEntry {
                // this same hash index as:
                // 58727 | ((arg as u64) << arg_bit_place): conflict
                current: 0xb98800,
                next: 0x60ca54,
                steps: 40471,
                state: HASH_STATE_USED,
                predecessors: 0,
            },
        );
        let unknown_fills = create_vec_of_atomic_u32(1 << (unknown_bits - unknown_fill_bits));
        unknown_fills[((arg as usize) << 2) | 1].store(8, atomic::Ordering::SeqCst);
        unknown_fills[((arg as usize) << 2) | 2].store(2, atomic::Ordering::SeqCst);
        unknown_fills[((arg as usize) << 2) | 3].store(10, atomic::Ordering::SeqCst);
        unknown_fills[0x357c00 >> (state_len - unknown_bits + unknown_fill_bits)].store(
            (0x357c00 >> (state_len - unknown_bits)) & ((1u32 << (unknown_fill_bits)) - 1),
            atomic::Ordering::SeqCst,
        );
        unknown_fills[0xb98800 >> (state_len - unknown_bits + unknown_fill_bits)].store(
            (0xb98800 >> (state_len - unknown_bits)) & ((1u32 << (unknown_fill_bits)) - 1),
            atomic::Ordering::SeqCst,
        );
        let resolved_unknowns = Arc::new(AtomicU64::new(
            unknown_fills
                .iter()
                .filter(|v| (v.load(atomic::Ordering::SeqCst) >> unknown_fill_bits) != 0)
                .count() as u64,
        ));
        let solution = Mutex::new(None);

        let mut expected_hashmap = vec![HashEntry::default(); 1 << hbits];
        hashmap_insert(
            state_len,
            hbits,
            &mut expected_hashmap,
            HashEntry {
                current: 4321 | ((arg as u64) << arg_bit_place),
                next: 0x23d5b2,
                steps: 1,
                state: HASH_STATE_USED,
                predecessors: 0,
            },
        );
        hashmap_insert(
            state_len,
            hbits,
            &mut expected_hashmap,
            HashEntry {
                current: 15720 | ((arg as u64) << arg_bit_place),
                next: 15720 | ((arg as u64) << arg_bit_place),
                steps: 1,
                state: HASH_STATE_STOPPED,
                predecessors: 0,
            },
        );
        hashmap_insert(
            state_len,
            hbits,
            &mut expected_hashmap,
            HashEntry {
                current: 32994 | ((arg as u64) << arg_bit_place),
                next: 0xffaa4c,
                steps: 1,
                state: HASH_STATE_USED,
                predecessors: 0,
            },
        );
        hashmap_insert(
            state_len,
            hbits,
            &mut expected_hashmap,
            HashEntry {
                // this same hash index as: 18056 | ((arg as u64) << arg_bit_place): conflict
                current: 0x39f5b,
                next: 0x1a0bc1,
                steps: 15,
                state: HASH_STATE_USED,
                predecessors: 10,
            },
        );
        hashmap_insert(
            state_len,
            hbits,
            &mut expected_hashmap,
            HashEntry {
                current: 21953 | ((arg as u64) << arg_bit_place),
                next: 0xd493a,
                steps: 1,
                state: HASH_STATE_USED,
                predecessors: 0,
            },
        );
        hashmap_insert(
            state_len,
            hbits,
            &mut expected_hashmap,
            HashEntry {
                // this same hash index as:
                // (((1 << 4) + 8) << 10) | ((arg as u64) << arg_bit_place): conflict
                current: (((1 << 4) + 8) << 10) | ((arg as u64) << arg_bit_place),
                next: 0x1ac7d3,
                steps: 1,
                state: HASH_STATE_USED,
                predecessors: 0,
            },
        );
        hashmap_insert(
            state_len,
            hbits,
            &mut expected_hashmap,
            HashEntry {
                // this same hash index as:
                // (((2 << 4) + 6) << 10) | ((arg as u64) << arg_bit_place): conflict
                current: (((2 << 4) + 6) << 10) | ((arg as u64) << arg_bit_place),
                next: (((2 << 4) + 6) << 10) | ((arg as u64) << arg_bit_place),
                steps: 1,
                state: HASH_STATE_LOOPED,
                predecessors: 0,
            },
        );
        hashmap_insert(
            state_len,
            hbits,
            &mut expected_hashmap,
            HashEntry {
                // this same hash index as:
                // (((3 << 4) + 6) << 10) | ((arg as u64) << arg_bit_place): conflict
                current: 0x357c00,
                next: 0xda10da,
                steps: 66581,
                state: HASH_STATE_USED,
                predecessors: 0,
            },
        );
        hashmap_insert(
            state_len,
            hbits,
            &mut expected_hashmap,
            HashEntry {
                // this same hash index as:
                // 58727 | ((arg as u64) << arg_bit_place): conflict
                current: 0xb98800,
                next: 0x60ca54,
                steps: 40471,
                state: HASH_STATE_USED,
                predecessors: 0,
            },
        );
        hashmap_insert(
            state_len,
            hbits,
            &mut expected_hashmap,
            HashEntry {
                current: 40691 | ((arg as u64) << arg_bit_place),
                next: 0x1956e3,
                steps: 1,
                state: HASH_STATE_STOPPED,
                predecessors: 0,
            },
        );
        let expected_unknown_fills =
            create_vec_of_atomic_u32(1 << (unknown_bits - unknown_fill_bits));
        for i in 0..1 << (unknown_bits - unknown_fill_bits) {
            expected_unknown_fills[i].store(
                unknown_fills[i].load(atomic::Ordering::SeqCst),
                atomic::Ordering::SeqCst,
            );
        }
        let expected_resolved_unknowns = expected_unknown_fills
            .iter()
            .filter(|v| (v.load(atomic::Ordering::SeqCst) >> unknown_fill_bits) != 0)
            .count() as u64;
        let expected_solution = None;

        AddToHashMapAndCheckSolutionData {
            state_len,
            arg_bit_place,
            arg: arg as u64,
            outputs,
            hashmap,
            expected_hashmap,
            unknown_bits,
            unknown_fill_bits,
            unknown_fills,
            resolved_unknowns,
            solution,
            expected_unknown_fills,
            expected_resolved_unknowns,
            expected_solution,
        }
    }

    #[test]
    fn test_add_to_hashmap_and_check_solution_cpu() {
        let AddToHashMapAndCheckSolutionData {
            state_len,
            arg_bit_place,
            arg,
            outputs,
            mut hashmap,
            expected_hashmap,
            unknown_bits,
            unknown_fill_bits,
            unknown_fills,
            resolved_unknowns,
            solution,
            expected_unknown_fills,
            expected_resolved_unknowns,
            expected_solution,
        } = add_to_hashmap_and_check_solution_data_1();
        let max_predecessors = 1;
        add_to_hashmap_and_check_solution_cpu(
            state_len,
            arg_bit_place,
            arg,
            &outputs,
            &mut hashmap,
            unknown_bits,
            unknown_fill_bits,
            unknown_fills.clone(),
            resolved_unknowns.clone(),
            &solution,
            max_predecessors,
            true,
        );
        for (i, he) in hashmap.into_iter().enumerate() {
            assert_eq!(expected_hashmap[i], he, "{}", i);
        }
        for (i, uf) in unknown_fills.iter().enumerate() {
            assert_eq!(
                expected_unknown_fills[i].load(atomic::Ordering::SeqCst),
                uf.load(atomic::Ordering::SeqCst),
                "{}",
                i
            );
        }
        assert_eq!(
            expected_resolved_unknowns,
            resolved_unknowns.load(atomic::Ordering::SeqCst)
        );
        assert_eq!(expected_solution, *solution.lock().unwrap());
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
