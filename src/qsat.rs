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
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{cl_ulong, CL_BLOCKING};

use std::collections::{BinaryHeap, HashMap};
use std::fmt::{Display, Formatter, Write};
use std::fs;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
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
    #[arg(help = "Circuit file")]
    circuit: String,
    #[arg(
        short = 'e',
        long,
        default_value_t = 24,
        help = "Power of two of number of elements executed at call"
    )]
    elem_inputs: usize,
    #[arg(
        short = 't',
        long,
        help = "Execution type: {cpu|opencl:{dev},cpu_and_openclXX}"
    )]
    exec_type: ExecType,
    #[arg(short = 'n', long, help = "Disable optimize_negs")]
    no_optimize_negs: bool,
    #[arg(short = 'G', long, help = "OpenCL group length")]
    opencl_group_len: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FinalResult {
    // reversed: if first quantifier is 'All' then is reversed (solution if not satisfiable)
    // otherwise solution is satisfiable.
    reversed: bool,
    solution_bits: usize,
    // only for first quantifier
    solution: Option<u128>,
}

impl FinalResult {
    fn join(self, second: Self) -> Self {
        assert_eq!(self.reversed, second.reversed);
        if let Some(self_sol) = self.solution {
            if let Some(second_sol) = second.solution {
                Self {
                    reversed: self.reversed,
                    solution_bits: self.solution_bits + second.solution_bits,
                    solution: Some(
                        (self_sol | (second_sol << self.solution_bits))
                            & ((1u128 << (self.solution_bits + second.solution_bits)) - 1),
                    ),
                }
            } else {
                panic!("Unexpected");
            }
        } else if second.solution.is_none() {
            Self {
                reversed: self.reversed,
                solution_bits: self.solution_bits + second.solution_bits,
                solution: None,
            }
        } else {
            panic!("Unexpected");
        }
    }

    fn set_solution_bits(self, sol_bits: usize) -> Self {
        Self {
            reversed: self.reversed,
            solution_bits: sol_bits,
            solution: self.solution,
        }
    }

    fn reverse(self) -> Self {
        Self {
            reversed: !self.reversed,
            solution_bits: self.solution_bits,
            solution: if self.solution.is_some() {
                None
            } else {
                Some(0)
            },
        }
    }
}

impl Display for FinalResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        if self.reversed {
            if let Some(sol) = self.solution {
                write!(f, "(Unsatisfiable: {1:00$b})", self.solution_bits, sol)
            } else {
                write!(f, "(Satisfiable)")
            }
        } else if let Some(sol) = self.solution {
            write!(f, "(Satisfiable: {1:00$b})", self.solution_bits, sol)
        } else {
            write!(f, "(Unsatisfiable)")
        }
    }
}

struct QuantReducer {
    // quants in boolean encoding: All=1, Exists=0
    quants: Vec<bool>, // reversed list of quantifiers (first is lowest)
    started: bool,
    start: u64,
    all_mask: u64,
    first_mask: u64, // for first quantifier
    other_mask: u64, // other than for first quantifier
    items: BinaryHeap<(std::cmp::Reverse<u64>, bool)>,
    result: Vec<bool>,
    solution: Option<u64>,
}

impl QuantReducer {
    fn new(quants: &[Quant]) -> Self {
        let first_quant = quants[0];
        // determine first quantifier length (bits)
        let first_quant_bits = quants.iter().take_while(|q| **q == first_quant).count();
        // to quants reversed and to bool:
        // initialize bits by rule: All=1, Exists=0 (and requires 1, or requires 0)
        let quants = quants
            .iter()
            .rev()
            .map(|q| *q == Quant::All)
            .collect::<Vec<_>>();
        let all_mask = u64::try_from((1u128 << quants.len()) - 1).unwrap();
        let first_mask = u64::try_from((1u128 << first_quant_bits) - 1).unwrap()
            << (quants.len() - first_quant_bits);
        Self {
            quants: quants.clone(),
            started: false,
            start: 0,
            all_mask,
            first_mask,
            other_mask: all_mask & !first_mask,
            items: BinaryHeap::new(),
            result: quants,
            solution: None,
        }
    }

    #[inline]
    fn is_end(&self) -> bool {
        self.started && (self.start & self.all_mask) == 0
    }

    fn push(&mut self, index: u64, item: bool) {
        assert!(!self.is_end());
        self.items.push((std::cmp::Reverse(index), item));
        self.flush();
    }

    fn start_prev(&self) -> u64 {
        self.start.overflowing_sub(1).0
    }

    // returns final result is
    fn flush(&mut self) {
        assert!(!self.is_end());
        while let Some((index, item)) = self.items.peek().copied() {
            if self.start == index.0 {
                let old_is_solution = self.solution.is_some();
                // if index is match then flush
                self.items.pop();
                self.apply(item);
                if !old_is_solution && self.solution.is_some() {
                    // VERY IMPORTANT for CPU quant reducer
                    // stop if solution found
                    break;
                }
            } else {
                break;
            }
        }
    }

    fn apply(&mut self, item: bool) {
        assert!(!self.is_end());
        self.started = true;
        let mut index = self.start;
        let mut prev = item;
        for (q, r) in self.quants.iter().zip(self.result.iter_mut()) {
            // if quants=true then use result & item, otherwise result | item.
            prev = (*r & prev) | ((*r ^ prev) & !q);
            *r = prev;
            if (index & 1) == 0 {
                break;
            }
            index >>= 1;
            *r = *q;
        }
        // if this all bits of other quantifiers are ones (last item in current value of first
        // quantifier) - then resolve solution -
        // result for first quantifier: all - last -> result=0, exists -> result=1
        if self.solution.is_none()
            && (self.start & self.other_mask) == self.other_mask
            && (prev ^ self.quants.last().unwrap())
        {
            let first_bits = self.first_mask.count_ones() as usize;
            // calculate solution in original order of bits.
            self.solution = Some(
                (self.start >> (self.quants.len() - first_bits)).reverse_bits()
                    >> (64 - first_bits),
            );
        }
        self.start = self.start.overflowing_add(1).0;
    }

    fn final_result(&self) -> Option<FinalResult> {
        if self.is_end() || self.solution.is_some() {
            Some(FinalResult {
                reversed: self.quants.last().copied().unwrap(),
                solution_bits: self.first_mask.count_ones() as usize,
                solution: self.solution.map(|x| x as u128),
            })
        } else {
            None
        }
    }
}

// CPU Code

const AGGR_OUTPUT_CPU_CODE: &str = r##"{
    // out format:
    // 0 .. (TYPE_LEN >> 5) - machine word that have solution
    // (TYPE_LEN >> 5) - final result bit
    // (TYPE_LEN >> 5) + 1 - non-zero if solution found
    // (TYPE_LEN >> 5) + 2 - low 32-bits of machine word index
    // (TYPE_LEN >> 5) + 3 - high 32-bits of machine word index
    uint32_t i;
    uint32_t temp[TYPE_LEN >> 5];
    uint32_t* out = (uint32_t*)output;
    uint32_t work_bit;
    uint32_t mod_idx = idx;
    GET_U32_ALL(temp, o0);
    for (i = 0; i < (TYPE_LEN >> 5); i++)
        temp[i] = temp[i] TYPE_QUANT_REDUCE_OP_0 (temp[i] >> 1);
    for (i = 0; i < (TYPE_LEN >> 5); i++)
        temp[i] = temp[i] TYPE_QUANT_REDUCE_OP_1 (temp[i] >> 2);
    for (i = 0; i < (TYPE_LEN >> 5); i++)
        temp[i] = temp[i] TYPE_QUANT_REDUCE_OP_2 (temp[i] >> 4);
    for (i = 0; i < (TYPE_LEN >> 5); i++)
        temp[i] = temp[i] TYPE_QUANT_REDUCE_OP_3 (temp[i] >> 8);
    for (i = 0; i < (TYPE_LEN >> 5); i++)
        temp[i] = temp[i] TYPE_QUANT_REDUCE_OP_4 (temp[i] >> 16);
    // continue reduction on machine word
#if TYPE_LEN > 32
    for (i = 0; i < (TYPE_LEN >> 5); i += 2) {
        temp[i] = temp[i] TYPE_QUANT_REDUCE_OP_5 temp[i + 1];
    }
#endif
#if TYPE_LEN > 64
    for (i = 0; i < (TYPE_LEN >> 5); i += 4) {
        temp[i] = temp[i] TYPE_QUANT_REDUCE_OP_6 temp[i + 2];
    }
#endif
#if TYPE_LEN > 128
    for (i = 0; i < (TYPE_LEN >> 5); i += 8) {
        temp[i] = temp[i] TYPE_QUANT_REDUCE_OP_7 temp[i + 4];
    }
#endif
#if TYPE_LEN > 256
    for (i = 0; i < (TYPE_LEN >> 5); i += 16) {
        temp[i] = temp[i] TYPE_QUANT_REDUCE_OP_8 temp[i + 8];
    }
#endif
#if TYPE_LEN > 512
    for (i = 0; i < (TYPE_LEN >> 5); i += 32) {
        temp[i] = temp[i] TYPE_QUANT_REDUCE_OP_9 temp[i + 16];
    }
#endif

    work_bit = (temp[0] & 1);

    if (idx == 0)
        out[(TYPE_LEN >> 5) + 1] = 0; // solution word

#ifdef WORK_QUANT_REDUCE_INIT_DATA
#define BASE (4 + (TYPE_LEN >> 5))
    if (idx == 0) {
        // initialization of work bits
        out[(TYPE_LEN >> 5)] = (WORK_QUANT_REDUCE_INIT_DATA >> (WORK_WORD_NUM_BITS - 1)) & 1;
        for (i = 0; i < WORK_WORD_NUM_BITS; i++) {
            out[BASE + i] = (WORK_QUANT_REDUCE_INIT_DATA >> i) & 1;
        }
    }
    // main loop to reduce single work
    for (i = 0; i < WORK_WORD_NUM_BITS; i++) {
        uint32_t r = out[BASE + i];
        work_bit = (r & work_bit) | ((r ^ work_bit) &
            (((~WORK_QUANT_REDUCE_INIT_DATA) >> i) & 1));
        out[BASE + i] = work_bit;
        if ((mod_idx & 1) == 0)
            break;
        mod_idx >>= 1;
        out[BASE + i] = ((WORK_QUANT_REDUCE_INIT_DATA >> i) & 1);
    }
    // finally write to work bit
    if ((work_bit ^ ((WORK_QUANT_REDUCE_INIT_DATA >> (WORK_WORD_NUM_BITS - 1)) & 1)) != 0) {
#ifdef WORK_HAVE_FIRST_QUANT
        if (out[(TYPE_LEN >> 5) + 1] == 0 && (idx & OTHER_MASK) == OTHER_MASK) {
            out[(TYPE_LEN >> 5) + 1] = 1;
            out[(TYPE_LEN >> 5) + 2] = idx & 0xffffffffU;
            out[(TYPE_LEN >> 5) + 3] = idx >> 32;
            GET_U32_ALL(out, o0);
            out[(TYPE_LEN >> 5)] = work_bit;
        }
#else
        if ((idx & ((1ULL << WORK_WORD_NUM_BITS) - 1ULL)) ==
                    ((1ULL << WORK_WORD_NUM_BITS) - 1ULL))
            out[(TYPE_LEN >> 5)] = work_bit;
#endif  // WORK_HAVE_FIRST_QUANT
    }
#undef BASE
#else // WORK_QUANT_REDUCE_INIT_DATA
    // if only one word to process - then copy
    GET_U32_ALL(out, o0);
    out[(TYPE_LEN >> 5)] = work_bit;
#ifdef WORK_HAVE_FIRST_QUANT
    if ((work_bit ^ TYPE_QUANT_REDUCE_QUANT_ALL) != 0) {
        out[(TYPE_LEN >> 5) + 1] = 1;
        out[(TYPE_LEN >> 5) + 2] = 0;
        out[(TYPE_LEN >> 5) + 3] = 0;
    }
#endif // WORK_HAVE_FIRST_QUANT
#endif // WORK_QUANT_REDUCE_INIT_DATA
}"##;

// aggr output code for OpenCL and OpenCL quant reducer code together
// QUANT_REDUCER - enables OpenCL Quant Reducer kernel code.
// major code is common.
const INIT_OPENCL_CODE: &str = "local uint local_results[GROUP_LEN];";
const AGGR_OUTPUT_OPENCL_CODE: &str = r##"
#ifdef QUANT_REDUCER

// OpenCL Quant Reducer kernel code
kernel void QUANT_REDUCER_NAME(unsigned long n, const global ushort* input, global ushort* output) {
    local uint local_results[GROUP_LEN];
    size_t idx = get_global_id(0);
    if (idx >= n) return;
    size_t lidx = get_local_id(0);
    global ushort* out = (global ushort*)output;
    uint work_bit;
    uint result1, result2;

    work_bit = input[idx] & 0x8000;

    // end of part of Quant Reducer kernel code
#else // QUANT_REDUCER

// normal aggegated output code for OpenCL
{
    uint i;
    size_t lidx = get_local_id(0);
    uint temp[TYPE_LEN >> 5];
    global ushort* out = (global ushort*)output;
    uint work_bit;
    uint result1, result2;
    GET_U32_ALL(temp, o0);
    for (i = 0; i < (TYPE_LEN >> 5); i++)
        temp[i] = temp[i] TYPE_QUANT_REDUCE_OP_0 (temp[i] >> 1);
    for (i = 0; i < (TYPE_LEN >> 5); i++)
        temp[i] = temp[i] TYPE_QUANT_REDUCE_OP_1 (temp[i] >> 2);
    for (i = 0; i < (TYPE_LEN >> 5); i++)
        temp[i] = temp[i] TYPE_QUANT_REDUCE_OP_2 (temp[i] >> 4);
    for (i = 0; i < (TYPE_LEN >> 5); i++)
        temp[i] = temp[i] TYPE_QUANT_REDUCE_OP_3 (temp[i] >> 8);
    for (i = 0; i < (TYPE_LEN >> 5); i++)
        temp[i] = temp[i] TYPE_QUANT_REDUCE_OP_4 (temp[i] >> 16);
    // continue reduction on machine word
#if TYPE_LEN > 32
    for (i = 0; i < (TYPE_LEN >> 5); i += 2) {
        temp[i] = temp[i] TYPE_QUANT_REDUCE_OP_5 temp[i + 1];
    }
#endif
#if TYPE_LEN > 64
    for (i = 0; i < (TYPE_LEN >> 5); i += 4) {
        temp[i] = temp[i] TYPE_QUANT_REDUCE_OP_6 temp[i + 2];
    }
#endif
#if TYPE_LEN > 128
    for (i = 0; i < (TYPE_LEN >> 5); i += 8) {
        temp[i] = temp[i] TYPE_QUANT_REDUCE_OP_7 temp[i + 4];
    }
#endif
#if TYPE_LEN > 256
    for (i = 0; i < (TYPE_LEN >> 5); i += 16) {
        temp[i] = temp[i] TYPE_QUANT_REDUCE_OP_8 temp[i + 8];
    }
#endif
#if TYPE_LEN > 512
    for (i = 0; i < (TYPE_LEN >> 5); i += 32) {
        temp[i] = temp[i] TYPE_QUANT_REDUCE_OP_9 temp[i + 16];
    }
#endif

    work_bit = (temp[0] & 1) << 15;

    // end of part of normal aggregated output code
#endif // not QUANT_REDUCER

///////////////////////////////////
// common part of code
///////////////////////////////////

// value on bits of local index (0-14 bits):
// 0-0x7ffe - local index if solution, 0x7fff - no local index if no solution
#if LOCAL_FIRST_QUANT_LEVEL == 0
    // routine performs before first quantifier bit - performs filtering of local index
    if (work_bit == LOCAL_FIRST_QUANT_PROPAGATE_CHECK)
        local_results[lidx] = work_bit | lidx;
    else
        local_results[lidx] = work_bit | 0x7fff;
#else
    // just copy result
    local_results[lidx] = work_bit;
#endif
    barrier(CLK_LOCAL_MEM_FENCE);

// routine performs before first quantifier bit - performs filtering of local index
#define LOCAL_BEFORE_FIRST_QUANT_LEVEL \
    work_bit = work_bit & 0x8000; \
    if (work_bit == LOCAL_FIRST_QUANT_PROPAGATE_CHECK) \
        local_results[lidx] = work_bit | lidx; \
    else \
        local_results[lidx] = work_bit | 0x7fff;

// peforms joining two results in first quantifier bits
#define LOCAL_QUANT_UPDATE_FIRST_QUANT \
        work_bit = work_bit & 0x8000; \
        if (work_bit == LOCAL_FIRST_QUANT_PROPAGATE_CHECK) \
            local_results[lidx] = work_bit | (min(result1 & 0x7fff, result2 & 0x7fff)); \
        else \
            local_results[lidx] = work_bit | 0x7fff;

// just join result bit without keeping local index becuase is unnecessary
#define LOCAL_QUANT_UPDATE \
        local_results[lidx] = work_bit;

    // it only check lidx + x n, because GROUP_LEN and same 'n' are power of two.
#if GROUP_LEN >= 2
    if (lidx + 1 < GROUP_LEN && lidx + 1 < n && (lidx & 1) == 0) {
        result1 = local_results[lidx];
        result2 = local_results[lidx + 1];
        work_bit = result1 LOCAL_QUANT_REDUCE_OP_0 result2;
#if LOCAL_FIRST_QUANT_LEVEL <= 0
        LOCAL_QUANT_UPDATE_FIRST_QUANT;
#else
#if LOCAL_FIRST_QUANT_LEVEL == 1
        LOCAL_BEFORE_FIRST_QUANT_LEVEL;
#else
        LOCAL_QUANT_UPDATE;
#endif
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if GROUP_LEN >= 4
    if (lidx + 2 < GROUP_LEN && lidx + 2 < n && (lidx & 3) == 0) {
        result1 = local_results[lidx];
        result2 = local_results[lidx + 2];
        work_bit = result1 LOCAL_QUANT_REDUCE_OP_1 result2;
#if LOCAL_FIRST_QUANT_LEVEL <= 1
        LOCAL_QUANT_UPDATE_FIRST_QUANT;
#else
#if LOCAL_FIRST_QUANT_LEVEL == 2
        LOCAL_BEFORE_FIRST_QUANT_LEVEL;
#else
        LOCAL_QUANT_UPDATE;
#endif
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if GROUP_LEN >= 8
    if (lidx + 4 < GROUP_LEN && lidx + 4 < n && (lidx & 7) == 0) {
        result1 = local_results[lidx];
        result2 = local_results[lidx + 4];
        work_bit = result1 LOCAL_QUANT_REDUCE_OP_2 result2;
#if LOCAL_FIRST_QUANT_LEVEL <= 2
        LOCAL_QUANT_UPDATE_FIRST_QUANT;
#else
#if LOCAL_FIRST_QUANT_LEVEL == 3
        LOCAL_BEFORE_FIRST_QUANT_LEVEL;
#else
        LOCAL_QUANT_UPDATE;
#endif
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if GROUP_LEN >= 16
    if (lidx + 8 < GROUP_LEN && lidx + 8 < n && (lidx & 15) == 0) {
        result1 = local_results[lidx];
        result2 = local_results[lidx + 8];
        work_bit = result1 LOCAL_QUANT_REDUCE_OP_3 result2;
#if LOCAL_FIRST_QUANT_LEVEL <= 3
        LOCAL_QUANT_UPDATE_FIRST_QUANT;
#else
#if LOCAL_FIRST_QUANT_LEVEL == 4
        LOCAL_BEFORE_FIRST_QUANT_LEVEL;
#else
        LOCAL_QUANT_UPDATE;
#endif
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if GROUP_LEN >= 32
    if (lidx + 16 < GROUP_LEN && lidx + 16 < n && (lidx & 31) == 0) {
        result1 = local_results[lidx];
        result2 = local_results[lidx + 16];
        work_bit = result1 LOCAL_QUANT_REDUCE_OP_4 result2;
#if LOCAL_FIRST_QUANT_LEVEL <= 4
        LOCAL_QUANT_UPDATE_FIRST_QUANT;
#else
#if LOCAL_FIRST_QUANT_LEVEL == 5
        LOCAL_BEFORE_FIRST_QUANT_LEVEL;
#else
        LOCAL_QUANT_UPDATE;
#endif
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if GROUP_LEN >= 64
    if (lidx + 32 < GROUP_LEN && lidx + 32 < n && (lidx & 63) == 0) {
        result1 = local_results[lidx];
        result2 = local_results[lidx + 32];
        work_bit = result1 LOCAL_QUANT_REDUCE_OP_5 result2;
#if LOCAL_FIRST_QUANT_LEVEL <= 5
        LOCAL_QUANT_UPDATE_FIRST_QUANT;
#else
#if LOCAL_FIRST_QUANT_LEVEL == 6
        LOCAL_BEFORE_FIRST_QUANT_LEVEL;
#else
        LOCAL_QUANT_UPDATE;
#endif
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if GROUP_LEN >= 128
    if (lidx + 64 < GROUP_LEN && lidx + 64 < n && (lidx & 127) == 0) {
        result1 = local_results[lidx];
        result2 = local_results[lidx + 64];
        work_bit = result1 LOCAL_QUANT_REDUCE_OP_6 result2;
#if LOCAL_FIRST_QUANT_LEVEL <= 6
        LOCAL_QUANT_UPDATE_FIRST_QUANT;
#else
#if LOCAL_FIRST_QUANT_LEVEL == 7
        LOCAL_BEFORE_FIRST_QUANT_LEVEL;
#else
        LOCAL_QUANT_UPDATE;
#endif
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if GROUP_LEN >= 256
    if (lidx + 128 < GROUP_LEN && lidx + 128 < n && (lidx & 255) == 0) {
        result1 = local_results[lidx];
        result2 = local_results[lidx + 128];
        work_bit = result1 LOCAL_QUANT_REDUCE_OP_7 result2;
#if LOCAL_FIRST_QUANT_LEVEL <= 7
        LOCAL_QUANT_UPDATE_FIRST_QUANT;
#else
#if LOCAL_FIRST_QUANT_LEVEL == 8
        LOCAL_BEFORE_FIRST_QUANT_LEVEL;
#else
        LOCAL_QUANT_UPDATE;
#endif
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if GROUP_LEN >= 512
    if (lidx + 256 < GROUP_LEN && lidx + 256 < n && (lidx & 511) == 0) {
        result1 = local_results[lidx];
        result2 = local_results[lidx + 256];
        work_bit = result1 LOCAL_QUANT_REDUCE_OP_8 result2;
#if LOCAL_FIRST_QUANT_LEVEL <= 8
        LOCAL_QUANT_UPDATE_FIRST_QUANT;
#else
#if LOCAL_FIRST_QUANT_LEVEL == 9
        LOCAL_BEFORE_FIRST_QUANT_LEVEL;
#else
        LOCAL_QUANT_UPDATE;
#endif
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if GROUP_LEN >= 1024
    if (lidx + 512 < GROUP_LEN && lidx + 512 < n && (lidx & 1023) == 0) {
        result1 = local_results[lidx];
        result2 = local_results[lidx + 512];
        work_bit = result1 LOCAL_QUANT_REDUCE_OP_9 result2;
#if LOCAL_FIRST_QUANT_LEVEL <= 9
        LOCAL_QUANT_UPDATE_FIRST_QUANT;
#else
#if LOCAL_FIRST_QUANT_LEVEL == 10
        LOCAL_BEFORE_FIRST_QUANT_LEVEL;
#else
        LOCAL_QUANT_UPDATE;
#endif
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if GROUP_LEN >= 2048
    if (lidx + 1024 < GROUP_LEN && lidx + 1024 < n && (lidx & 2047) == 0) {
        result1 = local_results[lidx];
        result2 = local_results[lidx + 1024];
        work_bit = result1 LOCAL_QUANT_REDUCE_OP_10 result2;
#if LOCAL_FIRST_QUANT_LEVEL <= 10
        LOCAL_QUANT_UPDATE_FIRST_QUANT;
#else
#if LOCAL_FIRST_QUANT_LEVEL == 11
        LOCAL_BEFORE_FIRST_QUANT_LEVEL;
#else
        LOCAL_QUANT_UPDATE;
#endif
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if GROUP_LEN >= 4096
    if (lidx + 2048 < GROUP_LEN && lidx + 2048 < n && (lidx & 4095) == 0) {
        result1 = local_results[lidx];
        result2 = local_results[lidx + 2048];
        work_bit = result1 LOCAL_QUANT_REDUCE_OP_11 result2;
#if LOCAL_FIRST_QUANT_LEVEL <= 11
        LOCAL_QUANT_UPDATE_FIRST_QUANT;
#else
        LOCAL_QUANT_UPDATE;
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    if (lidx == 0)
        out[idx >> GROUP_LEN_BITS] = local_results[0];
}
#undef LOCAL_BEFORE_FIRST_QUANT_LEVEL
#undef LOCAL_QUANT_UPDATE_FIRST_QUANT
#undef LOCAL_QUANT_UPDATE
"##;

fn get_aggr_output_cpu_code_defs(type_len: usize, elem_bits: usize, quants: &[Quant]) -> String {
    let first_quant = quants[0];
    // determine first quantifier length (bits)
    let first_quant_bits = quants.iter().take_while(|q| **q == first_quant).count();
    assert_eq!(type_len.count_ones(), 1);
    let type_len_bits = (usize::BITS - type_len.leading_zeros() - 1) as usize;
    assert!(elem_bits >= type_len_bits);
    let mut defs = String::new();
    let quants_len = quants.len();
    if elem_bits > type_len_bits {
        writeln!(
            defs,
            "#define WORK_QUANT_REDUCE_INIT_DATA ({}ULL)",
            quants[quants_len - elem_bits..quants_len - type_len_bits]
                .iter()
                .rev()
                .enumerate()
                .fold(0u64, |a, (b, q)| a | (u64::from(*q == Quant::All) << b))
        )
        .unwrap();
        if first_quant_bits > quants_len - elem_bits {
            let first_quant_bits = std::cmp::min(first_quant_bits, quants_len - type_len_bits);
            let rest_first_bits = first_quant_bits - (quants_len - elem_bits);
            writeln!(
                defs,
                "#define OTHER_MASK ({}ULL)",
                ((1u64 << (elem_bits - type_len_bits)) - 1)
                    & !(((1u64 << rest_first_bits) - 1)
                        << ((elem_bits - type_len_bits) - rest_first_bits))
            )
            .unwrap();
        }
    } else {
        writeln!(
            defs,
            "#define TYPE_QUANT_REDUCE_QUANT_ALL ({})",
            u64::from(quants[quants_len - type_len_bits] == Quant::All)
        )
        .unwrap();
    }
    for i in 0..type_len_bits {
        writeln!(
            defs,
            "#define TYPE_QUANT_REDUCE_OP_{} {}",
            i,
            match quants[quants_len - i - 1] {
                Quant::Exists => '|',
                Quant::All => '&',
            }
        )
        .unwrap();
    }
    writeln!(
        defs,
        "#define WORK_WORD_NUM_BITS ({})",
        elem_bits - type_len_bits
    )
    .unwrap();
    if first_quant_bits > quants_len - elem_bits {
        writeln!(defs, "#define WORK_HAVE_FIRST_QUANT").unwrap();
    }
    defs
}

fn get_aggr_output_opencl_code_defs(type_len: usize, group_len: usize, quants: &[Quant]) -> String {
    let first_quant = quants[0];
    assert_eq!(type_len.count_ones(), 1);
    // determine first quantifier length (bits)
    let first_quant_bits = quants.iter().take_while(|q| **q == first_quant).count();
    let type_len_bits = (usize::BITS - type_len.leading_zeros() - 1) as usize;
    assert_eq!(group_len.count_ones(), 1);
    let group_len_bits = (usize::BITS - group_len.leading_zeros() - 1) as usize;
    let quants_len = quants.len();
    let mut defs = String::new();
    writeln!(defs, "#define GROUP_LEN ({})", group_len).unwrap();
    writeln!(defs, "#define GROUP_LEN_BITS ({})", group_len_bits).unwrap();
    for i in 0..type_len_bits {
        writeln!(
            defs,
            "#define TYPE_QUANT_REDUCE_OP_{} {}",
            i,
            match quants[quants_len - i - 1] {
                Quant::Exists => '|',
                Quant::All => '&',
            }
        )
        .unwrap();
    }
    for i in 0..group_len_bits {
        writeln!(
            defs,
            "#define LOCAL_QUANT_REDUCE_OP_{} {}",
            i,
            match quants[quants_len - type_len_bits - i - 1] {
                Quant::Exists => '|',
                Quant::All => '&',
            }
        )
        .unwrap();
    }
    if quants_len - type_len_bits - group_len_bits < first_quant_bits {
        // if first quantifier in group_len quantifiers
        writeln!(
            defs,
            "#define LOCAL_FIRST_QUANT_LEVEL ({})",
            if quants_len - type_len_bits > first_quant_bits {
                quants_len - type_len_bits - first_quant_bits
            } else {
                0
            }
        )
        .unwrap();
        writeln!(
            defs,
            "#define LOCAL_FIRST_QUANT_PROPAGATE_CHECK ({})",
            match first_quant {
                Quant::Exists => 0x8000,
                Quant::All => 0,
            }
        )
        .unwrap();
    } else {
        writeln!(defs, "#define LOCAL_FIRST_QUANT_LEVEL (99999999)").unwrap();
    }
    defs
}

const QUANT_REDUCER_OPENCL_UNDEFS_CODE: &str = r##"
#undef GROUP_LEN
#undef GROUP_LEN_BITS
#undef LOCAL_QUANT_REDUCE_OP_0
#undef LOCAL_QUANT_REDUCE_OP_1
#undef LOCAL_QUANT_REDUCE_OP_2
#undef LOCAL_QUANT_REDUCE_OP_3
#undef LOCAL_QUANT_REDUCE_OP_4
#undef LOCAL_QUANT_REDUCE_OP_5
#undef LOCAL_QUANT_REDUCE_OP_6
#undef LOCAL_QUANT_REDUCE_OP_7
#undef LOCAL_QUANT_REDUCE_OP_8
#undef LOCAL_QUANT_REDUCE_OP_9
#undef LOCAL_QUANT_REDUCE_OP_10
#undef LOCAL_QUANT_REDUCE_OP_11
#undef LOCAL_FIRST_QUANT_LEVEL
#undef LOCAL_FIRST_QUANT_PROPAGATE_CHECK
#undef QUANT_REDUCER_NAME
"##;

//

fn get_final_results_from_cpu_outputs(
    type_len: usize,
    elem_bits: usize,
    quants: &[Quant],
    outputs: &[u32],
) -> (Option<FinalResult>, bool) {
    let type_len_bits = (usize::BITS - type_len.leading_zeros() - 1) as usize;
    let out_base = type_len >> 5;
    let quants_len = quants.len();
    let first_quant = *quants.first().unwrap();
    let first_quant_bits = quants.iter().take_while(|q| **q == first_quant).count();
    let work_first_bit = quants_len - elem_bits;
    let work_result = outputs[out_base] != 0;
    if outputs[out_base + 1] == 1 {
        // if solution found
        let first_quant_bits_in_work =
            std::cmp::min(first_quant_bits - work_first_bit, elem_bits - type_len_bits);
        let work_idx = (outputs[out_base + 2] as u64) | ((outputs[out_base + 3] as u64) << 32);
        let work_rev_idx = (work_idx.reverse_bits() >> (64 - (elem_bits - type_len_bits)))
            & ((1u64 << first_quant_bits_in_work) - 1);
        // join with values in type
        let final_rev_idx = if first_quant_bits > quants_len - type_len_bits {
            let mut qr = QuantReducer::new(&quants[quants_len - type_len_bits..]);
            for idx in 0..type_len {
                qr.push(idx as u64, ((outputs[idx >> 5] >> (idx & 31)) & 1) != 0);
            }
            let type_bit_rev_index =
                u64::try_from(qr.final_result().unwrap().solution.unwrap()).unwrap();
            work_rev_idx | (type_bit_rev_index << first_quant_bits_in_work)
        } else {
            work_rev_idx
        };
        (
            Some(FinalResult {
                solution_bits: first_quant_bits - work_first_bit,
                solution: Some(
                    (final_rev_idx as u128) & ((1u128 << (first_quant_bits - work_first_bit)) - 1),
                ),
                reversed: first_quant == Quant::All,
            }),
            work_result,
        )
    } else if first_quant_bits > work_first_bit {
        // if first quantifier in work
        (
            Some(FinalResult {
                solution_bits: first_quant_bits - work_first_bit,
                solution: None,
                reversed: first_quant == Quant::All,
            }),
            work_result,
        )
    } else {
        // no solution
        (None, work_result)
    }
}

fn adjust_opencl_group_len(group_len: usize) -> (usize, usize) {
    let group_len = std::cmp::min(group_len, 4096);
    let group_len_bits = (usize::BITS - group_len.leading_zeros() - 1) as usize;
    let group_len = if group_len.count_ones() != 1 {
        1usize << group_len_bits
    } else {
        group_len
    };
    (group_len, group_len_bits)
}

// OpenCLQuantReducer

struct OpenCLQuantReducer {
    cmd_queue: Arc<CommandQueue>,
    group_len: usize,
    group_len_bits: usize,
    kernels: Vec<Kernel>,
    input_len: usize,
    outputs: Vec<(Buffer<u16>, usize)>,
    reduce_start_bit: usize,
    // is some if first quantifiers after reduce_end_bit
    quants_after: Option<Vec<Quant>>,
    first_quant_bits: usize,
    // start bit position from which kernels starts reduction.
    quant_start_pos: usize,
    quants_start: Vec<Quant>,
    initial_input_group_len_bits: usize,
    is_first_quant_all: bool,
    have_first_quants: bool,
}

impl OpenCLQuantReducer {
    // form of quants:
    // before reduce_start_bit - reduction to do by argument level.
    // reduce_start_bit..reduce_end_bit - reduction done by this reducer.
    // reduce_start_bit..reduce_start_bit+quant_start_pos - reduction done by
    //     CPU in this reducer.
    // reduce_start_bit+quant_start_pos..reduce_end_bit - reduction done by these kernels.
    // reduce_end_bit..reduce_end_bit+initial_input_group_len_bits -
    //     reduction done by circuit kernel at local reduction.
    // reduce_end_bit+initial_input_group_len_bits..quants.len() - reduction done by
    //    circuit kernel at type reduction level.
    // reduce_start_bit..reduce_end_bit - bit to reduce by kernels
    // initial_input_group_len_bits - group length bits (reduction bits) from circuit kernel.
    fn new(
        reduce_start_bit: usize,
        reduce_end_bit: usize,
        initial_input_group_len_bits: usize,
        quants: &[Quant],
        context: Arc<Context>,
        cmd_queue: Arc<CommandQueue>,
        group_len: Option<usize>,
    ) -> Self {
        // println!("Start");
        assert!(1 <= reduce_start_bit);
        assert!(reduce_start_bit <= reduce_end_bit);
        assert_ne!(initial_input_group_len_bits, 0);
        assert!(reduce_end_bit + initial_input_group_len_bits <= quants.len());
        let is_first_quant_all = quants[0] == Quant::All;
        let first_quant = quants[0];
        let have_first_quants = quants[1..reduce_start_bit + 1]
            .iter()
            .all(|x| quants[0] == *x);
        let quants_after = quants[reduce_end_bit..].to_vec();
        let quants = &quants[reduce_start_bit..reduce_end_bit];
        let device = Device::new(context.devices()[0]);
        let group_len: usize =
            group_len.unwrap_or(usize::try_from(device.max_work_group_size().unwrap()).unwrap());
        let (group_len, group_len_bits) = adjust_opencl_group_len(group_len);
        // println!("GroupLen: {}", group_len);
        // println!("GroupLenBits: {}", group_len_bits);
        let quants_len = quants.len();
        // println!("QuantsLen: {}", quants_len);
        let kernel_num = quants_len / group_len_bits;
        let quant_start_pos = quants_len - kernel_num * group_len_bits;
        // println!("QuantStartPos: {}", quant_start_pos);
        let mut source_code = "#define QUANT_REDUCER 1\n".to_string();

        // determine first quantifier length (bits)
        let first_quant_bits = quants.iter().take_while(|q| **q == first_quant).count();
        let first_quant_bits = if !quants_after.is_empty() && quants_len == first_quant_bits {
            first_quant_bits
                + quants_after
                    .iter()
                    .take_while(|q| **q == first_quant)
                    .count()
        } else {
            first_quant_bits
        };
        for ki in 0..kernel_num {
            let quant_pos_end =
                std::cmp::min(quants_len, quant_start_pos + (ki + 1) * group_len_bits);
            // println!("QuantPosEnd: {} {}", ki, quant_pos_end);
            let defs = get_aggr_output_opencl_code_defs(1, group_len, &quants[0..quant_pos_end]);
            source_code += &format!("#define QUANT_REDUCER_NAME quant_reducer_{}\n", ki);
            source_code += &defs;
            source_code += "\n";
            source_code += AGGR_OUTPUT_OPENCL_CODE;
            source_code += "\n";
            source_code += QUANT_REDUCER_OPENCL_UNDEFS_CODE;
            source_code += "\n";
        }
        //println!("source: {}", source_code);
        let program = Program::create_and_build_from_source(&context, &source_code, "").unwrap();
        Self {
            cmd_queue: cmd_queue.clone(),
            group_len,
            group_len_bits,
            input_len: 1 << quants_len,
            kernels: (0..kernel_num)
                .map(|ki| Kernel::create(&program, &format!("quant_reducer_{}", ki)).unwrap())
                .collect::<Vec<_>>(),
            outputs: (0..kernel_num)
                .map(|ki| {
                    let output_len = 1usize << (quant_start_pos + ki * group_len_bits);
                    // println!("OutputLen: {} {}", ki, output_len);
                    (
                        unsafe {
                            Buffer::create(
                                &context,
                                CL_MEM_READ_WRITE,
                                output_len,
                                std::ptr::null_mut(),
                            )
                            .unwrap()
                        },
                        output_len,
                    )
                })
                .collect::<Vec<_>>(),
            reduce_start_bit,
            quants_after: if quants_len < first_quant_bits {
                Some(quants_after)
            } else {
                None
            },
            first_quant_bits,
            quant_start_pos,
            quants_start: quants[0..quant_start_pos].to_vec(),
            initial_input_group_len_bits,
            is_first_quant_all,
            have_first_quants,
        }
    }

    fn execute(&mut self, input: &Buffer<u32>) -> (Option<FinalResult>, bool) {
        let mut input_len = self.input_len;
        let mut next_input_buf = None;
        // call kernels
        for (kernel, (output, output_len)) in
            self.kernels.iter_mut().zip(self.outputs.iter_mut()).rev()
        {
            let cl_num = cl_ulong::try_from(input_len).unwrap();
            unsafe {
                if let Some(next_input) = next_input_buf {
                    ExecuteKernel::new(kernel)
                        .set_arg(&cl_num)
                        .set_arg(next_input)
                        .set_arg(output)
                        .set_local_work_size(self.group_len)
                        .set_global_work_size(
                            ((input_len + self.group_len - 1) / self.group_len) * self.group_len,
                        )
                        .enqueue_nd_range(&self.cmd_queue)
                        .unwrap();
                } else {
                    ExecuteKernel::new(kernel)
                        .set_arg(&cl_num)
                        .set_arg(input)
                        .set_arg(output)
                        .set_local_work_size(self.group_len)
                        .set_global_work_size(
                            ((input_len + self.group_len - 1) / self.group_len) * self.group_len,
                        )
                        .enqueue_nd_range(&self.cmd_queue)
                        .unwrap();
                }
            }
            input_len = *output_len;
            next_input_buf = Some(output);
        }
        self.cmd_queue.finish().unwrap();
        // retrieve results
        let last_output = if !self.outputs.is_empty() {
            // load from last output
            let mut last_output = vec![0u16; self.outputs[0].1];
            unsafe {
                self.cmd_queue
                    .enqueue_read_buffer(&self.outputs[0].0, CL_BLOCKING, 0, &mut last_output, &[])
                    .unwrap();
            }
            last_output
        } else {
            // load from input
            let mut last_output_32 = vec![0u32; (self.input_len + 1) >> 1];
            unsafe {
                self.cmd_queue
                    .enqueue_read_buffer(input, CL_BLOCKING, 0, &mut last_output_32, &[])
                    .unwrap();
            }
            // and convert to u16
            let mut last_output = vec![0u16; self.input_len];
            for (i, v) in last_output.iter_mut().enumerate() {
                *v = ((last_output_32[i >> 1] >> ((i & 1) << 4)) & 0xffff) as u16;
            }
            last_output
        };
        let quants_start_final_result = if self.quant_start_pos != 0 {
            // determine results by quantifier's reduction on CPU
            let mut qr = QuantReducer::new(&self.quants_start);
            for i in 0..1 << self.quant_start_pos {
                qr.push(i, (last_output[i as usize] >> 15) != 0);
            }
            let r = qr.final_result().unwrap();
            if r.reversed != self.is_first_quant_all {
                r.reverse()
            } else {
                r
            }
        } else {
            // generate final result with solution
            FinalResult {
                reversed: self.is_first_quant_all,
                solution_bits: 0,
                solution: Some(0),
            }
        };
        let result = if self.quant_start_pos != 0 {
            // if some first processed by CPU (QuantReducer)
            quants_start_final_result.solution.is_some() ^ self.is_first_quant_all
        } else {
            // get from last buffer
            ((last_output[0] >> 15) & 1) != 0
        };
        if !self.have_first_quants || !(self.is_first_quant_all ^ result) {
            // if no solution found
            return (None, result);
        }
        // solution found
        let first_quant_bits_in_reducer_and_inital_input = std::cmp::min(
            self.first_quant_bits,
            self.quant_start_pos
                + self.group_len_bits * self.kernels.len()
                + self.initial_input_group_len_bits,
        );
        if let Some(sol) = quants_start_final_result.solution {
            let mut new_sol = sol;
            let idx = if !self.kernels.is_empty() {
                // go get deeper first quant results
                let mut idx = usize::try_from(
                    (sol.reverse_bits()) >> (128 - quants_start_final_result.solution_bits),
                )
                .unwrap();
                // read last buffer
                // next buffer ....
                let output_num = std::cmp::min(
                    (self.first_quant_bits - self.quant_start_pos + self.group_len_bits - 1)
                        / self.group_len_bits,
                    self.outputs.len(),
                );
                for (oi, (buffer, _)) in self.outputs[0..output_num].iter().enumerate() {
                    let pass_bits = std::cmp::min(
                        self.first_quant_bits - oi * self.group_len_bits - self.quant_start_pos,
                        self.group_len_bits,
                    );
                    let mut buf_out = [0u16];
                    unsafe {
                        self.cmd_queue
                            .enqueue_read_buffer(&buffer, CL_BLOCKING, 2 * idx, &mut buf_out, &[])
                            .unwrap();
                    }
                    idx = ((buf_out[0] & 0x7fff) as usize) | (idx << self.group_len_bits);

                    let rev_idx = ((idx & 0x7fff).reverse_bits()
                        >> ((usize::BITS as usize) - self.group_len_bits))
                        & ((1usize << pass_bits) - 1);
                    if (idx & 0x7fff) != 0x7fff {
                        // update new sol
                        new_sol |=
                            (rev_idx as u128) << (self.quant_start_pos + oi * self.group_len_bits);
                    } else {
                        panic!("Unexpected");
                    }
                }
                idx
            } else {
                usize::try_from(
                    (sol.reverse_bits()) >> (128 - quants_start_final_result.solution_bits),
                )
                .unwrap()
            };
            if self.quants_after.is_some() {
                // go get deeper to input data
                let mut input_out = [0u32];
                unsafe {
                    self.cmd_queue
                        .enqueue_read_buffer(
                            input,
                            CL_BLOCKING,
                            4 * (idx >> 1),
                            &mut input_out,
                            &[],
                        )
                        .unwrap();
                }
                let idx = ((input_out[0] >> ((idx & 1) << 4)) & 0x7fff) as u16;
                let rev_idx = idx.reverse_bits() >> (16 - self.initial_input_group_len_bits);
                if idx != 0x7fff {
                    new_sol |= (rev_idx as u128)
                        << (self.quant_start_pos + self.outputs.len() * self.group_len_bits);
                } else {
                    panic!("Unexpected");
                }
            }
            (
                Some(FinalResult {
                    reversed: self.is_first_quant_all,
                    solution_bits: first_quant_bits_in_reducer_and_inital_input,
                    solution: Some(
                        new_sol & ((1u128 << first_quant_bits_in_reducer_and_inital_input) - 1),
                    ),
                }),
                !self.is_first_quant_all,
            )
        } else {
            panic!("Unexpected");
        }
    }

    fn final_result_with_circuit(
        &self,
        circuit: &Circuit<usize>,
        result: FinalResult,
    ) -> FinalResult {
        let max_sol_bits_with_init_group_len = self.quant_start_pos
            + self.group_len_bits * self.kernels.len()
            + self.initial_input_group_len_bits;
        if self.first_quant_bits > max_sol_bits_with_init_group_len
            && result.solution_bits == max_sol_bits_with_init_group_len + self.reduce_start_bit
        {
            // if circuit calculation needed
            if let Some(sol) = result.solution {
                let quants_after = self.quants_after.as_ref().unwrap();
                // println!("QuantsAfter: {:?}", quants_after);
                let mut qr = QuantReducer::new(&quants_after[self.initial_input_group_len_bits..]);
                let circuit_result_num =
                    1 << (quants_after.len() - self.initial_input_group_len_bits);
                // println!("circuit_result_num: {}", circuit_result_num);
                let mut circuit_results = vec![0u32; circuit_result_num >> 5];
                let circuit_results_word_bits =
                    quants_after.len() - self.initial_input_group_len_bits - 5;
                // println!("circuit_results_word_bits: {}", circuit_results_word_bits);
                for (i, v) in circuit_results.iter_mut().enumerate() {
                    // generate inputs for circuit
                    let inputs = (0..result.solution_bits)
                        .map(|b| (((sol >> b) & 1) as u32) * 0xffffffffu32)
                        .chain(
                            (0..circuit_results_word_bits)
                                .rev()
                                .map(|b| (((i >> b) & 1) as u32) * 0xffffffff),
                        )
                        .chain(
                            [0xffff0000, 0xff00ff00, 0xf0f0f0f0, 0xcccccccc, 0xaaaaaaaa]
                                .into_iter(),
                        )
                        .collect::<Vec<_>>();
                    // store circuit results
                    *v = circuit.eval(inputs)[0];
                }
                // evalute on QuantReducer
                for i in 0..1 << (quants_after.len() - self.initial_input_group_len_bits) {
                    qr.push(
                        i,
                        ((circuit_results[(i >> 5) as usize] >> (i & 31)) & 1) != 0,
                    );
                }
                result.join(qr.final_result().unwrap())
            } else {
                result
            }
        } else {
            result
        }
    }
}

//
// main quant reducers with arg reduction
//

struct MainCPUQuantReducer {
    type_len: usize,
    elem_bits: usize,
    qr: QuantReducer,
    quants: Vec<Quant>,
    work_results: HashMap<u64, FinalResult>,
    found_result: Option<FinalResult>,
    join_work_results: bool,
    solution_bits: usize,
}

impl MainCPUQuantReducer {
    fn new(elem_bits: usize, type_len: usize, quants: &[Quant]) -> Self {
        let first_quant = quants[0];
        let first_quant_bits = quants.iter().take_while(|q| **q == first_quant).count();
        assert_eq!(type_len.count_ones(), 1);
        Self {
            elem_bits,
            type_len,
            qr: QuantReducer::new(&quants[0..quants.len() - elem_bits]),
            quants: quants.to_vec(),
            work_results: HashMap::new(),
            found_result: None,
            join_work_results: first_quant_bits > quants.len() - elem_bits,
            solution_bits: first_quant_bits,
        }
    }

    fn eval(&mut self, arg: u64, outputs: &[u32]) -> Option<FinalResult> {
        if let Some(final_result) = self.found_result {
            return Some(final_result);
        }
        let (work_result, result) = get_final_results_from_cpu_outputs(
            self.type_len,
            self.elem_bits,
            &self.quants,
            outputs,
        );
        // put to work_results
        if let Some(work_result) = work_result {
            if work_result.solution.is_some() {
                self.work_results.insert(arg, work_result);
            }
        }
        self.qr.push(arg, result);
        let old_arg = self.qr.start_prev();
        if let Some(final_result) = self.qr.final_result() {
            // get correct work result from collection of previous work_results
            self.found_result = if let Some(old_work_result) = self.work_results.get(&old_arg) {
                if self.join_work_results {
                    Some(final_result.join(*old_work_result))
                } else {
                    Some(final_result.set_solution_bits(self.solution_bits))
                }
            } else {
                Some(final_result.set_solution_bits(self.solution_bits))
            };
            self.found_result
        } else {
            None
        }
    }
}

struct MainOpenCLQuantReducer {
    qr: QuantReducer,
    ocl_qr: OpenCLQuantReducer,
    found_result: Option<FinalResult>,
    join_work_results: bool,
    solution_bits: usize,
}

impl MainOpenCLQuantReducer {
    fn new(
        elem_bits: usize,
        type_len: usize,
        group_len: usize,
        quants: &[Quant],
        context: Arc<Context>,
        cmd_queue: Arc<CommandQueue>,
    ) -> Self {
        assert_eq!(type_len.count_ones(), 1);
        assert_eq!(group_len.count_ones(), 1);
        let type_len_bits = (usize::BITS - type_len.leading_zeros() - 1) as usize;
        let group_len_bits = (usize::BITS - group_len.leading_zeros() - 1) as usize;
        let first_quant = quants[0];
        let first_quant_bits = quants.iter().take_while(|q| **q == first_quant).count();
        Self {
            qr: QuantReducer::new(&quants[0..quants.len() - elem_bits]),
            ocl_qr: OpenCLQuantReducer::new(
                quants.len() - elem_bits,
                quants.len() - type_len_bits - group_len_bits,
                group_len_bits,
                quants,
                context.clone(),
                cmd_queue.clone(),
                Some(group_len),
            ),
            found_result: None,
            join_work_results: first_quant_bits > quants.len() - elem_bits,
            solution_bits: first_quant_bits,
        }
    }

    fn eval(
        &mut self,
        arg: u64,
        outputs: &Buffer<u32>,
        circuit: &Circuit<usize>,
    ) -> Option<FinalResult> {
        if let Some(final_result) = self.found_result {
            return Some(final_result);
        }
        let (work_result, result) = self.ocl_qr.execute(outputs);
        self.qr.push(arg, result);
        if let Some(final_result) = self.qr.final_result() {
            self.found_result = if let Some(work_result) = work_result {
                if self.join_work_results {
                    let work_result = final_result.join(work_result);
                    Some(self.ocl_qr.final_result_with_circuit(circuit, work_result))
                } else {
                    Some(final_result.set_solution_bits(self.solution_bits))
                }
            } else {
                Some(final_result.set_solution_bits(self.solution_bits))
            };
            self.found_result
        } else {
            None
        }
    }
}

struct MainCPUOpenCLQuantReducer {
    cpu_type_len: usize,
    elem_bits: usize,
    qr: QuantReducer,
    ocl_qrs: HashMap<usize, OpenCLQuantReducer>,
    quants: Vec<Quant>,
    work_results: HashMap<u64, (FinalResult, Option<usize>)>,
    found_result: Option<FinalResult>,
    join_work_results: bool,
    solution_bits: usize,
}

impl MainCPUOpenCLQuantReducer {
    fn new(
        elem_bits: usize,
        cpu_type_len: usize,
        quants: &[Quant],
        opencl_type_lens: &[usize],
        opencl_group_lens: &[usize],
        opencl_contexts: &[Arc<Context>],
        opencl_cmd_queues: &[Arc<CommandQueue>],
    ) -> Self {
        assert_eq!(cpu_type_len.count_ones(), 1);
        assert_eq!(opencl_type_lens.len(), opencl_group_lens.len());
        assert_eq!(opencl_type_lens.len(), opencl_contexts.len());
        assert_eq!(opencl_type_lens.len(), opencl_cmd_queues.len());
        let first_quant = quants[0];
        let first_quant_bits = quants.iter().take_while(|q| **q == first_quant).count();
        Self {
            elem_bits,
            cpu_type_len,
            qr: QuantReducer::new(&quants[0..quants.len() - elem_bits]),
            ocl_qrs: HashMap::from_iter((0..opencl_type_lens.len()).map(|i| {
                assert_eq!(opencl_type_lens[i].count_ones(), 1);
                assert_eq!(opencl_group_lens[i].count_ones(), 1);
                let opencl_type_len_bits =
                    (usize::BITS - opencl_type_lens[i].leading_zeros() - 1) as usize;
                let opencl_group_len_bits =
                    (usize::BITS - opencl_group_lens[i].leading_zeros() - 1) as usize;
                (
                    i,
                    OpenCLQuantReducer::new(
                        quants.len() - elem_bits,
                        quants.len() - opencl_type_len_bits - opencl_group_len_bits,
                        opencl_group_len_bits,
                        quants,
                        opencl_contexts[i].clone(),
                        opencl_cmd_queues[i].clone(),
                        Some(opencl_group_lens[i]),
                    ),
                )
            })),
            quants: quants.to_vec(),
            work_results: HashMap::new(),
            found_result: None,
            join_work_results: first_quant_bits > quants.len() - elem_bits,
            solution_bits: first_quant_bits,
        }
    }

    fn eval_end(
        &mut self,
        arg: u64,
        circuit: &Circuit<usize>,
        result: bool,
    ) -> Option<FinalResult> {
        self.qr.push(arg, result);
        let old_arg = self.qr.start_prev();
        if let Some(final_result) = self.qr.final_result() {
            // get correct work result from collection of previous work_results
            self.found_result = if let Some((old_work_result, dev_id)) =
                self.work_results.get(&old_arg)
            {
                if self.join_work_results {
                    if let Some(dev_id) = dev_id {
                        let work_result = final_result.join(*old_work_result);
                        // finalize by joining last bits of results from circuit values
                        // with this result.
                        Some(self.ocl_qrs[&dev_id].final_result_with_circuit(circuit, work_result))
                    } else {
                        Some(final_result.join(*old_work_result))
                    }
                } else {
                    Some(final_result.set_solution_bits(self.solution_bits))
                }
            } else {
                Some(final_result.set_solution_bits(self.solution_bits))
            };
            self.found_result
        } else {
            None
        }
    }

    fn eval_cpu(
        &mut self,
        arg: u64,
        outputs: &[u32],
        circuit: &Circuit<usize>,
    ) -> Option<FinalResult> {
        if let Some(final_result) = self.found_result {
            return Some(final_result);
        }
        let (work_result, result) = get_final_results_from_cpu_outputs(
            self.cpu_type_len,
            self.elem_bits,
            &self.quants,
            outputs,
        );
        // put to work_results
        if let Some(work_result) = work_result {
            if work_result.solution.is_some() {
                self.work_results.insert(arg, (work_result, None));
            }
        }
        self.eval_end(arg, circuit, result)
    }

    fn eval_opencl(
        &mut self,
        dev_id: usize,
        arg: u64,
        outputs: &Buffer<u32>,
        circuit: &Circuit<usize>,
    ) -> Option<FinalResult> {
        if let Some(final_result) = self.found_result {
            return Some(final_result);
        }
        let (work_result, result) = self.ocl_qrs.get_mut(&dev_id).unwrap().execute(outputs);
        // put to work_results
        if let Some(work_result) = work_result {
            if work_result.solution.is_some() {
                self.work_results.insert(arg, (work_result, Some(dev_id)));
            }
        }
        self.eval_end(arg, circuit, result)
    }
}

fn do_command_with_par_mapper<'a>(
    mut mapper: CPUParBasicMapperBuilder<'a>,
    qcircuit: QuantCircuit<usize>,
    elem_inputs: usize,
) -> FinalResult {
    let circuit = qcircuit.circuit();
    let input_len = circuit.input_len();
    let arg_steps = 1u128 << (input_len - elem_inputs);
    let type_len = mapper.type_len() as usize;
    assert_eq!(type_len.count_ones(), 1);
    let quants = qcircuit.quants();
    mapper.user_defs(&get_aggr_output_cpu_code_defs(
        type_len,
        elem_inputs,
        quants,
    ));
    mapper.add_with_config(
        "formula",
        circuit.clone(),
        CodeConfig::new()
            .elem_inputs(Some(
                &(input_len - elem_inputs..input_len)
                    .rev()
                    .collect::<Vec<usize>>(),
            ))
            .arg_inputs(Some(
                &(0..input_len - elem_inputs).rev().collect::<Vec<usize>>(),
            ))
            .aggr_output_code(Some(AGGR_OUTPUT_CPU_CODE))
            .aggr_output_len(Some((type_len >> 5) + 4 + 2 * elem_inputs)),
    );
    let mut execs = mapper.build().unwrap();
    let main_qr = Mutex::new(MainCPUQuantReducer::new(elem_inputs, type_len, quants));
    let input = execs[0].new_data(16);
    println!("Start execution");
    let start = SystemTime::now();
    let result = execs[0]
        .execute_direct(
            &input,
            None,
            |_, output, arg| {
                println!("Step: {} / {}", arg, arg_steps);
                main_qr.lock().unwrap().eval(arg, output)
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
    result.unwrap()
}

fn do_command_with_opencl_mapper<'a>(
    mut mapper: OpenCLBasicMapperBuilder<'a>,
    qcircuit: QuantCircuit<usize>,
    elem_inputs: usize,
    group_len: usize,
) -> FinalResult {
    let circuit = qcircuit.circuit();
    assert_eq!(group_len.count_ones(), 1);
    let input_len = circuit.input_len();
    let arg_steps = 1u128 << (input_len - elem_inputs);
    let type_len = mapper.type_len() as usize;
    assert_eq!(type_len.count_ones(), 1);
    let quants = qcircuit.quants();
    mapper.user_defs(&get_aggr_output_opencl_code_defs(
        type_len, group_len, quants,
    ));
    mapper.add_with_config(
        "formula",
        circuit.clone(),
        CodeConfig::new()
            .elem_inputs(Some(
                &(input_len - elem_inputs..input_len)
                    .rev()
                    .collect::<Vec<usize>>(),
            ))
            .arg_inputs(Some(
                &(0..input_len - elem_inputs).rev().collect::<Vec<usize>>(),
            ))
            .init_code(Some(INIT_OPENCL_CODE))
            .aggr_output_code(Some(AGGR_OUTPUT_OPENCL_CODE))
            .aggr_output_len(Some((1 << elem_inputs) / (group_len * type_len * 2))),
    );
    let mut execs = mapper.build().unwrap();
    let input = execs[0].new_data(16);
    let mut main_qr = MainOpenCLQuantReducer::new(
        elem_inputs,
        type_len,
        group_len,
        quants,
        unsafe { execs[0].executor().context() },
        unsafe { execs[0].executor().command_queue() },
    );
    println!("Start execution");
    let start = SystemTime::now();
    let result = execs[0]
        .execute(
            &input,
            None,
            |result, _, output, arg| {
                println!("Step: {} / {}", arg, arg_steps);
                if let Some(result) = result {
                    Some(result)
                } else {
                    main_qr.eval(arg, unsafe { output.buffer() }, circuit)
                }
            },
            |a| a.is_some(),
        )
        .unwrap();
    let time = start.elapsed().unwrap();
    println!("Time: {}", time.as_secs_f64());
    result.unwrap()
}

fn do_command_with_parseq_mapper<'a>(
    mut mapper: CPUOpenCLParSeqMapperBuilder<'a>,
    qcircuit: QuantCircuit<usize>,
    elem_inputs: usize,
    group_lens: &[usize],
) -> FinalResult {
    let circuit = qcircuit.circuit();
    let input_len = circuit.input_len();
    let arg_steps = 1u128 << (input_len - elem_inputs);
    let quants = qcircuit.quants();
    let cpu_type_len = mapper.type_len(ParSeqSelection::Par) as usize;
    assert_eq!(cpu_type_len.count_ones(), 1);
    let seq_builder_num = mapper.seq_builder_num();
    let opencl_type_lens = (0..seq_builder_num)
        .map(|i| {
            let ocl_type_len = mapper.type_len(ParSeqSelection::Seq(i)) as usize;
            assert_eq!(ocl_type_len.count_ones(), 1);
            ocl_type_len
        })
        .collect::<Vec<_>>();
    let cpu_user_def = get_aggr_output_cpu_code_defs(cpu_type_len, elem_inputs, quants);
    let ocl_user_defs = (0..seq_builder_num)
        .map(|i| get_aggr_output_opencl_code_defs(opencl_type_lens[i], group_lens[i], quants))
        .collect::<Vec<_>>();
    mapper.user_defs(|sel| match sel {
        ParSeqSelection::Par => &cpu_user_def,
        ParSeqSelection::Seq(i) => &ocl_user_defs[i],
    });
    mapper.add_with_config(
        "formula",
        circuit.clone(),
        &&(0..input_len - elem_inputs).rev().collect::<Vec<usize>>(),
        Some(
            &(input_len - elem_inputs..input_len)
                .rev()
                .collect::<Vec<usize>>(),
        ),
        |sel| match sel {
            ParSeqSelection::Par => ParSeqDynamicConfig::new()
                .aggr_output_code(Some(AGGR_OUTPUT_CPU_CODE))
                .aggr_output_len(Some((cpu_type_len >> 5) + 4 + 2 * elem_inputs)),
            ParSeqSelection::Seq(i) => ParSeqDynamicConfig::new()
                .init_code(Some(INIT_OPENCL_CODE))
                .aggr_output_code(Some(AGGR_OUTPUT_OPENCL_CODE))
                .aggr_output_len(Some(
                    (1 << elem_inputs) / (group_lens[i] * opencl_type_lens[i] * 2),
                )),
        },
    );
    let mut execs = mapper.build().unwrap();
    let main_qr = {
        let mut opencl_contexts = vec![None; seq_builder_num];
        let mut opencl_cmd_queues = vec![None; seq_builder_num];
        execs[0].with_executor(|exec| match exec {
            ParSeqObject::Seq((i, exec)) => {
                opencl_contexts[i] = unsafe { Some(exec.context()) };
                opencl_cmd_queues[i] = unsafe { Some(exec.command_queue()) };
            }
            _ => (),
        });
        let opencl_contexts = opencl_contexts
            .into_iter()
            .map(|c| c.unwrap())
            .collect::<Vec<_>>();
        let opencl_cmd_queues = opencl_cmd_queues
            .into_iter()
            .map(|c| c.unwrap())
            .collect::<Vec<_>>();
        Mutex::new(MainCPUOpenCLQuantReducer::new(
            elem_inputs,
            cpu_type_len,
            quants,
            &opencl_type_lens,
            group_lens,
            &opencl_contexts,
            &opencl_cmd_queues,
        ))
    };
    let input = execs[0].new_data(16);
    println!("Start execution");
    let start = SystemTime::now();
    let result = execs[0]
        .execute(
            &input,
            None,
            |sel, _, output, arg| {
                println!("Step: {} / {}", arg, arg_steps);
                match sel {
                    ParSeqSelection::Par => main_qr.lock().unwrap().eval_cpu(
                        arg,
                        output.par().unwrap().get().get(),
                        circuit,
                    ),
                    ParSeqSelection::Seq(i) => main_qr.lock().unwrap().eval_opencl(
                        i,
                        arg,
                        unsafe { output.seq().unwrap().buffer() },
                        circuit,
                    ),
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
    result.unwrap()
}

fn do_command(qcircuit: QuantCircuit<usize>, cmd_args: CommandArgs) {
    let circuit = qcircuit.circuit();
    let input_len = circuit.input_len();
    let result = if input_len >= 14 {
        let elem_inputs = if cmd_args.elem_inputs >= input_len {
            input_len - 1
        } else {
            cmd_args.elem_inputs
        };
        assert!(elem_inputs > 12 && elem_inputs <= 64);
        assert!(input_len - elem_inputs > 0 && input_len - elem_inputs <= 64);
        assert_eq!(circuit.outputs().len(), 1);
        println!("Elem inputs: {}", elem_inputs);

        let exec_type = cmd_args.exec_type;
        match exec_type {
            ExecType::CPU => {
                println!("Execute in CPU");
                let builder = ParBasicMapperBuilder::new(CPUBuilder::new(Some(
                    CPU_BUILDER_CONFIG_DEFAULT.optimize_negs(!cmd_args.no_optimize_negs),
                )));
                do_command_with_par_mapper(builder, qcircuit.clone(), elem_inputs)
            }
            ExecType::OpenCL(didx) => {
                println!("Execute in OpenCL device={}", didx);
                let device = Device::new(
                    *get_all_devices(CL_DEVICE_TYPE_GPU)
                        .unwrap()
                        .get(didx)
                        .unwrap(),
                );
                let group_len = cmd_args
                    .opencl_group_len
                    .unwrap_or(usize::try_from(device.max_work_group_size().unwrap()).unwrap());
                let (group_len, _) = adjust_opencl_group_len(group_len);
                println!("GroupLen: {}", group_len);
                let opencl_config = OPENCL_BUILDER_CONFIG_DEFAULT
                    .group_len(cmd_args.opencl_group_len)
                    .optimize_negs(!cmd_args.no_optimize_negs);
                let builder =
                    BasicMapperBuilder::new(OpenCLBuilder::new(&device, Some(opencl_config)));
                do_command_with_opencl_mapper(builder, qcircuit.clone(), elem_inputs, group_len)
            }
            ExecType::CPUAndOpenCL
            | ExecType::CPUAndOpenCLD
            | ExecType::CPUAndOpenCL1(_)
            | ExecType::CPUAndOpenCL1D(_) => {
                let par_builder = CPUBuilder::new(Some(
                    CPU_BUILDER_CONFIG_DEFAULT.optimize_negs(!cmd_args.no_optimize_negs),
                ));
                let seq_builders_and_group_lens = if let ExecType::CPUAndOpenCL1(didx) = exec_type {
                    println!("Execute in CPUAndOpenCL1");
                    get_all_devices(CL_DEVICE_TYPE_GPU).unwrap()[didx..=didx]
                        .into_iter()
                        .map(|dev_id| {
                            let device = Device::new(dev_id.clone());
                            let group_len = cmd_args.opencl_group_len.unwrap_or(
                                usize::try_from(device.max_work_group_size().unwrap()).unwrap(),
                            );
                            let (group_len, _) = adjust_opencl_group_len(group_len);
                            println!("GroupLen for {:?}: {}", dev_id, group_len);
                            let opencl_config = OPENCL_BUILDER_CONFIG_DEFAULT
                                .group_len(Some(group_len))
                                .optimize_negs(!cmd_args.no_optimize_negs);
                            (OpenCLBuilder::new(&device, Some(opencl_config)), group_len)
                        })
                        .collect::<Vec<_>>()
                } else if let ExecType::CPUAndOpenCL1D(didx) = exec_type {
                    println!("Execute in CPUAndOpenCL1D");
                    get_all_devices(CL_DEVICE_TYPE_GPU).unwrap()[didx..=didx]
                        .into_iter()
                        .map(|dev_id| {
                            let device = Device::new(dev_id.clone());
                            let group_len = cmd_args.opencl_group_len.unwrap_or(
                                usize::try_from(device.max_work_group_size().unwrap()).unwrap(),
                            );
                            let (group_len, _) = adjust_opencl_group_len(group_len);
                            println!("GroupLen for {:?}: {}", dev_id, group_len);
                            let opencl_config = OPENCL_BUILDER_CONFIG_DEFAULT
                                .group_len(Some(group_len))
                                .optimize_negs(!cmd_args.no_optimize_negs);
                            [
                                (
                                    OpenCLBuilder::new(&device, Some(opencl_config.clone())),
                                    group_len,
                                ),
                                (
                                    OpenCLBuilder::new(&device, Some(opencl_config.clone())),
                                    group_len,
                                ),
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
                            let group_len = cmd_args.opencl_group_len.unwrap_or(
                                usize::try_from(device.max_work_group_size().unwrap()).unwrap(),
                            );
                            let (group_len, _) = adjust_opencl_group_len(group_len);
                            println!("GroupLen for {:?}: {}", dev_id, group_len);
                            let opencl_config = OPENCL_BUILDER_CONFIG_DEFAULT
                                .group_len(Some(group_len))
                                .optimize_negs(!cmd_args.no_optimize_negs);
                            (OpenCLBuilder::new(&device, Some(opencl_config)), group_len)
                        })
                        .collect::<Vec<_>>()
                } else {
                    println!("Execute in CPUAndOpenCLD");
                    get_all_devices(CL_DEVICE_TYPE_GPU)
                        .unwrap()
                        .into_iter()
                        .map(|dev_id| {
                            let device = Device::new(dev_id);
                            let group_len = cmd_args.opencl_group_len.unwrap_or(
                                usize::try_from(device.max_work_group_size().unwrap()).unwrap(),
                            );
                            let (group_len, _) = adjust_opencl_group_len(group_len);
                            println!("GroupLen for {:?}: {}", dev_id, group_len);
                            let opencl_config = OPENCL_BUILDER_CONFIG_DEFAULT
                                .group_len(Some(group_len))
                                .optimize_negs(!cmd_args.no_optimize_negs);
                            [
                                (
                                    OpenCLBuilder::new(&device, Some(opencl_config.clone())),
                                    group_len,
                                ),
                                (
                                    OpenCLBuilder::new(&device, Some(opencl_config.clone())),
                                    group_len,
                                ),
                            ]
                        })
                        .flatten()
                        .collect::<Vec<_>>()
                };
                let group_lens = seq_builders_and_group_lens
                    .iter()
                    .map(|x| x.1)
                    .collect::<Vec<_>>();
                let seq_builders = seq_builders_and_group_lens
                    .into_iter()
                    .map(|x| x.0)
                    .collect::<Vec<_>>();
                let builder = ParSeqMapperBuilder::new(par_builder, seq_builders);
                do_command_with_parseq_mapper(builder, qcircuit.clone(), elem_inputs, &group_lens)
            }
        }
    } else {
        let mut qr = QuantReducer::new(qcircuit.quants());
        for v in 0..1 << input_len {
            let r = circuit.eval((0..input_len).rev().map(|b| (v >> b) & 1 != 0))[0];
            qr.push(v, r);
        }
        qr.final_result().unwrap()
    };
    println!("Result: {}", result);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn str_to_quants(s: &str) -> Vec<Quant> {
        s.chars()
            .filter(|c| {
                let c = c.to_ascii_lowercase();
                c == 'a' || c == 'e'
            })
            .map(|c| match c.to_ascii_lowercase() {
                'a' => Quant::All,
                'e' => Quant::Exists,
                _ => {
                    panic!("Unexpected");
                }
            })
            .collect::<Vec<_>>()
    }

    fn str_to_bools(s: &str) -> Vec<bool> {
        s.chars()
            .filter(|c| *c == '0' || *c == '1')
            .map(|c| match c {
                '0' => false,
                '1' => true,
                _ => {
                    panic!("Unexpected");
                }
            })
            .collect::<Vec<_>>()
    }

    #[test]
    fn test_quant_reducer() {
        // info: result solution is bit reversed!
        // indexes of items are reversed to original input of circuit!
        for (i, (quants, items, ordering, opt_result, result)) in [
            // 0
            (
                str_to_quants("eea"),
                str_to_bools("00000000"),
                (0..8).collect::<Vec<_>>(),
                None,
                FinalResult {
                    reversed: false,
                    solution_bits: 2,
                    solution: None,
                },
            ),
            // 1
            (
                str_to_quants("eea"),
                str_to_bools("00000100"),
                (0..8).collect::<Vec<_>>(),
                None,
                FinalResult {
                    reversed: false,
                    solution_bits: 2,
                    solution: None,
                },
            ),
            // 2
            (
                str_to_quants("eea"),
                str_to_bools("00001100"),
                (0..8).collect::<Vec<_>>(),
                Some(5),
                FinalResult {
                    reversed: false,
                    solution_bits: 2,
                    solution: Some(1),
                },
            ),
            // 3
            (
                str_to_quants("eea"),
                str_to_bools("00001100"),
                vec![4, 7, 3, 2, 0, 1, 6, 5],
                Some(7),
                FinalResult {
                    reversed: false,
                    solution_bits: 2,
                    solution: Some(1),
                },
            ),
            // 4
            (
                str_to_quants("eea"),
                str_to_bools("00110011"),
                (0..8).collect::<Vec<_>>(),
                Some(3),
                FinalResult {
                    reversed: false,
                    solution_bits: 2,
                    solution: Some(2),
                },
            ),
            // 5
            (
                str_to_quants("eea"),
                str_to_bools("00110011"),
                vec![0, 1, 5, 3, 4, 2, 6, 7],
                Some(5),
                FinalResult {
                    reversed: false,
                    solution_bits: 2,
                    solution: Some(2),
                },
            ),
            // changed quantifiers 1
            // 6
            (
                str_to_quants("aae"),
                str_to_bools("11111111"),
                (0..8).collect::<Vec<_>>(),
                None,
                FinalResult {
                    reversed: true,
                    solution_bits: 2,
                    solution: None,
                },
            ),
            // 7
            (
                str_to_quants("aae"),
                str_to_bools("11111011"),
                (0..8).collect::<Vec<_>>(),
                None,
                FinalResult {
                    reversed: true,
                    solution_bits: 2,
                    solution: None,
                },
            ),
            // 8
            (
                str_to_quants("aae"),
                str_to_bools("11110011"),
                (0..8).collect::<Vec<_>>(),
                Some(5),
                FinalResult {
                    reversed: true,
                    solution_bits: 2,
                    solution: Some(1),
                },
            ),
            // one quantifier
            // 9
            (
                str_to_quants("eee"),
                str_to_bools("00000000"),
                (0..8).collect::<Vec<_>>(),
                None,
                FinalResult {
                    reversed: false,
                    solution_bits: 3,
                    solution: None,
                },
            ),
            // 10
            (
                str_to_quants("eee"),
                str_to_bools("10000000"),
                (0..8).collect::<Vec<_>>(),
                Some(0),
                FinalResult {
                    reversed: false,
                    solution_bits: 3,
                    solution: Some(0),
                },
            ),
            // 11
            (
                str_to_quants("eee"),
                str_to_bools("01000000"),
                (0..8).collect::<Vec<_>>(),
                Some(1),
                FinalResult {
                    reversed: false,
                    solution_bits: 3,
                    solution: Some(4),
                },
            ),
            // 12
            (
                str_to_quants("eee"),
                str_to_bools("00100100"),
                (0..8).collect::<Vec<_>>(),
                Some(2),
                FinalResult {
                    reversed: false,
                    solution_bits: 3,
                    solution: Some(2),
                },
            ),
            // 13
            (
                str_to_quants("eee"),
                str_to_bools("00001000"),
                (0..8).collect::<Vec<_>>(),
                Some(4),
                FinalResult {
                    reversed: false,
                    solution_bits: 3,
                    solution: Some(1),
                },
            ),
            // 14
            (
                str_to_quants("eee"),
                str_to_bools("00000100"),
                (0..8).collect::<Vec<_>>(),
                Some(5),
                FinalResult {
                    reversed: false,
                    solution_bits: 3,
                    solution: Some(5),
                },
            ),
            // 15
            (
                str_to_quants("eee"),
                str_to_bools("00000010"),
                (0..8).collect::<Vec<_>>(),
                Some(6),
                FinalResult {
                    reversed: false,
                    solution_bits: 3,
                    solution: Some(3),
                },
            ),
            // one quantifier
            // 16
            (
                str_to_quants("aaa"),
                str_to_bools("11111101"),
                (0..8).collect::<Vec<_>>(),
                Some(6),
                FinalResult {
                    reversed: true,
                    solution_bits: 3,
                    solution: Some(3),
                },
            ),
            // 17
            (
                str_to_quants("eeee"),
                str_to_bools("00000000_00010000"),
                (0..16).collect::<Vec<_>>(),
                Some(11),
                FinalResult {
                    reversed: false,
                    solution_bits: 4,
                    solution: Some(0b1101),
                },
            ),
            // many quantifiers
            // 18
            (
                str_to_quants("eeaeaa"),
                str_to_bools(concat!(
                    "0000_0000.0000_0000:0000_0000.0000_0000",
                    "0000_0000.0000_0000:0000_0000.0000_0000"
                )),
                (0..64).collect::<Vec<_>>(),
                None,
                FinalResult {
                    reversed: false,
                    solution_bits: 2,
                    solution: None,
                },
            ),
            // 19
            (
                str_to_quants("eeaeaa"),
                str_to_bools(concat!(
                    "1110_0111.0011_1100:0011_0110.1110_0111",
                    "0111_0110.0100_0011:1100_1001.0101_0110"
                )),
                (0..64).collect::<Vec<_>>(),
                None,
                FinalResult {
                    reversed: false,
                    solution_bits: 2,
                    solution: None,
                },
            ),
            // 20
            (
                str_to_quants("eeaeaa"),
                str_to_bools(concat!(
                    "0000_1111.1111_0000:0000_0000.0000_0000",
                    "0000_0000.0000_0000:0000_0000.0000_0000"
                )),
                (0..64).collect::<Vec<_>>(),
                Some(15),
                FinalResult {
                    reversed: false,
                    solution_bits: 2,
                    solution: Some(0),
                },
            ),
            // 21
            (
                str_to_quants("eeaeaa"),
                str_to_bools(concat!(
                    "0000_0011.1001_0000:1111_0000.0000_1111",
                    "0000_0000.0000_0000:0000_1111.0000_1111"
                )),
                (0..64).collect::<Vec<_>>(),
                Some(31),
                FinalResult {
                    reversed: false,
                    solution_bits: 2,
                    solution: Some(2),
                },
            ),
            // 22
            (
                str_to_quants("eeaeaa"),
                str_to_bools(concat!(
                    "0000_0011.1001_0000:1000_0000.0000_0001",
                    "0000_1111.0000_1111:0000_1111.0000_1111"
                )),
                (0..64).collect::<Vec<_>>(),
                Some(47),
                FinalResult {
                    reversed: false,
                    solution_bits: 2,
                    solution: Some(1),
                },
            ),
            // 23
            (
                str_to_quants("eeaeaa"),
                str_to_bools(concat!(
                    "0000_0011.1001_0000:1000_0000.0000_0001",
                    "0000_1100.0000_1011:0000_1111.1111_1111"
                )),
                (0..64).collect::<Vec<_>>(),
                Some(63),
                FinalResult {
                    reversed: false,
                    solution_bits: 2,
                    solution: Some(3),
                },
            ),
            // 24
            (
                str_to_quants("eeeaeaa"),
                str_to_bools(concat!(
                    "0000_0011.1001_0000:1000_0000.0000_0001",
                    "0000_1100.0000_1011:0000_1001.1101_1011",
                    "0000_0011.1001_0100:1001_0010.0100_0001",
                    "1111_1100.0000_1111:0000_1001.1100_1101"
                )),
                (0..128).collect::<Vec<_>>(),
                Some(111),
                FinalResult {
                    reversed: false,
                    solution_bits: 3,
                    solution: Some(0b011),
                },
            ),
            // reverse quantifiers
            // 25
            (
                str_to_quants("aaeaee"),
                str_to_bools(concat!(
                    "0010_0100.0000_0000:0000_0000.0010_0010",
                    "0000_0000.0000_0000:0100_0010.0001_0010"
                )),
                (0..64).collect::<Vec<_>>(),
                Some(47),
                FinalResult {
                    reversed: true,
                    solution_bits: 2,
                    solution: Some(1),
                },
            ),
            // 26
            (
                str_to_quants("aaeaee"),
                str_to_bools(concat!(
                    "0010_0100.0000_0000:0000_0000.0010_0010",
                    "X0000_0100.0010_0000:0100_0010.0001_0010"
                )),
                (0..64).collect::<Vec<_>>(),
                Some(47),
                FinalResult {
                    reversed: true,
                    solution_bits: 2,
                    solution: Some(1),
                },
            ),
            // 27
            (
                str_to_quants("aaaeaee"),
                str_to_bools(concat!(
                    "0010_0100.0000_0000:0000_0000.0010_0010",
                    "0000_0100.0010_0001:0100_0010.0001_0010",
                    "0010_0100.0000_0000:0000_0000.0010_0010",
                    "X0000_0100.0010_0000:0100_0010.0001_0010"
                )),
                (0..128).collect::<Vec<_>>(),
                Some(111),
                FinalResult {
                    reversed: true,
                    solution_bits: 3,
                    solution: Some(3),
                },
            ),
            // 28: various ordering
            (
                str_to_quants("aaaeaee"),
                str_to_bools(concat!(
                    "0010_0100.0000_0000:0000_0000.0010_0010",
                    "0000_0100.0010_0001:0100_0010.0001_0010",
                    "0010_0100.0000_0000:0000_0000.0010_0010",
                    "X0000_0100.0010_0000:0100_0010.0001_0010"
                )),
                (0..96)
                    .chain((0..32).map(|i| 96 + ((i >> 1) | ((i << 4) & 16))))
                    .collect::<Vec<_>>(),
                Some(127),
                FinalResult {
                    reversed: true,
                    solution_bits: 3,
                    solution: Some(3),
                },
            ),
            // 29: bigger first
            (
                str_to_quants("eeeeeae"),
                str_to_bools(concat!(
                    "0000_0000.0000_0100:0000_0000.0000_0000",
                    "0000_0100.0000_0000:0000_0010.0000_0000",
                    "0000_0000.0000_0110:0000_0000.0000_0000",
                    "0000_0000.0000_0000:0000_0000.0000_0000",
                )),
                (0..128).collect::<Vec<_>>(),
                Some(79),
                FinalResult {
                    reversed: false,
                    solution_bits: 5,
                    solution: Some(0b11001),
                },
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let mut reducer = QuantReducer::new(&quants);
            for (j, ord) in ordering.into_iter().enumerate() {
                reducer.push(ord, items[usize::try_from(ord).unwrap()]);
                if let Some(push_index) = opt_result {
                    if j >= push_index {
                        assert_eq!(Some(result), reducer.final_result(), "{}", i);
                    }
                }
            }
            assert_eq!(result, reducer.final_result().unwrap(), "{}", i);
        }
    }

    #[test]
    fn test_get_aggr_output_cpu_code_defs() {
        assert_eq!(
            r##"#define WORK_QUANT_REDUCE_INIT_DATA (399ULL)
#define TYPE_QUANT_REDUCE_OP_0 |
#define TYPE_QUANT_REDUCE_OP_1 |
#define TYPE_QUANT_REDUCE_OP_2 |
#define TYPE_QUANT_REDUCE_OP_3 &
#define TYPE_QUANT_REDUCE_OP_4 |
#define TYPE_QUANT_REDUCE_OP_5 &
#define TYPE_QUANT_REDUCE_OP_6 |
#define TYPE_QUANT_REDUCE_OP_7 |
#define WORK_WORD_NUM_BITS (10)
"##,
            get_aggr_output_cpu_code_defs(256, 18, &str_to_quants("EEEEAAAEAAEEEAAAAEEAEAEEE"))
        );
        assert_eq!(
            r##"#define WORK_QUANT_REDUCE_INIT_DATA (399ULL)
#define OTHER_MASK (511ULL)
#define TYPE_QUANT_REDUCE_OP_0 |
#define TYPE_QUANT_REDUCE_OP_1 |
#define TYPE_QUANT_REDUCE_OP_2 |
#define TYPE_QUANT_REDUCE_OP_3 &
#define TYPE_QUANT_REDUCE_OP_4 |
#define TYPE_QUANT_REDUCE_OP_5 &
#define TYPE_QUANT_REDUCE_OP_6 |
#define TYPE_QUANT_REDUCE_OP_7 |
#define WORK_WORD_NUM_BITS (10)
#define WORK_HAVE_FIRST_QUANT
"##,
            get_aggr_output_cpu_code_defs(256, 18, &str_to_quants("EEEEEEEEAAEEEAAAAEEAEAEEE"))
        );
        assert_eq!(
            r##"#define WORK_QUANT_REDUCE_INIT_DATA (15ULL)
#define OTHER_MASK (15ULL)
#define TYPE_QUANT_REDUCE_OP_0 |
#define TYPE_QUANT_REDUCE_OP_1 |
#define TYPE_QUANT_REDUCE_OP_2 |
#define TYPE_QUANT_REDUCE_OP_3 &
#define TYPE_QUANT_REDUCE_OP_4 |
#define TYPE_QUANT_REDUCE_OP_5 &
#define TYPE_QUANT_REDUCE_OP_6 |
#define TYPE_QUANT_REDUCE_OP_7 |
#define WORK_WORD_NUM_BITS (10)
#define WORK_HAVE_FIRST_QUANT
"##,
            get_aggr_output_cpu_code_defs(256, 18, &str_to_quants("EEEEEEEEEEEEEAAAAEEAEAEEE"))
        );
        assert_eq!(
            r##"#define WORK_QUANT_REDUCE_INIT_DATA (911ULL)
#define TYPE_QUANT_REDUCE_OP_0 |
#define TYPE_QUANT_REDUCE_OP_1 |
#define TYPE_QUANT_REDUCE_OP_2 |
#define TYPE_QUANT_REDUCE_OP_3 &
#define TYPE_QUANT_REDUCE_OP_4 |
#define TYPE_QUANT_REDUCE_OP_5 &
#define TYPE_QUANT_REDUCE_OP_6 |
#define TYPE_QUANT_REDUCE_OP_7 |
#define WORK_WORD_NUM_BITS (10)
"##,
            get_aggr_output_cpu_code_defs(256, 18, &str_to_quants("EEEEEEEAAAEEEAAAAEEAEAEEE"))
        );
        assert_eq!(
            r##"#define TYPE_QUANT_REDUCE_QUANT_ALL (0)
#define TYPE_QUANT_REDUCE_OP_0 |
#define TYPE_QUANT_REDUCE_OP_1 |
#define TYPE_QUANT_REDUCE_OP_2 |
#define TYPE_QUANT_REDUCE_OP_3 &
#define TYPE_QUANT_REDUCE_OP_4 |
#define TYPE_QUANT_REDUCE_OP_5 &
#define TYPE_QUANT_REDUCE_OP_6 |
#define TYPE_QUANT_REDUCE_OP_7 |
#define WORK_WORD_NUM_BITS (0)
#define WORK_HAVE_FIRST_QUANT
"##,
            get_aggr_output_cpu_code_defs(256, 8, &str_to_quants("EEAEAEEE"))
        );
        assert_eq!(
            r##"#define TYPE_QUANT_REDUCE_QUANT_ALL (1)
#define TYPE_QUANT_REDUCE_OP_0 |
#define TYPE_QUANT_REDUCE_OP_1 |
#define TYPE_QUANT_REDUCE_OP_2 |
#define TYPE_QUANT_REDUCE_OP_3 &
#define TYPE_QUANT_REDUCE_OP_4 |
#define TYPE_QUANT_REDUCE_OP_5 &
#define TYPE_QUANT_REDUCE_OP_6 |
#define TYPE_QUANT_REDUCE_OP_7 &
#define WORK_WORD_NUM_BITS (0)
#define WORK_HAVE_FIRST_QUANT
"##,
            get_aggr_output_cpu_code_defs(256, 8, &str_to_quants("AEAEAEEE"))
        );
        assert_eq!(
            r##"#define WORK_QUANT_REDUCE_INIT_DATA (0ULL)
#define OTHER_MASK (0ULL)
#define TYPE_QUANT_REDUCE_OP_0 |
#define TYPE_QUANT_REDUCE_OP_1 |
#define TYPE_QUANT_REDUCE_OP_2 |
#define TYPE_QUANT_REDUCE_OP_3 &
#define TYPE_QUANT_REDUCE_OP_4 &
#define TYPE_QUANT_REDUCE_OP_5 &
#define TYPE_QUANT_REDUCE_OP_6 &
#define TYPE_QUANT_REDUCE_OP_7 &
#define WORK_WORD_NUM_BITS (10)
#define WORK_HAVE_FIRST_QUANT
"##,
            get_aggr_output_cpu_code_defs(256, 18, &str_to_quants("EEEEEEEEEEEEEEEEEAAAAAEEE"))
        );
        assert_eq!(
            r##"#define WORK_QUANT_REDUCE_INIT_DATA (0ULL)
#define OTHER_MASK (0ULL)
#define TYPE_QUANT_REDUCE_OP_0 |
#define TYPE_QUANT_REDUCE_OP_1 |
#define TYPE_QUANT_REDUCE_OP_2 |
#define TYPE_QUANT_REDUCE_OP_3 &
#define TYPE_QUANT_REDUCE_OP_4 &
#define TYPE_QUANT_REDUCE_OP_5 |
#define TYPE_QUANT_REDUCE_OP_6 |
#define TYPE_QUANT_REDUCE_OP_7 |
#define WORK_WORD_NUM_BITS (10)
#define WORK_HAVE_FIRST_QUANT
"##,
            get_aggr_output_cpu_code_defs(256, 18, &str_to_quants("EEEEEEEEEEEEEEEEEEEEAAEEE"))
        );
        assert_eq!(
            r##"#define WORK_QUANT_REDUCE_INIT_DATA (1023ULL)
#define OTHER_MASK (0ULL)
#define TYPE_QUANT_REDUCE_OP_0 &
#define TYPE_QUANT_REDUCE_OP_1 &
#define TYPE_QUANT_REDUCE_OP_2 &
#define TYPE_QUANT_REDUCE_OP_3 |
#define TYPE_QUANT_REDUCE_OP_4 |
#define TYPE_QUANT_REDUCE_OP_5 |
#define TYPE_QUANT_REDUCE_OP_6 |
#define TYPE_QUANT_REDUCE_OP_7 |
#define WORK_WORD_NUM_BITS (10)
#define WORK_HAVE_FIRST_QUANT
"##,
            get_aggr_output_cpu_code_defs(256, 18, &str_to_quants("AAAAAAAAAAAAAAAAAEEEEEAAA"))
        );
        assert_eq!(
            r##"#define WORK_QUANT_REDUCE_INIT_DATA (1023ULL)
#define OTHER_MASK (0ULL)
#define TYPE_QUANT_REDUCE_OP_0 &
#define TYPE_QUANT_REDUCE_OP_1 &
#define TYPE_QUANT_REDUCE_OP_2 &
#define TYPE_QUANT_REDUCE_OP_3 |
#define TYPE_QUANT_REDUCE_OP_4 &
#define TYPE_QUANT_REDUCE_OP_5 &
#define TYPE_QUANT_REDUCE_OP_6 &
#define TYPE_QUANT_REDUCE_OP_7 &
#define WORK_WORD_NUM_BITS (10)
#define WORK_HAVE_FIRST_QUANT
"##,
            get_aggr_output_cpu_code_defs(256, 18, &str_to_quants("AAAAAAAAAAAAAAAAAAAAAEAAA"))
        );
    }

    use gatenative::clang_writer::*;

    #[test]
    fn test_aggr_output_cpu_code() {
        let circuit = Circuit::<usize>::new(1, [], [(0, false)]).unwrap();
        for (i, (cpu_exts, clang_writer, quants, testcases)) in [
            (
                vec![CPUExtension::NoExtension],
                &CLANG_WRITER_U64,
                &str_to_quants("AAEAEA"),
                vec![
                    (
                        vec![
                            0b00000000_00000000_00000000_00000000u32,
                            0b00000000_00000000_00000000_00000000u32,
                        ],
                        false,
                    ),
                    (
                        vec![
                            0b00100000_00100000_00010000_00100010u32,
                            0b00000100_00001000_00010000_00100000u32,
                        ],
                        false,
                    ),
                    (
                        vec![
                            0b00100000_00111100_00010000_00100010u32,
                            0b00000100_00001000_00010000_00100000u32,
                        ],
                        false,
                    ),
                    (
                        vec![
                            0b00100000_00111100_00111100_00100010u32,
                            0b00000100_00001000_00010000_00100000u32,
                        ],
                        false,
                    ),
                    (
                        vec![
                            0b00100000_00111100_11010011_00100010u32,
                            0b11001100_00001000_00010000_00110011u32,
                        ],
                        true,
                    ),
                ],
            ),
            (
                vec![CPUExtension::NoExtension],
                &CLANG_WRITER_U64,
                &str_to_quants("EEAEAE"),
                vec![
                    (
                        vec![
                            0b00010000_00000000_00000000_00100000u32,
                            0b00000010_00000010_00001000_00000000u32,
                        ],
                        false,
                    ),
                    (
                        vec![
                            0b00010000_00000000_00000000_00100000u32,
                            0b10010010_00000110_00001000_00000000u32,
                        ],
                        true,
                    ),
                ],
            ),
            (
                vec![
                    CPUExtension::IntelAVX,
                    CPUExtension::IntelAVX2,
                    CPUExtension::IntelAVX512,
                ],
                &CLANG_WRITER_INTEL_AVX,
                &str_to_quants("EEAEAEAA"),
                vec![
                    (
                        vec![
                            0b00010000_00000000_00000000_00100000u32,
                            0b00000010_00000010_00001000_00000000u32,
                            0b00010000_00000000_00000000_00100000u32,
                            0b00000010_00000010_00001000_00000000u32,
                            0b00010000_00000000_00000000_00100000u32,
                            0b00000010_00000010_00001000_00000000u32,
                            0b00010000_00000000_00000000_00100000u32,
                            0b00000010_00000010_00001000_00000000u32,
                        ],
                        false,
                    ),
                    (
                        vec![
                            0b00010000_00000000_00000000_00100000u32,
                            0b00000010_00000010_00001000_00000000u32,
                            0b11110000_00001111_00000000_00100000u32,
                            0b00000010_00000010_00001111_11110000u32,
                            0b00010000_00000000_00000000_00100000u32,
                            0b00000010_00000010_00001000_00000000u32,
                            0b00010000_00000000_00000000_00100000u32,
                            0b00000010_00000010_00001000_00000000u32,
                        ],
                        true,
                    ),
                ],
            ),
            (
                vec![
                    CPUExtension::IntelAVX,
                    CPUExtension::IntelAVX2,
                    CPUExtension::IntelAVX512,
                ],
                &CLANG_WRITER_INTEL_AVX,
                &str_to_quants("AAAEEAAA"),
                vec![
                    (
                        vec![
                            0b00010000_00000000_00000000_00100000u32,
                            0b00000010_00000010_00001000_00000000u32,
                            0b00010000_00000000_00000000_00100000u32,
                            0b00000010_00000010_00001000_00000000u32,
                            0b00010000_00000000_00000000_00100000u32,
                            0b00000010_00000010_00001000_00000000u32,
                            0b00010000_00000000_00000000_00100000u32,
                            0b00000010_00000010_00001000_00000000u32,
                        ],
                        false,
                    ),
                    (
                        vec![
                            0b00010000_11111111_00000000_00100000u32,
                            0b00000010_00000010_11111111_00000000u32,
                            0b00010000_00000000_00000000_11111111u32,
                            0b00000010_11111111_00001000_00000000u32,
                            0b11111111_00000000_00000000_00100000u32,
                            0b00000010_00000010_00001000_11111111u32,
                            0b11111111_00000000_00000000_00100000u32,
                            0b00000010_00000010_11111111_00000000u32,
                        ],
                        true,
                    ),
                ],
            ),
            // Check for AVX2!!!
            (
                vec![CPUExtension::IntelAVX2, CPUExtension::IntelAVX512],
                &CLANG_WRITER_INTEL_AVX2,
                &str_to_quants("EEAEAEAA"),
                vec![
                    (
                        vec![
                            0b00010000_00000000_00000000_00100000u32,
                            0b00000010_00000010_00001000_00000000u32,
                            0b00010000_00000000_00000000_00100000u32,
                            0b00000010_00000010_00001000_00000000u32,
                            0b00010000_00000000_00000000_00100000u32,
                            0b00000010_00000010_00001000_00000000u32,
                            0b00010000_00000000_00000000_00100000u32,
                            0b00000010_00000010_00001000_00000000u32,
                        ],
                        false,
                    ),
                    (
                        vec![
                            0b00010000_00000000_00000000_00100000u32,
                            0b00000010_00000010_00001000_00000000u32,
                            0b11110000_00001111_00000000_00100000u32,
                            0b00000010_00000010_00001111_11110000u32,
                            0b00010000_00000000_00000000_00100000u32,
                            0b00000010_00000010_00001000_00000000u32,
                            0b00010000_00000000_00000000_00100000u32,
                            0b00000010_00000010_00001000_00000000u32,
                        ],
                        true,
                    ),
                ],
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let defs = get_aggr_output_cpu_code_defs(1 << quants.len(), quants.len(), &quants);
            if *cpu_exts.last().unwrap() != CPUExtension::NoExtension
                && !cpu_exts.iter().any(|ext| *ext == *CPU_EXTENSION)
            {
                continue;
            }
            let mut builder = CPUBuilder::new_with_cpu_ext_and_clang_config(
                *cpu_exts.first().unwrap(),
                clang_writer,
                None,
            );
            builder.user_defs(&defs);
            builder.add_with_config(
                "formula",
                circuit.clone(),
                CodeConfig::new()
                    .aggr_output_code(Some(AGGR_OUTPUT_CPU_CODE))
                    .aggr_output_len(Some(200)),
            );
            let mut execs = builder.build().unwrap();
            println!("Run {}", i);
            for (j, (data, result)) in testcases.into_iter().enumerate() {
                let input = execs[0].new_data_from_vec(data);
                let output = execs[0].execute(&input, 0).unwrap().release();
                assert_eq!(result, output[1 << (quants.len() - 5)] != 0, "{} {}", i, j);
            }
        }
    }

    #[test]
    fn test_aggr_output_cpu_code_2() {
        // info: index of solution is not reversed index of machine word in work.
        let circuit = Circuit::<usize>::new(1, [], [(0, false)]).unwrap();
        for (i, (elem_bits, quants, testcases)) in [
            (
                6,
                &str_to_quants("AAEAEA"),
                vec![
                    (
                        vec![
                            0b00000000_00000000_00000000_00000000u32,
                            0b00000000_00000000_00000000_00000000u32,
                        ],
                        vec![
                            0b00000000_00000000_00000000_00000000u32,
                            0b00000000_00000000_00000000_00000000u32,
                        ],
                        Some(0),
                        false,
                    ),
                    (
                        vec![
                            0b00100000_00111100_11010011_00100010u32,
                            0b11001100_00001000_00010000_00110011u32,
                        ],
                        vec![
                            0b00100000_00111100_11010011_00100010u32,
                            0b11001100_00001000_00010000_00110011u32,
                        ],
                        None,
                        true,
                    ),
                ],
            ),
            (
                9,
                &str_to_quants("EEEE_AAAAEE"),
                vec![
                    (
                        vec![
                            0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
                            0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
                        ],
                        vec![0, 0],
                        None,
                        false,
                    ),
                    (
                        vec![
                            0x0100, 0x0010, 0x0800, 0x0800, 0x0000, 0x0000, 0x557736bb, 0xd7952696,
                            0x0000, 0x0200, 0x4000, 0x0002, 0x0800, 0x0000, 0x0000, 0x0080,
                        ],
                        vec![0x557736bb, 0xd7952696],
                        Some(3),
                        true,
                    ),
                    (
                        vec![
                            0x0100, 0x0010, 0x0800, 0x0800, 0x0000, 0x0000, 0x557706bb, 0xd7959696,
                            0x0000, 0x0200, 0x4000, 0x0002, 0xfed756db, 0x56dab31e, 0x0000, 0x0080,
                        ],
                        vec![0xfed756db, 0x56dab31e],
                        Some(6),
                        true,
                    ),
                ],
            ),
            // with no arg part
            (
                9,
                &str_to_quants("EEE_AAAAEE"),
                vec![
                    (
                        vec![
                            0x0100, 0x0010, 0x0800, 0x0800, 0x0000, 0x0000, 0x557736bb, 0xd7952696,
                            0x0000, 0x0200, 0x4000, 0x0002, 0x0800, 0x0000, 0x0000, 0x0080,
                        ],
                        vec![0x557736bb, 0xd7952696],
                        Some(3),
                        true,
                    ),
                    (
                        vec![
                            0x0100, 0x0010, 0x0800, 0x0800, 0x0000, 0x0000, 0x557706bb, 0xd7959696,
                            0x0000, 0x0200, 0x4000, 0x0002, 0xfed756db, 0x56dab31e, 0x0000, 0x0080,
                        ],
                        vec![0xfed756db, 0x56dab31e],
                        Some(6),
                        true,
                    ),
                ],
            ),
            // no first quantifier in work bits
            (
                9,
                &str_to_quants("AEEE_AAAAEE"),
                vec![
                    (
                        vec![
                            0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
                            0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
                        ],
                        vec![0, 0],
                        None,
                        false,
                    ),
                    (
                        vec![
                            0x0100, 0x0010, 0x0800, 0x0800, 0x0000, 0x0000, 0x557736bb, 0xd7952696,
                            0x0000, 0x0200, 0x4000, 0x0002, 0x0800, 0x0000, 0x0000, 0x0080,
                        ],
                        vec![0, 0],
                        None,
                        true,
                    ),
                    (
                        vec![
                            0x0100, 0x0010, 0x0800, 0x0800, 0x0000, 0x0000, 0x557706bb, 0xd7959696,
                            0x0000, 0x0200, 0x4000, 0x0002, 0xfed756db, 0x56dab31e, 0x0000, 0x0080,
                        ],
                        vec![0, 0],
                        None,
                        true,
                    ),
                ],
            ),
            (
                9,
                &str_to_quants("AAAA_EEEEAA"),
                vec![
                    (
                        vec![
                            0x0f010,
                            0x0000,
                            0x0010,
                            0xf00500,
                            0x000f0,
                            0x00002000,
                            0x0010,
                            0x00f0000,
                            0x001f0000,
                            0x0000,
                            0x0110f00,
                            0x066000,
                            0x0000,
                            0x000f0100,
                            0x000f02000,
                            0x0000,
                        ],
                        vec![0, 0],
                        None,
                        true,
                    ),
                    (
                        vec![
                            0x0f010,
                            0x0000,
                            0x0010,
                            0xf00500,
                            0x000f0,
                            0x00002000,
                            0x0010,
                            0x00f0000,
                            0x00001000,
                            0x1e0000, // 4:
                            0x0110f00,
                            0x066000,
                            0x0000,
                            0x000f0100,
                            0x000f02000,
                            0x0000,
                        ],
                        vec![0x00001000, 0x1e0000],
                        Some(4),
                        false,
                    ),
                    (
                        vec![
                            0x0f010,
                            0x0000,
                            0x0010,
                            0xf00500,
                            0x000f0,
                            0x00002000,
                            0x0010,
                            0x00f0000,
                            0x00001000,
                            0x1ef000,
                            0x0110d00, // 5:
                            0x066000,
                            0x0000,
                            0x000f0100,
                            0x000f02000,
                            0x0000,
                        ],
                        vec![0x0110d00, 0x066000],
                        Some(5),
                        false,
                    ),
                ],
            ),
            // no first quantifier in work bits
            (
                9,
                &str_to_quants("EAAA_EEEEAA"),
                vec![
                    (
                        vec![
                            0x0f010,
                            0x0000,
                            0x0010,
                            0xf00500,
                            0x000f0,
                            0x00002000,
                            0x0010,
                            0x00f0000,
                            0x001f0000,
                            0x0000,
                            0x0110f00,
                            0x066000,
                            0x0000,
                            0x000f0100,
                            0x000f02000,
                            0x0000,
                        ],
                        vec![0, 0],
                        None,
                        true,
                    ),
                    (
                        vec![
                            0x0f010,
                            0x0000,
                            0x0010,
                            0xf00500,
                            0x000f0,
                            0x00002000,
                            0x0010,
                            0x00f0000,
                            0x00001000,
                            0x1e0000, // 4:
                            0x0110f00,
                            0x066000,
                            0x0000,
                            0x000f0100,
                            0x000f02000,
                            0x0000,
                        ],
                        vec![0, 0],
                        None,
                        false,
                    ),
                    (
                        vec![
                            0x0f010,
                            0x0000,
                            0x0010,
                            0xf00500,
                            0x000f0,
                            0x00002000,
                            0x0010,
                            0x00f0000,
                            0x00001000,
                            0x1ef000,
                            0x0110d00, // 5:
                            0x066000,
                            0x0000,
                            0x000f0100,
                            0x000f02000,
                            0x0000,
                        ],
                        vec![0, 0],
                        None,
                        false,
                    ),
                ],
            ),
            // more complex
            (
                12,
                &str_to_quants("AAAEEAE_EEEEEE"),
                vec![
                    (
                        vec![
                            0, 0, 0, 0, 0, 0, 0, 0, // 0: 0
                            0, 0, 0, 0, 0, 0, 0, 0, // 0: 1
                            0, 0, 3, 0, 0, 3, 0, 0, // 0: 2
                            0, 0, 0, 0, 0, 0, 0, 0, // 0: 3
                            0, 2, 0, 0, 0, 0, 7, 0, // 1: 0
                            0, 0, 0, 0, 0, 0, 0, 0, // 1: 1
                            0, 0, 0, 0, 0, 0, 0, 0, // 1: 2
                            0, 0, 0, 0, 0, 0, 0, 0, // 1: 3
                            0, 0, 0, 0, 0, 0, 0, 0, // 2: 0
                            0, 0, 0, 0, 0, 0, 0, 0, // 2: 1
                            0, 0, 0, 0, 0, 0, 0, 0, // 2: 2
                            2, 0, 0, 0, 0, 1, 0, 0, // 2: 3
                            0, 0, 0, 0, 0, 0, 0, 0, // 3: 0
                            9, 0, 0, 0, 0, 8, 0, 0, // 3: 1
                            0, 0, 0, 0, 0, 0, 0, 0, // 3: 2
                            0, 0, 0, 0, 0, 0, 0, 0, // 3: 3
                        ],
                        vec![0, 0],
                        None,
                        true,
                    ),
                    (
                        vec![
                            0, 0, 0, 0, 0, 0, 0, 0, // 0: 0
                            0, 0, 0, 0, 0, 0, 0, 0, // 0: 1
                            0, 0, 3, 0, 0, 3, 0, 0, // 0: 2
                            0, 0, 0, 0, 0, 0, 0, 0, // 0: 3
                            0, 2, 0, 0, 0, 0, 0, 0, // 1: 0
                            0, 0, 0, 0, 0, 0, 0, 0, // 1: 1
                            0, 0, 0, 0, 0, 0, 0, 0, // 1: 2
                            0, 0, 0, 0, 0, 0, 0, 0, // 1: 3
                            0, 0, 0, 0, 0, 0, 0, 0, // 2: 0
                            0, 0, 0, 0, 0, 0, 0, 0, // 2: 1
                            0, 0, 0, 0, 0, 0, 0, 0, // 2: 2
                            2, 0, 0, 0, 0, 1, 0, 0, // 2: 3
                            0, 0, 0, 0, 0, 0, 0, 0, // 3: 0
                            9, 0, 0, 0, 0, 8, 0, 0, // 3: 1
                            0, 0, 0, 0, 0, 0, 0, 0, // 3: 2
                            0, 0, 0, 0, 0, 0, 0, 0, // 3: 3
                        ],
                        vec![0, 0],
                        Some(31),
                        false,
                    ),
                ],
            ),
            (
                12,
                &str_to_quants("EEEAEAA_EEEEEE"),
                vec![
                    (
                        vec![
                            0, 1, 7, 0, 1, 0, 0, 4, // 0: 0
                            0, 4, 0, 6, 0, 8, 0, 0, // 0: 1
                            0, 0, 3, 0, 0, 3, 0, 0, // 0: 2
                            1, 0, 1, 0, 0, 0, 1, 2, // 0: 3
                            0, 1, 7, 0, 1, 0, 0, 4, // 1: 0
                            0, 4, 0, 6, 0, 8, 0, 0, // 1: 1
                            0, 0, 3, 0, 0, 0, 2, 0, // 1: 2
                            1, 0, 1, 0, 0, 1, 0, 0, // 1: 3
                            0, 1, 7, 1, 0, 0, 0, 4, // 2: 0
                            0, 4, 0, 0, 0, 8, 0, 1, // 2: 1
                            1, 0, 3, 2, 0, 3, 7, 0, // 2: 2
                            1, 2, 0, 2, 0, 0, 7, 0, // 2: 3
                            0, 1, 7, 0, 0, 0, 0, 4, // 3: 0
                            0, 0, 0, 6, 0, 8, 0, 1, // 3: 1
                            1, 0, 3, 0, 0, 3, 7, 0, // 3: 2
                            1, 0, 0, 3, 0, 1, 7, 0, // 3: 3
                        ],
                        vec![0, 0],
                        None,
                        false,
                    ),
                    (
                        vec![
                            0, 1, 7, 0, 1, 0, 0, 4, // 0: 0
                            0, 4, 0, 6, 0, 8, 0, 0, // 0: 1
                            0, 0, 3, 0, 0, 3, 0, 0, // 0: 2
                            1, 0, 1, 0, 0, 0, 1, 2, // 0: 3
                            0, 1, 7, 0, 1, 0, 0, 4, // 1: 0
                            0, 4, 0, 6, 0, 8, 0, 0, // 1: 1
                            0, 0, 3, 0, 0, 0, 2, 0, // 1: 2
                            1, 0, 1, 0, 0, 1, 0, 0, // 1: 3
                            0, 1, 7, 1, 2, 0, 0, 4, // 2: 0
                            0, 4, 3, 0, 0, 8, 0, 1, // 2: 1
                            1, 0, 3, 2, 0, 3, 7, 0, // 2: 2
                            1, 2, 0, 2, 0, 0, 112, 0, // 2: 3
                            0, 1, 7, 0, 0, 0, 0, 4, // 3: 0
                            0, 0, 0, 6, 0, 8, 0, 1, // 3: 1
                            1, 0, 3, 0, 0, 3, 7, 0, // 3: 2
                            1, 0, 0, 3, 0, 1, 7, 0, // 3: 3
                        ],
                        vec![112, 0],
                        Some(47),
                        true,
                    ),
                ],
            ),
            (
                9,
                &str_to_quants("EEEE_EAAAEE"),
                vec![
                    (
                        vec![
                            0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
                            0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
                        ],
                        vec![0, 0],
                        None,
                        false,
                    ),
                    (
                        vec![
                            0x0100, 0x0010, 0x0800, 0x0800, 0x0000, 0x0000, 0x557736bb, 0x0006,
                            0x0000, 0x0200, 0x4000, 0x0002, 0x0800, 0x0000, 0x0000, 0x0080,
                        ],
                        vec![0x557736bb, 0x0006],
                        Some(3),
                        true,
                    ),
                    (
                        vec![
                            0x0100, 0x0010, 0x0800, 0x0800, 0x0000, 0x0000, 0x557706bb, 0xd7909696,
                            0x0000, 0x0200, 0x4000, 0x0002, 0x00000000, 0x56dab31e, 0x0000, 0x0080,
                        ],
                        vec![0, 0x56dab31e],
                        Some(6),
                        true,
                    ),
                ],
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let defs = get_aggr_output_cpu_code_defs(64, elem_bits, &quants);
            let mut builder = CPUBuilder::new_with_cpu_ext_and_clang_config(
                CPUExtension::NoExtension,
                &CLANG_WRITER_U64,
                None,
            );
            builder.user_defs(&defs);
            builder.add_with_config(
                "formula",
                circuit.clone(),
                CodeConfig::new()
                    .aggr_output_code(Some(AGGR_OUTPUT_CPU_CODE))
                    .aggr_output_len(Some(200)),
            );
            let mut execs = builder.build().unwrap();
            println!("Run {}", i);
            let base = 2;
            for (j, (data, mword, found, result)) in testcases.into_iter().enumerate() {
                let input = execs[0].new_data_from_vec(data);
                let output = execs[0].execute(&input, 0).unwrap().release();
                assert_eq!(result, output[base] != 0, "{} {}", i, j);
                assert_eq!(found.is_some(), output[base + 1] != 0, "{} {}", i, j);
                assert_eq!(&mword, &output[0..base], "{} {}", i, j);
                if let Option::<u64>::Some(found) = found {
                    assert_eq!(
                        (found & ((1u64 << 32) - 1)) as u32,
                        output[base + 2],
                        "{} {}",
                        i,
                        j
                    );
                    assert_eq!((found >> 32) as u32, output[base + 3], "{} {}", i, j);
                }
            }
        }
    }

    #[test]
    fn test_get_aggr_output_opencl_code_defs() {
        assert_eq!(
            r##"#define GROUP_LEN (256)
#define GROUP_LEN_BITS (8)
#define TYPE_QUANT_REDUCE_OP_0 |
#define TYPE_QUANT_REDUCE_OP_1 |
#define TYPE_QUANT_REDUCE_OP_2 |
#define TYPE_QUANT_REDUCE_OP_3 &
#define TYPE_QUANT_REDUCE_OP_4 |
#define LOCAL_QUANT_REDUCE_OP_0 &
#define LOCAL_QUANT_REDUCE_OP_1 |
#define LOCAL_QUANT_REDUCE_OP_2 |
#define LOCAL_QUANT_REDUCE_OP_3 &
#define LOCAL_QUANT_REDUCE_OP_4 &
#define LOCAL_QUANT_REDUCE_OP_5 &
#define LOCAL_QUANT_REDUCE_OP_6 &
#define LOCAL_QUANT_REDUCE_OP_7 |
#define LOCAL_FIRST_QUANT_LEVEL (99999999)
"##,
            get_aggr_output_opencl_code_defs(32, 256, &str_to_quants("EEEEAAAEAAEEEAAAAEEAEAEEE"))
        );
        assert_eq!(
            r##"#define GROUP_LEN (256)
#define GROUP_LEN_BITS (8)
#define TYPE_QUANT_REDUCE_OP_0 |
#define TYPE_QUANT_REDUCE_OP_1 |
#define TYPE_QUANT_REDUCE_OP_2 |
#define TYPE_QUANT_REDUCE_OP_3 &
#define TYPE_QUANT_REDUCE_OP_4 |
#define LOCAL_QUANT_REDUCE_OP_0 &
#define LOCAL_QUANT_REDUCE_OP_1 |
#define LOCAL_QUANT_REDUCE_OP_2 |
#define LOCAL_QUANT_REDUCE_OP_3 &
#define LOCAL_QUANT_REDUCE_OP_4 &
#define LOCAL_QUANT_REDUCE_OP_5 &
#define LOCAL_QUANT_REDUCE_OP_6 &
#define LOCAL_QUANT_REDUCE_OP_7 |
#define LOCAL_FIRST_QUANT_LEVEL (7)
#define LOCAL_FIRST_QUANT_PROPAGATE_CHECK (32768)
"##,
            get_aggr_output_opencl_code_defs(
                32,
                256,
                &str_to_quants("EEEEEEEEEEEE_EAAAAEEA_EAEEE")
            )
        );
        assert_eq!(
            r##"#define GROUP_LEN (256)
#define GROUP_LEN_BITS (8)
#define TYPE_QUANT_REDUCE_OP_0 |
#define TYPE_QUANT_REDUCE_OP_1 |
#define TYPE_QUANT_REDUCE_OP_2 |
#define TYPE_QUANT_REDUCE_OP_3 &
#define TYPE_QUANT_REDUCE_OP_4 |
#define LOCAL_QUANT_REDUCE_OP_0 &
#define LOCAL_QUANT_REDUCE_OP_1 |
#define LOCAL_QUANT_REDUCE_OP_2 |
#define LOCAL_QUANT_REDUCE_OP_3 &
#define LOCAL_QUANT_REDUCE_OP_4 |
#define LOCAL_QUANT_REDUCE_OP_5 |
#define LOCAL_QUANT_REDUCE_OP_6 |
#define LOCAL_QUANT_REDUCE_OP_7 |
#define LOCAL_FIRST_QUANT_LEVEL (4)
#define LOCAL_FIRST_QUANT_PROPAGATE_CHECK (32768)
"##,
            get_aggr_output_opencl_code_defs(
                32,
                256,
                &str_to_quants("EEEEEEEEEEEE_EEEEAEEA_EAEEE")
            )
        );
        assert_eq!(
            r##"#define GROUP_LEN (256)
#define GROUP_LEN_BITS (8)
#define TYPE_QUANT_REDUCE_OP_0 |
#define TYPE_QUANT_REDUCE_OP_1 |
#define TYPE_QUANT_REDUCE_OP_2 |
#define TYPE_QUANT_REDUCE_OP_3 &
#define TYPE_QUANT_REDUCE_OP_4 |
#define LOCAL_QUANT_REDUCE_OP_0 |
#define LOCAL_QUANT_REDUCE_OP_1 |
#define LOCAL_QUANT_REDUCE_OP_2 |
#define LOCAL_QUANT_REDUCE_OP_3 |
#define LOCAL_QUANT_REDUCE_OP_4 |
#define LOCAL_QUANT_REDUCE_OP_5 |
#define LOCAL_QUANT_REDUCE_OP_6 |
#define LOCAL_QUANT_REDUCE_OP_7 |
#define LOCAL_FIRST_QUANT_LEVEL (0)
#define LOCAL_FIRST_QUANT_PROPAGATE_CHECK (32768)
"##,
            get_aggr_output_opencl_code_defs(
                32,
                256,
                &str_to_quants("EEEEEEEEEEEE_EEEEEEEE_EAEEE")
            )
        );
        assert_eq!(
            r##"#define GROUP_LEN (256)
#define GROUP_LEN_BITS (8)
#define TYPE_QUANT_REDUCE_OP_0 &
#define TYPE_QUANT_REDUCE_OP_1 &
#define TYPE_QUANT_REDUCE_OP_2 &
#define TYPE_QUANT_REDUCE_OP_3 |
#define TYPE_QUANT_REDUCE_OP_4 &
#define LOCAL_QUANT_REDUCE_OP_0 |
#define LOCAL_QUANT_REDUCE_OP_1 &
#define LOCAL_QUANT_REDUCE_OP_2 &
#define LOCAL_QUANT_REDUCE_OP_3 |
#define LOCAL_QUANT_REDUCE_OP_4 &
#define LOCAL_QUANT_REDUCE_OP_5 &
#define LOCAL_QUANT_REDUCE_OP_6 &
#define LOCAL_QUANT_REDUCE_OP_7 &
#define LOCAL_FIRST_QUANT_LEVEL (4)
#define LOCAL_FIRST_QUANT_PROPAGATE_CHECK (0)
"##,
            get_aggr_output_opencl_code_defs(
                32,
                256,
                &str_to_quants("AAAAAAAAAAAA_AAAAEAAE_AEAAA")
            )
        );
        assert_eq!(
            r##"#define GROUP_LEN (256)
#define GROUP_LEN_BITS (8)
#define LOCAL_QUANT_REDUCE_OP_0 |
#define LOCAL_QUANT_REDUCE_OP_1 &
#define LOCAL_QUANT_REDUCE_OP_2 &
#define LOCAL_QUANT_REDUCE_OP_3 |
#define LOCAL_QUANT_REDUCE_OP_4 &
#define LOCAL_QUANT_REDUCE_OP_5 &
#define LOCAL_QUANT_REDUCE_OP_6 &
#define LOCAL_QUANT_REDUCE_OP_7 &
#define LOCAL_FIRST_QUANT_LEVEL (4)
#define LOCAL_FIRST_QUANT_PROPAGATE_CHECK (0)
"##,
            get_aggr_output_opencl_code_defs(1, 256, &str_to_quants("AAAAAAAAAAAA_AAAAEAAE"))
        );
    }

    #[test]
    fn test_aggr_output_opencl_code() {
        let circuit = Circuit::<usize>::new(1, [], [(0, false)]).unwrap();
        let device = Device::new(*get_all_devices(CL_DEVICE_TYPE_GPU).unwrap().get(0).unwrap());
        for (i, (quants, testcases)) in [
            (
                &str_to_quants("EAA_AEAEA"),
                vec![
                    (vec![0b00000000_00000000_00000000_00000000u32], false, 0),
                    (vec![0b00100000_00100000_00010000_00100010u32], false, 0),
                    (vec![0b00100000_00111100_00010000_00100010u32], false, 0),
                    (vec![0b00100000_00111100_00111100_00100010u32], true, 0),
                    (vec![0b00100000_00111100_11010011_00100010u32], true, 0),
                ],
            ),
            (
                &str_to_quants("AAE_EAAEE"),
                vec![
                    (vec![0b00000000_00000000_00000000_00000000u32], false, 0),
                    (vec![0b00100000_00100001_00010000_00100010u32], false, 0),
                    (vec![0b01000000_00100100_00010010_00100000u32], false, 0),
                    (vec![0b00101000_00111100_00111100_00000010u32], true, 0),
                    (vec![0b01100100_00100100_01010010_00100001u32], true, 0),
                ],
            ),
            (
                &str_to_quants("AAA_AEAEA"),
                vec![
                    (vec![0b00000000_00000000_00000000_00000000u32], false, 0),
                    (vec![0b00100000_00100000_00010000_00100010u32], false, 0),
                    (vec![0b00100000_00111100_00010000_00100010u32], false, 0),
                    (vec![0b00100000_00111100_00111100_00100010u32], true, 0x7fff),
                    (vec![0b00100000_00111100_11010011_00100010u32], true, 0x7fff),
                ],
            ),
            (
                &str_to_quants("EEE_EAAEE"),
                vec![
                    (
                        vec![0b00000000_00000000_00000000_00000000u32],
                        false,
                        0x7fff,
                    ),
                    (
                        vec![0b00100000_00100001_00010000_00100010u32],
                        false,
                        0x7fff,
                    ),
                    (
                        vec![0b01000000_00100100_00010010_00100000u32],
                        false,
                        0x7fff,
                    ),
                    (vec![0b00101000_00111100_00111100_00000010u32], true, 0),
                    (vec![0b01100100_00100100_01010010_00100001u32], true, 0),
                ],
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let defs = get_aggr_output_opencl_code_defs(32, 1, &quants);
            let mut builder = OpenCLBuilder::new(
                &device,
                Some(OPENCL_BUILDER_CONFIG_DEFAULT.group_len(Some(1))),
            );
            assert_eq!(builder.type_len(), 32);
            builder.user_defs(&defs);
            //println!("Defs: {}", defs);
            builder.add_with_config(
                "formula",
                circuit.clone(),
                CodeConfig::new()
                    .init_code(Some(INIT_OPENCL_CODE))
                    .aggr_output_code(Some(AGGR_OUTPUT_OPENCL_CODE))
                    .aggr_output_len(Some(1)),
            );
            let mut execs = builder.build().unwrap();
            println!("Run {}", i);
            for (j, (data, result, lidx)) in testcases.into_iter().enumerate() {
                let input = execs[0].new_data_from_vec(data);
                let output = execs[0].execute(&input, 0).unwrap().release();
                assert_eq!(result, (output[0] >> 15) != 0, "{} {}", i, j);
                assert_eq!(lidx, (output[0] & 0x7fff), "{} {}", i, j);
            }
        }
    }

    #[test]
    fn test_aggr_output_opencl_code_2() {
        let circuit = Circuit::<usize>::new(1, [], [(0, false)]).unwrap();
        let device = Device::new(*get_all_devices(CL_DEVICE_TYPE_GPU).unwrap().get(0).unwrap());
        for (i, (quants, group_len, reduce_len, testcases)) in [
            // 0
            (
                &str_to_quants("EE_EEA_EEEEE"),
                8,
                1,
                vec![
                    (vec![0, 0, 0, 0, 0, 0, 0, 0], vec![(false, Some(0x7fff))]),
                    (vec![0, 0, 0, 1, 0, 0, 0, 0], vec![(false, Some(0x7fff))]),
                    (vec![0, 0, 1, 1, 0, 0, 0, 0], vec![(true, Some(2))]),
                    (vec![0, 0, 0, 1, 1, 2, 0, 0], vec![(true, Some(4))]),
                    (vec![0, 0, 0, 1, 1, 2, 3, 5], vec![(true, Some(4))]),
                    (vec![0, 0, 0, 1, 1, 0, 3, 5], vec![(true, Some(6))]),
                ],
            ),
            // 1
            (
                &str_to_quants("AA_AAE_EEEEE"),
                8,
                1,
                vec![
                    (vec![1, 1, 1, 1, 1, 1, 1, 1], vec![(true, Some(0x7fff))]),
                    (vec![0, 1, 1, 0, 1, 0, 0, 1], vec![(true, Some(0x7fff))]),
                    (vec![0, 1, 0, 0, 1, 0, 0, 1], vec![(false, Some(2))]),
                    (vec![0, 1, 1, 0, 0, 0, 0, 1], vec![(false, Some(4))]),
                    (vec![0, 1, 1, 0, 0, 0, 0, 0], vec![(false, Some(4))]),
                    (vec![0, 1, 1, 0, 0, 1, 0, 0], vec![(false, Some(6))]),
                ],
            ),
            // 2
            (
                &str_to_quants("AE_EEA_EEEEE"),
                8,
                1,
                vec![
                    (vec![0, 0, 0, 0, 0, 0, 0, 0], vec![(false, None)]),
                    (vec![0, 0, 0, 1, 0, 0, 0, 0], vec![(false, None)]),
                    (vec![0, 0, 1, 1, 0, 0, 0, 0], vec![(true, None)]),
                ],
            ),
            // 3
            (
                &str_to_quants("EA_AAE_EEEEE"),
                8,
                1,
                vec![
                    (vec![1, 1, 1, 1, 1, 1, 1, 1], vec![(true, None)]),
                    (vec![0, 1, 1, 0, 1, 0, 0, 1], vec![(true, None)]),
                    (vec![0, 1, 0, 0, 1, 0, 0, 1], vec![(false, None)]),
                ],
            ),
            // 4
            (
                &str_to_quants("EE_EEA_EEEEE"),
                8,
                2,
                vec![(
                    vec![0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 3, 5],
                    vec![(false, Some(0x7fff)), (true, Some(6))],
                )],
            ),
            // 5: many OpenCL groups
            (
                &str_to_quants("EE_EEA_EEEEE"),
                8,
                5,
                vec![(
                    vec![
                        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 0, 0, 0,
                        0, 0, 1, 1, 2, 3, 5, 0, 0, 0, 1, 1, 0, 3, 5,
                    ],
                    vec![
                        (false, Some(0x7fff)),
                        (true, Some(2)),
                        (true, Some(4)),
                        (true, Some(4)),
                        (true, Some(6)),
                    ],
                )],
            ),
            // 6: various machine words quantifiers
            (
                &str_to_quants("EE_EEA_EAEAA"),
                8,
                1,
                vec![
                    (vec![0, 0, 0, 0, 0, 0, 0, 0], vec![(false, Some(0x7fff))]),
                    (
                        vec![
                            0xf02344f1, 0x000eaf00, 0x13f94fe1, 0x1ef94fe1, 0xfaa240af, 0xaee0a11a,
                            0xfa03441e, 0x123ff444,
                        ],
                        vec![(false, Some(0x7fff))],
                    ),
                    (
                        vec![
                            0xf02344f1, 0x000eaf00, 0x13f94fe1, 0x1ef94fe1, 0xfaaf40af, 0xaee0f1fa,
                            0xfa03441e, 0x123ff444,
                        ],
                        vec![(true, Some(4))],
                    ),
                ],
            ),
            // 7: start quantifier only in work
            (
                &str_to_quants("EEA_EEEEE"),
                8,
                1,
                vec![
                    (vec![0, 0, 0, 0, 0, 0, 0, 0], vec![(false, Some(0x7fff))]),
                    (vec![0, 0, 0, 1, 0, 0, 0, 0], vec![(false, Some(0x7fff))]),
                    (vec![0, 0, 1, 1, 0, 0, 0, 0], vec![(true, Some(2))]),
                    (vec![0, 0, 0, 1, 1, 2, 0, 0], vec![(true, Some(4))]),
                    (vec![0, 0, 0, 1, 1, 2, 3, 5], vec![(true, Some(4))]),
                    (vec![0, 0, 0, 1, 1, 0, 3, 5], vec![(true, Some(6))]),
                ],
            ),
            // 8: greater group
            (
                &str_to_quants("EAAEAE_EEEEE"),
                64,
                1,
                vec![
                    (
                        vec![
                            0, 1, 1, 0, 0, 1, 0, 1, // 0
                            0, 1, 0, 1, 0, 0, 0, 1, // 8
                            0, 1, 0, 0, 0, 0, 0, 1, // 16   lack of 1
                            1, 1, 0, 1, 1, 1, 0, 0, // 24
                            0, 0, 0, 0, 1, 0, 1, 0, // 0
                            0, 0, 0, 0, 1, 0, 0, 1, // 8
                            0, 1, 1, 0, 0, 0, 0, 0, // 16
                            0, 0, 0, 0, 1, 0, 0, 1, // 24
                        ],
                        vec![(true, Some(32))],
                    ),
                    (
                        vec![
                            0, 1, 1, 0, 0, 1, 0, 1, // 0
                            0, 1, 0, 1, 0, 0, 0, 1, // 8
                            1, 1, 0, 0, 0, 0, 1, 1, // 16   lack of 1
                            1, 1, 0, 1, 1, 1, 0, 0, // 24
                            1, 1, 0, 0, 0, 0, 1, 1, // 0    lacks in all eight-items
                            0, 0, 1, 1, 1, 1, 0, 0, // 8
                            1, 1, 0, 0, 1, 1, 0, 0, // 16
                            0, 0, 1, 1, 0, 0, 1, 1, // 24
                        ],
                        vec![(false, Some(0x7fff))],
                    ),
                ],
            ),
            // 9: greater group
            (
                &str_to_quants("AAEAEE_EEEEE"),
                64,
                1,
                vec![
                    (
                        vec![
                            0, 0, 1, 0, 1, 0, 0, 0, // 0
                            0, 0, 0, 0, 0, 0, 0, 0, // 8
                            0, 0, 0, 0, 0, 0, 0, 0, // 16
                            1, 0, 0, 0, 0, 0, 1, 0, // 24
                            1, 0, 0, 0, 0, 1, 0, 0, // 32
                            0, 0, 0, 0, 0, 0, 0, 0, // 40
                            1, 0, 0, 0, 0, 0, 0, 0, // 48
                            0, 0, 1, 0, 1, 0, 0, 0, // 56
                        ],
                        vec![(true, Some(0x7fff))],
                    ),
                    (
                        vec![
                            1, 1, 1, 1, 1, 1, 0, 1, // 0
                            1, 0, 1, 0, 1, 1, 1, 1, // 8
                            1, 1, 0, 1, 1, 1, 1, 1, // 16
                            1, 1, 1, 1, 1, 0, 1, 1, // 24
                            1, 1, 1, 1, 0, 0, 0, 0, // 32
                            1, 1, 1, 1, 0, 0, 0, 0, // 40
                            1, 1, 1, 1, 1, 1, 1, 1, // 48
                            1, 0, 1, 1, 1, 1, 1, 1, // 56
                        ],
                        vec![(false, Some(32))],
                    ),
                ],
            ),
            // 10: greater group
            (
                &str_to_quants("EAAEAE_EEEEE"),
                64,
                3,
                vec![(
                    vec![
                        0, 1, 1, 0, 0, 1, 0, 1, // 0
                        0, 1, 0, 1, 0, 0, 0, 1, // 8
                        0, 1, 0, 0, 0, 0, 0, 1, // 16   lack of 1
                        1, 1, 0, 1, 1, 1, 0, 0, // 24
                        0, 0, 0, 0, 1, 0, 1, 0, // 0
                        0, 0, 0, 0, 1, 0, 0, 1, // 8
                        0, 1, 1, 0, 0, 0, 0, 0, // 16
                        0, 0, 0, 0, 1, 0, 0, 1, // 24
                        0, 1, 1, 0, 0, 1, 0, 1, // 0
                        0, 1, 0, 1, 0, 0, 0, 1, // 8
                        1, 1, 0, 0, 0, 0, 1, 1, // 16   lack of 1
                        1, 1, 0, 1, 1, 1, 0, 0, // 24
                        1, 1, 0, 0, 0, 0, 1, 1, // 0    lacks in all eight-items
                        0, 0, 1, 1, 1, 1, 0, 0, // 8
                        1, 1, 0, 0, 1, 1, 0, 0, // 16
                        0, 0, 1, 1, 0, 0, 1, 1, // 24
                        0, 1, 1, 0, 0, 1, 0, 1, // 0
                        0, 1, 0, 1, 0, 0, 0, 1, // 8
                        0, 1, 0, 1, 0, 0, 0, 1, // 16
                        1, 1, 0, 1, 1, 1, 0, 0, // 24
                        0, 0, 0, 0, 1, 0, 1, 0, // 0
                        0, 0, 0, 0, 1, 0, 0, 1, // 8
                        0, 1, 1, 0, 0, 0, 0, 0, // 16
                        0, 0, 0, 0, 1, 0, 0, 1, // 24
                    ],
                    vec![(true, Some(32)), (false, Some(0x7fff)), (true, Some(0))],
                )],
            ),
            // 11: greater group
            (
                &str_to_quants("EEEAAEAA_EEEEE"),
                256,
                1,
                vec![
                    (
                        vec![
                            0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, // 0
                            0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, // 1
                            1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, // 2
                            1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, // 3
                            1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, // 4
                            1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 5
                            1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, // 6
                            0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, // 7
                            0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, // 8
                            0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, // 9
                            0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, // 10
                            0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, // 11
                            0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, // 12
                            0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, // 13
                            0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, // 14
                            0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, // 15
                        ],
                        vec![(true, Some(96))],
                    ),
                    (
                        vec![
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 0
                            1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 1
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, // 2
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 3
                            1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 4
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 5
                            1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, // 6
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 7
                            1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 8
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 9
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 10
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, // 11
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, // 12
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 13
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 14
                            0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 15
                        ],
                        vec![(false, None)],
                    ),
                ],
            ),
            // 0
            (
                &str_to_quants("EE_EEE_EEAAA"),
                8,
                1,
                vec![
                    (vec![0, 0, 0, 0, 0, 0, 0, 0], vec![(false, Some(0x7fff))]),
                    (vec![0, 0, 0, 1, 0, 0, 0, 0], vec![(false, Some(0x7fff))]),
                    (vec![0, 0, 0xff00, 0, 0, 0, 0, 0], vec![(true, Some(2))]),
                    (vec![0, 0, 0, 1, 0xff0000, 111, 0, 0], vec![(true, Some(4))]),
                    (
                        vec![0, 0, 0, 1, 0xfe00, 0x1ff10, 3, 5],
                        vec![(true, Some(5))],
                    ),
                    (
                        vec![0, 0, 0, 0xef, 0xee, 0, 0xeeff10, 5],
                        vec![(true, Some(6))],
                    ),
                    (
                        vec![0, 0, 0, 0xef, 0xee, 0, 0xeeff100, 5],
                        vec![(false, Some(0x7fff))],
                    ),
                ],
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let defs = get_aggr_output_opencl_code_defs(32, group_len, &quants);
            let mut builder = OpenCLBuilder::new(
                &device,
                Some(OPENCL_BUILDER_CONFIG_DEFAULT.group_len(Some(group_len))),
            );
            assert_eq!(builder.type_len(), 32);
            builder.user_defs(&defs);
            //println!("Defs: {}", defs);
            builder.add_with_config(
                "formula",
                circuit.clone(),
                CodeConfig::new()
                    .init_code(Some(INIT_OPENCL_CODE))
                    .aggr_output_code(Some(AGGR_OUTPUT_OPENCL_CODE))
                    .aggr_output_len(Some((reduce_len + 1) >> 1)),
            );
            let mut execs = builder.build().unwrap();
            println!("Run {}", i);
            for (j, (data, results)) in testcases.into_iter().enumerate() {
                assert_eq!(data.len(), group_len * reduce_len);
                let input = execs[0].new_data_from_vec(data);
                let output = execs[0].execute(&input, 0).unwrap().release();
                assert_eq!(results.len(), reduce_len);
                for k in 0..reduce_len {
                    let reduced = (output[k >> 1] >> ((k & 1) << 4)) & 0xffff;
                    let (result, lidx) = results[k];
                    assert_eq!(result, (reduced >> 15) != 0, "{} {}", i, j);
                    if let Some(lidx) = lidx {
                        assert_eq!(lidx, (reduced & 0x7fff), "{} {}", i, j);
                    }
                }
            }
        }
    }

    #[test]
    fn test_get_final_results_from_cpu_outputs() {
        assert_eq!(
            (None, false),
            get_final_results_from_cpu_outputs(
                256,
                18,
                &str_to_quants("AAEE_AAEEAEAAEE_EEAEAEEA"),
                &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][..]
            )
        );
        assert_eq!(
            (None, true),
            get_final_results_from_cpu_outputs(
                256,
                18,
                &str_to_quants("AAEE_AAEEAEAAEE_EEAEAEEA"),
                &[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0][..]
            )
        );
        assert_eq!(
            (None, false),
            get_final_results_from_cpu_outputs(
                256,
                18,
                &str_to_quants("EEEE_AAEEAEAAEE_EEAEAEEA"),
                &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][..]
            )
        );
        // with first quantifier
        assert_eq!(
            (
                Some(FinalResult {
                    reversed: false,
                    solution_bits: 3,
                    solution: None
                }),
                false
            ),
            get_final_results_from_cpu_outputs(
                256,
                18,
                &str_to_quants("EEEE_EEEAAEAAEE_EEAEAEEA"),
                &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][..]
            )
        );
        assert_eq!(
            (
                Some(FinalResult {
                    reversed: true,
                    solution_bits: 3,
                    solution: None
                }),
                true
            ),
            get_final_results_from_cpu_outputs(
                256,
                18,
                &str_to_quants("AAAA_AAAEEEAAEE_EEAEAEEA"),
                &[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0][..]
            )
        );
        assert_eq!(
            (
                Some(FinalResult {
                    reversed: false,
                    solution_bits: 3,
                    solution: Some(0)
                }),
                true
            ),
            get_final_results_from_cpu_outputs(
                256,
                18,
                &str_to_quants("EEEE_EEEAAEAAEE_EEAEAEEA"),
                &[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0][..]
            )
        );
        assert_eq!(
            (
                Some(FinalResult {
                    reversed: false,
                    solution_bits: 3,
                    solution: Some(3)
                }),
                true
            ),
            get_final_results_from_cpu_outputs(
                256,
                18,
                &str_to_quants("EEEE_EEEAAEAAEE_EEAEAEEA"),
                &[0, 0, 11, 770, 0, 110500, 0, 17, 1, 1, 6 << 7, 0][..]
            )
        );
        assert_eq!(
            (
                Some(FinalResult {
                    reversed: true,
                    solution_bits: 6,
                    solution: Some(37)
                }),
                false
            ),
            get_final_results_from_cpu_outputs(
                256,
                18,
                &str_to_quants("AAAA_AAAAAAEAEE_EEAEAEEA"),
                &[0, 49, 0, 0, 456, 21200, 0, 133, 0, 1, 41 << 4, 0][..]
            )
        );
        assert_eq!(
            (
                Some(FinalResult {
                    reversed: false,
                    solution_bits: 10,
                    solution: Some(902)
                }),
                true
            ),
            get_final_results_from_cpu_outputs(
                256,
                18,
                &str_to_quants("EEEE_EEEEEEEEEE_AEAEAEEA"),
                &[0, 0, 11, 770, 0, 110500, 0, 17, 1, 1, 391, 0][..]
            )
        );
        assert_eq!(
            (
                Some(FinalResult {
                    reversed: false,
                    solution_bits: 15,
                    solution: Some(0b00011010_1110000110)
                }),
                true
            ),
            get_final_results_from_cpu_outputs(
                256,
                18,
                &str_to_quants("EEEE_EEEEEEEEEE_EEEEEAAA"),
                &[0, 0, 0xff000000, 770, 0, 110500, 0, 17, 1, 1, 391, 0][..]
            )
        );
        assert_eq!(
            (
                Some(FinalResult {
                    reversed: true,
                    solution_bits: 15,
                    solution: Some(0b1001_1101000110)
                }),
                false
            ),
            get_final_results_from_cpu_outputs(
                256,
                18,
                &str_to_quants("AAAA_AAAAAAAAAA_AAAAAEEE"),
                &[
                    0x02030607, 0x02030607, 0x02030607, 0x02030607, 0x02000607, 0x02030607,
                    0x02030607, 0x02030607, 0, 1, 395, 0
                ][..]
            )
        );
        assert_eq!(
            (
                Some(FinalResult {
                    reversed: false,
                    solution_bits: 18,
                    solution: Some(0b11011010_1110000110)
                }),
                true
            ),
            get_final_results_from_cpu_outputs(
                256,
                18,
                &str_to_quants("EEEE_EEEEEEEEEE_EEEEEEEE"),
                &[0, 0, 0xf8000000, 770, 0, 110500, 0, 17, 1, 1, 391, 0][..]
            )
        );
    }

    #[test]
    fn test_opencl_quant_reducer() {
        use std::iter;
        let device = Device::new(*get_all_devices(CL_DEVICE_TYPE_GPU).unwrap().get(0).unwrap());
        let context = Arc::new(Context::from_device(&device).unwrap());
        #[allow(deprecated)]
        let cmd_queue =
            Arc::new(unsafe { CommandQueue::create(&context, device.id(), 0).unwrap() });
        for (
            i,
            (reduce_start_bit, reduce_end_bit, init_group_len_bits, quants, group_len, testcases),
        ) in [
            // 0
            (
                2,
                2,
                4,
                &str_to_quants("AE__EEAE_AAEAA"),
                64,
                vec![
                    (vec![0u16], (Option::<FinalResult>::None, false)),
                    (vec![0x8000], (None, true)),
                ],
            ),
            // 1
            (
                2,
                2,
                4,
                &str_to_quants("EE__EEAE_AAEAA"),
                64,
                vec![
                    (vec![0u16], (None, false)),
                    (
                        vec![0x800a],
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 2,
                                solution: Some(1),
                            }),
                            true,
                        ),
                    ),
                    (
                        vec![0x8004],
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 2,
                                solution: Some(2),
                            }),
                            true,
                        ),
                    ),
                ],
            ),
            // 2
            (
                2,
                2,
                4,
                &str_to_quants("EE__EEEE_EEEAA"),
                64,
                vec![
                    (vec![0u16], (None, false)),
                    (
                        vec![0x800a],
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 4,
                                solution: Some(5),
                            }),
                            true,
                        ),
                    ),
                ],
            ),
            // 3
            (
                2,
                2,
                4,
                &str_to_quants("AA__AAEE_AAEAA"),
                64,
                vec![
                    (vec![0x8000], (None, true)),
                    (
                        vec![0x000a],
                        (
                            Some(FinalResult {
                                reversed: true,
                                solution_bits: 2,
                                solution: Some(1),
                            }),
                            false,
                        ),
                    ),
                ],
            ),
            // 4
            (
                2,
                2,
                4,
                &str_to_quants("AA__AAAA_AAEAA"),
                64,
                vec![
                    (vec![0x8000], (None, true)),
                    (
                        vec![0x000a],
                        (
                            Some(FinalResult {
                                reversed: true,
                                solution_bits: 4,
                                solution: Some(5),
                            }),
                            false,
                        ),
                    ),
                ],
            ),
            // 5
            (
                2,
                5,
                4,
                &str_to_quants("AE_AAA_EEAE_AAEAA"),
                64,
                vec![
                    (vec![0u16; 8], (Option::<FinalResult>::None, false)),
                    (vec![0x8000u16; 8], (None, true)),
                ],
            ),
            // 6
            (
                2,
                5,
                4,
                &str_to_quants("AE_AEE_EEAE_AAEAA"),
                64,
                vec![
                    (vec![0u16; 8], (None, false)),
                    (vec![0, 0x8000, 0, 0, 0, 0, 0x8000, 0], (None, true)),
                ],
            ),
            // 7
            (
                2,
                5,
                4,
                &str_to_quants("EA_EAA_EEAE_AAEAA"),
                64,
                vec![
                    (vec![0u16; 8], (None, false)),
                    (
                        vec![0, 0, 0, 0, 0x8000, 0x8000, 0x8000, 0x8000],
                        (None, true),
                    ),
                ],
            ),
            // 8
            (
                2,
                5,
                4,
                &str_to_quants("EE_EEA_EEAE_AAEAA"),
                64,
                vec![
                    (vec![0u16; 8], (None, false)),
                    (vec![0, 0, 0x8000, 0, 0x0, 0x8000, 0, 0], (None, false)),
                    (
                        vec![0, 0, 0, 0x8000, 0x8000, 0x8000, 0, 0],
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 2,
                                solution: Some(1),
                            }),
                            true,
                        ),
                    ),
                    (
                        vec![0, 0, 0x8000, 0x8000, 0x0, 0x8000, 0, 0],
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 2,
                                solution: Some(2),
                            }),
                            true,
                        ),
                    ),
                ],
            ),
            // 9
            (
                2,
                5,
                4,
                &str_to_quants("AA_AAE_EEAE_AAEAA"),
                64,
                vec![
                    (vec![0x8000; 8], (None, true)),
                    (
                        vec![0x8000, 0, 0x8000, 0, 0x8000, 0, 0, 0x8000],
                        (None, true),
                    ),
                    (
                        vec![0, 0x8000, 0, 0x8000, 0, 0, 0x8000, 0],
                        (
                            Some(FinalResult {
                                reversed: true,
                                solution_bits: 2,
                                solution: Some(1),
                            }),
                            false,
                        ),
                    ),
                    (
                        vec![0, 0x8000, 0, 0, 0, 0x8000, 0, 0x8000],
                        (
                            Some(FinalResult {
                                reversed: true,
                                solution_bits: 2,
                                solution: Some(2),
                            }),
                            false,
                        ),
                    ),
                ],
            ),
            // 10
            (
                2,
                5,
                4,
                &str_to_quants("EE_EEE_EEAE_AAEAA"),
                64,
                vec![
                    (vec![0u16; 8], (None, false)),
                    (
                        vec![0, 0, 0, 0x8003, 0, 0, 0, 0],
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 5,
                                solution: Some(0b00110),
                            }),
                            true,
                        ),
                    ),
                    (
                        vec![0, 0, 0, 0x800c, 0, 0, 0, 0],
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 5,
                                solution: Some(0b11110),
                            }),
                            true,
                        ),
                    ),
                    (
                        vec![0, 0, 0x8005, 0x8000, 0x0, 0x8000, 0, 0],
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 5,
                                solution: Some(0b10010),
                            }),
                            true,
                        ),
                    ),
                ],
            ),
            // 11
            (
                2,
                5,
                4,
                &str_to_quants("EE_EEE_EEEE_EEEAA"),
                64,
                vec![
                    (vec![0u16; 8], (None, false)),
                    (
                        vec![0, 0, 0, 0x8003, 0, 0, 0, 0],
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 7,
                                solution: Some(0b1100110),
                            }),
                            true,
                        ),
                    ),
                    (
                        vec![0, 0, 0x8009, 0x8000, 0x0, 0x8000, 0, 0],
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 7,
                                solution: Some(0b1001010),
                            }),
                            true,
                        ),
                    ),
                ],
            ),
            // 12
            (
                2,
                5,
                4,
                &str_to_quants("AA_AAA_AAAA_AAAEE"),
                64,
                vec![
                    (vec![0x8000; 8], (None, true)),
                    (
                        vec![0x8000, 0x8000, 0x8000, 0x4, 0x8000, 0x8000, 0x8000, 0x8000],
                        (
                            Some(FinalResult {
                                reversed: true,
                                solution_bits: 7,
                                solution: Some(0b10110),
                            }),
                            false,
                        ),
                    ),
                    (
                        vec![0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 7, 9, 0x8000],
                        (
                            Some(FinalResult {
                                reversed: true,
                                solution_bits: 7,
                                solution: Some(0b1110101),
                            }),
                            false,
                        ),
                    ),
                ],
            ),
            // 13
            (
                2,
                8,
                7,
                &str_to_quants("EA_EEEAEE_AAAEAAE_EEAEE"),
                64,
                vec![
                    (vec![0u16; 64], (None, false)),
                    (
                        vec![
                            0, 0, 0, 0, 0, 0, 0, 0, // 0
                            0, 0, 0, 0x8000, 0, 0, 0, 0, // 1
                            0, 0, 0, 0, 0, 0x8000, 0, 0, // 2
                            0x8000, 0x8000, 0, 0x8000, 0, 0, 0, 0, // 3
                            0, 0, 0, 0, 0x8000, 0x8000, 0, 0x8000, // 4
                            0, 0, 0, 0, 0x8000, 0x8000, 0, 0, // 5
                            0, 0, 0, 0, 0, 0, 0x8000, 0, // 6
                            0, 0, 0, 0x8000, 0, 0, 0, 0, // 7
                        ],
                        (None, false),
                    ),
                    (
                        vec![
                            0, 0, 0, 0, 0, 0, 0, 0, // 0
                            0, 0, 0, 0x8000, 0, 0, 0, 0, // 1
                            0, 0, 0, 0, 0, 0x8000, 0, 0, // 2
                            0x8000, 0x8000, 0, 0x8000, 0, 0, 0, 0, // 3
                            0, 0, 0, 0, 0x8000, 0x8000, 0, 0x8000, // 4
                            0, 0x8000, 0, 0, 0x8000, 0x8000, 0, 0, // 5
                            0, 0, 0, 0, 0, 0, 0x8000, 0, // 6
                            0, 0, 0, 0x8000, 0, 0, 0, 0, // 7
                        ],
                        (None, true),
                    ),
                ],
            ),
            // 14
            (
                2,
                8,
                7,
                &str_to_quants("AE_AAAEAA_EEAEEAE_AAEAA"),
                64,
                vec![
                    (vec![0x8000; 64], (None, true)),
                    (
                        vec![
                            0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
                            0x8000, // 0
                            0, 0, 0x8000, 0, 0x8000, 0x8000, 0x8000, 0x8000, // 1
                            0x8000, 0x8000, 0x8000, 0x8000, 0, 0, 0x8000, 0x8000, // 2
                            0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0, 0x8000, 0x8000, // 3
                            0x8000, 0, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, // 4
                            0x8000, 0, 0, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, // 5
                            0, 0, 0, 0, 0x8000, 0x8000, 0x8000, 0x8000, // 6
                            0x8000, 0x8000, 0x8000, 0x8000, 0, 0, 0, 0, // 6
                        ],
                        (None, true),
                    ),
                    (
                        vec![
                            0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
                            0x8000, // 0
                            0, 0, 0x8000, 0, 0x8000, 0x8000, 0x8000, 0x8000, // 1
                            0x8000, 0x8000, 0x8000, 0, 0, 0, 0x8000, 0x8000, // 2
                            0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0, 0x8000, 0x8000, // 3
                            0x8000, 0, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, // 4
                            0x8000, 0, 0, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, // 5
                            0, 0, 0, 0, 0x8000, 0x8000, 0x8000, 0x8000, // 6
                            0x8000, 0x8000, 0x8000, 0x8000, 0, 0, 0, 0, // 7
                        ],
                        (None, false),
                    ),
                ],
            ),
            // 15
            (
                2,
                8,
                7,
                &str_to_quants("EE_EEEEAA_EEAEEAE_AAEAA"),
                64,
                vec![
                    (vec![0u16; 64], (None, false)),
                    (
                        vec![
                            0, 0, 0, 0, 0, 0, 0, 0, // 0
                            0, 0, 0, 0x8000, 0, 0, 0, 0, // 1
                            0, 0, 0, 0, 0, 0x8000, 0, 0, // 2
                            0x8000, 0x8000, 0, 0x8000, 0, 0, 0, 0, // 3
                            0, 0, 0, 0, 0x8000, 0x8000, 0, 0x8000, // 4
                            0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0, 0, // 5
                            0, 0, 0, 0, 0, 0, 0x8000, 0, // 6
                            0, 0, 0, 0x8000, 0, 0, 0, 0, // 7
                        ],
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 4,
                                solution: Some(0b0101),
                            }),
                            true,
                        ),
                    ),
                    (
                        vec![
                            0, 0, 0, 0, 0, 0, 0, 0, // 0
                            0, 0, 0, 0x8000, 0, 0, 0, 0, // 1
                            0, 0, 0, 0, 0, 0x8000, 0, 0, // 2
                            0, 0, 0, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, // 3
                            0, 0, 0, 0, 0x8000, 0x8000, 0, 0x8000, // 4
                            0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0, 0, // 5
                            0, 0, 0, 0, 0, 0, 0x8000, 0, // 6
                            0, 0, 0, 0x8000, 0, 0, 0, 0, // 7
                        ],
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 4,
                                solution: Some(0b1110),
                            }),
                            true,
                        ),
                    ),
                ],
            ),
            // 16
            (
                2,
                8,
                7,
                &str_to_quants("AA_AAAAAE_EEAEEAE_AAEAA"),
                64,
                vec![
                    (vec![0x8000; 64], (None, true)),
                    (
                        vec![
                            0x8000, 0, 0x8000, 0, 0, 0x8000, 0x8000, 0, // 0
                            0x8000, 0x8000, 0, 0x8000, 0x8000, 0, 0x8000, 0, // 1
                            0, 0x8000, 0, 0x8000, 0, 0x8000, 0, 0x8000, // 2
                            0, 0x8000, 0, 0, 0x8000, 0, 0x8000, 0, // 3: this point
                            0x8000, 0x8000, 0, 0x8000, 0x8000, 0x8000, 0x8000, 0, // 4
                            0x8000, 0x8000, 0, 0x8000, 0x8000, 0x8000, 0, 0x8000, // 5
                            0x8000, 0, 0x8000, 0x8000, 0x8000, 0, 0x8000, 0x8000, // 6
                            0x8000, 0, 0x8000, 0x8000, 0x8000, 0, 0x8000, 0, // 7
                        ],
                        (
                            Some(FinalResult {
                                reversed: true,
                                solution_bits: 5,
                                solution: Some(0b10110),
                            }),
                            false,
                        ),
                    ),
                    (
                        vec![
                            0x8000, 0, 0x8000, 0, 0, 0x8000, 0x8000, 0, // 0
                            0x8000, 0x8000, 0, 0x8000, 0x8000, 0, 0x8000, 0, // 1
                            0, 0x8000, 0, 0x8000, 0, 0x8000, 0, 0x8000, // 2
                            0, 0x8000, 0x8000, 0, 0x8000, 0, 0x8000, 0, // 3
                            0x8000, 0x8000, 0, 0x8000, 0x8000, 0x8000, 0x8000, 0, // 4
                            0x8000, 0x8000, 0, 0x8000, 0, 0, 0, 0x8000, // 5: this point
                            0x8000, 0, 0x8000, 0x8000, 0x8000, 0, 0x8000, 0x8000, // 6
                            0x8000, 0, 0, 0, 0x8000, 0, 0x8000, 0, // 7
                        ],
                        (
                            Some(FinalResult {
                                reversed: true,
                                solution_bits: 5,
                                solution: Some(0b01101),
                            }),
                            false,
                        ),
                    ),
                ],
            ),
            // 17
            (
                2,
                8,
                7,
                &str_to_quants("EE_EEEEEE_EEEAAAE_AAEAA"),
                64,
                vec![
                    (vec![0u16; 64], (None, false)),
                    (
                        iter::repeat(0)
                            .take(22)
                            .chain(iter::once(0x8060))
                            .chain(iter::repeat(0).take(64 - 22 - 1))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 9,
                                solution: Some(0b011_011010),
                            }),
                            true,
                        ),
                    ),
                    (
                        iter::repeat(0)
                            .take(43)
                            .chain(iter::once(0x8030))
                            .chain(iter::repeat(0).take(64 - 43 - 1))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 9,
                                solution: Some(0b110_110101),
                            }),
                            true,
                        ),
                    ),
                    (
                        iter::repeat(0)
                            .take(17)
                            .chain(iter::once(0x8040))
                            .chain(iter::repeat(0).take(21))
                            .chain(iter::once(0x8010))
                            .chain(iter::repeat(0).take(64 - 21 - 1 - 17 - 1))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 9,
                                solution: Some(0b001_100010),
                            }),
                            true,
                        ),
                    ),
                ],
            ),
            // 18
            (
                2,
                8,
                7,
                &str_to_quants("EE_EEEEEE_EEEEEEE_EEAEA"),
                64,
                vec![
                    (vec![0u16; 64], (None, false)),
                    (
                        iter::repeat(0)
                            .take(22)
                            .chain(iter::once(0x802d))
                            .chain(iter::repeat(0).take(64 - 22 - 1))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 13,
                                solution: Some(0b1011010_011010),
                            }),
                            true,
                        ),
                    ),
                    (
                        iter::repeat(0)
                            .take(43)
                            .chain(iter::once(0x805a))
                            .chain(iter::repeat(0).take(64 - 43 - 1))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 13,
                                solution: Some(0b0101101_110101),
                            }),
                            true,
                        ),
                    ),
                ],
            ),
            // 19
            (
                2,
                8,
                7,
                &str_to_quants("AA_AAAAAA_AAAAEEA_EAEAA"),
                64,
                vec![
                    (vec![0x8000; 64], (None, true)),
                    (
                        iter::repeat(0x8000)
                            .take(27)
                            .chain(iter::once(0x68))
                            .chain(iter::repeat(0x8000).take(64 - 27 - 1))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: true,
                                solution_bits: 10,
                                solution: Some(0b01011_110110),
                            }),
                            false,
                        ),
                    ),
                ],
            ),
            // 20
            (
                2,
                11,
                7,
                &str_to_quants("EE_EEE_EEEEAA_EEAEEAE_AAEAA"),
                64,
                vec![
                    (vec![0u16; 64 * 8], (None, false)),
                    (
                        iter::repeat(0)
                            .take(141)
                            .chain(iter::once(0x8022))
                            .chain(iter::repeat(0).take(64 * 8 - 141 - 1))
                            .collect::<Vec<_>>(),
                        (None, false),
                    ),
                    (
                        iter::repeat(0)
                            .take(21 * 8)
                            .chain(iter::repeat(0x8022).take(8))
                            .chain(iter::repeat(0).take(64 * 8 - 21 * 8 - 8))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 7,
                                solution: Some(0b101_010),
                            }),
                            true,
                        ),
                    ),
                    (
                        iter::repeat(0)
                            .take(46 * 8)
                            .chain(iter::repeat(0x8022).take(8))
                            .chain(iter::repeat(0).take(64 * 8 - 46 * 8 - 8))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 7,
                                solution: Some(0b011_101),
                            }),
                            true,
                        ),
                    ),
                ],
            ),
            // 21
            (
                2,
                11,
                7,
                &str_to_quants("EE_EEE_EEEEEE_EEEEEAA_AAEAA"),
                64,
                vec![
                    (vec![0u16; 64 * 8], (None, false)),
                    (
                        iter::repeat(0)
                            .take(141)
                            .chain(iter::once(0x806c))
                            .chain(iter::repeat(0).take(64 * 8 - 141 - 1))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 14,
                                solution: Some(0b11011_101100_010),
                            }),
                            true,
                        ),
                    ),
                    (
                        iter::repeat(0)
                            .take(184)
                            .chain(iter::once(0x8038))
                            .chain(iter::repeat(0).take(173))
                            .chain(iter::once(0x8014))
                            .chain(iter::repeat(0).take(64 * 8 - 173 - 1 - 184 - 1))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 14,
                                solution: Some(0b1110_000111_010),
                            }),
                            true,
                        ),
                    ),
                ],
            ),
            // 22
            (
                2,
                11,
                7,
                &str_to_quants("AA_AAA_AAAAAA_AAAAEEA_AAEAA"),
                64,
                vec![
                    (vec![0x8000; 64 * 8], (None, true)),
                    (
                        iter::repeat(0x8000)
                            .take(141)
                            .chain(iter::once(0x6e))
                            .chain(iter::repeat(0x8000).take(64 * 8 - 141 - 1))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: true,
                                solution_bits: 13,
                                solution: Some(0b1011_101100_010),
                            }),
                            false,
                        ),
                    ),
                    (
                        iter::repeat(0x8000)
                            .take(188)
                            .chain(iter::once(0x38))
                            .chain(iter::repeat(0x8000).take(173))
                            .chain(iter::once(0x14))
                            .chain(iter::repeat(0x8000).take(64 * 8 - 173 - 1 - 188 - 1))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: true,
                                solution_bits: 13,
                                solution: Some(0b1110_001111_010),
                            }),
                            false,
                        ),
                    ),
                ],
            ),
            // 23
            (
                2,
                11,
                7,
                &str_to_quants("EE_EEA_AAEEAA_EEAEEAE_AAEAA"),
                64,
                vec![
                    (vec![0u16; 64 * 8], (None, false)),
                    (
                        iter::repeat(0)
                            .take(141)
                            .chain(iter::once(0x8022))
                            .chain(iter::repeat(0).take(64 * 8 - 141 - 1))
                            .collect::<Vec<_>>(),
                        (None, false),
                    ),
                    (
                        iter::repeat(0)
                            .take(128)
                            .chain(
                                iter::repeat([
                                    0, 0, 0x8000, 0, 0, 0, 0, 0, 0x8000, 0x8000, 0x8000, 0x8000, 0,
                                    0, 0, 0,
                                ])
                                .take(8)
                                .flatten(),
                            )
                            .chain(iter::repeat(0).take(128 * 2))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 2,
                                solution: Some(2),
                            }),
                            true,
                        ),
                    ),
                    (
                        iter::repeat(0)
                            .take(128 * 2)
                            .chain(
                                iter::repeat([
                                    0, 0, 0x8000, 0, 0, 0, 0, 0, 0x8000, 0x8000, 0x8000, 0x8000, 0,
                                    0, 0, 0,
                                ])
                                .take(8)
                                .flatten(),
                            )
                            .chain(iter::repeat(0).take(128))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 2,
                                solution: Some(1),
                            }),
                            true,
                        ),
                    ),
                ],
            ),
            // 24
            (
                2,
                11,
                7,
                &str_to_quants("AA_AAA_AAAAAA_EAEAAEA_AAEAA"),
                64,
                vec![
                    (vec![0x8000; 64 * 8], (None, true)),
                    (
                        iter::repeat(0x8000)
                            .take(141)
                            .chain(iter::once(0x6e))
                            .chain(iter::repeat(0x8000).take(64 * 8 - 141 - 1))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: true,
                                solution_bits: 9,
                                solution: Some(0b101100_010),
                            }),
                            false,
                        ),
                    ),
                ],
            ),
            // 25
            (
                2,
                11,
                7,
                &str_to_quants("EE_EEE_AAEEAA_EEAEEAE_AAEAA"),
                64,
                vec![
                    (vec![0u16; 64 * 8], (None, false)),
                    (
                        iter::repeat(0)
                            .take(141)
                            .chain(iter::once(0x8022))
                            .chain(iter::repeat(0).take(64 * 8 - 141 - 1))
                            .collect::<Vec<_>>(),
                        (None, false),
                    ),
                    (
                        iter::repeat(0)
                            .take(64 * 6)
                            .chain(
                                iter::repeat([
                                    0, 0, 0x8000, 0, 0, 0, 0, 0, 0x8000, 0x8000, 0x8000, 0x8000, 0,
                                    0, 0, 0,
                                ])
                                .take(4)
                                .flatten(),
                            )
                            .chain(iter::repeat(0).take(64))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 3,
                                solution: Some(3),
                            }),
                            true,
                        ),
                    ),
                ],
            ),
            // 26
            (
                2,
                14,
                7,
                &str_to_quants("AE_AAAEAA_AEAEEA_EEAEEAE_AAEAA"),
                64,
                vec![
                    (vec![0u16; 64 * 64], (None, false)),
                    (
                        (0..64)
                            .map(|i| {
                                if (i & 4) == 0 {
                                    (0..64)
                                        .map(|j| {
                                            (((0x300c00000000c003u64.rotate_left(i * 32) >> j) & 1)
                                                << 15)
                                                as u16
                                        })
                                        .collect::<Vec<_>>()
                                } else {
                                    vec![0; 64]
                                }
                            })
                            .flatten()
                            .collect(),
                        (None, true),
                    ),
                ],
            ),
            // 27
            (
                2,
                14,
                7,
                &str_to_quants("EE_EEEEAA_AEAEEA_EEAEEAE_AAEAA"),
                64,
                vec![
                    (vec![0u16; 64 * 64], (None, false)),
                    (
                        iter::repeat(0)
                            .take(256 * 5)
                            .chain(
                                vec![
                                    (0..64)
                                        .map(|j| {
                                            (((0x300c00000000c003u64 >> j) & 1) << 15) as u16
                                        })
                                        .collect::<Vec<_>>();
                                    4
                                ]
                                .into_iter()
                                .flatten(),
                            )
                            .chain(iter::repeat(0).take(64 * 64 - 256 * 5 - 256))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 4,
                                solution: Some(0b1010),
                            }),
                            true,
                        ),
                    ),
                    (
                        iter::repeat(0)
                            .take(256 * 11)
                            .chain(
                                vec![
                                    (0..64)
                                        .map(|j| {
                                            (((0x300c00000000c003u64 >> j) & 1) << 15) as u16
                                        })
                                        .collect::<Vec<_>>();
                                    4
                                ]
                                .into_iter()
                                .flatten(),
                            )
                            .chain(iter::repeat(0).take(64 * 64 - 256 * 11 - 256))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 4,
                                solution: Some(0b1101),
                            }),
                            true,
                        ),
                    ),
                ],
            ),
            // 28
            (
                2,
                14,
                7,
                &str_to_quants("EE_EEEEEE_AEAEEA_EEAEEAE_AAEAA"),
                64,
                vec![
                    (vec![0u16; 64 * 64], (None, false)),
                    (
                        iter::repeat(0)
                            .take(64 * 13)
                            .chain(
                                (0..64)
                                    .map(|j| (((0x300c00000000c003u64 >> j) & 1) << 15) as u16)
                                    .collect::<Vec<_>>(),
                            )
                            .chain(iter::repeat(0).take(64 * 12))
                            .chain(
                                (0..64)
                                    .map(|j| (((0x300c00000000c003u64 >> j) & 1) << 15) as u16)
                                    .collect::<Vec<_>>(),
                            )
                            .chain(iter::repeat(0).take(64 * (64 - 13 - 2 - 12)))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 6,
                                solution: Some(0b101100),
                            }),
                            true,
                        ),
                    ),
                    (
                        iter::repeat(0)
                            .take(64 * 50)
                            .chain(
                                (0..64)
                                    .map(|j| (((0x300c00000000c003u64 >> j) & 1) << 15) as u16)
                                    .collect::<Vec<_>>(),
                            )
                            .chain(iter::repeat(0).take(64 * 64 - 64 * 50 - 64))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 6,
                                solution: Some(0b010011),
                            }),
                            true,
                        ),
                    ),
                ],
            ),
            // 29
            (
                2,
                14,
                7,
                &str_to_quants("EE_EEEEEE_EEEEAE_EEAEEAE_AAEAA"),
                64,
                vec![
                    (vec![0u16; 64 * 64], (None, false)),
                    (
                        iter::repeat(0)
                            .take(379 * 4)
                            .chain([0, 0x8000, 0x8000, 0].into_iter())
                            .chain(iter::repeat(0).take(4 * (1024 - 379 - 1)))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 10,
                                solution: Some(0b1101_111010),
                            }),
                            true,
                        ),
                    ),
                    (
                        iter::repeat(0)
                            .take(999 * 4)
                            .chain([0, 0x8000, 0x8000, 0].into_iter())
                            .chain(iter::repeat(0).take(4 * (1024 - 999 - 1)))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 10,
                                solution: Some(0b1110_011111),
                            }),
                            true,
                        ),
                    ),
                ],
            ),
            // 30
            (
                2,
                14,
                7,
                &str_to_quants("EE_EEEEEE_EEEEEE_AAAEEAE_AAEAA"),
                64,
                vec![
                    (vec![0u16; 64 * 64], (None, false)),
                    (
                        iter::repeat(0)
                            .take(2631)
                            .chain(iter::once(0x807f))
                            .chain(iter::repeat(0).take(4096 - 2631 - 1))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 12,
                                solution: Some(0b111000_100101),
                            }),
                            true,
                        ),
                    ),
                    (
                        iter::repeat(0)
                            .take(915)
                            .chain(iter::once(0x807f))
                            .chain(iter::repeat(0).take(4096 - 915 - 1))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 12,
                                solution: Some(0b110010_011100),
                            }),
                            true,
                        ),
                    ),
                ],
            ),
            // 31
            (
                2,
                14,
                7,
                &str_to_quants("EE_EEEEEE_EEEEEE_EEEEEAE_AAEAA"),
                64,
                vec![
                    (vec![0u16; 64 * 64], (None, false)),
                    (
                        iter::repeat(0)
                            .take(2276)
                            .chain(iter::once(0x805c))
                            .chain(iter::repeat(0).take(4096 - 2276 - 1))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 17,
                                solution: Some(0b11101_001001_110001),
                            }),
                            true,
                        ),
                    ),
                    (
                        iter::repeat(0)
                            .take(877)
                            .chain(iter::once(0x8034))
                            .chain(iter::repeat(0).take(4096 - 877 - 1))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 17,
                                solution: Some(0b10110_101101_101100),
                            }),
                            true,
                        ),
                    ),
                ],
            ),
            // 32
            (
                2,
                14,
                7,
                &str_to_quants("EE_EEEEEE_EEEEEE_EEEEEEE_EAEAA"),
                64,
                vec![
                    (vec![0u16; 64 * 64], (None, false)),
                    (
                        iter::repeat(0)
                            .take(2276)
                            .chain(iter::once(0x805d))
                            .chain(iter::repeat(0).take(4096 - 2276 - 1))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 19,
                                solution: Some(0b1011101_001001_110001),
                            }),
                            true,
                        ),
                    ),
                ],
            ),
            // 33
            (
                2,
                17,
                4,
                &str_to_quants("AE_AAE_AAAEAA_AEAEEA_EEAE_AAEAA"),
                64,
                vec![
                    (vec![0u16; 4096 * 8], (None, false)),
                    (
                        (0..8)
                            .map(|h| {
                                if (h & 1) == 0 {
                                    (0..64)
                                        .map(|i| {
                                            if (i & 4) == 0 {
                                                (0..64)
                                                    .map(|j| {
                                                        (((0x300c00000000c003u64
                                                            .rotate_left(i * 32)
                                                            >> j)
                                                            & 1)
                                                            << 15)
                                                            as u16
                                                    })
                                                    .collect::<Vec<_>>()
                                            } else {
                                                vec![0; 64]
                                            }
                                        })
                                        .flatten()
                                        .collect()
                                } else {
                                    vec![0; 4096]
                                }
                            })
                            .flatten()
                            .collect(),
                        (None, true),
                    ),
                ],
            ),
            // 34
            (
                2,
                17,
                4,
                &str_to_quants("EE_EEE_AAAEAA_AEAEEA_EEAE_AAEAA"),
                64,
                vec![
                    (vec![0u16; 4096 * 8], (None, false)),
                    (
                        (0..8)
                            .map(|h| {
                                if h == 6 {
                                    (0..64)
                                        .map(|i| {
                                            if (i & 4) == 0 {
                                                (0..64)
                                                    .map(|j| {
                                                        (((0x300c00000000c003u64
                                                            .rotate_left(i * 32)
                                                            >> j)
                                                            & 1)
                                                            << 15)
                                                            as u16
                                                    })
                                                    .collect::<Vec<_>>()
                                            } else {
                                                vec![0; 64]
                                            }
                                        })
                                        .flatten()
                                        .collect()
                                } else {
                                    vec![0; 4096]
                                }
                            })
                            .flatten()
                            .collect(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 3,
                                solution: Some(3),
                            }),
                            true,
                        ),
                    ),
                ],
            ),
            // 35
            (
                2,
                17,
                4,
                &str_to_quants("EE_EEE_EEEEAA_AEAEEA_EEAE_AAEAA"),
                64,
                vec![
                    (vec![0u16; 4096 * 8], (None, false)),
                    (
                        iter::repeat(0)
                            .take(256 * 23)
                            .chain(
                                vec![
                                    (0..64)
                                        .map(|j| {
                                            (((0x300c00000000c003u64 >> j) & 1) << 15) as u16
                                        })
                                        .collect::<Vec<_>>();
                                    4
                                ]
                                .into_iter()
                                .flatten(),
                            )
                            .chain(iter::repeat(0).take(8 * 4096 - 256 * 23 - 256))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 7,
                                solution: Some(0b1110_100),
                            }),
                            true,
                        ),
                    ),
                    (
                        iter::repeat(0)
                            .take(256 * 41)
                            .chain(
                                vec![
                                    (0..64)
                                        .map(|j| {
                                            (((0x300c00000000c003u64 >> j) & 1) << 15) as u16
                                        })
                                        .collect::<Vec<_>>();
                                    4
                                ]
                                .into_iter()
                                .flatten(),
                            )
                            .chain(iter::repeat(0).take(8 * 4096 - 256 * 41 - 256))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 7,
                                solution: Some(0b1001_010),
                            }),
                            true,
                        ),
                    ),
                    (
                        iter::repeat(0)
                            .take(256 * 102)
                            .chain(
                                vec![
                                    (0..64)
                                        .map(|j| {
                                            (((0x300c00000000c003u64 >> j) & 1) << 15) as u16
                                        })
                                        .collect::<Vec<_>>();
                                    4
                                ]
                                .into_iter()
                                .flatten(),
                            )
                            .chain(iter::repeat(0).take(8 * 4096 - 256 * 102 - 256))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 7,
                                solution: Some(0b0110_011),
                            }),
                            true,
                        ),
                    ),
                ],
            ),
            // 36
            (
                2,
                17,
                4,
                &str_to_quants("EE_EEE_EEEEEE_AEAEEA_EEAE_AAEAA"),
                64,
                vec![
                    (vec![0u16; 8 * 4096], (None, false)),
                    (
                        iter::repeat(0)
                            .take(64 * 169)
                            .chain(
                                (0..64)
                                    .map(|j| (((0x300c00000000c003u64 >> j) & 1) << 15) as u16)
                                    .collect::<Vec<_>>(),
                            )
                            .chain(iter::repeat(0).take(64 * 206))
                            .chain(
                                (0..64)
                                    .map(|j| (((0x300c00000000c003u64 >> j) & 1) << 15) as u16)
                                    .collect::<Vec<_>>(),
                            )
                            .chain(iter::repeat(0).take(64 * (512 - 169 - 2 - 206)))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 9,
                                solution: Some(0b100101_010),
                            }),
                            true,
                        ),
                    ),
                    (
                        iter::repeat(0)
                            .take(64 * 418)
                            .chain(
                                (0..64)
                                    .map(|j| (((0x300c00000000c003u64 >> j) & 1) << 15) as u16)
                                    .collect::<Vec<_>>(),
                            )
                            .chain(iter::repeat(0).take(64 * (512 - 418 - 1)))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 9,
                                solution: Some(0b010001_011),
                            }),
                            true,
                        ),
                    ),
                ],
            ),
            // 37
            (
                2,
                17,
                4,
                &str_to_quants("EE_EEE_EEEEEE_EEEEEA_EEAE_AAEAA"),
                64,
                vec![
                    (vec![0u16; 8 * 4096], (None, false)),
                    (
                        iter::repeat(0)
                            .take(11677 * 2)
                            .chain([0x8000, 0x8000].into_iter())
                            .chain(iter::repeat(0).take(2 * (4096 * 4 - 11677 - 1)))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 14,
                                solution: Some(0b10111_001101_101),
                            }),
                            true,
                        ),
                    ),
                    (
                        iter::repeat(0)
                            .take(3791 * 2)
                            .chain([0x8000, 0x8000].into_iter())
                            .chain(iter::repeat(0).take(2 * (4096 * 4 - 3791 - 1)))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 14,
                                solution: Some(0b11110_011011_100),
                            }),
                            true,
                        ),
                    ),
                ],
            ),
            // 38
            (
                2,
                17,
                4,
                &str_to_quants("EE_EEE_EEEEEE_EEEEEE_AAAE_AAEAA"),
                64,
                vec![
                    (vec![0u16; 8 * 4096], (None, false)),
                    (
                        iter::repeat(0)
                            .take(14061)
                            .chain(iter::once(0x807f))
                            .chain(iter::repeat(0).take(4096 * 8 - 14061 - 1))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 15,
                                solution: Some(0b101101_110110_110),
                            }),
                            true,
                        ),
                    ),
                    (
                        iter::repeat(0)
                            .take(29762)
                            .chain(iter::once(0x807f))
                            .chain(iter::repeat(0).take(4096 * 8 - 29762 - 1))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 15,
                                solution: Some(0b10000_100010_111),
                            }),
                            true,
                        ),
                    ),
                ],
            ),
            // 39
            (
                2,
                17,
                4,
                &str_to_quants("EE_EEE_EEEEEE_EEEEEE_EEEA_AAEAA"),
                64,
                vec![
                    (vec![0u16; 8 * 4096], (None, false)),
                    (
                        iter::repeat(0)
                            .take(14367)
                            .chain(iter::once(0x800c))
                            .chain(iter::repeat(0).take(4096 * 8 - 14367 - 1))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 18,
                                solution: Some(0b11_111110_000001_110),
                            }),
                            true,
                        ),
                    ),
                    (
                        iter::repeat(0)
                            .take(31722)
                            .chain(iter::once(0x8006))
                            .chain(iter::repeat(0).take(4096 * 8 - 31722 - 1))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 18,
                                solution: Some(0b110_010101_111101_111),
                            }),
                            true,
                        ),
                    ),
                ],
            ),
            // 40
            (
                2,
                16,
                4,
                &str_to_quants("EE_EE_EEEEEE_EEEEEE_EEEA_AAEAA"),
                64,
                vec![
                    (vec![0u16; 4 * 4096], (None, false)),
                    (
                        iter::repeat(0)
                            .take(11779)
                            .chain(iter::once(0x800c))
                            .chain(iter::repeat(0).take(4096 * 4 - 11779 - 1))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 17,
                                solution: Some(0b011_11000_000011_101),
                            }),
                            true,
                        ),
                    ),
                ],
            ),
            // 41
            (
                2,
                20,
                4,
                &str_to_quants("EE_EEEEEE_EEEEEE_EEEEEA_EEAE_AAEAA"),
                64,
                vec![
                    (vec![0u16; 64 * 4096], (None, false)),
                    (
                        iter::repeat(0)
                            .take(76993 * 2)
                            .chain([0x807f, 0x80ff].into_iter())
                            .chain(iter::repeat(0).take(2 * (4096 * 32 - 76993 - 1)))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 17,
                                solution: Some(0b10000_011001_101001),
                            }),
                            true,
                        ),
                    ),
                    (
                        iter::repeat(0)
                            .take(47606 * 2)
                            .chain([0x80ff, 0x80ff].into_iter())
                            .chain(iter::repeat(0).take(2 * (4096 * 32 - 47606 - 1)))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 17,
                                solution: Some(0b01101_111100_111010),
                            }),
                            true,
                        ),
                    ),
                ],
            ),
            // 42
            (
                2,
                20,
                4,
                &str_to_quants("EE_EEEEEE_EEEEEE_EEEEEE_EEEA_AAEAA"),
                64,
                vec![
                    (vec![0u16; 64 * 4096], (None, false)),
                    (
                        iter::repeat(0)
                            .take(116093)
                            .chain(iter::once(0x8002))
                            .chain(iter::repeat(0).take(4096 * 64 - 116093 - 1))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 21,
                                solution: Some(0b100_101111_101010_001110),
                            }),
                            true,
                        ),
                    ),
                    (
                        iter::repeat(0)
                            .take(195601)
                            .chain(iter::once(0x800c))
                            .chain(iter::repeat(0).take(4096 * 64 - 195601 - 1))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 21,
                                solution: Some(0b11_100010_000011_111101),
                            }),
                            true,
                        ),
                    ),
                ],
            ),
            // 43
            (
                2,
                20,
                4,
                &str_to_quants("EE_EEEEEE_EEEEEE_AEAEEA_EEAE_AAEAA"),
                64,
                vec![
                    (vec![0u16; 64 * 4096], (None, false)),
                    (
                        iter::repeat(0)
                            .take(64 * 2167)
                            .chain(
                                (0..64)
                                    .map(|j| {
                                        ((((0x300c00000000c003u64 >> j) & 1) << 15) + 255) as u16
                                    })
                                    .collect::<Vec<_>>(),
                            )
                            .chain(iter::repeat(0).take(64 * (4096 - 2167 - 1)))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 12,
                                solution: Some(0b111011_100001),
                            }),
                            true,
                        ),
                    ),
                    (
                        iter::repeat(0)
                            .take(64 * 1578)
                            .chain(
                                (0..64)
                                    .map(|j| {
                                        ((((0x300c00000000c003u64 >> j) & 1) + 255) << 15) as u16
                                    })
                                    .collect::<Vec<_>>(),
                            )
                            .chain(iter::repeat(0).take(64 * (4096 - 1578 - 1)))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 12,
                                solution: Some(0b010101_000110),
                            }),
                            true,
                        ),
                    ),
                ],
            ),
            // 44
            (
                2,
                20,
                4,
                &str_to_quants("EE_EEEEEE_EEEEAA_AEAEEA_EEAE_AAEAA"),
                64,
                vec![
                    (vec![0u16; 4096 * 64], (None, false)),
                    (
                        iter::repeat(0)
                            .take(256 * 741)
                            .chain(
                                vec![
                                    (0..64)
                                        .map(|j| {
                                            (((0x300c00000000c003u64 >> j) & 1) << 15) as u16
                                        })
                                        .collect::<Vec<_>>();
                                    4
                                ]
                                .into_iter()
                                .flatten(),
                            )
                            .chain(iter::repeat(0).take(64 * 4096 - 256 * 741 - 256))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 10,
                                solution: Some(0b1010_011101),
                            }),
                            true,
                        ),
                    ),
                ],
            ),
            // 45
            (
                2,
                24,
                4,
                &str_to_quants("EE_EEEE_EEEEEE_EEEEEE_EEEEEA_EEAE_AAEAA"),
                64,
                vec![
                    (vec![0u16; 1024 * 4096], (None, false)),
                    (
                        iter::repeat(0)
                            .take(568217 * 2)
                            .chain([0x80ff, 0x80ff].into_iter())
                            .chain(iter::repeat(0).take(2 * (4096 * 512 - 568217 - 1)))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 21,
                                solution: Some(0b010011_001110_101010_0010),
                            }),
                            true,
                        ),
                    ),
                    (
                        iter::repeat(0)
                            .take(1784317 * 2)
                            .chain([0x80ff, 0x80ff].into_iter())
                            .chain(iter::repeat(0).take(2 * (4096 * 512 - 1784317 - 1)))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 21,
                                solution: Some(0b010111_111100_111001_1011),
                            }),
                            true,
                        ),
                    ),
                ],
            ),
            // 46
            (
                2,
                23,
                4,
                &str_to_quants("EE_EEE_EEEEEE_EEEEEE_EEEEEE_EEEA_AAEAA"),
                64,
                vec![
                    (vec![0u16; 512 * 4096], (None, false)),
                    (
                        iter::repeat(0)
                            .take(794531)
                            .chain(iter::once(0x8002))
                            .chain(iter::repeat(0).take(4096 * 512 - 794531 - 1))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 24,
                                solution: Some(0b100_110001_011111_100000_110),
                            }),
                            true,
                        ),
                    ),
                    (
                        iter::repeat(0)
                            .take(1859312)
                            .chain(iter::once(0x800c))
                            .chain(iter::repeat(0).take(4096 * 512 - 1859312 - 1))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 24,
                                solution: Some(0b011_000011_110111_101000_111),
                            }),
                            true,
                        ),
                    ),
                ],
            ),
            // 47
            (
                2,
                22,
                4,
                &str_to_quants("EE_EE_EEEEEE_EEEEAA_AEAEEA_EEAE_AAEAA"),
                64,
                vec![
                    (vec![0u16; 4096 * 256], (None, false)),
                    (
                        iter::repeat(0)
                            .take(256 * 2593)
                            .chain(
                                vec![
                                    (0..64)
                                        .map(|j| {
                                            (((0x300c00000000c003u64 >> j) & 1) << 15) as u16
                                        })
                                        .collect::<Vec<_>>();
                                    4
                                ]
                                .into_iter()
                                .flatten(),
                            )
                            .chain(iter::repeat(0).take(256 * 4096 - 256 * 2593 - 256))
                            .collect::<Vec<_>>(),
                        (
                            Some(FinalResult {
                                reversed: false,
                                solution_bits: 12,
                                solution: Some(0b1000_010001_01),
                            }),
                            true,
                        ),
                    ),
                ],
            ),
            // 48
            (
                2,
                5,
                4,
                &str_to_quants("EE_AEE_EEAE_AAEAA"),
                64,
                vec![
                    (vec![0u16; 8], (None, false)),
                    (vec![0, 0x8000, 0, 0, 0x8000, 0, 0, 0x8000], (None, true)),
                ],
            ),
            // 49
            (
                2,
                5,
                4,
                &str_to_quants("AA_EEA_EEAE_AAEAA"),
                64,
                vec![
                    (vec![0x8000; 8], (None, true)),
                    (vec![0, 0, 0, 0, 0x8000, 0x8000, 0, 0x8000], (None, true)),
                ],
            ),
            // 50
            (
                2,
                10,
                4,
                &str_to_quants("EE_AA_EEAEEA_EEAE_AAEAA"),
                64,
                vec![
                    (vec![0u16; 256], (None, false)),
                    (vec![0x8000; 256], (None, true)),
                ],
            ),
            // 51
            (
                2,
                10,
                4,
                &str_to_quants("AA_EE_AAEEAA_EEAE_AAEAA"),
                64,
                vec![
                    (vec![0x8000; 256], (None, true)),
                    (vec![0x0; 256], (None, false)),
                ],
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let mut ocl_qr = OpenCLQuantReducer::new(
                reduce_start_bit,
                reduce_end_bit,
                init_group_len_bits,
                quants,
                context.clone(),
                cmd_queue.clone(),
                Some(group_len),
            );
            let input_len = 1usize << (reduce_end_bit - reduce_start_bit);
            let mut input_buffer = unsafe {
                Buffer::<u32>::create(
                    &context,
                    CL_MEM_READ_WRITE,
                    (input_len + 1) >> 1,
                    std::ptr::null_mut(),
                )
                .unwrap()
            };
            for (j, (input, exp_result)) in testcases.into_iter().enumerate() {
                println!("Test {} {}", i, j);
                assert_eq!(input.len(), input_len);
                let mut input_u32 = vec![0u32; (input_len + 1) >> 1];
                for i in 0..input_len {
                    input_u32[i >> 1] |= (input[i] as u32) << ((i & 1) << 4);
                }
                unsafe {
                    cmd_queue
                        .enqueue_write_buffer(&mut input_buffer, CL_BLOCKING, 0, &input_u32, &[])
                        .unwrap();
                    cmd_queue.finish().unwrap();
                }
                let result = ocl_qr.execute(&input_buffer);
                assert_eq!(exp_result, result, "{} {}", i, j);
            }
        }
    }

    fn gen_circuit(solution_bits: usize, solution: u128, values: u32) -> Circuit<usize> {
        use gategen::boolexpr::*;
        use gategen::dynintexpr::*;
        use gateutil::*;
        let ec = ExprCreatorSys::new();
        let input = UDynExprNode::variable(ec.clone(), solution_bits + 5);
        let out = dynint_booltable(
            input.subvalue(solution_bits, 5),
            (0..32u32).map(|b| {
                BoolExprNode::single_value(
                    ec.clone(),
                    ((values >> (b.reverse_bits() >> (32 - 5))) & 1) != 0,
                )
            }),
        ) & input
            .subvalue(0, solution_bits)
            .equal(UDynExprNode::try_constant_n(ec.clone(), solution_bits, solution).unwrap());
        let (circuit, input_map) = out.to_circuit();
        let input_list = input_map_to_input_list(input_map, input.iter());
        translate_inputs_rev(circuit, input_list)
    }

    #[test]
    fn test_opencl_quant_reducer_final_result_with_circuit() {
        let device = Device::new(*get_all_devices(CL_DEVICE_TYPE_GPU).unwrap().get(0).unwrap());
        let context = Arc::new(Context::from_device(&device).unwrap());
        #[allow(deprecated)]
        let cmd_queue =
            Arc::new(unsafe { CommandQueue::create(&context, device.id(), 0).unwrap() });
        for (
            i,
            (reduce_start_bit, reduce_end_bit, init_group_len_bits, quants, group_len, testcases),
        ) in [
            // 0
            (
                2,
                24,
                4,
                &str_to_quants("EE_EEEE_EEEEEE_EEEEEE_EEEEEE_EEEE_EEEAA"),
                64,
                vec![
                    (
                        0x68e681du128,
                        0xb4a85fb8u32,
                        FinalResult {
                            reversed: false,
                            solution_bits: 2 + 26 + 3,
                            solution: Some(0b010_0110100011100110100000011101),
                        },
                    ),
                    (
                        0x68e689eu128,
                        0xb4f852b8u32,
                        FinalResult {
                            reversed: false,
                            solution_bits: 2 + 26 + 3,
                            solution: Some(0b101_0110100011100110100010011110),
                        },
                    ),
                    (
                        0x68e685fu128,
                        0xb4ef52b8u32,
                        FinalResult {
                            reversed: false,
                            solution_bits: 2 + 26 + 3,
                            solution: Some(0b001_0110100011100110100001011111),
                        },
                    ),
                ],
            ),
            // 1
            (
                2,
                24,
                4,
                &str_to_quants("EE_EEEE_EEEEEE_EEEEEE_EEEEEE_EEEE_EEEEE"),
                64,
                vec![
                    (
                        0xa9a6c79u128,
                        1 << 27,
                        FinalResult {
                            reversed: false,
                            solution_bits: 2 + 26 + 5,
                            solution: Some(0b11011_1010100110100110110001111001),
                        },
                    ),
                    (
                        0x841bad8u128,
                        1 << 22,
                        FinalResult {
                            reversed: false,
                            solution_bits: 2 + 26 + 5,
                            solution: Some(0b01101_1000010000011011101011011000),
                        },
                    ),
                ],
            ),
            // 2
            (
                2,
                24,
                4,
                &str_to_quants("EE_EEEE_EEEEEE_EEEEEE_EEEEEE_EEEE_AAAAA"),
                64,
                vec![(
                    0xa9a6c7bu128,
                    0x4b854e2u32, // doesn't matter
                    FinalResult {
                        reversed: false,
                        solution_bits: 2 + 26,
                        solution: Some(0b1010100110100110110001111011),
                    },
                )],
            ),
            // 3
            (
                2,
                24,
                4,
                &str_to_quants("AA_AAAA_AAAAAA_AAAAAA_AAAAAA_AAAA_AAAEE"),
                64,
                vec![(
                    0x76312ceu128,
                    0x10111111u32,
                    FinalResult {
                        reversed: true,
                        solution_bits: 2 + 26 + 3,
                        solution: Some(0b110_111011000110001001011001110),
                    },
                )],
            ),
            // 4
            (
                2,
                20,
                4,
                &str_to_quants("AA_AAAAAA_AAAAAA_AAAAAA_AAAA_AAAEA"),
                64,
                vec![(
                    0xeb346bu128,
                    0xfebc6b73u32,
                    FinalResult {
                        reversed: true,
                        solution_bits: 2 + 22 + 3,
                        solution: Some(0b110_111010110011010001101011),
                    },
                )],
            ),
            // 5
            (
                2,
                20,
                4,
                &str_to_quants("AA_AAAAAA_AAAAAA_AAAAAA_AAAA_EEEEA"),
                64,
                vec![(
                    0xeb3469u128,
                    0xfebc6b73u32, // doesn't matter
                    FinalResult {
                        reversed: true,
                        solution_bits: 2 + 22,
                        solution: Some(0b111010110011010001101001),
                    },
                )],
            ),
            // 6
            (
                2,
                20,
                4,
                &str_to_quants("AA_AAAAAA_AAAAAA_AAAAAA_AAAA_AAAEA"),
                64,
                vec![(
                    0xeb346au128,  // doesn't matter
                    0xfebc6b73u32, // doesn't matter
                    FinalResult {
                        reversed: true,
                        solution_bits: 2 + 22,
                        solution: None,
                    },
                )],
            ),
            // 7
            (
                2,
                24,
                4,
                &str_to_quants("EE_EEEE_EEEEEE_EEEEEE_EEEEEE_EEAA_AAAAA"),
                64,
                vec![(
                    0x29a6c7bu128,
                    0x4b854e2u32, // doesn't matter
                    FinalResult {
                        reversed: false,
                        solution_bits: 2 + 24,
                        solution: Some(0b10100110100110110001111011),
                    },
                )],
            ),
            // 8
            (
                2,
                24,
                4,
                &str_to_quants("EE_EEEE_EEEEEE_EEEEEE_EEEEEE_EEEE_EEEAA"),
                64,
                vec![(
                    0x68e681cu128,
                    0xb4a85fb8u32, // doesn't matter
                    FinalResult {
                        reversed: false,
                        solution_bits: 2 + 26,
                        solution: None,
                    },
                )],
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let ocl_qr = OpenCLQuantReducer::new(
                reduce_start_bit,
                reduce_end_bit,
                init_group_len_bits,
                quants,
                context.clone(),
                cmd_queue.clone(),
                Some(group_len),
            );
            let first_quant = quants[0];
            let first_quant_bits =
                quants.iter().take_while(|q| **q == first_quant).count() - reduce_start_bit;
            let solution_bits = std::cmp::min(
                reduce_end_bit + init_group_len_bits,
                reduce_start_bit + first_quant_bits,
            );
            println!("Data: {}: {} {}", i, first_quant_bits, solution_bits);
            for (j, (solution, circuit_values, exp_result)) in testcases.into_iter().enumerate() {
                println!("Test {} {}", i, j);
                let circuit = gen_circuit(solution_bits, solution, circuit_values);
                let final_result = ocl_qr.final_result_with_circuit(
                    &circuit,
                    FinalResult {
                        reversed: first_quant == Quant::All,
                        solution_bits,
                        solution: if exp_result.solution.is_some() {
                            Some(solution)
                        } else {
                            None
                        },
                    },
                );
                println!("Result: {}", final_result);
                assert_eq!(final_result, exp_result, "{} {}", i, j);
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "OpenCL devices: {:?}",
        get_all_devices(CL_DEVICE_TYPE_GPU).unwrap_or(vec![])
    );
    let cmd_args = CommandArgs::parse();
    let circuit_str = fs::read_to_string(cmd_args.circuit.clone())?;
    let qcircuit = match QuantCircuit::<usize>::from_str(&circuit_str) {
        Ok(c) => c,
        Err(_) => {
            println!("Load as circuit with quantifiers");
            let circuit = Circuit::<usize>::from_str(&circuit_str)?;
            let input_len = circuit.input_len();
            QuantCircuit::new(std::iter::repeat(Quant::Exists).take(input_len), circuit).unwrap()
        }
    };
    do_command(qcircuit, cmd_args);
    Ok(())
}
