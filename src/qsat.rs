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

use std::collections::BinaryHeap;
use std::fmt::{Display, Formatter, Write};
use std::fs;
use std::str::FromStr;
use std::sync::Arc;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FinalResult {
    // reversed: if first quantifier is 'All' then is reversed (solution if not satisfiable)
    // otherwise solution is satisfiable.
    reversed: bool,
    solution_bits: usize,
    // only for first quantifier
    solution: Option<u128>,
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

    // returns final result is
    fn flush(&mut self) {
        assert!(!self.is_end());
        while let Some((index, item)) = self.items.peek().copied() {
            if self.start == index.0 {
                // if index is match then flush
                self.items.pop();
                self.apply(item);
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
local uint local_results[GROUP_LEN];
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
        let work_rev_idx = work_idx.reverse_bits() >> (64 - first_quant_bits_in_work);
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

// OpenCLQuantReducer

struct OpenCLQuantReducer {
    cmd_queue: Arc<CommandQueue>,
    group_len: usize,
    group_len_bits: usize,
    kernels: Vec<Kernel>,
    input_len: usize,
    outputs: Vec<(Buffer<u16>, usize)>,
    // is some if first quantifiers after reduce_end_bit
    quants_after: Option<Vec<Quant>>,
    first_quant_bits: usize,
    // start bit position from which kernels starts reduction.
    quant_start_pos: usize,
    quants_start: Vec<Quant>,
    total_reduce_bits: usize,
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
        assert!(1 < reduce_start_bit);
        assert!(reduce_start_bit <= reduce_end_bit);
        assert_ne!(initial_input_group_len_bits, 0);
        assert!(reduce_start_bit + reduce_end_bit + initial_input_group_len_bits <= quants.len());
        let is_first_quant_all = quants[0] == Quant::All;
        let first_quant = quants[0];
        let have_first_quants = quants[1..reduce_start_bit + 1]
            .iter()
            .all(|x| quants[0] == *x);
        let quants_after = quants[reduce_end_bit..].to_vec();
        let total_reduce_bits = quants.len() - reduce_start_bit;
        let quants = &quants[reduce_start_bit..reduce_end_bit];
        let device = Device::new(context.devices()[0]);
        let group_len: usize =
            group_len.unwrap_or(usize::try_from(device.max_work_group_size().unwrap()).unwrap());
        let group_len = std::cmp::min(group_len, 4096);
        let group_len_bits = (usize::BITS - group_len.leading_zeros() - 1) as usize;
        let group_len = if group_len.count_ones() != 1 {
            1usize << group_len_bits
        } else {
            group_len
        };
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
            quants_after: if quants_len < first_quant_bits {
                Some(quants_after)
            } else {
                None
            },
            first_quant_bits,
            quant_start_pos,
            quants_start: quants[0..quant_start_pos].to_vec(),
            total_reduce_bits,
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
            qr.final_result().unwrap()
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
                let mut cur_first_quant_bits = self.first_quant_bits - self.quant_start_pos;
                // go get deeper first quant results
                let mut idx = usize::try_from(
                    (sol.reverse_bits()) >> (128 - quants_start_final_result.solution_bits),
                )
                .unwrap();
                // read last buffer
                // next buffer ....
                for (oi, (buffer, _)) in self.outputs.iter().rev().enumerate() {
                    let pass_bits = std::cmp::min(cur_first_quant_bits, self.group_len_bits);
                    let mut buf_out = [0u16];
                    unsafe {
                        self.cmd_queue
                            .enqueue_read_buffer(&buffer, CL_BLOCKING, 2 * idx, &mut buf_out, &[])
                            .unwrap();
                    }
                    idx = (buf_out[0] & 0x7fff) as usize;

                    let rev_idx = (idx.reverse_bits()
                        >> ((usize::BITS as usize) - self.group_len_bits))
                        & ((1usize << pass_bits) - 1);
                    if idx != 0x7fff {
                        // update new sol
                        new_sol |=
                            (rev_idx as u128) << (self.quant_start_pos + oi * self.group_len);
                    } else {
                        panic!("Unexpected");
                    }
                    if cur_first_quant_bits <= self.group_len_bits {
                        break;
                    }
                    cur_first_quant_bits -= self.group_len_bits;
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
                        << (self.quant_start_pos + self.outputs.len() * self.group_len);
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
            (
                Some(FinalResult {
                    reversed: self.is_first_quant_all,
                    solution_bits: first_quant_bits_in_reducer_and_inital_input,
                    solution: None,
                }),
                self.is_first_quant_all,
            )
        }
    }
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
                Some(OpenCLBuilderConfig {
                    optimize_negs: true,
                    group_vec: false,
                    group_len: Some(1),
                }),
            );
            assert_eq!(builder.type_len(), 32);
            builder.user_defs(&defs);
            //println!("Defs: {}", defs);
            builder.add_with_config(
                "formula",
                circuit.clone(),
                CodeConfig::new()
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
                Some(OpenCLBuilderConfig {
                    optimize_negs: true,
                    group_vec: false,
                    group_len: Some(group_len),
                }),
            );
            assert_eq!(builder.type_len(), 32);
            builder.user_defs(&defs);
            //println!("Defs: {}", defs);
            builder.add_with_config(
                "formula",
                circuit.clone(),
                CodeConfig::new()
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
                &[0, 0, 11, 770, 0, 110500, 0, 17, 1, 1, 6, 0][..]
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
                &[0, 49, 0, 0, 456, 21200, 0, 133, 0, 1, 41, 0][..]
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
            (
                2,
                8,
                7,
                &str_to_quants("EE_EEEEEE_EEEAAAE_AAEAA"),
                64,
                vec![],
            ),
            (
                2,
                8,
                7,
                &str_to_quants("EE_EEEEEE_EEEEEEE_EEAEA"),
                64,
                vec![],
            ),
            (
                2,
                14,
                7,
                &str_to_quants("AE_AAAEAA_AEAEEA_EEAEEAE_AAEAA"),
                64,
                vec![],
            ),
            (
                3,
                17,
                4,
                &str_to_quants("AEE_AA_AAAEAA_AEAEEA_EEAE_AAEAA"),
                64,
                vec![],
            ),
            (
                3,
                23,
                4,
                &str_to_quants("AEE_AA_AAAEAA_AEAEEA_EEAAEA_EEAE_AAEAA"),
                64,
                vec![],
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
