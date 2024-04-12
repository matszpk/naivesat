use gatenative::cpu_build_exec::*;
use gatenative::mapper::*;
use gatenative::opencl_build_exec::*;
use gatenative::parseq_mapper::*;
use gatenative::*;
use gatesim::*;

use clap::Parser;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};

use std::collections::BinaryHeap;
use std::fmt::{Display, Formatter, Write};
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
        let quant_pos = 0;
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
                reversed: *self.quants.last().unwrap(),
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
    for (i = 0; i < (TYPE_LEN >> 5); i++) {
        temp[i] = temp[i] TYPE_QUANT_REDUCE_OP_0 (temp[i] >> 1);
    }
    for (i = 0; i < (TYPE_LEN >> 5); i++) {
        temp[i] = temp[i] TYPE_QUANT_REDUCE_OP_1 (temp[i] >> 2);
    }
    for (i = 0; i < (TYPE_LEN >> 5); i++) {
        temp[i] = temp[i] TYPE_QUANT_REDUCE_OP_2 (temp[i] >> 4);
    }
    for (i = 0; i < (TYPE_LEN >> 5); i++) {
        temp[i] = temp[i] TYPE_QUANT_REDUCE_OP_3 (temp[i] >> 8);
    }
    for (i = 0; i < (TYPE_LEN >> 5); i++) {
        temp[i] = temp[i] TYPE_QUANT_REDUCE_OP_4 (temp[i] >> 16);
    }
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

#ifdef WORK_QUANT_REDUCE_INIT_DATA
#define BASE (4 + (TYPE_LEN >> 5))
    if (idx == 0) {
        // initialization of work bits
        out[0] = 0;
        out[1] = 0; // solution word
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
    if ((idx & ((1ULL << WORK_WORD_NUM_BITS) - 1ULL)) != ((1ULL << WORK_WORD_NUM_BITS) - 1ULL))
        out[(TYPE_LEN >> 5)] = work_bit;
#ifdef WORK_HAVE_FIRST_QUANT
    if (out[(TYPE_LEN >> 5) + 1] == 0 && (idx & OTHER_MASK) == OTHER_MASK &&
        (work_bit ^ ((WORK_QUANT_REDUCE_INIT_DATA >> (WORK_WORD_NUM_BITS - 1)) & 1)) != 0) {
        out[(TYPE_LEN >> 5) + 1] = 1;
        out[(TYPE_LEN >> 5) + 2] = idx & 0xffffffffU;
        out[(TYPE_LEN >> 5) + 3] = idx >> 32;
        // store machine word that have solution
        GET_U32_ALL(out, o0);
    }
#endif
#undef BASE
#undef PBASE
#else
    GET_U32_ALL(out, o0);
    out[(TYPE_LEN >> 5)] = work_bit;
#endif
}"##;

fn get_aggr_output_code_defs(type_len: usize, elem_bits: usize, quants: &[Quant]) -> String {
    let first_quant = quants[0];
    // determine first quantifier length (bits)
    let first_quant_bits = quants.iter().take_while(|q| **q == first_quant).count();
    assert_eq!(type_len.count_ones(), 1);
    let type_len_bits = (usize::BITS - type_len.leading_zeros() - 1) as usize;
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
            writeln!(
                defs,
                "#define OTHER_MASK ({}ULL)",
                ((1u64 << (elem_bits - type_len_bits)) - 1)
                    & !((1u64 << (first_quant_bits - (quants_len - elem_bits))) - 1)
            )
            .unwrap();
        }
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
    if first_quant_bits > quants_len - elem_bits && elem_bits > type_len_bits {
        writeln!(defs, "#define WORK_HAVE_FIRST_QUANT").unwrap();
    }
    defs
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
    fn test_get_aggr_output_code_defs() {
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
            get_aggr_output_code_defs(256, 18, &str_to_quants("EEEEAAAEAAEEEAAAAEEAEAEEE"))
        );
        assert_eq!(
            r##"#define WORK_QUANT_REDUCE_INIT_DATA (399ULL)
#define OTHER_MASK (1022ULL)
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
            get_aggr_output_code_defs(256, 18, &str_to_quants("EEEEEEEEAAEEEAAAAEEAEAEEE"))
        );
        assert_eq!(
            r##"#define WORK_QUANT_REDUCE_INIT_DATA (15ULL)
#define OTHER_MASK (960ULL)
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
            get_aggr_output_code_defs(256, 18, &str_to_quants("EEEEEEEEEEEEEAAAAEEAEAEEE"))
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
            get_aggr_output_code_defs(256, 18, &str_to_quants("EEEEEEEAAAEEEAAAAEEAEAEEE"))
        );
        assert_eq!(
            r##"#define TYPE_QUANT_REDUCE_OP_0 |
#define TYPE_QUANT_REDUCE_OP_1 |
#define TYPE_QUANT_REDUCE_OP_2 |
#define TYPE_QUANT_REDUCE_OP_3 &
#define TYPE_QUANT_REDUCE_OP_4 |
#define TYPE_QUANT_REDUCE_OP_5 &
#define TYPE_QUANT_REDUCE_OP_6 |
#define TYPE_QUANT_REDUCE_OP_7 |
#define WORK_WORD_NUM_BITS (0)
"##,
            get_aggr_output_code_defs(256, 8, &str_to_quants("EEAEAEEE"))
        );
    }

    use gatenative::clang_writer::*;

    #[test]
    fn test_get_aggr_output_cpu_code() {
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
                vec![(
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
                )],
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let defs = get_aggr_output_code_defs(1 << quants.len(), quants.len(), &quants);
            if *cpu_exts.last().unwrap() != CPUExtension::NoExtension
                && !cpu_exts.iter().any(|ext| *ext == *CPU_EXTENSION)
            {
                continue;
            }
            let mut builder = CPUBuilder::new_with_cpu_ext_and_clang_config(
                *cpu_exts.last().unwrap(),
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
