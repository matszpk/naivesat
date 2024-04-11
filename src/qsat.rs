use gatenative::cpu_build_exec::*;
use gatenative::mapper::*;
use gatenative::opencl_build_exec::*;
use gatenative::parseq_mapper::*;
use gatenative::*;
use gatesim::*;

use clap::Parser;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};

use std::collections::BinaryHeap;
use std::fmt::{Display, Formatter};
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
    solution: Option<u64>,
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
                solution: self.solution,
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn str_to_quants(s: &str) -> Vec<Quant> {
        s.chars()
            .filter(|c| *c == 'a' || *c == 'e')
            .map(|c| match c {
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
