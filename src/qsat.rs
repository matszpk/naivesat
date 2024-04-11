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
    mask: u64,
    items: BinaryHeap<(std::cmp::Reverse<u64>, bool)>,
    result: Vec<bool>,
}

impl QuantReducer {
    fn new(quants: &[Quant]) -> Self {
        // initialize bits by rule: All=1, Exists=0 (and requires 1, or requires 0)
        let quants = quants
            .iter()
            .rev()
            .map(|q| *q == Quant::All)
            .collect::<Vec<_>>();
        Self {
            quants: quants.clone(),
            started: false,
            start: 0,
            mask: u64::try_from((1u128 << quants.len()) - 1).unwrap(),
            items: BinaryHeap::new(),
            result: quants,
        }
    }

    #[inline]
    fn is_end(&self) -> bool {
        self.started && (self.start & self.mask) == 0
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
        self.start = self.start.overflowing_add(1).0;
    }

    fn final_result(&self) -> Option<bool> {
        if self.is_end() {
            Some(*self.result.last().unwrap())
        } else {
            None
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
