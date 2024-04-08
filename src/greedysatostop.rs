use gatenative::cpu_build_exec::*;
//use gatenative::opencl_build_exec::*;
use gatenative::*;
use gatesim::*;

use clap::Parser;
// use opencl3::command_queue::CommandQueue;
// use opencl3::context::Context;
// use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
// use opencl3::kernel::{ExecuteKernel, Kernel};
// use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
// use opencl3::program::Program;
// use opencl3::types::{cl_ulong, CL_BLOCKING};

use rayon::prelude::*;

use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::ops::Range;
use std::str::FromStr;
use std::sync::atomic::{self, AtomicU32, AtomicU64};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Parser, Debug)]
#[command(version, about, long_about = None)]
struct CommandArgs {
    circuit: String,
    unknowns: usize,
    #[arg(short = 'p', long)]
    partitions: Option<usize>,
    #[arg(short = 'v', long)]
    verify: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Solution {
    start: u64,
    end: u64,
}

#[derive(Clone, Copy, Debug)]
enum FinalResult {
    Solution(Solution),
    NoSolution,
}

struct AtomicU32Array<'a> {
    original: Vec<u32>,
    atomic: &'a [AtomicU32],
}

impl From<Vec<u32>> for AtomicU32Array<'_> {
    fn from(mut t: Vec<u32>) -> Self {
        let atomic = unsafe {
            &*std::ptr::slice_from_raw_parts(
                t.as_mut_slice().as_mut_ptr().cast::<AtomicU32>(),
                t.len(),
            )
        };
        Self {
            original: t,
            atomic,
        }
    }
}

impl<'a> AtomicU32Array<'a> {
    #[inline]
    fn get(&self, i: usize) -> &'a AtomicU32 {
        &self.atomic[i]
    }

    #[inline]
    fn as_slice(&self) -> &'a [AtomicU32] {
        &self.atomic
    }

    #[inline]
    fn len(&self) -> usize {
        self.original.len()
    }
}

struct AtomicU64Array<'a> {
    original: Vec<u32>,
    atomic: &'a [AtomicU64],
}

impl From<Vec<u32>> for AtomicU64Array<'_> {
    fn from(mut t: Vec<u32>) -> Self {
        let atomic = unsafe {
            &*std::ptr::slice_from_raw_parts(
                t.as_mut_slice().as_mut_ptr().cast::<AtomicU64>(),
                t.len(),
            )
        };
        Self {
            original: t,
            atomic,
        }
    }
}

impl<'a> AtomicU64Array<'a> {
    #[inline]
    fn get(&self, i: usize) -> &'a AtomicU64 {
        &self.atomic[i]
    }

    #[inline]
    fn as_slice(&self) -> &'a [AtomicU64] {
        &self.atomic
    }

    #[inline]
    fn len(&self) -> usize {
        self.original.len()
    }
}

//
// join nexts
//

fn join_nexts(input_len: usize, nexts: Arc<AtomicU32Array>) {
    let cpu_num = rayon::current_num_threads();
    let chunk_num = std::cmp::min(std::cmp::max(cpu_num * 8, 64), nexts.len() >> 6);
    let chunk_len = nexts.len() / chunk_num;
    let stop_mask = 1u32 << input_len;
    let next_mask = stop_mask - 1u32;
    nexts
        .as_slice()
        .chunks(chunk_len)
        .par_bridge()
        .for_each(|chunk| {
            for cell in chunk {
                std::sync::atomic::fence(atomic::Ordering::SeqCst);
                let old_value = cell.load(atomic::Ordering::SeqCst);
                let old_next = old_value & next_mask;
                std::sync::atomic::fence(atomic::Ordering::SeqCst);
                if (old_value & stop_mask) == 0 {
                    cell.store(
                        nexts.get(old_next as usize).load(atomic::Ordering::SeqCst),
                        atomic::Ordering::SeqCst,
                    );
                }
                std::sync::atomic::fence(atomic::Ordering::SeqCst);
            }
        });
}

fn join_nexts_64(input_len: usize, nexts: Arc<AtomicU64Array>) {
    let cpu_num = rayon::current_num_threads();
    let chunk_num = std::cmp::min(std::cmp::max(cpu_num * 8, 64), nexts.len() >> 6);
    let chunk_len = nexts.len() / chunk_num;
    let stop_mask = 1u64 << input_len;
    let next_mask = stop_mask - 1u64;
    nexts
        .as_slice()
        .chunks(chunk_len)
        .par_bridge()
        .for_each(|chunk| {
            for cell in chunk {
                std::sync::atomic::fence(atomic::Ordering::SeqCst);
                let old_value = cell.load(atomic::Ordering::SeqCst);
                let old_next = old_value & next_mask;
                std::sync::atomic::fence(atomic::Ordering::SeqCst);
                if (old_value & stop_mask) == 0 {
                    cell.store(
                        nexts.get(old_next as usize).load(atomic::Ordering::SeqCst),
                        atomic::Ordering::SeqCst,
                    );
                }
                std::sync::atomic::fence(atomic::Ordering::SeqCst);
            }
        });
}

fn check_stop(input_len: usize, nexts: Arc<AtomicU32Array>) -> bool {
    let cpu_num = rayon::current_num_threads();
    let chunk_num = std::cmp::min(std::cmp::max(cpu_num * 8, 64), nexts.len() >> 6);
    let chunk_len = nexts.len() / chunk_num;
    let stop_mask = 1u32 << input_len;
    let stop = Arc::new(AtomicU32::new(0));
    nexts
        .as_slice()
        .chunks(chunk_len)
        .par_bridge()
        .for_each(|chunk| {
            if chunk
                .iter()
                .any(|cell| (cell.load(atomic::Ordering::SeqCst) & stop_mask) != 0)
            {
                stop.fetch_or(1, atomic::Ordering::SeqCst);
            }
        });
    stop.load(atomic::Ordering::SeqCst) != 0
}

fn check_stop_64(input_len: usize, nexts: Arc<AtomicU64Array>) -> bool {
    let cpu_num = rayon::current_num_threads();
    let chunk_num = std::cmp::min(std::cmp::max(cpu_num * 8, 64), nexts.len() >> 6);
    let chunk_len = nexts.len() / chunk_num;
    let stop_mask = 1u64 << input_len;
    let stop = Arc::new(AtomicU32::new(0));
    nexts
        .as_slice()
        .chunks(chunk_len)
        .par_bridge()
        .for_each(|chunk| {
            if chunk
                .iter()
                .any(|cell| (cell.load(atomic::Ordering::SeqCst) & stop_mask) != 0)
            {
                stop.fetch_or(1, atomic::Ordering::SeqCst);
            }
        });
    stop.load(atomic::Ordering::SeqCst) != 0
}

fn find_solution(
    input_len: usize,
    unknowns: usize,
    nexts: Arc<AtomicU32Array>,
) -> Option<Solution> {
    let unknown_state_num = 1 << unknowns;
    let cpu_num = rayon::current_num_threads();
    let chunk_num = std::cmp::min(std::cmp::max(cpu_num * 8, 64), unknown_state_num >> 6);
    let chunk_len = unknown_state_num / chunk_num;
    let stop_mask = 1u32 << input_len;
    let next_mask = stop_mask - 1;
    let unknowns_mult = 1 << (input_len - unknowns);
    let result = Mutex::new(None);
    nexts
        .as_slice()
        .chunks(unknowns_mult * chunk_len)
        .enumerate()
        .par_bridge()
        .for_each(|(ch_idx, chunk)| {
            for (i, v) in chunk.chunks(unknowns_mult).enumerate() {
                let value = v[0].load(atomic::Ordering::SeqCst);
                if (value & stop_mask) != 0 {
                    let mut r = result.lock().unwrap();
                    *r = Some(Solution {
                        start: ((i + (ch_idx * chunk_len)) as u64) << (input_len - unknowns),
                        end: (value & next_mask) as u64,
                    });
                }
            }
        });
    result.into_inner().unwrap()
}

fn find_solution_64(
    input_len: usize,
    unknowns: usize,
    nexts: Arc<AtomicU64Array>,
) -> Option<Solution> {
    let unknown_state_num = 1 << unknowns;
    let cpu_num = rayon::current_num_threads();
    let chunk_num = std::cmp::min(std::cmp::max(cpu_num * 8, 64), unknown_state_num >> 6);
    let chunk_len = unknown_state_num / chunk_num;
    let stop_mask = 1u64 << input_len;
    let next_mask = stop_mask - 1;
    let unknowns_mult = 1 << (input_len - unknowns);
    let result = Mutex::new(None);
    nexts
        .as_slice()
        .chunks(unknowns_mult * chunk_len)
        .enumerate()
        .par_bridge()
        .for_each(|(ch_idx, chunk)| {
            for (i, v) in chunk.chunks(unknowns_mult).enumerate() {
                let value = v[0].load(atomic::Ordering::SeqCst);
                if (value & stop_mask) != 0 {
                    let mut r = result.lock().unwrap();
                    *r = Some(Solution {
                        start: ((i + (ch_idx * chunk_len)) as u64) << (input_len - unknowns),
                        end: value & next_mask,
                    });
                }
            }
        });
    result.into_inner().unwrap()
}

//
// partition code
//

struct MemImage {
    state_len: usize,
    data: Vec<u8>,
}

impl MemImage {
    fn new(state_len: usize, len: usize) -> Self {
        assert!(state_len < 64);
        assert_eq!(len & 7, 0);
        Self {
            state_len,
            data: vec![0u8; (len * (state_len + 1)) >> 3],
        }
    }

    #[inline]
    fn slice(&self) -> &[u8] {
        &self.data
    }

    #[inline]
    fn slice_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    fn get(&self, i: usize) -> u64 {
        let mut idx = (i * self.state_len) >> 3;
        let mut bit = (i * self.state_len) & 7;
        let mut bit_count = 0;
        let mut value = 0u64;
        if (bit & 7) != 0 {
            value |= (self.data[idx] >> bit) as u64;
            bit_count += 8 - bit;
            idx += 1;
        }
        while bit_count + 7 < self.state_len {
            value |= (self.data[idx] as u64) << bit_count;
            bit_count += 8;
            idx += 1;
        }
        if bit_count < self.state_len {
            let mask = 1u8 << (self.state_len - bit_count) - 1;
            value |= ((self.data[idx] & mask) as u64) << bit_count;
        }
        value
    }

    fn set(&mut self, i: usize, value: u64) {
        let mut idx = (i * self.state_len) >> 3;
        let mut bit = (i * self.state_len) & 7;
        let mut bit_count = 0;
        let value_bytes = value.to_le_bytes();
        if (bit & 7) != 0 {
            self.data[idx] = (self.data[idx] & ((1u8 << bit) - 1)) | (value_bytes[0] << bit);
            bit_count += 8 - bit;
            idx += 1;
        }
        while bit_count + 7 < self.state_len {
            self.data[idx] = (value_bytes[bit_count >> 3] >> (8 - bit));
            if bit != 0 {
                self.data[idx] |= (value_bytes[(bit_count + 8) >> 3] << bit);
            }
            bit_count += 8;
            idx += 1;
        }
        if bit_count < self.state_len {
            let mask = (1u8 << (self.state_len - bit_count)) - 1;
            self.data[idx] =
                (self.data[idx] & !mask) | ((value_bytes[bit_count >> 3] >> (8 - bit)) & mask);
            if bit != 0 && bit < self.state_len - bit_count {
                self.data[idx] =
                    (self.data[idx] & !mask) | ((value_bytes[(bit_count + 8) >> 3] << bit) & mask);
            }
        }
    }
}

struct FileImage {
    state_len: usize,
    partitions: usize,
    file: File,
    path: String,
    partition_len: usize,
    count: u64,
}

impl FileImage {
    fn new(state_len: usize, partitions: usize, prefix: &str) -> io::Result<Self> {
        assert!(state_len < 64);
        assert_eq!(partitions.count_ones(), 1);
        let path = format!(
            "{}greedy_temp_{}",
            prefix,
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let mut file = File::create(&path)?;
        Ok(Self {
            state_len,
            partitions,
            file,
            path,
            partition_len: (((1 << state_len) / partitions) * (state_len + 1)) >> 3,
            count: 0,
        })
    }

    fn load_partition(&mut self, part: usize, out: &mut [u8]) -> io::Result<()> {
        assert!(part < self.partitions);
        assert_eq!(out.len(), self.partition_len);
        assert_eq!(self.count, 1 << self.state_len);
        let pos = self.partition_len as u64;
        let new_pos = self.file.seek(SeekFrom::Start(pos as u64))?;
        assert_eq!(pos, new_pos);
        self.file.read_exact(out)?;
        Ok(())
    }

    fn save_partition(&mut self, part: usize, out: &[u8]) -> io::Result<()> {
        assert!(part < self.partitions);
        assert_eq!(out.len(), self.partition_len);
        assert_eq!(self.count, 1 << self.state_len);
        let pos = self.partition_len as u64;
        let new_pos = self.file.seek(SeekFrom::Start(pos as u64))?;
        assert_eq!(pos, new_pos);
        self.file.write_all(out)?;
        Ok(())
    }

    fn save_chunk(&mut self, out: &[u8]) -> io::Result<()> {
        let add = (out.len() / (self.state_len + 1)) as u64 * 8;
        assert!(self.count + add <= 1u64 << self.state_len);
        assert_eq!(out.len() % (self.state_len + 1), 0);
        self.file.write_all(out)?;
        self.count += add;
        Ok(())
    }
}

impl Drop for FileImage {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

//
// main solver code
//

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

fn do_solve_with_cpu_builder(circuit: Circuit<usize>, cmd_args: &CommandArgs) -> FinalResult {
    let input_len = circuit.input_len();
    let output_len = input_len + 1;
    let words_per_elem = (output_len + 31) >> 5;
    let (output, start) = {
        let mut builder = CPUBuilder::new_parallel(None, Some(2048));
        builder.transform_helpers();
        builder.user_defs(&format!("#define OUTPUT_NUM ({})\n", output_len));
        builder.user_defs(&gen_output_transform_code(output_len));
        builder.add_with_config(
            "formula",
            circuit,
            CodeConfig::new()
                .elem_inputs(Some(&(0..input_len).collect::<Vec<usize>>()))
                .aggr_output_code(Some(AGGR_OUTPUT_CPU_CODE))
                .aggr_output_len(Some(words_per_elem * (1 << input_len)))
                .dont_clear_outputs(true),
        );
        let mut execs = builder.build().unwrap();
        let start = SystemTime::now();
        let input = execs[0].new_data(16);
        println!("Calculate first nexts");
        (execs[0].execute(&input, 0).unwrap().release(), start)
    };
    let mut final_result = FinalResult::NoSolution;
    if input_len < 32 {
        let nexts = Arc::new(AtomicU32Array::from(output));
        if check_stop(input_len, nexts.clone()) {
            for i in 0..input_len {
                println!("Joining nexts: Stage: {} / {}", i, input_len);
                join_nexts(input_len, nexts.clone());
                if let Some(sol) = find_solution(input_len, cmd_args.unknowns, nexts.clone()) {
                    final_result = FinalResult::Solution(sol);
                    break;
                }
            }
        }
    } else {
        let nexts = Arc::new(AtomicU64Array::from(output));
        if check_stop_64(input_len, nexts.clone()) {
            for i in 0..input_len {
                println!("Joining nexts: Stage: {} / {}", i, input_len);
                join_nexts_64(input_len, nexts.clone());
                if let Some(sol) = find_solution_64(input_len, cmd_args.unknowns, nexts.clone()) {
                    final_result = FinalResult::Solution(sol);
                    break;
                }
            }
        }
    }
    let time = start.elapsed().unwrap();
    println!("Time: {}", time.as_secs_f64());
    final_result
}

fn do_solve_with_cpu_builder_with_partitions(
    circuit: Circuit<usize>,
    cmd_args: &CommandArgs,
) -> FinalResult {
    FinalResult::NoSolution
}

fn do_solve(circuit: Circuit<usize>, cmd_args: CommandArgs) {
    let input_len = circuit.input_len();
    assert!(input_len < 64);
    let result = do_solve_with_cpu_builder(circuit.clone(), &cmd_args);
    if let FinalResult::Solution(sol) = result {
        println!("Solution: {:?}", sol);
        if cmd_args.verify {
            let mut state = sol.start;
            loop {
                let state_vec = (0..input_len)
                    .map(|b| ((state >> b) & 1) != 0)
                    .collect::<Vec<_>>();
                let output_vec = circuit.eval(state_vec.clone());
                state = output_vec[0..input_len]
                    .iter()
                    .take(input_len)
                    .enumerate()
                    .fold(0u64, |a, (i, x)| a | (u64::from(*x) << i));
                if output_vec[input_len] {
                    break;
                }
            }
            println!("Verified: end={}", state);
            if sol.end != state {
                println!("INCORRECT");
            }
        }
    } else {
        println!("Unsatisfiable!");
    }
}

fn main() {
    // for x in get_all_devices(CL_DEVICE_TYPE_GPU).unwrap() {
    //     println!("OpenCLDevice: {:?}", x);
    // }
    let cmd_args = CommandArgs::parse();
    let circuit_str = fs::read_to_string(cmd_args.circuit.clone()).unwrap();
    let circuit = Circuit::<usize>::from_str(&circuit_str).unwrap();
    let input_len = circuit.input_len();
    assert_eq!(input_len + 1, circuit.outputs().len());
    assert!(cmd_args.unknowns <= input_len);
    do_solve(circuit, cmd_args);
}
