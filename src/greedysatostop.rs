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

use std::fmt::Debug;
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
        assert_eq!(t.len() & 1, 0);
        let atomic = unsafe {
            &*std::ptr::slice_from_raw_parts(
                t.as_mut_slice().as_mut_ptr().cast::<AtomicU64>(),
                t.len() >> 1,
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
        self.original.len() >> 1
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
    start: u64,
    data: Vec<u64>,
    mask: u64,
    len: usize,
}

impl MemImage {
    fn new(state_len: usize, start: u64, len: usize) -> Self {
        assert!(state_len < 64);
        assert!(start < 1u64 << state_len);
        assert!(start + (len as u64) < (1u64 << state_len));
        assert_eq!(len & 63, 0);
        Self {
            state_len,
            start,
            data: vec![0u64; (len * (state_len + 1)) >> 6],
            mask: if state_len + 1 < 64 {
                u64::try_from((1u128 << (state_len + 1)) - 1).unwrap()
            } else {
                u64::MAX
            },
            len,
        }
    }

    #[inline]
    fn slice(&self) -> &[u64] {
        &self.data
    }

    #[inline]
    fn slice_mut(&mut self) -> &mut [u64] {
        &mut self.data
    }

    #[inline]
    fn get(&self, i: usize) -> u64 {
        let idx = (i * (self.state_len + 1)) >> 6;
        let bit = (i * (self.state_len + 1)) & 63;
        if 64 - bit >= self.state_len + 1 {
            (self.data[idx] >> bit) & self.mask
        } else {
            ((self.data[idx] >> bit) | (self.data[idx + 1] << (64 - bit))) & self.mask
        }
    }

    #[inline]
    fn set(&mut self, i: usize, value: u64) {
        let value = value & self.mask;
        let idx = (i * (self.state_len + 1)) >> 6;
        let bit = (i * (self.state_len + 1)) & 63;
        self.data[idx] = (self.data[idx] & !(self.mask << bit)) | (value << bit);
        if 64 - bit < self.state_len + 1 {
            self.data[idx + 1] =
                (self.data[idx + 1] & !(self.mask >> (64 - bit))) | (value >> (64 - bit));
        }
    }

    fn from_slice<T>(state_len: usize, start: u64, data: &[T]) -> Self
    where
        T: Clone,
        u64: TryFrom<T>,
        <u64 as TryFrom<T>>::Error: Debug,
    {
        assert_eq!(data.len() & 63, 0);
        let mut m = MemImage::new(state_len, start, data.len());
        for (i, v) in data.iter().enumerate() {
            m.set(i, u64::try_from(v.clone()).unwrap());
        }
        m
    }

    fn join_nexts(&mut self, second: &MemImage) {
        assert_eq!(self.state_len, second.state_len);
        let state_mask = self.mask >> 1;
        for i in 0..self.len {
            let old_value = self.get(i);
            let old_next = old_value & state_mask;
            if (old_value >> self.state_len) == 0
                && second.start <= old_next
                && old_next < second.start + (second.len as u64)
            {
                // if no stopped state and next in range of second MemImage then update
                self.set(i, second.get((old_next - second.start) as usize));
            }
        }
    }

    fn find_solution(&self, unknowns: usize) -> Option<Solution> {
        assert!(unknowns <= self.state_len);
        let before_unknowns_mask = (1u64 << (self.state_len - unknowns)) - 1;
        let unknowns_step = before_unknowns_mask + 1;
        let unknown_state_start = if (self.start & before_unknowns_mask) != 0 {
            // add one unknowns step if start is not zero in before unknown bits
            (self.start & !before_unknowns_mask) + unknowns_step
        } else {
            self.start & !before_unknowns_mask
        };
        let end = self.start + (self.len as u64);
        let mut unknown_state = unknown_state_start;
        let state_mask = self.mask >> 1;
        while unknown_state < end {
            let value = self.get((unknown_state - self.start) as usize);
            if (value >> self.state_len) != 0 {
                return Some(Solution {
                    start: unknown_state,
                    end: value & state_mask,
                });
            }
            unknown_state += unknowns_step;
        }
        None
    }
}

const FILE_BUFFER_LEN: usize = 1024 * 64;

fn read_u64_from_file(file: &mut File, buf: &mut [u64]) -> io::Result<()> {
    let mut byte_buf = vec![0u8; FILE_BUFFER_LEN];
    for chunk in buf.chunks_mut(FILE_BUFFER_LEN >> 3) {
        file.read_exact(&mut byte_buf[0..chunk.len() << 3])?;
        for (i, v) in chunk.iter_mut().enumerate() {
            let mut bytes: [u8; 8] = [0u8; 8];
            bytes
                .as_mut_slice()
                .copy_from_slice(&byte_buf[i << 3..(i + 1) << 3]);
            *v = u64::from_le_bytes(bytes);
        }
    }
    Ok(())
}

fn write_u64_to_file(file: &mut File, buf: &[u64]) -> io::Result<()> {
    let mut byte_buf = vec![0u8; FILE_BUFFER_LEN];
    for chunk in buf.chunks(FILE_BUFFER_LEN >> 3) {
        for (i, v) in chunk.iter().enumerate() {
            let bytes = v.to_le_bytes();
            byte_buf[i << 3..(i + 1) << 3].copy_from_slice(bytes.as_slice());
        }
        file.write_all(&byte_buf[0..chunk.len() << 3])?;
    }
    Ok(())
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
        assert!(partitions * 64 <= 1 << state_len);
        let path = format!(
            "{}greedy_temp_{}",
            prefix,
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let file = File::options()
            .create(true)
            .read(true)
            .write(true)
            .open(&path)?;
        Ok(Self {
            state_len,
            partitions,
            file,
            path,
            partition_len: (((1 << state_len) / partitions) * (state_len + 1)) >> 3,
            count: 0,
        })
    }

    fn load_partition(&mut self, part: usize, out: &mut MemImage) -> io::Result<()> {
        assert!(part < self.partitions);
        assert_eq!(out.slice().len() << 3, self.partition_len);
        assert_eq!(self.count, 1 << self.state_len);
        let pos = (part as u64) * self.partition_len as u64;
        let new_pos = self.file.seek(SeekFrom::Start(pos as u64))?;
        assert_eq!(pos, new_pos);
        read_u64_from_file(&mut self.file, out.slice_mut())?;
        out.start = (part * (1 << self.state_len) / self.partitions) as u64;
        Ok(())
    }

    fn save_partition(&mut self, part: usize, out: &MemImage) -> io::Result<()> {
        assert!(part < self.partitions);
        assert_eq!(out.slice().len() << 3, self.partition_len);
        assert_eq!(self.count, 1 << self.state_len);
        let pos = (part as u64) * self.partition_len as u64;
        let new_pos = self.file.seek(SeekFrom::Start(pos as u64))?;
        assert_eq!(pos, new_pos);
        write_u64_to_file(&mut self.file, out.slice())?;
        Ok(())
    }

    fn save_chunk(&mut self, out: &MemImage) -> io::Result<()> {
        let add = ((out.slice().len() << 3) / (self.state_len + 1)) as u64 * 8;
        assert!(self.count + add <= 1u64 << self.state_len);
        assert_eq!((out.slice().len() << 3) % (self.state_len + 1), 0);
        write_u64_to_file(&mut self.file, out.slice())?;
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
#if OUTPUT_NUM <= 32
    uint32_t* output_u = ((uint32_t*)output) + idx * TYPE_LEN;
    OUTPUT_TRANSFORM_FIRST_32(output_u);
#else
    uint64_t* output_u = ((uint64_t*)output) + idx * TYPE_LEN;
    uint32_t i;
    uint32_t temp[((OUTPUT_NUM + 31) >> 5) * TYPE_LEN];
    OUTPUT_TRANSFORM_FIRST_32(temp);
    OUTPUT_TRANSFORM_SECOND_32(temp + 32 * (TYPE_LEN >> 5));
    for (i = 0; i < TYPE_LEN; i++) {
        output_u[i] = ((uint64_t)temp[i]) | (((uint64_t)temp[i + TYPE_LEN]) << 32);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mem_image() {
        for (k, (state_len, mult, add, len)) in [
            (10, 3, 25, 256),
            (16, 67, 441, 1024),
            (19, 49414, 9481778, 64 * 80),
            (31, 96069201, 9669093056, 64 * 222),
            (32, 139696069201, 9669093056, 64 * 222),
            (40, 166139696069201, 966771239093056, 64 * 320),
            (45, 16171069201, 966771239093056, 64 * 400),
        ]
        .into_iter()
        .enumerate()
        {
            let mut mi = MemImage::new(state_len, 0, len);
            // initialize mem image by messy content
            for (i, v) in mi.slice_mut().iter_mut().enumerate() {
                *v = (i as u64)
                    .overflowing_mul(5884491582115956921)
                    .0
                    .overflowing_add(5560939029013487713)
                    .0;
            }
            assert_eq!(mi.slice().len(), (state_len + 1) * (len >> 6));
            assert_eq!(mi.mask, (1 << (state_len + 1)) - 1);
            let mask = (1 << (state_len + 1)) - 1;
            // filling
            for i in 0..len {
                mi.set(i, (i as u64).overflowing_mul(mult).0.overflowing_add(add).0);
            }
            // checking internal content
            for i in 0..len {
                // get bit by bit from mem image slice.
                let res = (0..state_len + 1).fold(0, |a, dest_bit| {
                    let idx = (((state_len + 1) * i) + dest_bit) >> 6;
                    let bit = (((state_len + 1) * i) + dest_bit) & 63;
                    a | (((mi.slice()[idx] >> bit) & 1) << dest_bit)
                });
                assert_eq!(
                    ((i as u64).overflowing_mul(mult).0.overflowing_add(add).0) & mask,
                    res,
                    "{} {}",
                    k,
                    i
                );
            }
            // compare from get
            for i in 0..len {
                assert_eq!(
                    ((i as u64).overflowing_mul(mult).0.overflowing_add(add).0) & mask,
                    mi.get(i),
                    "{} {}",
                    k,
                    i
                );
            }
        }
    }

    #[test]
    fn test_mem_image_find_solution() {
        for (i, (state_len, start, len, unknowns, pos_sol, next, exp_sol)) in [
            (24, 3 << 20, 1 << 20, 11, 63491, 58586, None),
            (
                24,
                3 << 20,
                1 << 20,
                11,
                79 * 8192,
                58591,
                Some(Solution {
                    start: (3 << 20) + 79 * 8192,
                    end: 58591,
                }),
            ),
            (
                24,
                11 << 20,
                1 << 20,
                22,
                45821 << 2,
                679116,
                Some(Solution {
                    start: (11 << 20) + (45821 << 2),
                    end: 679116,
                }),
            ),
            (24, 11 << 20, 1 << 20, 2, 0, 1133454, None),
            (24, 12 << 20, 1 << 20, 2, 1, 1233454, None),
            (
                24,
                12 << 20,
                1 << 20,
                2,
                0,
                1133454,
                Some(Solution {
                    start: 12 << 20,
                    end: 1133454,
                }),
            ),
            (24, 12 << 20, 1 << 20, 1, 0, 1133454, None),
            (24, 4 << 20, 1 << 20, 1, 0, 1133454, None),
            (24, 8 << 20, 1 << 20, 1, 1, 1133454, None),
            (
                24,
                8 << 20,
                1 << 20,
                1,
                0,
                9955001,
                Some(Solution {
                    start: 8 << 20,
                    end: 9955001,
                }),
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let mut mi = MemImage::new(state_len, start, len);
            mi.set(pos_sol, (1 << state_len) | (next & ((1 << state_len) - 1)));
            assert_eq!(exp_sol, mi.find_solution(unknowns), "{}", i);
        }
    }

    #[test]
    fn test_file_image() {
        for (k, (state_len, mult, add, partitions, changer)) in
            [(26, 4849217, 3455641, 16, 9450290114)]
                .into_iter()
                .enumerate()
        {
            let mut fi = FileImage::new(state_len, partitions, "").unwrap();
            let save_parts = partitions << 1;
            let chunk_len = (1usize << state_len) / save_parts;
            // save chunks
            for part in 0..save_parts {
                let mut chunk_mi = MemImage::new(state_len, 0, chunk_len);
                for ci in 0..chunk_len {
                    let i = chunk_len * part + ci;
                    chunk_mi.set(
                        ci,
                        (i as u64).overflowing_mul(mult).0.overflowing_add(add).0,
                    );
                }
                fi.save_chunk(&chunk_mi).unwrap();
            }
            // check partitions
            let chunk_len = (1usize << state_len) / partitions;
            let mask = (1u64 << (state_len + 1)) - 1;
            for part in (0..partitions).rev() {
                let mut part_chunk = MemImage::new(state_len, 0, chunk_len);
                fi.load_partition(part, &mut part_chunk).unwrap();
                assert_eq!((chunk_len as u64) * (part as u64), part_chunk.start);
                // check partition
                for ci in 0..chunk_len {
                    let i = chunk_len * part + ci;
                    assert_eq!(
                        ((i as u64).overflowing_mul(mult).0.overflowing_add(add).0) & mask,
                        part_chunk.get(ci),
                        "{} {}",
                        part,
                        ci
                    );
                }
            }
            // write new partitions
            for part in (0..partitions).rev() {
                // save new partition
                let mut part_chunk = MemImage::new(state_len, 0, chunk_len);
                for ci in 0..chunk_len {
                    let i = chunk_len * part + ci;
                    part_chunk.set(
                        ci,
                        ((i + changer) as u64)
                            .overflowing_mul(mult)
                            .0
                            .overflowing_add(add)
                            .0,
                    );
                }
                fi.save_partition(part, &part_chunk).unwrap();
            }
            // check new partitions
            for part in (0..partitions).rev() {
                let mut part_chunk = MemImage::new(state_len, 0, chunk_len);
                fi.load_partition(part, &mut part_chunk).unwrap();
                assert_eq!((chunk_len as u64) * (part as u64), part_chunk.start);
                // check partition
                for ci in 0..chunk_len {
                    let i = chunk_len * part + ci;
                    assert_eq!(
                        (((i + changer) as u64)
                            .overflowing_mul(mult)
                            .0
                            .overflowing_add(add)
                            .0)
                            & mask,
                        part_chunk.get(ci),
                        "new: {} {}",
                        part,
                        ci
                    );
                }
            }
        }
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
