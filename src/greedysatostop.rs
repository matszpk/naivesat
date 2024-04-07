use gatenative::cpu_build_exec::*;
use gatenative::mapper::*;
use gatenative::opencl_build_exec::*;
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

use rayon::prelude::*;

use std::cell::UnsafeCell;
use std::fs;
use std::ops::Range;
use std::str::FromStr;
use std::sync::atomic::{self, AtomicU32, AtomicU64};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

#[derive(Clone, Parser, Debug)]
#[command(version, about, long_about = None)]
struct CommandArgs {
    circuit: String,
    unknowns: usize,
    #[arg(short = 'C', long)]
    opencl: Option<usize>,
    #[arg(short = 'H', long)]
    opencl_max_hashmap_size: Option<u64>,
    #[arg(short = 'v', long)]
    verify: bool,
}

fn main() {
    for x in get_all_devices(CL_DEVICE_TYPE_GPU).unwrap() {
        println!("OpenCLDevice: {:?}", x);
    }
    let cmd_args = CommandArgs::parse();
}
