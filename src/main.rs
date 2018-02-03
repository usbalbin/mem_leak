extern crate ocl;

use ocl::flags;

const ITERATION_COUNT: i32 = 1;
const ELEM_COUNT: usize = 1_000_000;
const DEVICE_TYPE: ocl::flags::DeviceType = flags::DEVICE_TYPE_GPU;

fn main() {
    leak();
    println!();
    no_leak();
}

fn leak() -> () {
    let (queue, mut kernel) = setup();

    for i in 0..ITERATION_COUNT {
        let mut buf = ocl::Buffer::<f32>::builder()
            .queue(queue.queue().clone())
            .flags(ocl::MemFlags::new().read_write())
            .dims(ELEM_COUNT)
            .build().unwrap();
        print_ref_count("creation", i, &buf);   //Buffer i created, reference count = 1

        kernel.set_arg_buf_named("out", Some(&mut buf))
            .unwrap();
        //Buffer i - 1 reference count decremented
        //For my AMD Vega 56 GPU it is now 1, not freed
        //For my Intel 3930k CPU it is now 0, thus freed
        //
        //Buffer i reference count incremented to 2
        print_ref_count("setting argument", i, &buf);


        unsafe {    //Enqueue kernel WITHOUT specifying event with enew()
            kernel
                .cmd()
                .gws(ELEM_COUNT)
                .enq().unwrap();
        }
        //Buffer i reference count incremented to 3
        //(happens for my AMD Vega 56 GPU but not Intel 3930k CPU)
        print_ref_count("enqueueing kernel", i, &buf);


        // ... Do stuff with buf ...

        // buf is dropped
        //Buffer i reference count decremented to 2
    }
}

fn no_leak() -> () {
    let (queue, mut kernel) = setup();

    for i in 0..ITERATION_COUNT {
        let mut buf = ocl::Buffer::<f32>::builder()
            .queue(queue.queue().clone())
            .flags(ocl::MemFlags::new().read_write())
            .dims(ELEM_COUNT)
            .build().unwrap();
        print_ref_count("creation", i, &buf);

        kernel.set_arg_buf_named("out", Some(&mut buf))
            .unwrap();
        print_ref_count("setting argument", i, &buf);


        unsafe {
            let mut event = ocl::Event::empty();
            kernel
                .cmd()
                .enew(&mut event)
                .gws(ELEM_COUNT)
                .enq().unwrap();
            event.wait_for().unwrap();
        }
        print_ref_count("waiting for kernel", i, &buf);
    }
}

fn setup() -> (ocl::ProQue, ocl::Kernel) {
    let (platform, device) = get_device(DEVICE_TYPE).unwrap();
    let src = "kernel void simple(global float* out) {\
        out[get_global_id(0)] = get_global_id(0);  \
    }";
    let queue = ocl::ProQue::builder()
        .platform(platform)
        .device(device)
        .src(src)
        .build()
        .unwrap();
    let kernel = queue.create_kernel("simple").unwrap()
        .arg_buf_named::<f32, ocl::Buffer<f32>>("out", None);

    (queue, kernel)
}

fn get_device(device_type: ocl::flags::DeviceType) -> Option<(ocl::Platform, ocl::Device)> {
    let platforms = ocl::Platform::list();
    for platform in platforms {
        let devices =
            match ocl::Device::list(platform, Some(device_type)) {
                Ok(d) => d,
                Err(_) => continue,
            };
        for d in devices {
            return Some((platform, d));
        }
    }
    None
}

fn print_ref_count(s: &str, i: i32, buffer: &ocl::Buffer<f32>) {
    use ocl::enums::{MemInfo, MemInfoResult };

    if let MemInfoResult::ReferenceCount(rc) = buffer.mem_info(MemInfo::ReferenceCount) {
        println!("RC for buffer created in iteration {}, after {} is: {}", i, s, rc);
    } else {
        println!("Failed to get RC for buffer created in iteration {}", i);
    }
}