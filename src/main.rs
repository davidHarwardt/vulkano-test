use vulkano::{
    VulkanLibrary,
    instance::{
        Instance,
        InstanceCreateInfo,
    }, 
    device::{
        QueueFlags, 
        Device, 
        DeviceCreateInfo, 
        QueueCreateInfo,
    }, 
    memory::allocator::{
        StandardMemoryAllocator, 
        AllocationCreateInfo,
        MemoryUsage,
    },
    buffer::{
        Buffer,
        BufferCreateInfo,
        BufferUsage,
    },
    command_buffer::{allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo}, AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo}, sync::{self, GpuFuture},
};
    

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("warn,shadertoy=info")
    .init();

    // let event_loop = EventLoop::new();
    let lib = VulkanLibrary::new().expect("could not load vulkan lib");
    let instance = Instance::new(lib, InstanceCreateInfo::default()).expect("could not create instance");

    let physical_dev = instance.enumerate_physical_devices()
        .expect("could not enumerate devices")
        .next()
    .expect("no devices available");

    let queue_family_index = physical_dev
        .queue_family_properties()
        .iter()
        .position(|props| props.queue_flags.contains(QueueFlags::GRAPHICS))
    .expect("could not find graphics queue") as u32;

    let (dev, mut queues) = Device::new(
        physical_dev,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    ).expect("failed to create device");

    let queue = queues.next().unwrap();

    let alloc = StandardMemoryAllocator::new_default(dev.clone());

    let src = Buffer::from_iter(
        &alloc,
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..Default::default()
        },
        0i32..64,
    ).expect("could not create src buffer");

    let dest = Buffer::from_iter(
        &alloc,
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Download,
            ..Default::default()
        },
        (0..64).map(|_| 0i32),
    ).expect("could not create dest buffer");

    let cmd_buf_alloc = StandardCommandBufferAllocator::new(
        dev.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );

    let mut builder = AutoCommandBufferBuilder::primary(
        &cmd_buf_alloc,
        queue_family_index,
        CommandBufferUsage::OneTimeSubmit,
    ).expect("could not create cmb_buffer_builder");

    builder
        .copy_buffer(CopyBufferInfo::buffers(src.clone(), dest.clone()))
        .expect("could not write commands to buffer");

    let cmd_buf = builder.build().expect("could not build command buffer");

    let fut = sync::now(dev.clone())
        .then_execute(queue.clone(), cmd_buf)
        .expect("could not execute cmd_buffer")
        .then_signal_fence_and_flush()
    .expect("failed to flush");

    fut.wait(None).expect("error waiting for future");

    let src_data = src.read().unwrap();
    let dest_data = dest.read().unwrap();
    assert_eq!(&*src_data, &*dest_data);

    tracing::info!("ok");
}
