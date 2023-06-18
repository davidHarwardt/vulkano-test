#![allow(unused)]

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
    command_buffer::{allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo}, AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo}, sync::{self, GpuFuture}, pipeline::{ComputePipeline, Pipeline, PipelineBindPoint}, descriptor_set::{allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet}, image::{StorageImage, ImageDimensions}, format::Format,
};
    
mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 460

            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

            layout(set = 0, binding = 0) buffer Data {
                uint data[];
            } buf;

            void main() {
                uint idx = gl_GlobalInvocationID.x;
                buf.data[idx] *= 12;
            }
        ",
    }
}

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

    let data_buf = Buffer::from_iter(
        &alloc,
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..Default::default()
        },
        0..65536u32,
    ).expect("could not create data_buffer");

    let shader = cs::load(dev.clone())
        .expect("failed to create shader module");

    let compute_pipeline = ComputePipeline::new(
        dev.clone(),
        shader.entry_point("main").unwrap(),
        &(),
        None,
        |_| {},
    ).expect("could not create compute pipeline");

    let desc_set_alloc = StandardDescriptorSetAllocator::new(dev.clone());
    let pipeline_layout = compute_pipeline.layout();
    let desc_set_layouts = pipeline_layout.set_layouts();

    let desc_set_layout_idx = 0;
    let desc_set_layout = desc_set_layouts
        .get(desc_set_layout_idx)
    .unwrap();
    let desc_set = PersistentDescriptorSet::new(
        &desc_set_alloc,
        desc_set_layout.clone(),
        [WriteDescriptorSet::buffer(0, data_buf.clone())],
    ).expect("could not create desc_set");

    let cmd_buf_alloc = StandardCommandBufferAllocator::new(
        dev.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );

    let img = StorageImage::new(
        &alloc,
        ImageDimensions::Dim2d {
            width: 1024,
            height: 1024,
            array_layers: 1,
        },
        Format::R8G8B8A8_UNORM,
        Some(queue.queue_family_index()),
    ).expect("could not create image");

    let mut cmd_buf_builder = AutoCommandBufferBuilder::primary(
        &cmd_buf_alloc,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    ).expect("could not create cmd_buf_builder");

    let wg_count = [1024, 1, 1];

    cmd_buf_builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            desc_set_layout_idx as _,
            desc_set,
        )
        .dispatch(wg_count)
    .expect("cs dispatch failed");

    let cmd_buf = cmd_buf_builder
        .build()
    .expect("could not build command buffer");

    let fut = sync::now(dev.clone())
        .then_execute(queue.clone(), cmd_buf)
        .expect("coult not execute cmd_buf")
        .then_signal_fence_and_flush()
    .expect("failed to flush");
    fut.wait(None).expect("waiting for fut failed");

    let content = data_buf.read().expect("could not read data_buf");
    for (n, val) in content.iter().enumerate() {
        assert_eq!(*val, n as u32 * 12);
    }

    tracing::info!("ok");
}



