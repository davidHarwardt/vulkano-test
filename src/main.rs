#![allow(unused)]

use image::{ImageBuffer, Rgba};
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
        BufferUsage, BufferContents,
    },
    command_buffer::{
        allocator::{
            StandardCommandBufferAllocator,
            StandardCommandBufferAllocatorCreateInfo
        },
        AutoCommandBufferBuilder,
        CommandBufferUsage, 
        CopyBufferInfo, 
        ClearColorImageInfo, 
        CopyImageToBufferInfo, RenderPassBeginInfo, SubpassContents,
    }, 
    sync::{
        self, 
        GpuFuture,
    }, 
    pipeline::{
        ComputePipeline, 
        Pipeline, 
        PipelineBindPoint, graphics::{vertex_input::Vertex, viewport::{Viewport, ViewportState}, input_assembly::InputAssemblyState}, GraphicsPipeline,
    }, 
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, 
        PersistentDescriptorSet, 
        WriteDescriptorSet,
    }, 
    image::{
        StorageImage, 
        ImageDimensions, view::ImageView,
    }, 
    format::{
        Format, 
        ClearColorValue,
    }, render_pass::{Framebuffer, FramebufferCreateInfo, Subpass},
};
    
mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 460

            layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

            layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;

            void main() {
                vec2 norm_coords = (gl_GlobalInvocationID.xy + vec2(0.5)) / vec2(imageSize(img));
                vec2 c = (norm_coords - vec2(0.5)) * 2.0 - vec2(1.0, 0.0);

                vec2 z = vec2(0.0, 0.0);
                float i;
                for(i = 0.0; i < 1.0; i += 0.005) {
                    z = vec2(
                        z.x * z.x - z.y * z.y + c.x,
                        z.y * z.x + z.x * z.y + c.y
                    );

                    if(length(z) > 4.0) break;
                }

                vec4 col = vec4(vec3(i), 1.0);
                imageStore(img, ivec2(gl_GlobalInvocationID.xy), col);
            }
        ",
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 460

            layout(location = 0) in vec2 pos;

            void main() {
                gl_Position = vec4(pos, 0.0, 1.0);
            }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 460

            layout(location = 0) out vec4 f_color;

            void main() {
                f_color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        ",
    }
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct Vert {
    #[format(R32G32_SFLOAT)]
    pos: [f32; 2],
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

    let verts = vec![
        Vert { pos: [-0.5, -0.5] },
        Vert { pos: [ 0.0,  0.5] },
        Vert { pos: [ 0.5, -0.25] },
    ];

    let vs = vs::load(dev.clone()).expect("failed to create vertex shader");
    let fs = fs::load(dev.clone()).expect("failed to create fragment shader");

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [1024.0, 1024.0],
        depth_range: 0.0..1.0,
    };

    let img_buffer = Buffer::from_iter(
        &alloc,
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Download,
            ..Default::default()
        },
        (0..(1024 * 1024 * 4)).map(|_| 0u8)
    ).expect("failed to create img buffer");

    let vert_buffer = Buffer::from_iter(
        &alloc,
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..Default::default()
        },
        verts,
    ).expect("could not create vertex buffer");

    let pass = vulkano::single_pass_renderpass! {
        dev.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: Format::R8G8B8A8_UNORM,
                samples: 1,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {},
        },
    }.expect("could not create render_pass");

    let pipeline = GraphicsPipeline::start()
        .vertex_input_state(Vert::per_vertex())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        .render_pass(Subpass::from(pass.clone(), 0).unwrap())
    .build(dev.clone()).expect("could not build graphics pipeline");

    let img = StorageImage::new(
        &alloc,
        ImageDimensions::Dim2d { width: 1024, height: 1024, array_layers: 1 },
        Format::R8G8B8A8_UNORM,
        Some(queue.queue_family_index()),
    ).expect("could not create img");
    let view = ImageView::new_default(img.clone()).expect("could not create image view");

    let framebuffer = Framebuffer::new(
        pass.clone(),
        FramebufferCreateInfo {
            attachments: vec![view],
            ..Default::default()
        },
    ).expect("could not create framebuffer");

    let cmd_buf_alloc = StandardCommandBufferAllocator::new(
        dev.clone(),
        StandardCommandBufferAllocatorCreateInfo::default()
    );
    let mut cmd_builder = AutoCommandBufferBuilder::primary(
        &cmd_buf_alloc,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    ).expect("could not create cmd_builder");

    cmd_builder
        .begin_render_pass(RenderPassBeginInfo {
            clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
            ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
        }, SubpassContents::Inline)
        .expect("could not begin renderpass")
        .bind_pipeline_graphics(pipeline.clone())
        .bind_vertex_buffers(0, vert_buffer.clone())
        .draw(3, 1, 0, 0)
        .expect("could not draw")
        .end_render_pass()
        .expect("could not end renderpass")
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(img, img_buffer.clone()))
    .expect("could not copy framebuffer to img_buffer");

    let cmd_buf = cmd_builder.build().expect("could not build command buffer");
    let fut = sync::now(dev.clone())
        .then_execute(queue.clone(), cmd_buf)
        .expect("could not execute cmd_buf")
        .then_signal_fence_and_flush()
    .expect("could not flush");
    fut.wait(None).unwrap();

    let buf_img = img_buffer.read().expect("could not read image buffer");
    let img = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buf_img[..])
        .expect("could not create img from data");

    img.save("img.png").expect("could not save img");

    tracing::info!("ok");
}



