#![allow(unused)]

use std::sync::Arc;

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
        QueueCreateInfo, DeviceExtensions, physical::PhysicalDevice, Queue,
    }, 
    memory::allocator::{
        StandardMemoryAllocator, 
        AllocationCreateInfo,
        MemoryUsage,
    },
    buffer::{
        Buffer,
        BufferCreateInfo,
        BufferUsage, BufferContents, Subbuffer,
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
        CopyImageToBufferInfo, RenderPassBeginInfo, SubpassContents, PrimaryAutoCommandBuffer,
    }, 
    sync::{
        self, 
        GpuFuture, FlushError,
    }, 
    pipeline::{
        ComputePipeline, 
        Pipeline, 
        PipelineBindPoint, graphics::{vertex_input::{Vertex, BuffersDefinition}, viewport::{Viewport, ViewportState}, input_assembly::InputAssemblyState}, GraphicsPipeline,
    }, 
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, 
        PersistentDescriptorSet, 
        WriteDescriptorSet,
    }, 
    image::{
        StorageImage, 
        ImageDimensions, view::ImageView, ImageUsage, SwapchainImage,
    }, 
    format::{
        Format, 
        ClearColorValue,
    }, render_pass::{Framebuffer, FramebufferCreateInfo, Subpass, RenderPass}, swapchain::{Surface, Swapchain, SwapchainCreateInfo, SwapchainCreationError, self, AcquireError, SwapchainPresentInfo, PresentInfo}, shader::ShaderModule,
};
use vulkano_win::VkSurfaceBuild;
use winit::{event_loop::EventLoop, window::{WindowBuilder, Window}};
    
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

impl Vert {
    fn triangle_ex() -> [Self; 3] {
        [
            Self { pos: [-0.5, -0.5 ] },
            Self { pos: [ 0.0, -0.5 ] },
            Self { pos: [ 0.5, -0.25] },
        ]
    }
}

fn select_physical_device(
    instance: &Arc<Instance>,
    surface: &Arc<Surface>,
    dev_exts: &DeviceExtensions,
) -> (Arc<PhysicalDevice>, u32) {
    instance.enumerate_physical_devices()
        .expect("could not enumerate devices")
        .filter(|p| p.supported_extensions().contains(&dev_exts))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.contains(QueueFlags::GRAPHICS)
                        && p.surface_support(i as _, &surface).unwrap_or(false)
                })
            .map(|q| (p, q as u32))
        })
        .min_by_key(|(p, _)| {
            use vulkano::device::physical::PhysicalDeviceType::*;
            match p.properties().device_type {
                DiscreteGpu => 0,
                IntegratedGpu => 1,
                VirtualGpu => 2,
                Cpu => 3,
                _ => 4,
            }
        })
    .expect("no devices available")
}

fn get_render_pass(device: Arc<Device>, swapchain: &Arc<Swapchain>) -> Arc<RenderPass> {
    vulkano::single_pass_renderpass! {
        device,
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.image_format(),
                samples: 1,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {},
        },
    }.unwrap()
}

fn get_framebuffers(
    images: &[Arc<SwapchainImage>],
    render_pass: &Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    images.iter()
        .map(|img| {
            let view = ImageView::new_default(img.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            ).unwrap()
        })
    .collect()
}

fn get_pipeline(
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
) -> Arc<GraphicsPipeline> {
    GraphicsPipeline::start()
        .vertex_input_state(Vert::per_vertex())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .render_pass(Subpass::from(render_pass, 0).unwrap())
        .build(device)
    .expect("could not create gfx pipeline")
}

fn get_cmd_buffers(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    cmd_alloc: &StandardCommandBufferAllocator,
    pipeline: &Arc<GraphicsPipeline>,
    framebuffers: &Vec<Arc<Framebuffer>>,
    vertex_buffer: &Subbuffer<[Vert]>,
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    framebuffers.iter()
        .map(|fb| {
            let mut builder = AutoCommandBufferBuilder::primary(
                cmd_alloc,
                queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            ).expect("could not create cmd buffer builder");

            builder
                .begin_render_pass(RenderPassBeginInfo {
                    clear_values: vec![Some([0.1, 0.1, 0.1, 1.0].into())],
                    ..RenderPassBeginInfo::framebuffer(fb.clone())
                }, SubpassContents::Inline)
                .expect("could not begin render pass")
                .bind_pipeline_graphics(pipeline.clone())
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .draw(vertex_buffer.len() as _, 1, 0, 0)
                .expect("could not draw")
                .end_render_pass()
            .expect("could not end renderpass");

            Arc::new(builder.build().expect("could not build cmd_buf"))
        })
    .collect()
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("warn,shadertoy=info")
    .init();

    let lib = VulkanLibrary::new().expect("could not load vulkan lib");
    let required_exts = vulkano_win::required_extensions(&lib);
    let instance = Instance::new(lib, InstanceCreateInfo {
        enabled_extensions: required_exts,
        ..Default::default()
    }).expect("could not create instance");

    let event_loop = EventLoop::new();
    let window = Arc::new(WindowBuilder::new()
        .build(&event_loop)
    .expect("could not create surface"));

    let surface = vulkano_win::create_surface_from_winit(window.clone(), instance.clone())
        .expect("could not get surface");

    let dev_exts = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };

    let (physical_dev, queue_family_index) = select_physical_device(
        &instance,
        &surface,
        &dev_exts,
    );

    let (dev, mut queues) = Device::new(
        physical_dev.clone(),
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: dev_exts,
            ..Default::default()
        },
    ).expect("failed to create device");
    let queue = queues.next().unwrap();

    let caps = physical_dev
        .surface_capabilities(&surface, Default::default())
    .expect("could not get surface caps");

    let dims = window.inner_size();
    let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
    let image_format = Some(
        physical_dev
            .surface_formats(&surface, Default::default())
            .unwrap()[0].0,
    );

    let (mut swapchain, images) = Swapchain::new(
        dev.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: caps.min_image_count + 1,
            image_format,
            image_extent: dims.into(),
            image_usage: ImageUsage::COLOR_ATTACHMENT,
            composite_alpha,
            ..Default::default()
        },
    ).expect("could not create swapchain");

    let render_pass = get_render_pass(dev.clone(), &swapchain);
    let framebuffers = get_framebuffers(&images, &render_pass);

    let alloc = StandardMemoryAllocator::new_default(dev.clone());

    let vert_buf = Buffer::from_iter(
        &alloc,
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..Default::default()
        },
        Vert::triangle_ex()
    ).expect("could not create buffer");

    let vs = vs::load(dev.clone()).expect("could not load vertex shader");
    let fs = fs::load(dev.clone()).expect("could not load fragment shader");

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: window.inner_size().into(),
        depth_range: 0.0..1.0,
    };

    let pipeline = get_pipeline(
        dev.clone(),
        vs.clone(),
        fs.clone(),
        render_pass.clone(),
        viewport.clone(),
    );

    let cmd_alloc = StandardCommandBufferAllocator::new(dev.clone(), Default::default());

    let mut cmd_buf = get_cmd_buffers(
        &dev,
        &queue,
        &cmd_alloc,
        &pipeline,
        &framebuffers,
        &vert_buf,
    );

    let mut window_resized = false;
    let mut recreate_swapchain = false;

    tracing::info!("ok");
    event_loop.run(move |event, _, ctrl| {
        use winit::event::{Event, WindowEvent};
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                ctrl.set_exit();
            },
            Event::WindowEvent { event: WindowEvent::Resized(_), .. } => {
                window_resized = true;
            },
            Event::MainEventsCleared => {},
            Event::RedrawRequested(_) => {
                let (image_i, suboptimal, acquire_fut) = match swapchain::acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    },
                    Err(e) => panic!("failed to acquire next image"),
                };

                if suboptimal { recreate_swapchain = true }
                let execution = sync::now(dev.clone())
                    .join(acquire_fut)
                    .then_execute(queue.clone(), cmd_buf[image_i as usize].clone())
                    .expect("could not execute")
                    .then_swapchain_present(
                        queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_i),
                    )
                .then_signal_fence_and_flush();

                match execution {
                    Ok(fut) => fut.wait(None).unwrap(),
                    Err(FlushError::OutOfDate) => recreate_swapchain = true,
                    Err(e) => tracing::warn!("failed to flush future: {e}"),
                }
            },
            Event::RedrawEventsCleared => {
                if window_resized || recreate_swapchain {
                    recreate_swapchain = false;
                    let new_dim = window.inner_size();

                    let (new_swapchain, new_images) = match swapchain.recreate(SwapchainCreateInfo {
                        image_extent: new_dim.into(),
                        ..swapchain.create_info()
                    }) {
                        Ok(r) => r,
                        Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                        Err(e) => panic!("failed to recreate swapchain"),
                    };
                    swapchain = new_swapchain;
                    let new_framebuffers = get_framebuffers(&new_images, &render_pass);

                    if window_resized {
                        window_resized = false;

                        viewport.dimensions = new_dim.into();
                        let new_pipeline = get_pipeline(
                            dev.clone(),
                            vs.clone(),
                            fs.clone(),
                            render_pass.clone(),
                            viewport.clone(),
                        );

                        cmd_buf = get_cmd_buffers(
                            &dev,
                            &queue,
                            &cmd_alloc,
                            &new_pipeline,
                            &new_framebuffers,
                            &vert_buf,
                        );
                    }
                }
            },
            _ => (),
        }
    });

    tracing::info!("exiting");
}





