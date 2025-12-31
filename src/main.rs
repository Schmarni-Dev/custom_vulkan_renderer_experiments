use std::sync::Arc;

use glam::{Mat4, Quat, Vec3, vec3};
use tracing::info;
use vulkano::{
    Validated, VulkanError, VulkanLibrary,
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, allocator::StandardCommandBufferAllocator,
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, QueueFlags,
        physical::PhysicalDeviceType,
    },
    image::{Image, ImageUsage, view::ImageView},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::StandardMemoryAllocator,
    swapchain::{
        CompositeAlpha, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
        acquire_next_image,
    },
    sync::GpuFuture as _,
};
use winit::{application::ApplicationHandler, event_loop::EventLoop, window::Window};

use crate::renderer::{RenderPipeline, Renderer, View};

pub mod mesh;
pub mod renderer;
// pub mod stardust_backend;

fn main() {
    tracing_subscriber::fmt().init();
    info!("Hello, world!");
    let event_loop = EventLoop::new().unwrap();
    let library = VulkanLibrary::new().unwrap();
    let required_extensions =
        Surface::required_extensions(&event_loop).unwrap() | Renderer::required_instance_exts();
    if library.api_version() < Renderer::required_vulkan_version() {
        panic!("vulkan 1.2 not supported")
    }
    let instance = Instance::new(
        library.clone(),
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )
    .unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    } | Renderer::required_device_exts();
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.intersects(QueueFlags::GRAPHICS)
                        && p.presentation_support(i as u32, &event_loop).unwrap()
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .unwrap();

    info!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    let (device, mut queues) = Device::new(
        physical_device.clone(),
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_features: Renderer::required_device_features(),
            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        Default::default(),
    ));
    let mut app = WinitApp {
        renderer: Renderer::new(
            instance,
            physical_device,
            device,
            memory_allocator,
            queue,
            command_buffer_allocator,
        ),
        render_reqs: None,
        a: 0.0,
        recreate_swapchain: false,
    };
    event_loop.run_app(&mut app).unwrap();
}
struct RenderRequirements {
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    swap_images: Vec<Arc<Image>>,
    render_pipeline: RenderPipeline,
}
struct WinitApp {
    renderer: Renderer,
    render_reqs: Option<RenderRequirements>,
    a: f32,
    recreate_swapchain: bool,
}
impl ApplicationHandler for WinitApp {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );
        let surface = Surface::from_window(self.renderer.instance.clone(), window.clone()).unwrap();
        let window_size = window.inner_size();
        info!(?window_size);

        let (image_format, _) = self
            .renderer
            .dev
            .physical_device()
            .surface_formats(&surface, Default::default())
            .unwrap()[0];
        let (swapchain, images) = {
            let surface_capabilities = self
                .renderer
                .dev
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();

            Swapchain::new(
                self.renderer.dev.clone(),
                surface,
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count.max(2),
                    image_format,
                    image_extent: window_size.into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST,
                    composite_alpha: CompositeAlpha::Opaque,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let render_pipeline = RenderPipeline::new(&self.renderer, image_format, None);

        self.render_reqs = Some(RenderRequirements {
            window,
            swapchain,
            swap_images: images,
            render_pipeline,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            winit::event::WindowEvent::Resized(_physical_size) => {
                self.recreate_swapchain = true;
            }
            winit::event::WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            winit::event::WindowEvent::Destroyed => {
                event_loop.exit();
            }
            // winit::event::WindowEvent::KeyboardInput {
            //     device_id,
            //     event,
            //     is_synthetic,
            // } => {}
            winit::event::WindowEvent::RedrawRequested => {
                if let Some(reqs) = self.render_reqs.as_mut() {
                    if self.recreate_swapchain {
                        self.recreate_swapchain = false;
                        let (swap, images) = reqs
                            .swapchain
                            .recreate(SwapchainCreateInfo {
                                image_extent: reqs.window.inner_size().into(),
                                ..reqs.swapchain.create_info()
                            })
                            .unwrap();
                        reqs.swapchain = swap;
                        reqs.swap_images = images;
                    }
                    let (index, suboptimal, acquire_future) =
                        match acquire_next_image(reqs.swapchain.clone(), None)
                            .map_err(Validated::unwrap)
                        {
                            Ok(v) => v,
                            Err(VulkanError::OutOfDate) => {
                                self.recreate_swapchain = true;
                                return;
                            }
                            Err(err) => panic!("{}", err),
                        };
                    self.recreate_swapchain |= suboptimal;
                    let vertex_positions = &[
                        vec3(-1.0, 0.0, -0.5),
                        vec3(-1.0, 1.0, -0.5),
                        vec3(1.0, 1.0, -0.5),
                    ];
                    let mut builder = AutoCommandBufferBuilder::primary(
                        self.renderer.cballoc.clone(),
                        self.renderer.render_queue.queue_family_index(),
                        CommandBufferUsage::OneTimeSubmit,
                    )
                    .unwrap();
                    let target_image = reqs.swap_images[index as usize].clone();
                    let ratio = target_image.extent()[0] as f32 / target_image.extent()[1] as f32;
                    let mat = Mat4::perspective_rh(90f32.to_radians(), ratio, 0.0, 1000.0)
                        * (Mat4::from_quat(Quat::from_rotation_z(0.3 * self.a))
                            * Mat4::from_translation((Vec3::Y + Vec3::Z) * self.a))
                        .inverse();
                    self.renderer.record_render_commands(
                        &[View { world_to_clip: mat }],
                        ImageView::new_default(target_image).unwrap(),
                        vertex_positions,
                        &reqs.render_pipeline,
                        &mut builder,
                    );
                    self.a += 0.001;
                    let command_buffer = builder.build().unwrap();

                    let future = acquire_future
                        .then_execute(self.renderer.render_queue.clone(), command_buffer)
                        .unwrap()
                        .then_swapchain_present(
                            self.renderer.render_queue.clone(),
                            SwapchainPresentInfo::swapchain_image_index(
                                reqs.swapchain.clone(),
                                index,
                            ),
                        )
                        .then_signal_fence_and_flush()
                        .unwrap();

                    future.wait(None).unwrap();
                }
            }
            _ => {}
        }
    }
    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        self.render_reqs.as_ref().unwrap().window.request_redraw();
    }
}
