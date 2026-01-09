use std::{sync::Arc, time::{Duration, Instant}};

use glam::{Mat4, UVec2, vec3};
use timeline_syncobj::{render_node::DrmRenderNode, timeline_syncobj::TimelineSyncObj};
use tracing::{info, warn};
use vulkano::{
    VulkanLibrary,
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferSubmitInfo, CommandBufferUsage, SemaphoreSubmitInfo,
        SubmitInfo, allocator::StandardCommandBufferAllocator,
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, QueueCreateInfo, QueueFlags,
        physical::{PhysicalDevice, PhysicalDeviceType},
    },
    format::Format,
    image::{Image, ImageUsage, view::ImageView},
    instance::{Instance, InstanceCreateInfo, InstanceExtensions},
    memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator},
    sync::{
        PipelineStages,
        semaphore::{ExternalSemaphoreHandleType, ImportSemaphoreFdInfo, Semaphore, SemaphoreType},
    },
};

use crate::renderer::{RenderPipeline, Renderer, View};

pub mod mesh;
pub mod renderer;
// pub mod stardust_backend;
// pub mod winit_backend;

pub trait Backend {
    type Init;
    fn create_init() -> Self::Init;
    fn required_instance_exts(init: &Self::Init) -> InstanceExtensions;
    fn required_device_exts(init: &Self::Init) -> DeviceExtensions;
    fn required_device_features(init: &Self::Init) -> DeviceFeatures;
    fn supports_queue_type(
        init: &Self::Init,
        phys_dev: &Arc<PhysicalDevice>,
        type_index: u32,
    ) -> bool;
    fn create(init: Self::Init, renderer: Arc<Renderer>) -> Self;
}

fn main() {
    tracing_subscriber::fmt().init();
    info!("Hello, world!");

    let library = VulkanLibrary::new().unwrap();
    let required_extensions = Renderer::required_instance_exts();
    if library.api_version() < Renderer::required_vulkan_version() {
        panic!("vulkan 1.2 not supported")
    }
    let instance = Instance::new(
        library.clone(),
        InstanceCreateInfo {
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )
    .unwrap();

    let device_extensions = DeviceExtensions {
        // TODO: move this to stardust specific code
        ext_image_drm_format_modifier: true,
        ext_external_memory_dma_buf: true,
        khr_external_memory: true,
        khr_external_memory_fd: true,
        khr_external_semaphore: true,
        khr_external_semaphore_fd: true,
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
                .position(|(_i, q)| q.queue_flags.intersects(QueueFlags::GRAPHICS))
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
    let renderer = Arc::new(Renderer::new(
        instance,
        physical_device,
        device,
        memory_allocator,
        queue,
        command_buffer_allocator,
    ));
    stardust_loop(renderer);
}

fn stardust_loop(renderer: Arc<Renderer>) {
    let res = UVec2::new(512, 512);
    let render_node_id = 128;
    let render_node = DrmRenderNode::new(render_node_id).unwrap();
    info!("drm node");
    let render_pipeline = RenderPipeline::new(&renderer, Format::R8G8B8A8_SRGB, None);
    info!("pipeline");
    let image = Image::new(
        renderer.malloc.clone(),
        vulkano::image::ImageCreateInfo {
            format: Format::R8G8B8A8_SRGB,
            extent: [res.x, res.y, 1],
            usage: ImageUsage::COLOR_ATTACHMENT,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
    .unwrap();
    info!("image");
    let syncobj = TimelineSyncObj::create(&render_node).unwrap();
    info!("syncobj");
    let semaphore = Arc::new(
        Semaphore::new(
            renderer.dev.clone(),
            vulkano::sync::semaphore::SemaphoreCreateInfo {
                semaphore_type: SemaphoreType::Timeline,
                ..Default::default()
            },
        )
        .unwrap(),
    );
    info!("pre semaphore import info");
    unsafe {
        semaphore
            .import_fd(ImportSemaphoreFdInfo {
                file: Some(syncobj.export().unwrap().into()),
                ..ImportSemaphoreFdInfo::handle_type(ExternalSemaphoreHandleType::OpaqueFd)
            })
            .unwrap()
    };
    info!("post semaphore import info");
    let mut wait_point = 1;
    loop {
        let ratio = res.x as f32 / res.y as f32;
        let mat = Mat4::perspective_rh(90f32.to_radians(), ratio, 0.0, 1000.0)
            // * (Mat4::from_quat(Quat::from_rotation_z(0.3 * frame_info.elapsed))
            //     * Mat4::from_translation((Vec3::Y + Vec3::Z) * frame_info.elapsed))
            .inverse();
        let vertex_positions = &[
            vec3(-1.0, 0.0, -0.5),
            vec3(-1.0, 1.0, -0.5),
            vec3(1.0, 1.0, -0.5),
        ];
        let mut builder = AutoCommandBufferBuilder::primary(
            renderer.cballoc.clone(),
            renderer.render_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        renderer.record_render_commands(
            &[View { world_to_clip: mat }],
            ImageView::new_default(image.clone()).unwrap(),
            vertex_positions,
            &render_pipeline,
            &mut builder,
        );
        let buffer = builder.build().unwrap();
        renderer.render_queue.with(|mut queue| unsafe {
            queue
                .submit(
                    &[SubmitInfo {
                        command_buffers: vec![CommandBufferSubmitInfo::new(buffer)],
                        signal_semaphores: vec![SemaphoreSubmitInfo {
                            value: wait_point,
                            stages: PipelineStages::ALL_COMMANDS,
                            ..SemaphoreSubmitInfo::new(semaphore.clone())
                        }],
                        ..Default::default()
                    }],
                    None,
                )
                .unwrap()
        });
        let time = Instant::now();

        syncobj.blocking_wait(wait_point, None).unwrap();
        let diff = time.elapsed().as_secs_f64() * 1000.0;
        let time = Instant::now();
        let diff2 = time.elapsed().as_secs_f64() * 1000.0;

        println!("render time: {diff}ms, baseline time: {diff2}");
        wait_point += 1;
    }
}
