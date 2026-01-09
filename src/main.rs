use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use glam::{Mat4, Quat, UVec2, Vec3, vec3};
use stardust_xr_fusion::{
    AsyncEventHandle, Client, ClientHandle,
    drawable::{MaterialParameter, Model, ModelPartAspect, get_primary_render_device_id},
    project_local_resources,
    root::{RootAspect, RootEvent},
    spatial::Transform,
    values::ResourceID,
};
use timeline_syncobj::render_node::DrmRenderNode;
use tokio::time::sleep;
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
    image::{ImageUsage, view::ImageView},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions},
    memory::allocator::StandardMemoryAllocator,
    sync::{
        PipelineStages,
        semaphore::{Semaphore, SemaphoreWaitFlags, SemaphoreWaitInfo},
    },
};

use crate::{
    renderer::{RenderPipeline, Renderer, View},
    stardust_backend::{DmatexFormat, Swapchain, get_phys_dev_node_id},
    winit_backend::WinitBackend,
};

pub mod mesh;
pub mod renderer;
pub mod stardust_backend;
pub mod winit_backend;

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

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().init();
    info!("Hello, world!");
    let client = Client::connect().await.unwrap();
    client
        .setup_resources(&[&project_local_resources!("res")])
        .unwrap();
    let async_loop = client.async_event_loop();
    let client = async_loop.client_handle.clone();
    let render_node_id = get_primary_render_device_id(&client).await.unwrap();

    let winit_init = WinitBackend::create_init();
    let library = VulkanLibrary::new().unwrap();
    let required_extensions =
        WinitBackend::required_instance_exts(&winit_init) | Renderer::required_instance_exts();
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
        khr_swapchain: true,
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
        .filter(|p| get_phys_dev_node_id(p) == render_node_id)
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.intersects(QueueFlags::GRAPHICS)
                        && WinitBackend::supports_queue_type(&winit_init, &p, i as u32)
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
    let renderer = Arc::new(Renderer::new(
        instance,
        physical_device,
        device,
        memory_allocator,
        queue,
        command_buffer_allocator,
    ));
    tokio::spawn(stardust_loop(
        async_loop.get_event_handle(),
        client,
        renderer.clone(),
    ))
    .await
    .unwrap();

    // tokio::task::block_in_place(|| {
    //     WinitBackend::create(winit_init, renderer.clone()).run();
    // });
}

async fn stardust_loop(
    event_handle: AsyncEventHandle,
    client: Arc<ClientHandle>,
    renderer: Arc<Renderer>,
) {
    let res = UVec2::new(512, 512);
    let render_node_id = get_primary_render_device_id(&client).await.unwrap();
    let render_node = DrmRenderNode::new(render_node_id).unwrap();
    let model = Model::create(
        client.get_root(),
        Transform::identity(),
        &ResourceID::new_namespaced("vk", "panel"),
    )
    .unwrap();
    let formats = DmatexFormat::enumerate(&client, render_node_id)
        .await
        .unwrap();
    let format = formats.get(&Format::R8G8B8A8_SRGB).unwrap();
    let render_pipeline = RenderPipeline::new(&renderer, Format::R8G8B8A8_SRGB, None);
    let mut swapchain = Swapchain::new(
        &client,
        &renderer.dev,
        &render_node,
        stardust_xr_fusion::drawable::DmatexSize::Dim2D(res.into()),
        format,
        None,
        ImageUsage::COLOR_ATTACHMENT,
    );
    let panel = model.part("Panel").unwrap();
    sleep(Duration::from_millis(100)).await;
    loop {
        event_handle.wait().await;
        let frame_info = match client.get_root().recv_root_event() {
            Some(RootEvent::Ping { response }) => {
                response.send_ok(());
                continue;
            }
            Some(RootEvent::Frame { info }) => info,
            _ => {
                continue;
            }
        };
        info!("about to wait on next image");
        let info = swapchain.next().await;
        info!("waited on next image");

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
            ImageView::new_default(info.image).unwrap(),
            vertex_positions,
            &render_pipeline,
            &mut builder,
        );
        let buffer = builder.build().unwrap();
        let timeline_semaphore = Arc::new(
            Semaphore::new(
                renderer.dev.clone(),
                vulkano::sync::semaphore::SemaphoreCreateInfo {
                    semaphore_type: vulkano::sync::semaphore::SemaphoreType::Timeline,
                    initial_value: 0,
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        renderer.render_queue.with(|mut queue| unsafe {
            queue
                .submit(
                    &[SubmitInfo {
                        command_buffers: vec![CommandBufferSubmitInfo::new(buffer)],
                        signal_semaphores: vec![
                            // SemaphoreSubmitInfo {
                            //     stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT,
                            //     ..SemaphoreSubmitInfo::new(info.server_acquire_semaphore)
                            // },
                            SemaphoreSubmitInfo {
                                value: 1,
                                stages: PipelineStages::ALL_COMMANDS,
                                ..SemaphoreSubmitInfo::new(timeline_semaphore.clone())
                            },
                        ],
                        ..Default::default()
                    }],
                    None,
                )
                .unwrap()
        });
        let time = Instant::now();
        timeline_semaphore
            .wait(
                SemaphoreWaitInfo {
                    value: 1,
                    ..Default::default()
                },
                None,
            )
            .unwrap();
        let diff = time.elapsed().as_secs_f64() * 1000.0;

        warn!("render time: {diff}ms");
        unsafe { info.tex.timeline.signal(info.material_param.acquire_point) }.unwrap();
        panel
            .set_material_parameter("diffuse", MaterialParameter::Dmatex(info.material_param))
            .unwrap();
        info!("param set");
    }
}
