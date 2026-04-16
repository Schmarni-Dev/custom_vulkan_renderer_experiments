use std::{sync::Arc, time::Instant};

use glam::{Mat4, UVec2, Vec3, vec3};
use stardust_xr_cme::{
    format::DmatexFormat, get_phys_dev_node_id, render_device::RenderDevice, swapchain::Swapchain,
};
use stardust_xr_fusion::{
    AsyncEventHandle, Client, ClientHandle,
    drawable::{MaterialParameter, Model, ModelPartAspect},
    project_local_resources,
    root::{RootAspect, RootEvent},
    spatial::Transform,
    values::ResourceID,
};
use tracing::info;
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
    instance::{Instance, InstanceCreateInfo, InstanceExtensions},
    memory::allocator::StandardMemoryAllocator,
    swapchain::{PresentInfo, SwapchainPresentInfo},
};

use crate::{
    renderer::{RenderPipeline, Renderer, View},
    winit_backend::WinitBackend,
};

pub mod mesh;
pub mod renderer;
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
    let render_dev = RenderDevice::primary_server_device(&client).await.unwrap();
    let render_node_id = render_dev.drm_node_id();

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
        render_dev,
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
    render_dev: RenderDevice,
    client: Arc<ClientHandle>,
    renderer: Arc<Renderer>,
) {
    let res = UVec2::new(512, 512);
    let model = Model::create(
        client.get_root(),
        Transform::from_scale([0.1; 3]),
        &ResourceID::new_namespaced("vk", "panel"),
    )
    .unwrap();
    let formats = DmatexFormat::enumerate(&client, &render_dev).await.unwrap();
    let format = formats.get(&Format::R8G8B8A8_SRGB).unwrap();
    let render_pipeline = RenderPipeline::new(&renderer, Format::R8G8B8A8_SRGB, None);

    let mut swapchain = Swapchain::new(
        &client,
        &renderer.dev,
        &render_dev,
        stardust_xr_fusion::drawable::DmatexSize::Dim2D(res.into()),
        format,
        None,
        ImageUsage::COLOR_ATTACHMENT,
    );
    let panel = model.part("Panel").unwrap();
    let mut last_frame = Instant::now();
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

        let ratio = res.x as f32 / res.y as f32;
        let mat = Mat4::perspective_rh(90f32.to_radians(), ratio, 0.0, 1000.0)
            // * (Mat4::from_quat(Quat::from_rotation_z(0.3 * frame_info.elapsed))
                * Mat4::from_translation( Vec3::Z * frame_info.elapsed * 0.1)
            .inverse();
        let vertex_positions = &[
            vec3(-1.0, 0.0, -0.5),
            vec3(-1.0, 1.0, -0.5),
            vec3(1.0, 1.0, -0.5),
        ];
        let info = swapchain.prepare_next_image();
        let mut builder = AutoCommandBufferBuilder::primary(
            renderer.cballoc.clone(),
            renderer.render_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        renderer.record_render_commands(
            &[View { world_to_clip: mat }],
            ImageView::new_default(info.image()).unwrap(),
            vertex_positions,
            &render_pipeline,
            &mut builder,
            [1.0, 0.0, 1.0, 1.0],
        );
        let buffer = builder.build().unwrap();
        let param = info.submit(
            &renderer.dev,
            &renderer.render_queue,
            |wait_semaphore, mut queue, submit_semaphore| unsafe {
                queue
                    .submit(
                        &[SubmitInfo {
                            command_buffers: vec![CommandBufferSubmitInfo::new(buffer)],
                            wait_semaphores: vec![SemaphoreSubmitInfo::new(wait_semaphore)],
                            signal_semaphores: vec![SemaphoreSubmitInfo::new(submit_semaphore)],
                            ..Default::default()
                        }],
                        None,
                    )
                    .unwrap();
            },
        );
        let delta = last_frame.elapsed().as_secs_f64() * 1000.0;
        info!("stardust frametime: {delta}ms");
        panel
            .set_material_parameter("diffuse", MaterialParameter::Dmatex(param))
            .unwrap();
        last_frame = Instant::now();
    }
}
