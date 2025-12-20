use std::sync::Arc;

use glam::{Mat4, Vec3};
use vulkano::{
    Version,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    command_buffer::{
        AutoCommandBufferBuilder, RenderPassBeginInfo, allocator::StandardCommandBufferAllocator,
    },
    device::{Device, DeviceExtensions, DeviceFeatures, Queue, physical::PhysicalDevice},
    image::{Image, ImageLayout, SampleCount, view::ImageView},
    instance::{Instance, InstanceExtensions},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition as _},
            viewport::{Scissor, Viewport, ViewportState},
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    render_pass::{
        AttachmentDescription, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp,
        Framebuffer, FramebufferCreateInfo, RenderPass, RenderPassCreateInfo, Subpass,
        SubpassDescription,
    },
};

pub struct Renderer {
    pub instance: Arc<Instance>,
    pub phys_dev: Arc<PhysicalDevice>,
    pub dev: Arc<Device>,
    pub malloc: Arc<StandardMemoryAllocator>,
    pub cballoc: Arc<StandardCommandBufferAllocator>,
    pub render_queue: Arc<Queue>,
}

impl Renderer {
    pub const fn required_instance_exts() -> InstanceExtensions {
        InstanceExtensions {
            // khr_external_memory_capabilities: (),
            // khr_external_semaphore_capabilities: (),
            ..InstanceExtensions::empty()
        }
    }
    // required vulkan version: 1.2
    pub const fn required_device_exts() -> DeviceExtensions {
        DeviceExtensions {
            // khr_draw_indirect_count: true,
            khr_fragment_shading_rate: true,
            khr_global_priority: false,
            khr_line_rasterization: false,
            // khr_performance_query: (),
            khr_push_descriptor: false,
            khr_synchronization2: true,
            // khr_timeline_semaphore: true,
            ext_external_memory_acquire_unmodified: false,
            ext_external_memory_dma_buf: false,
            // ext_host_image_copy: false,
            // ext_image_compression_control: (),
            // ext_image_compression_control_swapchain: (),
            ext_image_drm_format_modifier: false,
            // ext_multi_draw: (),
            ..DeviceExtensions::empty()
        }
    }
    pub const fn required_vulkan_version() -> Version {
        Version::V1_2
    }
    pub const fn required_device_features() -> DeviceFeatures {
        DeviceFeatures {
            multiview: true,
            ..DeviceFeatures::empty()
        }
    }
    pub fn new(
        instance: Arc<Instance>,
        phys_dev: Arc<PhysicalDevice>,
        dev: Arc<Device>,
        malloc: Arc<StandardMemoryAllocator>,
        render_queue: Arc<Queue>,
        cballoc: Arc<StandardCommandBufferAllocator>,
    ) -> Self {
        Self {
            instance,
            phys_dev,
            dev,
            malloc,
            render_queue,
            cballoc,
        }
    }
    pub fn record_render_commands<L>(
        &self,
        // views: &[View],
        vertex_positions: &[Vec3],
        render_pipeline: &RenderPipeline,
        builder: &mut AutoCommandBufferBuilder<L>,
    ) {
        let vertex_buffer = Buffer::from_iter(
            self.malloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertex_positions.iter().map(|pos| VertexData {
                position: pos.to_array(),
            }),
        )
        .unwrap();

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                    ..RenderPassBeginInfo::framebuffer(render_pipeline.framebuffer.clone())
                },
                Default::default(),
            )
            .unwrap()
            .bind_pipeline_graphics(render_pipeline.pipeline.clone())
            .unwrap()
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .unwrap();

        // Drawing commands are broadcast to each view in the view mask of the active renderpass
        // which means only a single draw call is needed to draw to multiple layers of the
        // framebuffer.
        unsafe { builder.draw(vertex_buffer.len() as u32, 1, 0, 0) }.unwrap();

        builder.end_render_pass(Default::default()).unwrap();
    }
}
pub struct RenderPipeline {
    framebuffer: Arc<Framebuffer>,
    pipeline: Arc<GraphicsPipeline>,
}
impl RenderPipeline {
    pub fn new(vk: &Renderer, views: &[View]) -> Self {
        let render_pass = RenderPass::new(
            vk.dev.clone(),
            RenderPassCreateInfo {
                attachments: views
                    .iter()
                    .map(|view| AttachmentDescription {
                        format: view.target.format(),
                        samples: SampleCount::Sample1,
                        load_op: AttachmentLoadOp::Clear,
                        store_op: AttachmentStoreOp::Store,
                        initial_layout: ImageLayout::General,
                        final_layout: ImageLayout::General,
                        ..Default::default()
                    })
                    .collect(),
                subpasses: vec![SubpassDescription {
                    // The view mask indicates which layers of the framebuffer should be rendered for
                    // each subpass.
                    view_mask: (views.len() > 1)
                        .then(|| 2u32.pow(views.len() as u32 + 1) - 1)
                        .unwrap_or(0),
                    color_attachments: views
                        .iter()
                        .enumerate()
                        .map(|(i, _)| {
                            Some(AttachmentReference {
                                attachment: i as u32,
                                layout: ImageLayout::General,
                                ..Default::default()
                            })
                        })
                        .collect(),
                    ..Default::default()
                }],
                // The correlated view masks indicate sets of views that may be more efficient to render
                // concurrently.
                correlated_view_masks: (views.len() > 1)
                    .then(|| vec![2u32.pow(views.len() as u32 + 1) - 1])
                    .unwrap_or_default(),

                ..Default::default()
            },
        )
        .unwrap();
        let framebuffer = unsafe {
            Framebuffer::new_unchecked(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: views.iter().map(|view| view.target.clone()).collect(),
                    layers: views.len() as u32,
                    ..Default::default()
                },
            )
        }
        .unwrap();
        let pipeline = {
            let vs = vs::load(vk.dev.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = fs::load(vk.dev.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let vertex_input_state = VertexData::per_vertex().definition(&vs).unwrap();
            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];
            let layout = PipelineLayout::new(
                vk.dev.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(vk.dev.clone())
                    .unwrap(),
            )
            .unwrap();
            let subpass = Subpass::from(render_pass, 0).unwrap();
            GraphicsPipeline::new(
                vk.dev.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState {
                        viewports: views
                            .iter()
                            .map(|v| Viewport {
                                offset: [0.0, 0.0],
                                extent: [
                                    v.target_backing.extent()[0] as f32,
                                    v.target_backing.extent()[1] as f32,
                                ],
                                depth_range: 0.0..=1.0,
                            })
                            .into_iter()
                            .collect(),
                        scissors: views.iter().map(|_| Scissor::default()).collect(),
                        ..Default::default()
                    }),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments(),
                        ColorBlendAttachmentState::default(),
                    )),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };
        Self {
            framebuffer,
            pipeline,
        }
    }
}
mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
                #version 450
                #extension GL_EXT_multiview : enable

                layout(location = 0) in vec3 position;

                void main() {
                    gl_Position = vec4(position, 1.0) + gl_ViewIndex * vec4(0.25, 0.25, 0.25, 0.0);
                }
            ",
    }
}
mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
                #version 450

                layout(location = 0) out vec4 f_color;

                void main() {
                    f_color = vec4(1.0, 0.0, 0.0, 1.0);
                }
            ",
    }
}
pub struct View {
    pub world_to_view: Mat4,
    pub target_backing: Arc<Image>,
    pub target: Arc<ImageView>,
}
#[derive(BufferContents, Vertex, Clone, Copy)]
#[repr(C)]
struct VertexData {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
}
