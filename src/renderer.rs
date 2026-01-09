use std::{collections::HashSet, sync::Arc};

use glam::{Mat4, Vec3};
use vulkano::{
    Version,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    command_buffer::{
        AutoCommandBufferBuilder, RenderingAttachmentInfo, RenderingInfo,
        allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::{WriteDescriptorSet, layout::DescriptorSetLayoutCreateFlags},
    device::{Device, DeviceExtensions, DeviceFeatures, Queue, physical::PhysicalDevice},
    format::Format,
    image::view::ImageView,
    instance::{Instance, InstanceExtensions},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        DynamicState, GraphicsPipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            subpass::{PipelineRenderingCreateInfo, PipelineSubpassType},
            vertex_input::{Vertex, VertexDefinition as _},
            viewport::{Scissor, Viewport, ViewportState},
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    render_pass::{AttachmentLoadOp, AttachmentStoreOp},
};

#[derive(Debug)]
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
            khr_dynamic_rendering: true,
            khr_dynamic_rendering_local_read: false,
            ext_extended_dynamic_state: true,
            khr_push_descriptor: true,
            // khr_draw_indirect_count: true,
            khr_global_priority: false,
            // khr_performance_query: (),
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
            dynamic_rendering: true,
            extended_dynamic_state: true,
            runtime_descriptor_array: true,
            timeline_semaphore: true,
            synchronization2: true,
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
        views: &[View],
        render_target: Arc<ImageView>,
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
        let views_buffer = Buffer::from_iter(
            self.malloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            views.iter().map(|v| ViewData {
                world_to_clip: v.world_to_clip.to_cols_array_2d(),
            }),
        )
        .unwrap();

        builder
            .begin_rendering(RenderingInfo {
                layer_count: 0,
                view_mask: mask_from_len(views.len()),
                color_attachments: vec![Some(RenderingAttachmentInfo {
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                    clear_value: Some([0.0, 1.0, 1.0, 1.0].into()),
                    ..RenderingAttachmentInfo::image_view(render_target.clone())
                })],
                depth_attachment: None,
                stencil_attachment: None,
                ..Default::default()
            })
            .unwrap()
            .set_viewport_with_count(
                [Viewport {
                    offset: [0.0, render_target.image().extent()[1] as f32],
                    extent: [
                        render_target.image().extent()[0] as f32,
                        -(render_target.image().extent()[1] as f32),
                    ],
                    depth_range: 0.0..=1.0,
                }]
                .into_iter()
                .collect(),
            )
            .unwrap()
            .set_scissor_with_count([Scissor::default()].into_iter().collect())
            .unwrap()
            .bind_pipeline_graphics(render_pipeline.pipeline.clone())
            .unwrap()
            .push_descriptor_set(
                vulkano::pipeline::PipelineBindPoint::Graphics,
                render_pipeline.pipeline.layout().clone(),
                1,
                [WriteDescriptorSet::buffer(0, views_buffer)]
                    .into_iter()
                    .collect(),
            )
            .unwrap()
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .unwrap();

        // Drawing commands are broadcast to each view in the view mask of the active renderpass
        // which means only a single draw call is needed to draw to multiple layers of the
        // framebuffer.
        unsafe { builder.draw(vertex_buffer.len() as u32, 1, 0, 0) }.unwrap();

        builder.end_rendering().unwrap();
    }
}
fn mask_from_len(len: usize) -> u32 {
    2u32.pow(len as u32) - 1
}
pub struct RenderPipeline {
    pipeline: Arc<GraphicsPipeline>,
}
impl RenderPipeline {
    pub fn new(vk: &Renderer, view_format: Format, multiview_amount: Option<usize>) -> Self {
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
            let layout = PipelineLayout::new(vk.dev.clone(), {
                let mut v = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);
                v.set_layouts[1].flags |= DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR;
                v.set_layouts[1]
                    .bindings
                    .get_mut(&0)
                    .unwrap()
                    .descriptor_count = 1;
                v.into_pipeline_layout_create_info(vk.dev.clone()).unwrap()
            })
            .unwrap();
            GraphicsPipeline::new(
                vk.dev.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState {
                        viewports: Default::default(),
                        scissors: Default::default(),
                        ..Default::default()
                    }),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        1,
                        ColorBlendAttachmentState::default(),
                    )),
                    dynamic_state: HashSet::from_iter([
                        DynamicState::ViewportWithCount,
                        DynamicState::ScissorWithCount,
                    ]),
                    subpass: Some(PipelineSubpassType::BeginRendering(
                        PipelineRenderingCreateInfo {
                            view_mask: multiview_amount.map(|v| mask_from_len(v)).unwrap_or(1),
                            color_attachment_formats: vec![Some(view_format)],
                            ..PipelineRenderingCreateInfo::default()
                        },
                    )),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };
        Self { pipeline }
    }
}
mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
                #version 450
                #extension GL_EXT_multiview : enable
                #extension GL_EXT_nonuniform_qualifier : enable

                layout(location = 0) in vec3 position;

                layout(set = 1, binding = 0) uniform InData {
                    mat4 world_to_clip;
                } views[];

                void main() {
                    gl_Position = views[gl_ViewIndex].world_to_clip * vec4(position, 1.0);
                }
            ",
    }
}
mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
                #version 450
                #extension GL_EXT_multiview : enable

                layout(location = 0) out vec4 f_color;

                void main() {
                    f_color = vec4(1.0, 0.0, float(gl_ViewIndex)*0.25, 1.0);
                }
            ",
    }
}
pub struct View {
    pub world_to_clip: Mat4,
}
#[derive(BufferContents, Vertex, Clone, Copy)]
#[repr(C)]
struct VertexData {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
}
#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct ViewData {
    world_to_clip: [[f32; 4]; 4],
}
