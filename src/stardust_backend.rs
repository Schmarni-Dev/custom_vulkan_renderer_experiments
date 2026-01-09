use std::{collections::HashMap, fs::File, os::fd::OwnedFd, sync::Arc};

use stardust_xr_fusion::{
    ClientHandle,
    drawable::{
        DmatexMaterialParam, DmatexPlane, DmatexSize, enumerate_dmatex_formats,
        get_primary_render_device_id, import_dmatex, unregister_dmatex,
    },
    node::NodeResult,
};
use timeline_syncobj::{render_node::DrmRenderNode, timeline_syncobj::TimelineSyncObj};
use tracing::{error, info, warn};
use vulkano::{
    device::{Device, DeviceOwned, physical::PhysicalDevice},
    format::Format,
    image::{
        Image, ImageCreateFlags, ImageCreateInfo, ImageFormatInfo, ImageTiling, ImageType,
        ImageUsage, SampleCount, sys::RawImage,
    },
    memory::{
        DedicatedAllocation, DeviceMemory, ExternalMemoryHandleType, ExternalMemoryHandleTypes,
        MemoryAllocateInfo, MemoryPropertyFlags, ResourceMemory,
    },
    sync::{
        GpuFuture, now,
        semaphore::{
            ExternalSemaphoreHandleType, ExternalSemaphoreHandleTypes, ImportSemaphoreFdInfo,
            Semaphore, SemaphoreCreateInfo,
        },
    },
};

pub struct DmatexHandle {
    pub image: Arc<Image>,
    pub timeline: TimelineSyncObj,
    pub dmatex_id: u64,
    client: Arc<ClientHandle>,
}
impl DmatexHandle {
    // TODO: error handling
    pub fn new(
        client: &Arc<ClientHandle>,
        dev: &Arc<Device>,
        render_node: &DrmRenderNode,
        size: DmatexSize,
        format: &DmatexFormat,
        array_layers: Option<u32>,
        usage: ImageUsage,
    ) -> Self {
        let modifiers = dev
            .physical_device()
            .format_properties(format.format)
            .unwrap()
            .drm_format_modifier_properties
            .into_iter()
            .map(|v| v.drm_format_modifier)
            .collect::<Vec<_>>();
        let raw_image = RawImage::new(
            dev.clone(),
            ImageCreateInfo {
                flags: ImageCreateFlags::empty(),
                image_type: match &size {
                    DmatexSize::Dim1D(_) => ImageType::Dim1d,
                    DmatexSize::Dim2D(_) => ImageType::Dim2d,
                    DmatexSize::Dim3D(_) => ImageType::Dim3d,
                },
                format: format.format,
                view_formats: vec![],
                extent: match &size {
                    DmatexSize::Dim1D(v) => [*v, 1, 1],
                    DmatexSize::Dim2D(v) => [v.x, v.y, 1],
                    DmatexSize::Dim3D(v) => (*v).into(),
                },
                array_layers: array_layers.unwrap_or(1),
                tiling: ImageTiling::DrmFormatModifier,
                usage: ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::SAMPLED
                    | ImageUsage::TRANSFER_DST,
                drm_format_modifiers: modifiers,
                external_memory_handle_types: ExternalMemoryHandleTypes::DMA_BUF,
                ..Default::default()
            },
        )
        .unwrap();
        let (modifier, planes) = raw_image.drm_format_modifier().unwrap();
        let mem_reqs = raw_image.memory_requirements();
        info!("modifier {modifier} needs {planes} planes");
        let mems = mem_reqs
            .iter()
            .map(|v| {
                let wants_decicated =
                    v.prefers_dedicated_allocation || v.requires_dedicated_allocation;
                if !wants_decicated {
                    warn!("dmatex image doesn't want a dedicated alloc, too bad");
                }
                let Some((type_index, _)) = dev
                    .physical_device()
                    .memory_properties()
                    .memory_types
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| v.memory_type_bits & (1 << i) != 0)
                    .find(|(_, p)| {
                        // nvidia doesn't put the device local mem first
                        dbg!(   p.property_flags.contains(MemoryPropertyFlags::DEVICE_LOCAL))
                    // not sure if this is even needed, just in case
                        && !p.property_flags.contains(MemoryPropertyFlags::PROTECTED)
                    })
                else {
                    warn!("unable to find memory type for dmatex plane");
                    return None;
                };
                vulkano::memory::DeviceMemory::allocate(
                    dev.clone(),
                    MemoryAllocateInfo {
                        allocation_size: dbg!(v.layout.size()),
                        memory_type_index: type_index as u32,
                        dedicated_allocation: Some(DedicatedAllocation::Image(&raw_image)),
                        export_handle_types: ExternalMemoryHandleTypes::DMA_BUF,
                        ..MemoryAllocateInfo::default()
                    },
                )
                .inspect_err(|err| error!("failed to allocate mem for dmatex plane: {err}"))
                .ok()
            })
            .collect::<Option<Vec<DeviceMemory>>>();
        let mems = mems.unwrap();
        let fds = mems
            .iter()
            .map(|v| v.export_fd(ExternalMemoryHandleType::DmaBuf))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let image = match raw_image.bind_memory(mems.into_iter().map(ResourceMemory::new_dedicated))
        {
            Ok(v) => v,
            Err((err, _, _)) => panic!("failed to bind image mem: {err}"),
        };
        let timeline = TimelineSyncObj::create(render_node).unwrap();
        let dmatex_id = client.generate_id();
        let first_fd = fds[0].try_clone().unwrap();
        let planes = fds
            .into_iter()
            .chain([first_fd])
            .enumerate()
            .map(|(i, v)| {
                let aspect = match i {
                    0 => vulkano::image::ImageAspect::MemoryPlane0,
                    1 => vulkano::image::ImageAspect::MemoryPlane1,
                    2 => vulkano::image::ImageAspect::MemoryPlane2,
                    3 => vulkano::image::ImageAspect::MemoryPlane3,
                    _ => vulkano::image::ImageAspect::Color,
                };
                let layout = image.subresource_layout(aspect, 0, 0).unwrap();
                DmatexPlane {
                    dmabuf_fd: OwnedFd::from(v).into(),
                    offset: layout.offset as u32,
                    row_size: layout.row_pitch as u32,
                    array_element_size: layout.array_pitch.unwrap_or(0) as u32,
                    depth_slice_size: layout.depth_pitch.unwrap_or(0) as u32,
                }
            })
            .collect::<Vec<_>>();
        info!(?planes);
        import_dmatex(
            client,
            dmatex_id,
            size,
            format.fourcc,
            modifier,
            format!("{:?}", format.format).contains("SRGB"),
            array_layers,
            &planes,
            timeline.export().unwrap().into(),
        )
        .unwrap();

        Self {
            image: Arc::new(image),
            timeline,
            dmatex_id,
            client: client.clone(),
        }
    }
    pub fn point_semaphore(&self, point: u64) -> Semaphore {
        // let fd = self.timeline.export_sync_file_point(point).unwrap();
        let semaphore = Semaphore::new(
            self.image.device().clone(),
            SemaphoreCreateInfo {
                export_handle_types: ExternalSemaphoreHandleTypes::SYNC_FD,
                ..Default::default()
            },
        )
        .unwrap();
        // unsafe {
        //     semaphore.import_fd(ImportSemaphoreFdInfo {
        //         file: Some(File::from(fd)),
        //         ..ImportSemaphoreFdInfo::handle_type(
        //             vulkano::sync::semaphore::ExternalSemaphoreHandleType::SyncFd,
        //         )
        //     })
        // }
        // .unwrap();
        // let fd = unsafe { semaphore.export_fd(ExternalSemaphoreHandleType::SyncFd) }.unwrap();
        // self.timeline
        //     .import_sync_file_point(fd.into(), point)
        //     .unwrap();
        semaphore
    }
}
// pub async fn is_phys_dev_allowed(phys_dev: &Arc<PhysicalDevice>, client: &Arc<Client>) {
//     let id = get_primary_render_device_id(client);
// }
pub fn get_phys_dev_node_id(phys_dev: &Arc<PhysicalDevice>) -> u64 {
    let props = phys_dev.properties();
    // Create dev_t from the primary node major/minor numbers
    let major = props.render_major.unwrap() as u64;
    let minor = props.render_minor.unwrap() as u64;
    // On Linux, dev_t is created with makedev(major, minor)
    // which is ((major & 0xfffff000) << 32) | ((major & 0xfff) << 8) | (minor & 0xff)
    ((major & 0xfffff000) << 32) | ((major & 0xfff) << 8) | (minor & 0xff)
}
pub struct Swapchain<const IMAGES: usize = 3> {
    images: [(Arc<DmatexHandle>, u64); IMAGES],
    next_image: usize,
}

impl Swapchain {
    pub fn new(
        client: &Arc<ClientHandle>,
        dev: &Arc<Device>,
        render_node: &DrmRenderNode,
        size: DmatexSize,
        format: &DmatexFormat,
        array_layers: Option<u32>,
        usage: ImageUsage,
    ) -> Self {
        let images = [(); _]
            .map(|_| {
                DmatexHandle::new(
                    client,
                    dev,
                    render_node,
                    size.clone(),
                    format,
                    array_layers,
                    usage,
                )
                .into()
            })
            .map(|v| (v, 0));
        Self {
            images,
            next_image: 0,
        }
    }
    pub async fn next(&mut self) -> SwapchainFrameHandle {
        let (image, previous_release) = &mut self.images[self.next_image];
        self.next_image += 1;
        self.next_image %= 3;
        let acquire_point = if *previous_release != 0 {
            image.timeline.wait_async(*previous_release).unwrap().await;
            *previous_release + 1
        } else {
            0
        };
        let acquire_semaphore = image.point_semaphore(acquire_point);
        *previous_release = acquire_point + 1;
        SwapchainFrameHandle {
            image: image.image.clone(),
            server_acquire_semaphore: Arc::new(acquire_semaphore),
            material_param: DmatexMaterialParam {
                dmatex_id: image.dmatex_id,
                acquire_point,
                release_point: *previous_release,
            },
            tex: image.clone(),
        }
    }
}
pub struct SwapchainFrameHandle {
    pub image: Arc<Image>,
    pub server_acquire_semaphore: Arc<Semaphore>,
    pub material_param: DmatexMaterialParam,
    pub tex: Arc<DmatexHandle>,
}
impl Drop for DmatexHandle {
    fn drop(&mut self) {
        if let Err(err) = unregister_dmatex(&self.client, self.dmatex_id) {
            warn!("failed to unregister dmatex on drop: {err}");
        }
    }
}

// TODO: Docs
#[derive(Debug, Clone)]
pub struct DmatexFormat {
    format: Format,
    fourcc: u32,
    variants: Vec<DmatexFormatVariant>,
}
impl DmatexFormat {
    pub async fn enumerate(
        client: &Arc<ClientHandle>,
        render_node_id: u64,
    ) -> NodeResult<HashMap<Format, DmatexFormat>> {
        let formats = enumerate_dmatex_formats(client, render_node_id).await?;
        let mut out = HashMap::new();
        for v in formats {
            let Some(format) = drm_fourcc::DrmFourcc::try_from(v.format)
                .ok()
                .and_then(Format::from_drm_fourcc)
            else {
                continue;
            };
            if format == Format::R8G8B8A8_UNORM {
                info!(v.format);
            }
            // info!("srgb path: {:?}", format);
            let format = if v.is_srgb {
                let Some(format) = format.to_srgb() else {
                    continue;
                };
                format
            } else {
                format
            };
            out.entry(format)
                .or_insert_with(|| DmatexFormat {
                    format,
                    fourcc: v.format,
                    variants: vec![],
                })
                .variants
                .push(DmatexFormatVariant {
                    modifier: v.drm_modifier,
                    planes: v.planes,
                });
        }

        Ok(out)
    }
}
#[derive(Debug, Clone, Copy)]
pub struct DmatexFormatVariant {
    pub modifier: u64,
    pub planes: u32,
}

pub trait VulkanoFormatExtension: Sized {
    fn from_drm_fourcc(drm_format: drm_fourcc::DrmFourcc) -> Option<Self>;
    fn to_srgb(&self) -> Option<Self>;
}
impl VulkanoFormatExtension for Format {
    fn from_drm_fourcc(drm_format: drm_fourcc::DrmFourcc) -> Option<Self> {
        use Format as F;
        use drm_fourcc::DrmFourcc as D;
        Some(match drm_format {
            D::Abgr1555 | D::Xbgr1555 => F::R5G5B5A1_UNORM_PACK16,
            D::Abgr2101010 | D::Xbgr2101010 => F::A2B10G10R10_UNORM_PACK32,
            D::Abgr4444 | D::Xbgr4444 => F::A4B4G4R4_UNORM_PACK16,
            D::Abgr8888 | D::Xbgr8888 => F::R8G8B8A8_UNORM,
            D::Argb1555 | D::Xrgb1555 => F::A1R5G5B5_UNORM_PACK16,
            D::Argb2101010 | D::Xrgb2101010 => F::A2R10G10B10_UNORM_PACK32,
            D::Argb4444 | D::Xrgb4444 => F::B4G4R4A4_UNORM_PACK16,
            D::Argb8888 | D::Xrgb8888 => F::B8G8R8A8_UNORM,
            D::Bgr565 => F::B5G6R5_UNORM_PACK16,
            D::Bgr888 => F::B8G8R8_UNORM,
            // D::Bgr888_a8 => F::B8G8R8A8_UNORM,
            D::Bgra4444 | D::Bgrx4444 => F::B4G4R4A4_UNORM_PACK16,
            D::Bgra5551 | D::Bgrx5551 => F::B5G5R5A1_UNORM_PACK16,
            D::Bgra8888 | D::Bgrx8888 => F::B8G8R8A8_UNORM,
            D::R16 => F::R16_UNORM,
            D::R8 => F::R8_UNORM,
            D::Rg1616 => F::R16G16_UNORM,
            D::Rg88 => F::R8G8_UNORM,
            D::Rgb565 => F::R5G6B5_UNORM_PACK16,
            D::Rgb888 => F::R8G8B8_UNORM,
            // D::Rgb888_a8 => F::R8G8B8A8_UNORM,
            D::Rgba4444 | D::Rgbx4444 => F::R4G4B4A4_UNORM_PACK16,
            D::Rgba5551 | D::Rgbx5551 => F::R5G5B5A1_UNORM_PACK16,
            D::Rgba8888 | D::Rgbx8888 => F::R8G8B8A8_UNORM,
            D::Abgr16161616f => F::R16G16B16A16_SFLOAT,
            _ => return None,
        })
    }
    fn to_srgb(&self) -> Option<Self> {
        use Format as F;
        Some(match self {
            F::R8_UNORM => F::R8_SRGB,
            F::R8G8_UNORM => F::R8G8_SRGB,
            F::R8G8B8_UNORM => F::R8G8B8_SRGB,
            F::B8G8R8_UNORM => F::B8G8R8_SRGB,
            F::R8G8B8A8_UNORM => F::R8G8B8A8_SRGB,
            F::B8G8R8A8_UNORM => F::B8G8R8A8_SRGB,
            F::A8B8G8R8_UNORM_PACK32 => F::A8B8G8R8_SRGB_PACK32,
            F::BC1_RGB_UNORM_BLOCK => F::BC1_RGB_SRGB_BLOCK,
            F::BC1_RGBA_UNORM_BLOCK => F::BC1_RGBA_SRGB_BLOCK,
            F::BC2_UNORM_BLOCK => F::BC2_SRGB_BLOCK,
            F::BC3_UNORM_BLOCK => F::BC3_SRGB_BLOCK,
            F::BC7_UNORM_BLOCK => F::BC7_SRGB_BLOCK,
            F::ETC2_R8G8B8_UNORM_BLOCK => F::ETC2_R8G8B8_SRGB_BLOCK,
            F::ETC2_R8G8B8A1_UNORM_BLOCK => F::ETC2_R8G8B8A1_SRGB_BLOCK,
            F::ETC2_R8G8B8A8_UNORM_BLOCK => F::ETC2_R8G8B8A8_SRGB_BLOCK,
            _ => return None,
        })
    }
}
