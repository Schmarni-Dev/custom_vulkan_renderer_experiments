use std::borrow::Cow;

use glam::{Mat4, Vec3A};


pub struct Mesh {
    vertecies: Cow<'static, [Vec3A]>,
    transform: Mat4,
}
