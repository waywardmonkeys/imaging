// Copyright 2026 the Imaging Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Vello hybrid backend for `imaging`.
//!
//! This crate provides a headless CPU/GPU renderer that consumes `imaging::Scene` (or accepts
//! commands directly via `imaging::Sink`) and produces an RGBA8 image buffer using
//! `vello_hybrid` + `wgpu`.
//!
//! For reuse, you can also compile an `imaging::Scene` into a backend-native [`CompiledScene`]
//! and render that compiled representation.

#![deny(unsafe_code)]
#![cfg_attr(not(test), warn(unused_crate_dependencies))]

use imaging::{Clip, Composite, Draw, Geometry, Group, Scene, Sink, replay};
use kurbo::{Affine, Shape as _};
use peniko::Brush;
use std::sync::mpsc;
use vello_hybrid::{RenderError, RenderSize, RenderTargetConfig};
use wgpu::{
    CommandEncoderDescriptor, Extent3d, TextureDescriptor, TextureDimension, TextureFormat,
};

/// Errors that can occur when rendering via Vello hybrid.
#[derive(Debug)]
pub enum Error {
    /// The scene is invalid (unbalanced stacks).
    InvalidScene(imaging::ValidateError),
    /// An image brush was encountered; this backend does not support images yet.
    UnsupportedImageBrush,
    /// A filter configuration could not be translated.
    UnsupportedFilter,
    /// No suitable GPU adapter was found.
    NoAdapter,
    /// A GPU device could not be created.
    RequestDevice,
    /// Vello hybrid returned a render error.
    Render(RenderError),
    /// An internal invariant was violated.
    Internal(&'static str),
}

/// Errors that can occur while compiling an `imaging::Scene` into a `vello_hybrid::Scene`.
#[derive(Debug)]
pub enum CompileError {
    /// The input scene is invalid (unbalanced stacks).
    InvalidScene(imaging::ValidateError),
    /// An image brush was encountered; this backend does not support images yet.
    UnsupportedImageBrush,
    /// A filter configuration could not be translated.
    UnsupportedFilter,
    /// An internal invariant was violated.
    Internal(&'static str),
}

/// Backend-native compiled scene type.
pub type CompiledScene = vello_hybrid::Scene;

/// Options that affect compilation from `imaging::Scene` into [`CompiledScene`].
#[non_exhaustive]
#[derive(Copy, Clone, Debug)]
pub struct CompileOptions {
    width: u16,
    height: u16,
    tolerance: f64,
    validate: bool,
}

impl CompileOptions {
    /// Create compile options for a fixed-size target.
    pub fn new(width: u16, height: u16) -> Self {
        Self {
            width,
            height,
            tolerance: 0.1,
            validate: true,
        }
    }

    /// Set the tolerance used when converting shapes to paths.
    #[must_use]
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Enable/disable stack validation of the input `imaging::Scene`.
    #[must_use]
    pub fn with_validate_scene(mut self, validate: bool) -> Self {
        self.validate = validate;
        self
    }

    /// Target width in pixels.
    pub fn width(&self) -> u16 {
        self.width
    }

    /// Target height in pixels.
    pub fn height(&self) -> u16 {
        self.height
    }

    /// Shape-to-path conversion tolerance.
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// Whether `compile` validates `Scene` stack well-nestedness.
    pub fn validate_scene(&self) -> bool {
        self.validate
    }
}

/// Options that affect rendering a compiled scene.
///
/// This is separate from [`CompileOptions`] because some backend options apply equally to
/// immediate-mode execution and rendering precompiled scenes.
#[non_exhaustive]
#[derive(Copy, Clone, Debug, Default)]
pub struct RenderOptions {}

/// Compile an `imaging::Scene` into a backend-native [`CompiledScene`].
///
/// Note: `vello_hybrid::Scene` is sized at creation time, so [`CompileOptions::new`] requires a
/// fixed target size.
pub fn compile(scene: &Scene, opts: &CompileOptions) -> Result<CompiledScene, CompileError> {
    if opts.validate {
        scene.validate().map_err(CompileError::InvalidScene)?;
    }

    let mut compiled = vello_hybrid::Scene::new(opts.width, opts.height);
    compiled.reset();

    let mut recorder = VelloHybridRecorder {
        scene: &mut compiled,
        tolerance: opts.tolerance,
        error: None,
        clip_depth: 0,
        group_depth: 0,
    };
    replay(scene, &mut recorder);

    if let Some(err) = recorder.error.take() {
        return Err(err);
    }
    if recorder.clip_depth != 0 {
        return Err(CompileError::Internal("unbalanced clip stack"));
    }
    if recorder.group_depth != 0 {
        return Err(CompileError::Internal("unbalanced group stack"));
    }

    Ok(compiled)
}

/// Renderer that executes `imaging` commands using `vello_hybrid` + `wgpu`.
#[derive(Debug)]
pub struct VelloHybridRenderer {
    scene: vello_hybrid::Scene,
    renderer: vello_hybrid::Renderer,

    device: wgpu::Device,
    queue: wgpu::Queue,

    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    readback: wgpu::Buffer,
    bytes_per_row: u32,

    width: u16,
    height: u16,
    tolerance: f64,
    error: Option<Error>,
    clip_depth: u32,
    group_depth: u32,
}

impl VelloHybridRenderer {
    /// Create a renderer for a fixed-size target.
    pub fn new(width: u16, height: u16) -> Self {
        Self::try_new(width, height).expect("create imaging_vello_hybrid renderer")
    }

    /// Create a renderer for a fixed-size target.
    ///
    /// This is fallible because `wgpu` may not be able to find a compatible adapter/device
    /// in some sandboxed or headless environments.
    pub fn try_new(width: u16, height: u16) -> Result<Self, Error> {
        let (device, queue) = pollster::block_on(init_device_and_queue())?;
        let (texture, texture_view, readback, bytes_per_row) =
            create_targets(&device, width, height);

        let mut scene = vello_hybrid::Scene::new(width, height);
        scene.reset();

        let renderer = vello_hybrid::Renderer::new(
            &device,
            &RenderTargetConfig {
                format: TextureFormat::Rgba8Unorm,
                width: u32::from(width),
                height: u32::from(height),
            },
        );

        Ok(Self {
            scene,
            renderer,
            device,
            queue,
            texture,
            texture_view,
            readback,
            bytes_per_row,
            width,
            height,
            tolerance: 0.1,
            error: None,
            clip_depth: 0,
            group_depth: 0,
        })
    }

    /// Set the tolerance used when converting shapes to paths.
    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.tolerance = tolerance;
    }

    /// Reset the internal scene and local error state.
    pub fn reset(&mut self) {
        self.scene.reset();
        self.error = None;
        self.clip_depth = 0;
        self.group_depth = 0;
    }

    /// Render a recorded scene and return an RGBA8 buffer (unpremultiplied).
    pub fn render_scene_rgba8(&mut self, scene: &Scene) -> Result<Vec<u8>, Error> {
        scene.validate().map_err(Error::InvalidScene)?;
        self.reset();
        replay(scene, self);
        self.finish_rgba8()
    }

    /// Render a compiled scene and return an RGBA8 buffer (unpremultiplied).
    pub fn render_compiled_rgba8(
        &mut self,
        scene: &CompiledScene,
        _opts: &RenderOptions,
    ) -> Result<Vec<u8>, Error> {
        self.reset();
        render_hybrid_scene_rgba8(
            &mut self.renderer,
            scene,
            &self.device,
            &self.queue,
            &self.texture,
            &self.texture_view,
            &self.readback,
            self.bytes_per_row,
            self.width,
            self.height,
        )
    }

    /// Finish rendering the current command stream and return an RGBA8 buffer (unpremultiplied).
    pub fn finish_rgba8(&mut self) -> Result<Vec<u8>, Error> {
        if let Some(err) = self.error.take() {
            return Err(err);
        }
        if self.clip_depth != 0 {
            return Err(Error::Internal("unbalanced clip stack"));
        }
        if self.group_depth != 0 {
            return Err(Error::Internal("unbalanced group stack"));
        }

        render_hybrid_scene_rgba8(
            &mut self.renderer,
            &self.scene,
            &self.device,
            &self.queue,
            &self.texture,
            &self.texture_view,
            &self.readback,
            self.bytes_per_row,
            self.width,
            self.height,
        )
    }

    fn set_error_once(&mut self, err: Error) {
        if self.error.is_none() {
            self.error = Some(err);
        }
    }

    fn brush_to_paint(
        &mut self,
        brush: Brush,
        composite: Composite,
    ) -> Option<vello_common::paint::PaintType> {
        let brush = brush.multiply_alpha(composite.alpha);
        match brush {
            Brush::Solid(c) => Some(Brush::Solid(c)),
            Brush::Gradient(g) => Some(Brush::Gradient(g)),
            Brush::Image(_) => {
                self.set_error_once(Error::UnsupportedImageBrush);
                None
            }
        }
    }

    fn geometry_to_path(&self, geom: &Geometry) -> kurbo::BezPath {
        match geom {
            Geometry::Rect(r) => r.to_path(self.tolerance),
            Geometry::RoundedRect(rr) => rr.to_path(self.tolerance),
            Geometry::Path(p) => p.clone(),
        }
    }

    fn clip_to_path(&mut self, clip: &Clip) -> (Affine, kurbo::BezPath, peniko::Fill) {
        match clip {
            Clip::Fill {
                transform,
                shape,
                fill_rule,
            } => (*transform, self.geometry_to_path(shape), *fill_rule),
            Clip::Stroke {
                transform,
                shape,
                stroke,
            } => {
                let path = self.geometry_to_path(shape);
                let outline = kurbo::stroke(
                    path.iter(),
                    stroke,
                    &kurbo::StrokeOpts::default(),
                    self.tolerance,
                );
                (*transform, outline, peniko::Fill::NonZero)
            }
        }
    }
}

impl Sink for VelloHybridRenderer {
    fn push_clip(&mut self, clip: Clip) {
        if self.error.is_some() {
            return;
        }
        let (xf, path, fill_rule) = self.clip_to_path(&clip);
        self.scene.set_transform(xf);
        self.scene.set_fill_rule(fill_rule);
        self.scene.push_clip_path(&path);
        self.clip_depth += 1;
    }

    fn pop_clip(&mut self) {
        if self.error.is_some() {
            return;
        }
        if self.clip_depth == 0 {
            self.set_error_once(Error::Internal("pop_clip underflow"));
            return;
        }
        self.scene.pop_clip_path();
        self.clip_depth -= 1;
    }

    fn push_group(&mut self, group: Group) {
        if self.error.is_some() {
            return;
        }
        if !group.filters.is_empty() {
            // vello_hybrid does not support filter layers yet.
            self.set_error_once(Error::UnsupportedFilter);
            return;
        }
        let clip_path = group.clip.as_ref().map(|clip| {
            let (xf, path, fill_rule) = self.clip_to_path(clip);
            self.scene.set_transform(xf);
            self.scene.set_fill_rule(fill_rule);
            path
        });

        let blend = Some(group.composite.blend);
        let opacity = Some(group.composite.alpha);
        self.scene
            .push_layer(clip_path.as_ref(), blend, opacity, None, None);
        self.group_depth += 1;
    }

    fn pop_group(&mut self) {
        if self.error.is_some() {
            return;
        }
        if self.group_depth == 0 {
            self.set_error_once(Error::Internal("pop_group underflow"));
            return;
        }
        self.scene.pop_layer();
        self.group_depth -= 1;
    }

    fn draw(&mut self, draw: Draw) {
        if self.error.is_some() {
            return;
        }

        match draw {
            Draw::Fill {
                transform,
                fill_rule,
                paint,
                paint_transform,
                shape,
                composite,
            } => {
                let Some(paint) = self.brush_to_paint(paint, composite) else {
                    return;
                };
                self.scene.set_transform(transform);
                self.scene.set_fill_rule(fill_rule);
                self.scene
                    .set_paint_transform(paint_transform.unwrap_or(Affine::IDENTITY));

                // Workaround for vello#1408:
                // `Compose::Copy` with a fully transparent solid source is semantically a clear,
                // but vello_hybrid currently treats fully transparent solid paints as "not
                // visible" and skips generating any strips. Avoid that by mapping to `Clear` with
                // an arbitrary opaque paint.
                let (blend, paint) = match (&paint, composite.blend.compose) {
                    (Brush::Solid(c), peniko::Compose::Copy) if c.components[3] == 0.0 => (
                        peniko::BlendMode::new(peniko::Mix::Normal, peniko::Compose::Clear),
                        Brush::Solid(peniko::Color::from_rgba8(0, 0, 0, 255)),
                    ),
                    _ => (composite.blend, paint),
                };

                self.scene.set_blend_mode(blend);
                self.scene.set_paint(paint);

                match shape {
                    Geometry::Rect(r) => self.scene.fill_rect(&r),
                    Geometry::RoundedRect(rr) => {
                        let path = rr.to_path(self.tolerance);
                        self.scene.fill_path(&path);
                    }
                    Geometry::Path(p) => self.scene.fill_path(&p),
                }
            }
            Draw::Stroke {
                transform,
                stroke,
                paint,
                paint_transform,
                shape,
                composite,
            } => {
                let Some(paint) = self.brush_to_paint(paint, composite) else {
                    return;
                };
                self.scene.set_transform(transform);
                self.scene.set_stroke(stroke);
                self.scene
                    .set_paint_transform(paint_transform.unwrap_or(Affine::IDENTITY));
                // Workaround for vello#1408: see the fill path case above.
                let (blend, paint) = match (&paint, composite.blend.compose) {
                    (Brush::Solid(c), peniko::Compose::Copy) if c.components[3] == 0.0 => (
                        peniko::BlendMode::new(peniko::Mix::Normal, peniko::Compose::Clear),
                        Brush::Solid(peniko::Color::from_rgba8(0, 0, 0, 255)),
                    ),
                    _ => (composite.blend, paint),
                };

                self.scene.set_blend_mode(blend);
                self.scene.set_paint(paint);

                match shape {
                    Geometry::Rect(r) => self.scene.stroke_rect(&r),
                    Geometry::RoundedRect(rr) => {
                        let path = rr.to_path(self.tolerance);
                        self.scene.stroke_path(&path);
                    }
                    Geometry::Path(p) => self.scene.stroke_path(&p),
                }
            }
        }
    }
}

fn render_hybrid_scene_rgba8(
    renderer: &mut vello_hybrid::Renderer,
    scene: &vello_hybrid::Scene,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    texture_view: &wgpu::TextureView,
    readback: &wgpu::Buffer,
    bytes_per_row: u32,
    width: u16,
    height: u16,
) -> Result<Vec<u8>, Error> {
    // NOTE: the compiled scene must match this renderer's target size.
    let render_size = RenderSize {
        width: u32::from(width),
        height: u32::from(height),
    };
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("imaging_vello_hybrid render"),
    });

    renderer
        .render(
            scene,
            device,
            queue,
            &mut encoder,
            &render_size,
            texture_view,
        )
        .map_err(Error::Render)?;

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: None,
            },
        },
        Extent3d {
            width: u32::from(width),
            height: u32::from(height),
            depth_or_array_layers: 1,
        },
    );

    queue.submit([encoder.finish()]);

    let slice = readback.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|_| Error::Internal("device poll failed"))?;
    rx.recv()
        .map_err(|_| Error::Internal("map_async callback dropped"))?
        .map_err(|_| Error::Internal("buffer map failed"))?;

    let mapped = slice.get_mapped_range();
    let width_bytes = usize::from(width) * 4;
    let mut pixels = Vec::with_capacity(usize::from(width) * usize::from(height));
    for row in mapped.chunks_exact(bytes_per_row as usize) {
        for px in row[..width_bytes].chunks_exact(4) {
            pixels.push(peniko::color::PremulRgba8::from_u8_array([
                px[0], px[1], px[2], px[3],
            ]));
        }
    }
    drop(mapped);
    readback.unmap();

    let pixmap = vello_common::pixmap::Pixmap::from_parts(pixels, width, height);
    let unpremul = pixmap.take_unpremultiplied();

    let mut bytes = Vec::with_capacity(unpremul.len() * 4);
    for p in unpremul {
        bytes.extend_from_slice(&[p.r, p.g, p.b, p.a]);
    }
    Ok(bytes)
}

#[derive(Debug)]
struct VelloHybridRecorder<'a> {
    scene: &'a mut vello_hybrid::Scene,
    tolerance: f64,
    error: Option<CompileError>,
    clip_depth: u32,
    group_depth: u32,
}

impl VelloHybridRecorder<'_> {
    fn set_error_once(&mut self, err: CompileError) {
        if self.error.is_none() {
            self.error = Some(err);
        }
    }

    fn brush_to_paint(
        &mut self,
        brush: Brush,
        composite: Composite,
    ) -> Option<vello_common::paint::PaintType> {
        let brush = brush.multiply_alpha(composite.alpha);
        match brush {
            Brush::Solid(c) => Some(Brush::Solid(c)),
            Brush::Gradient(g) => Some(Brush::Gradient(g)),
            Brush::Image(_) => {
                self.set_error_once(CompileError::UnsupportedImageBrush);
                None
            }
        }
    }

    fn geometry_to_path(&self, geom: &Geometry) -> kurbo::BezPath {
        match geom {
            Geometry::Rect(r) => r.to_path(self.tolerance),
            Geometry::RoundedRect(rr) => rr.to_path(self.tolerance),
            Geometry::Path(p) => p.clone(),
        }
    }

    fn clip_to_path(&mut self, clip: &Clip) -> (Affine, kurbo::BezPath, peniko::Fill) {
        match clip {
            Clip::Fill {
                transform,
                shape,
                fill_rule,
            } => (*transform, self.geometry_to_path(shape), *fill_rule),
            Clip::Stroke {
                transform,
                shape,
                stroke,
            } => {
                let path = self.geometry_to_path(shape);
                let outline = kurbo::stroke(
                    path.iter(),
                    stroke,
                    &kurbo::StrokeOpts::default(),
                    self.tolerance,
                );
                (*transform, outline, peniko::Fill::NonZero)
            }
        }
    }
}

impl Sink for VelloHybridRecorder<'_> {
    fn push_clip(&mut self, clip: Clip) {
        if self.error.is_some() {
            return;
        }
        let (xf, path, fill_rule) = self.clip_to_path(&clip);
        self.scene.set_transform(xf);
        self.scene.set_fill_rule(fill_rule);
        self.scene.push_clip_path(&path);
        self.clip_depth += 1;
    }

    fn pop_clip(&mut self) {
        if self.error.is_some() {
            return;
        }
        if self.clip_depth == 0 {
            self.set_error_once(CompileError::Internal("pop_clip underflow"));
            return;
        }
        self.scene.pop_clip_path();
        self.clip_depth -= 1;
    }

    fn push_group(&mut self, group: Group) {
        if self.error.is_some() {
            return;
        }
        if !group.filters.is_empty() {
            // vello_hybrid does not support filter layers yet.
            self.set_error_once(CompileError::UnsupportedFilter);
            return;
        }
        let clip_path = group.clip.as_ref().map(|clip| {
            let (xf, path, fill_rule) = self.clip_to_path(clip);
            self.scene.set_transform(xf);
            self.scene.set_fill_rule(fill_rule);
            path
        });

        let blend = Some(group.composite.blend);
        let opacity = Some(group.composite.alpha);
        self.scene
            .push_layer(clip_path.as_ref(), blend, opacity, None, None);
        self.group_depth += 1;
    }

    fn pop_group(&mut self) {
        if self.error.is_some() {
            return;
        }
        if self.group_depth == 0 {
            self.set_error_once(CompileError::Internal("pop_group underflow"));
            return;
        }
        self.scene.pop_layer();
        self.group_depth -= 1;
    }

    fn draw(&mut self, draw: Draw) {
        if self.error.is_some() {
            return;
        }

        match draw {
            Draw::Fill {
                transform,
                fill_rule,
                paint,
                paint_transform,
                shape,
                composite,
            } => {
                let Some(paint) = self.brush_to_paint(paint, composite) else {
                    return;
                };
                self.scene.set_transform(transform);
                self.scene.set_fill_rule(fill_rule);
                self.scene
                    .set_paint_transform(paint_transform.unwrap_or(Affine::IDENTITY));

                // Workaround for vello#1408:
                // `Compose::Copy` with a fully transparent solid source is semantically a clear,
                // but vello_hybrid currently treats fully transparent solid paints as "not
                // visible" and skips generating any strips. Avoid that by mapping to `Clear` with
                // an arbitrary opaque paint.
                let (blend, paint) = match (&paint, composite.blend.compose) {
                    (Brush::Solid(c), peniko::Compose::Copy) if c.components[3] == 0.0 => (
                        peniko::BlendMode::new(peniko::Mix::Normal, peniko::Compose::Clear),
                        Brush::Solid(peniko::Color::from_rgba8(0, 0, 0, 255)),
                    ),
                    _ => (composite.blend, paint),
                };

                self.scene.set_blend_mode(blend);
                self.scene.set_paint(paint);

                match shape {
                    Geometry::Rect(r) => self.scene.fill_rect(&r),
                    Geometry::RoundedRect(rr) => {
                        let path = rr.to_path(self.tolerance);
                        self.scene.fill_path(&path);
                    }
                    Geometry::Path(p) => self.scene.fill_path(&p),
                }
            }
            Draw::Stroke {
                transform,
                stroke,
                paint,
                paint_transform,
                shape,
                composite,
            } => {
                let Some(paint) = self.brush_to_paint(paint, composite) else {
                    return;
                };
                self.scene.set_transform(transform);
                self.scene.set_stroke(stroke);
                self.scene
                    .set_paint_transform(paint_transform.unwrap_or(Affine::IDENTITY));
                // Workaround for vello#1408: see the fill path case above.
                let (blend, paint) = match (&paint, composite.blend.compose) {
                    (Brush::Solid(c), peniko::Compose::Copy) if c.components[3] == 0.0 => (
                        peniko::BlendMode::new(peniko::Mix::Normal, peniko::Compose::Clear),
                        Brush::Solid(peniko::Color::from_rgba8(0, 0, 0, 255)),
                    ),
                    _ => (composite.blend, paint),
                };

                self.scene.set_blend_mode(blend);
                self.scene.set_paint(paint);

                match shape {
                    Geometry::Rect(r) => self.scene.stroke_rect(&r),
                    Geometry::RoundedRect(rr) => {
                        let path = rr.to_path(self.tolerance);
                        self.scene.stroke_path(&path);
                    }
                    Geometry::Path(p) => self.scene.stroke_path(&p),
                }
            }
        }
    }
}

async fn init_device_and_queue() -> Result<(wgpu::Device, wgpu::Queue), Error> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .map_err(|_| Error::NoAdapter)?;

    adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("imaging_vello_hybrid device"),
            required_features: wgpu::Features::empty(),
            ..Default::default()
        })
        .await
        .map_err(|_| Error::RequestDevice)
}

fn create_targets(
    device: &wgpu::Device,
    width: u16,
    height: u16,
) -> (wgpu::Texture, wgpu::TextureView, wgpu::Buffer, u32) {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("imaging_vello_hybrid render target"),
        size: Extent3d {
            width: u32::from(width),
            height: u32::from(height),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let bytes_per_row = (u32::from(width) * 4).next_multiple_of(256);
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("imaging_vello_hybrid readback buffer"),
        size: u64::from(bytes_per_row) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    (texture, texture_view, readback, bytes_per_row)
}

#[cfg(test)]
mod tests {
    use super::*;
    use imaging::{Clip, Draw, Filter, Geometry, Group, Paint};
    use kurbo::{Affine, Rect};

    #[test]
    fn compile_smoke() {
        let mut s = Scene::new();
        s.draw(Draw::Fill {
            transform: Affine::IDENTITY,
            fill_rule: imaging::FillRule::NonZero,
            paint: Paint::Solid(peniko::Color::WHITE),
            paint_transform: None,
            shape: Geometry::Rect(Rect::new(0.0, 0.0, 1.0, 1.0)),
            composite: Composite::default(),
        });

        let opts = CompileOptions::new(32, 32);
        let compiled = compile(&s, &opts);
        assert!(compiled.is_ok());
    }

    #[test]
    fn compile_rejects_unbalanced_scene_by_default() {
        let mut s = Scene::new();
        let _ = s.push_clip(Clip::Fill {
            transform: Affine::IDENTITY,
            shape: Geometry::Rect(Rect::new(0.0, 0.0, 1.0, 1.0)),
            fill_rule: imaging::FillRule::NonZero,
        });

        let opts = CompileOptions::new(32, 32);
        let err = compile(&s, &opts).unwrap_err();
        assert!(matches!(err, CompileError::InvalidScene(_)));
    }

    #[test]
    fn compile_can_report_unsupported_filter_without_scene_validation() {
        let mut s = Scene::new();
        let mut g = Group::default();
        g.filters.push(Filter::blur(1.0));
        let _ = s.push_group(g);

        let opts = CompileOptions::new(32, 32).with_validate_scene(false);
        let err = compile(&s, &opts).unwrap_err();
        assert!(matches!(err, CompileError::UnsupportedFilter));
    }
}
