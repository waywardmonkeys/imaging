// Copyright 2026 the Imaging Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! SVG export backend for `imaging`.
//!
//! This crate provides a small [`imaging::Sink`] implementation that records `imaging` commands
//! and can export them as an SVG document.
//!
//! This backend is intended for debugging/inspection rather than pixel-perfect rendering:
//! - Composition modes other than `src-over` are not faithfully representable in SVG.
//! - Sweep gradients are approximated (SVG has no standard conic gradient).
//! - Image brushes are emitted as placeholders.

#![no_std]
#![deny(unsafe_code)]
#![cfg_attr(not(test), warn(unused_crate_dependencies))]

extern crate alloc;

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write as _;

use imaging::{Clip, Draw, Filter, Geometry, Group, Scene, Sink, replay};
use kurbo::{Affine, BezPath};
use peniko::{BlendMode, Brush, Color, Compose, Extend, GradientKind, Mix};

const DEFAULT_TOLERANCE: f64 = 0.1;

/// Errors that can occur when exporting SVG.
#[derive(Clone, Debug)]
pub enum Error {
    /// The scene is invalid (unbalanced stacks).
    InvalidScene(imaging::ValidateError),
    /// An internal invariant was violated.
    Internal(&'static str),
}

/// A backend that records `imaging` commands and exports an SVG document.
#[derive(Debug, Default)]
pub struct SvgRenderer {
    defs: String,
    body: String,
    clip_stack: Vec<()>,
    group_stack: Vec<()>,
    clip_counter: u64,
    filter_counter: u64,
    gradient_counter: u64,
}

impl SvgRenderer {
    /// Create an empty renderer.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Clear recorded output and reset internal counters/stacks.
    pub fn reset(&mut self) {
        self.defs.clear();
        self.body.clear();
        self.clip_stack.clear();
        self.group_stack.clear();
        self.clip_counter = 0;
        self.filter_counter = 0;
        self.gradient_counter = 0;
    }

    /// Export a recorded [`Scene`] as an SVG document.
    pub fn render_scene(
        &mut self,
        scene: &Scene,
        width: u32,
        height: u32,
    ) -> Result<String, Error> {
        scene.validate().map_err(Error::InvalidScene)?;
        self.reset();
        replay(scene, self);
        self.finish_svg(width, height)
    }

    /// Finish exporting the current command stream.
    pub fn finish_svg(&mut self, width: u32, height: u32) -> Result<String, Error> {
        if !self.clip_stack.is_empty() {
            return Err(Error::Internal("unbalanced clip stack"));
        }
        if !self.group_stack.is_empty() {
            return Err(Error::Internal("unbalanced group stack"));
        }

        let mut svg = String::new();
        let _ = writeln!(
            svg,
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">"
        );
        if !self.defs.is_empty() {
            svg.push_str("<defs>");
            svg.push_str(&self.defs);
            svg.push_str("</defs>");
        }
        svg.push_str(&self.body);
        svg.push_str("</svg>");
        Ok(svg)
    }

    fn geometry_to_path(&self, geom: &Geometry) -> BezPath {
        geom.to_path(DEFAULT_TOLERANCE)
    }

    fn next_clip_id(&mut self) -> String {
        self.clip_counter += 1;
        format!("clip{}", self.clip_counter)
    }

    fn next_filter_id(&mut self) -> String {
        self.filter_counter += 1;
        format!("filter{}", self.filter_counter)
    }

    fn next_gradient_id(&mut self) -> String {
        self.gradient_counter += 1;
        format!("grad{}", self.gradient_counter)
    }

    fn write_clip_def(&mut self, id: &str, clip: &Clip) {
        let _ = write!(
            self.defs,
            "<clipPath id=\"{id}\" clipPathUnits=\"userSpaceOnUse\">"
        );
        match clip {
            Clip::Fill {
                transform,
                shape,
                fill_rule,
            } => {
                let mut path = self.geometry_to_path(shape);
                path.apply_affine(*transform);
                let d = bez_path_to_svg_d(&path);
                let _ = write!(
                    self.defs,
                    "<path d=\"{d}\" clip-rule=\"{}\"/>",
                    fill_rule_svg(*fill_rule)
                );
            }
            Clip::Stroke {
                transform, shape, ..
            } => {
                // Best-effort: SVG clip paths don't include strokes; represent as the fill shape.
                let mut path = self.geometry_to_path(shape);
                path.apply_affine(*transform);
                let d = bez_path_to_svg_d(&path);
                let _ = write!(self.defs, "<path d=\"{d}\" clip-rule=\"nonzero\"/>");
            }
        }
        self.defs.push_str("</clipPath>");
    }

    fn write_filter_def(&mut self, id: &str, filters: &[Filter], transform: Affine) {
        // Use a generous region to avoid clipping blur/shadow output.
        let _ = write!(
            self.defs,
            "<filter id=\"{id}\" x=\"-50%\" y=\"-50%\" width=\"200%\" height=\"200%\">"
        );

        // Match other backends: filter parameters are specified in user space and scaled by the
        // current transform when the filter is applied.
        let [a, b, c, d, _e, _f] = transform.as_coeffs();
        let (scale_x, scale_y) = approx_axis_scales(a, b, c, d);

        let mut input: Option<String> = None;
        for (i, f) in filters.iter().enumerate() {
            let in_attr = input.as_deref().unwrap_or("SourceGraphic");
            let result = format!("r{i}");
            match *f {
                Filter::Flood { color } => {
                    let (rgb, alpha) = color_to_svg(color);
                    let _ = write!(
                        self.defs,
                        "<feFlood flood-color=\"{rgb}\" flood-opacity=\"{}\" result=\"{result}\"/>",
                        fmt_f32(alpha)
                    );
                }
                Filter::Blur {
                    std_deviation_x,
                    std_deviation_y,
                } => {
                    let _ = write!(
                        self.defs,
                        "<feGaussianBlur in=\"{in_attr}\" stdDeviation=\"{} {}\" result=\"{result}\"/>",
                        fmt_f32(std_deviation_x * scale_x),
                        fmt_f32(std_deviation_y * scale_y),
                    );
                }
                Filter::DropShadow {
                    dx,
                    dy,
                    std_deviation_x,
                    std_deviation_y,
                    color,
                } => {
                    let offset_x = dx * f64_to_f32(a) + dy * f64_to_f32(c);
                    let offset_y = dx * f64_to_f32(b) + dy * f64_to_f32(d);
                    let (rgb, alpha) = color_to_svg(color);
                    let _ = write!(
                        self.defs,
                        "<feDropShadow in=\"{in_attr}\" dx=\"{}\" dy=\"{}\" stdDeviation=\"{} {}\" flood-color=\"{rgb}\" flood-opacity=\"{}\" result=\"{result}\"/>",
                        fmt_f32(offset_x),
                        fmt_f32(offset_y),
                        fmt_f32(std_deviation_x * scale_x),
                        fmt_f32(std_deviation_y * scale_y),
                        fmt_f32(alpha),
                    );
                }
                Filter::Offset { dx, dy } => {
                    let offset_x = dx * f64_to_f32(a) + dy * f64_to_f32(c);
                    let offset_y = dx * f64_to_f32(b) + dy * f64_to_f32(d);
                    let _ = write!(
                        self.defs,
                        "<feOffset in=\"{in_attr}\" dx=\"{}\" dy=\"{}\" result=\"{result}\"/>",
                        fmt_f32(offset_x),
                        fmt_f32(offset_y)
                    );
                }
            }
            input = Some(result);
        }

        self.defs.push_str("</filter>");
    }

    fn style_for_brush(
        &mut self,
        brush: Brush,
        kind: PaintKind,
        paint_transform: Option<Affine>,
    ) -> String {
        match brush {
            Brush::Solid(color) => style_for_solid_color(color, kind),
            Brush::Gradient(gradient) => {
                let id = self.next_gradient_id();
                let url = format!("url(#{id})");
                self.write_gradient_def(&id, &gradient, paint_transform);

                let mut out = String::new();
                match kind {
                    PaintKind::Fill => {
                        let _ = write!(out, " fill=\"{url}\"");
                    }
                    PaintKind::Stroke => {
                        let _ = write!(out, " stroke=\"{url}\" fill=\"none\"");
                    }
                }
                out
            }
            Brush::Image(_) => {
                // Placeholder.
                let mut out = String::new();
                match kind {
                    PaintKind::Fill => out.push_str(" fill=\"#ff00ff\" fill-opacity=\"0.25\""),
                    PaintKind::Stroke => {
                        out.push_str(" stroke=\"#ff00ff\" stroke-opacity=\"0.75\" fill=\"none\"");
                    }
                }
                out
            }
        }
    }

    fn write_gradient_def(
        &mut self,
        id: &str,
        gradient: &peniko::Gradient,
        paint_transform: Option<Affine>,
    ) {
        let spread = match gradient.extend {
            Extend::Pad => "pad",
            Extend::Repeat => "repeat",
            Extend::Reflect => "reflect",
        };

        let transform_attr = paint_transform
            .filter(|xf| *xf != Affine::IDENTITY)
            .map(|xf| format!(" gradientTransform=\"{}\"", affine_to_svg_matrix(xf)))
            .unwrap_or_default();

        let closing = match gradient.kind {
            GradientKind::Linear(pos) => {
                let _ = write!(
                    self.defs,
                    "<linearGradient id=\"{id}\" gradientUnits=\"userSpaceOnUse\" x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" spreadMethod=\"{spread}\"{transform_attr}>",
                    fmt_f64_to_f32(pos.start.x),
                    fmt_f64_to_f32(pos.start.y),
                    fmt_f64_to_f32(pos.end.x),
                    fmt_f64_to_f32(pos.end.y),
                );
                "</linearGradient>"
            }
            GradientKind::Radial(pos) => {
                let fr_attr = if pos.start_radius > 0.0 {
                    format!(" fr=\"{}\"", fmt_f32(pos.start_radius))
                } else {
                    String::new()
                };
                let _ = write!(
                    self.defs,
                    "<radialGradient id=\"{id}\" gradientUnits=\"userSpaceOnUse\" cx=\"{}\" cy=\"{}\" r=\"{}\" fx=\"{}\" fy=\"{}\"{fr_attr} spreadMethod=\"{spread}\"{transform_attr}>",
                    fmt_f64_to_f32(pos.end_center.x),
                    fmt_f64_to_f32(pos.end_center.y),
                    fmt_f32(pos.end_radius),
                    fmt_f64_to_f32(pos.start_center.x),
                    fmt_f64_to_f32(pos.start_center.y),
                );
                "</radialGradient>"
            }
            GradientKind::Sweep(_pos) => {
                // Best-effort: SVG has no standard sweep/conic gradient.
                // Emit a tiny linear gradient so the output stays usable.
                let _ = write!(
                    self.defs,
                    "<linearGradient id=\"{id}\" gradientUnits=\"userSpaceOnUse\" x1=\"0\" y1=\"0\" x2=\"1\" y2=\"0\" spreadMethod=\"{spread}\"{transform_attr}>"
                );
                "</linearGradient>"
            }
        };

        for stop in gradient.stops.iter() {
            let (rgb, alpha) = dynamic_color_to_svg(stop.color);
            let _ = write!(
                self.defs,
                "<stop offset=\"{}\" stop-color=\"{rgb}\" stop-opacity=\"{}\"/>",
                fmt_f32(stop.offset),
                fmt_f32(alpha)
            );
        }
        self.defs.push_str(closing);
    }
}

impl Sink for SvgRenderer {
    fn push_clip(&mut self, clip: Clip) {
        let id = self.next_clip_id();
        self.write_clip_def(&id, &clip);
        let _ = write!(self.body, "<g clip-path=\"url(#{id})\">");
        self.clip_stack.push(());
    }

    fn pop_clip(&mut self) {
        if self.clip_stack.pop().is_none() {
            return;
        }
        self.body.push_str("</g>");
    }

    fn push_group(&mut self, group: Group) {
        let mut attrs = String::new();

        // Always isolate, matching the IR semantics.
        let mut style = String::from("isolation:isolate");

        if group.composite.alpha < 1.0 {
            let _ = write!(attrs, " opacity=\"{}\"", fmt_f32(group.composite.alpha));
        }
        if let Some(css) = blend_mode_css(group.composite.blend) {
            let _ = write!(style, ";mix-blend-mode:{css}");
        }

        if let Some(clip) = group.clip.as_ref() {
            let id = self.next_clip_id();
            self.write_clip_def(&id, clip);
            let _ = write!(attrs, " clip-path=\"url(#{id})\"");
        }

        if !group.filters.is_empty() {
            let id = self.next_filter_id();
            let clip_transform = group
                .clip
                .as_ref()
                .map(clip_transform)
                .unwrap_or(Affine::IDENTITY);
            self.write_filter_def(&id, &group.filters, clip_transform);
            let _ = write!(attrs, " filter=\"url(#{id})\"");
        }

        let _ = write!(self.body, "<g{attrs} style=\"{style}\">");
        self.group_stack.push(());
    }

    fn pop_group(&mut self) {
        if self.group_stack.pop().is_none() {
            return;
        }
        self.body.push_str("</g>");
    }

    fn draw(&mut self, draw: Draw) {
        match draw {
            Draw::Fill {
                transform,
                fill_rule,
                paint,
                paint_transform,
                shape,
                composite,
            } => {
                let mut path = self.geometry_to_path(&shape);
                path.apply_affine(transform);
                let d = bez_path_to_svg_d(&path);

                let brush = paint.multiply_alpha(composite.alpha);
                let paint_attrs = self.style_for_brush(brush, PaintKind::Fill, paint_transform);

                let mut attrs = String::new();
                if let Some(css) = blend_mode_css(composite.blend) {
                    let _ = write!(attrs, " style=\"mix-blend-mode:{css}\"");
                }
                let _ = write!(
                    self.body,
                    "<path d=\"{d}\" fill-rule=\"{}\"{paint_attrs}{attrs}/>",
                    fill_rule_svg(fill_rule)
                );
            }
            Draw::Stroke {
                transform,
                stroke,
                paint,
                paint_transform,
                shape,
                composite,
            } => {
                let mut path = self.geometry_to_path(&shape);
                path.apply_affine(transform);
                let d = bez_path_to_svg_d(&path);

                let brush = paint.multiply_alpha(composite.alpha);
                let mut paint_attrs =
                    self.style_for_brush(brush, PaintKind::Stroke, paint_transform);
                paint_attrs.push_str(&stroke_style_attrs(&stroke));

                let mut attrs = String::new();
                if let Some(css) = blend_mode_css(composite.blend) {
                    let _ = write!(attrs, " style=\"mix-blend-mode:{css}\"");
                }
                let _ = write!(self.body, "<path d=\"{d}\"{paint_attrs}{attrs}/>");
            }
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum PaintKind {
    Fill,
    Stroke,
}

fn clip_transform(clip: &Clip) -> Affine {
    match clip {
        Clip::Fill { transform, .. } | Clip::Stroke { transform, .. } => *transform,
    }
}

fn fill_rule_svg(rule: imaging::FillRule) -> &'static str {
    match rule {
        imaging::FillRule::NonZero => "nonzero",
        imaging::FillRule::EvenOdd => "evenodd",
    }
}

fn blend_mode_css(mode: BlendMode) -> Option<&'static str> {
    if mode.compose != Compose::SrcOver {
        return None;
    }
    match mode.mix {
        Mix::Normal => None,
        Mix::Multiply => Some("multiply"),
        Mix::Screen => Some("screen"),
        Mix::Overlay => Some("overlay"),
        Mix::Darken => Some("darken"),
        Mix::Lighten => Some("lighten"),
        Mix::ColorDodge => Some("color-dodge"),
        Mix::ColorBurn => Some("color-burn"),
        Mix::HardLight => Some("hard-light"),
        Mix::SoftLight => Some("soft-light"),
        Mix::Difference => Some("difference"),
        Mix::Exclusion => Some("exclusion"),
        Mix::Hue => Some("hue"),
        Mix::Saturation => Some("saturation"),
        Mix::Color => Some("color"),
        Mix::Luminosity => Some("luminosity"),
    }
}

fn style_for_solid_color(color: Color, kind: PaintKind) -> String {
    let rgba = color.to_rgba8();
    let a = (rgba.a as f32) / 255.0;
    let rgb = format!("#{:02x}{:02x}{:02x}", rgba.r, rgba.g, rgba.b);

    let mut out = String::new();
    match kind {
        PaintKind::Fill => {
            let _ = write!(out, " fill=\"{rgb}\"");
            if a < 1.0 {
                let _ = write!(out, " fill-opacity=\"{}\"", fmt_f32(a));
            }
        }
        PaintKind::Stroke => {
            let _ = write!(out, " stroke=\"{rgb}\" fill=\"none\"");
            if a < 1.0 {
                let _ = write!(out, " stroke-opacity=\"{}\"", fmt_f32(a));
            }
        }
    }
    out
}

fn dynamic_color_to_svg(color: peniko::color::DynamicColor) -> (String, f32) {
    let c: peniko::color::AlphaColor<peniko::color::Srgb> = color.to_alpha_color();
    let [r, g, b, a] = c.components;
    let rgb = format!(
        "#{:02x}{:02x}{:02x}",
        f32_to_u8(r),
        f32_to_u8(g),
        f32_to_u8(b)
    );
    (rgb, a.clamp(0.0, 1.0))
}

fn color_to_svg(color: Color) -> (String, f32) {
    let rgba = color.to_rgba8();
    let a = (rgba.a as f32) / 255.0;
    (format!("#{:02x}{:02x}{:02x}", rgba.r, rgba.g, rgba.b), a)
}

fn stroke_style_attrs(stroke: &imaging::StrokeStyle) -> String {
    use kurbo::{Cap, Join};

    let mut out = String::new();
    let _ = write!(out, " stroke-width=\"{}\"", fmt_f64_to_f32(stroke.width));
    let _ = write!(
        out,
        " stroke-linecap=\"{}\"",
        match stroke.start_cap {
            Cap::Butt => "butt",
            Cap::Round => "round",
            Cap::Square => "square",
        }
    );
    let _ = write!(
        out,
        " stroke-linejoin=\"{}\"",
        match stroke.join {
            Join::Miter => "miter",
            Join::Round => "round",
            Join::Bevel => "bevel",
        }
    );
    if stroke.miter_limit.is_finite() && matches!(stroke.join, Join::Miter) {
        let _ = write!(
            out,
            " stroke-miterlimit=\"{}\"",
            fmt_f64_to_f32(stroke.miter_limit)
        );
    }
    if !stroke.dash_pattern.is_empty() {
        out.push_str(" stroke-dasharray=\"");
        for (i, v) in stroke.dash_pattern.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&fmt_f64_to_f32(*v));
        }
        out.push('"');
    }
    if stroke.dash_offset != 0.0 {
        let _ = write!(
            out,
            " stroke-dashoffset=\"{}\"",
            fmt_f64_to_f32(stroke.dash_offset)
        );
    }
    out
}

fn bez_path_to_svg_d(path: &BezPath) -> String {
    use kurbo::PathEl;
    let mut d = String::new();
    for el in path.iter() {
        match el {
            PathEl::MoveTo(p) => {
                let _ = write!(d, "M{} {}", fmt_f64_to_f32(p.x), fmt_f64_to_f32(p.y));
            }
            PathEl::LineTo(p) => {
                let _ = write!(d, "L{} {}", fmt_f64_to_f32(p.x), fmt_f64_to_f32(p.y));
            }
            PathEl::QuadTo(p1, p2) => {
                let _ = write!(
                    d,
                    "Q{} {} {} {}",
                    fmt_f64_to_f32(p1.x),
                    fmt_f64_to_f32(p1.y),
                    fmt_f64_to_f32(p2.x),
                    fmt_f64_to_f32(p2.y)
                );
            }
            PathEl::CurveTo(p1, p2, p3) => {
                let _ = write!(
                    d,
                    "C{} {} {} {} {} {}",
                    fmt_f64_to_f32(p1.x),
                    fmt_f64_to_f32(p1.y),
                    fmt_f64_to_f32(p2.x),
                    fmt_f64_to_f32(p2.y),
                    fmt_f64_to_f32(p3.x),
                    fmt_f64_to_f32(p3.y)
                );
            }
            PathEl::ClosePath => d.push('Z'),
        }
    }
    d
}

fn affine_to_svg_matrix(xf: Affine) -> String {
    // kurbo::Affine stores [a, b, c, d, e, f] corresponding to:
    // [ a c e ]
    // [ b d f ]
    // [ 0 0 1 ]
    let c = xf.as_coeffs();
    format!(
        "matrix({} {} {} {} {} {})",
        fmt_f64_to_f32(c[0]),
        fmt_f64_to_f32(c[1]),
        fmt_f64_to_f32(c[2]),
        fmt_f64_to_f32(c[3]),
        fmt_f64_to_f32(c[4]),
        fmt_f64_to_f32(c[5]),
    )
}

fn approx_axis_scales(a: f64, b: f64, c: f64, d: f64) -> (f32, f32) {
    // Best-effort approximation without requiring `libm`.
    let sx = f64_to_f32(a.abs().max(b.abs())).max(0.0);
    let sy = f64_to_f32(c.abs().max(d.abs())).max(0.0);
    (sx, sy)
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "SVG output uses f32 in 0..=1 for colors"
)]
fn f32_to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "SVG output is best-effort; scalars are formatted as f32"
)]
fn f64_to_f32(v: f64) -> f32 {
    v as f32
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "SVG uses f32-like scalar formatting"
)]
fn fmt_f64_to_f32(v: f64) -> String {
    fmt_f32(v as f32)
}

fn fmt_f32(v: f32) -> String {
    // Keep output readable and stable enough for debugging.
    if v.is_finite() {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "best-effort pretty formatting"
        )]
        let i = v as i32;
        let diff = (i as f32) - v;
        if diff > -1e-6 && diff < 1e-6 {
            return format!("{i}");
        }
    } else {
        return format!("{v}");
    }

    let mut s = format!("{:.3}", v);
    while s.contains('.') && s.ends_with('0') {
        s.pop();
    }
    if s.ends_with('.') {
        s.pop();
    }
    s
}
