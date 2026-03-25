// Copyright 2026 the Imaging Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! SVG snapshot tests for `imaging_svg`.

#![cfg(feature = "svg")]

use imaging_snapshot_tests::cases::{
    DEFAULT_HEIGHT, DEFAULT_WIDTH, build_scene, selected_cases_for_backend,
};
use imaging_svg::SvgRenderer;

mod common;

#[test]
fn snapshots() {
    let width = u32::from(DEFAULT_WIDTH);
    let height = u32::from(DEFAULT_HEIGHT);
    let w = f64::from(DEFAULT_WIDTH);
    let h = f64::from(DEFAULT_HEIGHT);

    let mut errors = Vec::new();
    for case in selected_cases_for_backend("svg") {
        let scene = build_scene(case, w, h);
        let mut renderer = SvgRenderer::new();
        let svg = renderer
            .render_scene(&scene, width, height)
            .expect("render svg scene");
        common::check_text_snapshot("svg", case.name(), "svg", &svg, &mut errors);
    }
    common::assert_no_snapshot_errors(errors);
}
