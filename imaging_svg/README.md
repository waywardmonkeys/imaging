# imaging_svg

SVG export backend for the `imaging` command stream.

This backend is intended for debugging/inspection rather than pixel-perfect rendering. It
approximates blend modes and filter effects using SVG `<g>` compositing and filter primitives.

