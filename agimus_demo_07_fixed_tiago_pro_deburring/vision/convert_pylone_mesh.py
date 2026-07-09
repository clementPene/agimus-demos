#!/usr/bin/env python3
"""
Convert the pylone STL (meters, no color) to a colored PLY for MegaPose.

MegaPose's run_inference_on_example expects meshes in millimeters
(mesh_units="mm" is hardcoded in make_object_dataset), hence the x1000
scaling by default. The color should roughly match the real object: the
real pylone is black, but a pure black kills shading in the render-and
-compare, so default is a very dark gray.

Requires: pip install trimesh

Usage:
    python3 convert_pylone_mesh.py
    python3 convert_pylone_mesh.py --stl ../urdf/visual/pylone.stl \
        --output ./pylone.ply --color 40 40 40 --units mm
"""

import argparse
import os

import numpy as np
import trimesh

DEFAULT_STL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'urdf', 'visual', 'pylone.stl')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stl', default=DEFAULT_STL)
    parser.add_argument('--output', default='pylone.ply')
    parser.add_argument('--color', type=int, nargs=3, default=[40, 40, 40],
                        metavar=('R', 'G', 'B'))
    parser.add_argument('--units', choices=['mm', 'm'], default='mm',
                        help='Output units (mm for run_inference_on_example)')
    args = parser.parse_args()

    mesh = trimesh.load(args.stl, force='mesh')
    print(f'Loaded {args.stl}: {len(mesh.faces)} faces, '
          f'extents {mesh.extents.round(3)} (assumed meters)')

    if args.units == 'mm':
        mesh.apply_scale(1000.0)

    rgba = np.array(args.color + [255], dtype=np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh, vertex_colors=np.tile(rgba, (len(mesh.vertices), 1)))

    mesh.export(args.output)
    print(f'Saved {args.output} ({args.units}, color RGB {args.color})')


if __name__ == '__main__':
    main()
