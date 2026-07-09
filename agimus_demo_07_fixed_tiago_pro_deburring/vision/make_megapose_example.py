#!/usr/bin/env python3
"""
Build a happypose "example" directory for MegaPose inference from a captured
image, its camera_data.json and the pylone PLY mesh. The bounding box is
drawn with the mouse (cv2.selectROI) unless given with --bbox.

Layout produced (happypose run_inference_on_example format):
    $HAPPYPOSE_DATA_DIR/examples/<label>/
        image_rgb.png
        camera_data.json
        inputs/object_data.json
        meshes/<label>/<label>.ply

Then run on the GPU machine:
    python -m happypose.pose_estimators.megapose.scripts.run_inference_on_example \
        <label> --run-inference --vis-outputs

Usage:
    python3 make_megapose_example.py --capture ./capture_01 --mesh ./pylone.ply
    python3 make_megapose_example.py --capture ./capture_01 --mesh ./pylone.ply \
        --bbox 640 360 1200 700
"""

import argparse
import json
import os
import shutil
import sys

import cv2


def select_bbox(image_path):
    img = cv2.imread(image_path)
    if img is None:
        sys.exit(f'Cannot read {image_path}')
    print('Draw the box around the pylone (ENTER to validate, c to cancel)')
    x, y, w, h = cv2.selectROI('select pylone', img, showCrosshair=True)
    cv2.destroyAllWindows()
    if w == 0 or h == 0:
        sys.exit('Empty bbox, aborted')
    return [int(x), int(y), int(x + w), int(y + h)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', required=True,
                        help='Directory with image_rgb.png + camera_data.json '
                             '(output of capture_image.py)')
    parser.add_argument('--mesh', required=True, help='Colored PLY in mm '
                        '(output of convert_pylone_mesh.py)')
    parser.add_argument('--label', default='pylone')
    parser.add_argument('--bbox', type=int, nargs=4, default=None,
                        metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'))
    parser.add_argument('--data-dir',
                        default=os.environ.get('HAPPYPOSE_DATA_DIR'),
                        help='Defaults to $HAPPYPOSE_DATA_DIR')
    args = parser.parse_args()

    if not args.data_dir:
        sys.exit('Set HAPPYPOSE_DATA_DIR or pass --data-dir')

    image_path = os.path.join(args.capture, 'image_rgb.png')
    camera_path = os.path.join(args.capture, 'camera_data.json')
    for p in (image_path, camera_path, args.mesh):
        if not os.path.exists(p):
            sys.exit(f'Missing file: {p}')

    bbox = args.bbox or select_bbox(image_path)
    print(f'bbox_modal: {bbox}')

    example_dir = os.path.join(args.data_dir, 'examples', args.label)
    os.makedirs(os.path.join(example_dir, 'inputs'), exist_ok=True)
    mesh_dir = os.path.join(example_dir, 'meshes', args.label)
    os.makedirs(mesh_dir, exist_ok=True)

    shutil.copy(image_path, os.path.join(example_dir, 'image_rgb.png'))
    shutil.copy(camera_path, os.path.join(example_dir, 'camera_data.json'))
    shutil.copy(args.mesh,
                os.path.join(mesh_dir, args.label + '.ply'))

    object_data = [{'label': args.label, 'bbox_modal': bbox}]
    with open(os.path.join(example_dir, 'inputs', 'object_data.json'),
              'w') as f:
        json.dump(object_data, f, indent=2)

    print(f'Example ready in {example_dir}')
    print('Run inference with:')
    print('  python -m happypose.pose_estimators.megapose.scripts.'
          f'run_inference_on_example {args.label} '
          '--run-inference --vis-outputs')


if __name__ == '__main__':
    main()
