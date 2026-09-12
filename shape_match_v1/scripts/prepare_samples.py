#!/usr/bin/env python3
"""Convert supplied PNGs without resampling; preserve template alpha as a domain.
Not part of matching: no detection, segmentation, or position labels are used here.
"""
from pathlib import Path
import argparse
from PIL import Image

def convert(source: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('original', 'template_ring', 'template_nut'):
        with Image.open(source / f'{name}.png') as image:
            image.convert('L').save(destination / f'{name}.pgm')
            if 'A' in image.getbands():
                image.getchannel('A').point(lambda x: 255 if x else 0).save(destination / f'{name}_mask.pgm')
            elif name.startswith('template'):
                Image.new('L', image.size, 255).save(destination / f'{name}_mask.pgm')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/reference'))
    parser.add_argument('--destination', type=Path, default=Path('data'))
    args = parser.parse_args()
    convert(args.source, args.destination)
