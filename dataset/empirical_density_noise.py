"""Global empirical density sampling, fixed per sample, without spatial signal.

Counts every voxel in every canonical training crop. Values are represented in
float16, matching the existing materialized noise/crop convention; no crop or
voxel subsampling, histogram binning, Gaussian fit, or per-complex distribution.
Validation/probe samples never contribute to the distribution.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import zlib

import numpy as np


class EmpiricalDensityNoise:
    def __init__(self, path):
        with np.load(path) as data:
            self.values = data['values'].astype(np.float32)
            counts = data['counts'].astype(np.float64)
        if (self.values.ndim != 1 or counts.shape != self.values.shape
                or not np.isfinite(self.values).all() or not np.isfinite(counts).all()
                or (counts <= 0).any() or not counts.size):
            raise ValueError('Invalid empirical density distribution')
        self.cdf = np.cumsum(counts / counts.sum())
        self.cdf[-1] = 1.0

    def draw(self, seed, shape):
        rng = np.random.default_rng(seed)
        return self.values[np.searchsorted(self.cdf, rng.random(shape))]


def main():
    import torch
    from omegaconf import OmegaConf
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dataset import _make_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--probe-source', default=str(Path(__file__).resolve().parent / 'data' / 'pdbbind' / 'voxels_v5'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = OmegaConf.load(args.config)
    output = Path(cfg.dset.density_noise_distribution)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        build_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        build_cfg.dset.density_noise_distribution = ''
        build_cfg.with_gradmag = False
        dataset = _make_dataset(build_cfg, 'train', aug=False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=None,
                    num_workers=args.workers, prefetch_factor=2 if args.workers else None)
        counts = np.zeros(65536, dtype=np.int64)
        for i, sample in enumerate(loader):
            if not bool(sample['xray_available']):
                raise RuntimeError(f'Missing training density at sample {i}')
            values = sample['xray_density'].numpy().astype(np.float16)
            if not np.isfinite(values).all():
                raise ValueError(f'Nonfinite float16 density at sample {i}')
            counts += np.bincount(values.view(np.uint16).ravel(), minlength=65536)
            if (i + 1) % 500 == 0:
                print(f'[distribution] {i+1}/{len(dataset)} crops', flush=True)
        active = np.flatnonzero(counts)
        values = active.astype(np.uint16).view(np.float16)
        order = np.argsort(values)
        temp = output.with_name(output.stem + '.tmp.npz')
        np.savez(temp, values=values[order], counts=counts[active][order])
        os.replace(temp, output)
        output.with_suffix('.json').write_text(json.dumps({
            'source_config': OmegaConf.to_container(build_cfg, resolve=True),
            'samples': len(dataset), 'voxels': int(counts.sum()),
            'representation': 'float16, all canonical train crop voxels; no subsampling',
        }, indent=2))
        print(f'[distribution] saved {output}', flush=True)
    noise = EmpiricalDensityNoise(output)
    source = Path(args.probe_source)
    dest = output.parent / 'probe'
    (dest / 'density').mkdir(parents=True, exist_ok=True)
    # Preserve required voxel metadata and record the ablation explicitly.
    stats = json.loads((source / 'stats.json').read_text())
    stats['density_ablation'] = {'distribution': str(output.resolve()),
                                'seed': cfg.dset.density_noise_seed,
                                'gradmag': 'derived from noise at feature extraction'}
    (dest / 'stats.json').write_text(json.dumps(stats, indent=2))
    files = sorted((source / 'density').glob('*.npy'))
    if not files:
        raise RuntimeError(f'No probe density crops at {source}')
    for path in files:
        shape = np.load(path, mmap_mode='r').shape
        seed = int(cfg.dset.density_noise_seed) + zlib.crc32(path.stem.encode())
        np.save(dest / 'density' / path.name, noise.draw(seed, shape))
    print(f'[probe noise] saved {len(files)} crops to {dest}', flush=True)


if __name__ == '__main__':
    main()
