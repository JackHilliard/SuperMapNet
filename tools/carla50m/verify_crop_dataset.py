"""Verify the 50 m -> 30 m crop dataset: alignment, coverage, determinism.

Builds the train and val datasets straight from a config's ``EXPConfig``
(so it checks the crop kwargs the training job will actually use), pulls
crops through ``Carla50mCropDataset`` and reports the invariants that
matter (the SuperMapNet twin of MapTR's ``tools/carla50m/verify_crop_dataset.py``):

  * GT vertices sit ON LiDAR returns (points and GT received the same pose);
  * every crop is fully populated and everything is inside the +-half box;
  * the train yaw is uniform;
  * the val crop is identical across calls while the train crop is not;
  * the point width equals the encoder's ``ppdim``.

Run inside the container::

    python3 tools/carla50m/verify_crop_dataset.py \
        configs/supermapnet_carla_lidar_50m_crop_2cls.py \
        [--data-root <export root>] [--n 40]
"""
import argparse
import importlib.util
import os
import sys
import warnings

import numpy as np


def load_exp_config(path):
    spec = importlib.util.spec_from_file_location('cfg_module', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['cfg_module'] = mod
    spec.loader.exec_module(mod)
    return mod.EXPConfig


def build(exp_cfg, split, data_root, split_dirs):
    from mapmaster.dataset.carla50m_crop_dataset import Carla50mCropDataset
    key = {'train': 'ann_file_train', 'val': 'ann_file_val',
           'test': 'ann_file_test'}[split]
    ann = exp_cfg.map_conf[key]
    if data_root:
        ann = os.path.join(data_root, split_dirs[split], 'manifest.json')
    crop = exp_cfg.crop_train if split == 'train' else exp_cfg.crop_eval
    return Carla50mCropDataset(
        ann_file=ann, map_conf=exp_cfg.map_conf,
        point_conf=exp_cfg.pivot_conf, transforms=None, data_split=split,
        **exp_cfg.dataset_setup, **crop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--data-root', default=None,
                    help="override the config's DATA_ROOT (export root)")
    ap.add_argument('--n', type=int, default=40, help='crops to sample')
    ap.add_argument('--train-split', default='train',
                    help='with --data-root: split dir for the train-mode '
                         'checks (an export with only a test split: pass '
                         '"test")')
    ap.add_argument('--val-split', default='test',
                    help='with --data-root: split dir for the frozen '
                         'val-mode checks')
    ap.add_argument('--full-eval-scan', action='store_true',
                    help='also count GT-empty frozen eval crops over the '
                         'whole val split')
    args = ap.parse_args()
    warnings.filterwarnings('ignore')
    exp_cfg = load_exp_config(args.config)
    classes = list(exp_cfg.map_conf['map_classes'])
    ppdim = exp_cfg.model_setup['lidar_encoder']['net_kwargs']['ppdim']

    print('=' * 78 + '\n1. BUILD\n' + '=' * 78)
    split_dirs = dict(train=args.train_split, val=args.val_split,
                      test=args.val_split)
    tr = build(exp_cfg, 'train', args.data_root, split_dirs)
    te = build(exp_cfg, 'val', args.data_root, split_dirs)
    print(f'  train: {len(tr)} samples   val: {len(te)} samples   '
          f'classes: {classes}')
    print(f'  train max_shift={tr.max_shift:.3f} m  need={tr._need:.3f} m')
    print(f'  val   max_shift={te.max_shift:.3f} m  need={te._need:.3f} m')
    assert 'crop_R' in te.infos[0], 'val crops must be frozen'
    assert 'crop_R' not in tr.infos[0], 'train crops must be fresh'
    assert tr.point_dim == ppdim, \
        f'dataset point_dim {tr.point_dim} != encoder ppdim {ppdim}'

    half = tr.half
    slack = 4.0 * float(exp_cfg.crop_train['noise_std']) + 1e-3
    n = min(args.n, len(tr))
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(tr), n, replace=False)

    print('\n' + '=' * 78 + '\n2. POINT/GT ALIGNMENT\n' + '=' * 78)
    d_all, occ, npts, ngt, bad_p, bad_g, dims, trimmed = \
        [], [], [], [], 0, 0, set(), 0
    for i in idxs:
        i = int(i)
        info = tr.infos[i]
        _, R, sh = tr.crop_pose(i)
        pts, mask = tr.get_crop_lidar(info, R, sh)
        P = pts.numpy()[mask > 0]
        if int(mask.sum()) == tr.max_lidar_points:
            trimmed += 1
        dims.add(P.shape[1])
        npts.append(len(P))
        G = [v['pts'] for v in tr.get_crop_vectors(info, R, sh)]
        ngt.append(len(G))
        if len(P) and np.abs(P[:, :2]).max() > half + slack:
            bad_p += 1
        bad_g += sum(np.abs(g[:, :2]).max() > half + 1e-3 for g in G)
        if len(P) == 0 or not G:
            continue
        V = np.concatenate(G, 0)
        d_all.append(np.linalg.norm(V[:, None, :2] - P[None, :, :2],
                                    axis=-1).min(1))
        q = [((P[:, 0] > 0) & (P[:, 1] > 0)), ((P[:, 0] <= 0) & (P[:, 1] > 0)),
             ((P[:, 0] > 0) & (P[:, 1] <= 0)),
             ((P[:, 0] <= 0) & (P[:, 1] <= 0))]
        occ.append(sum(1 for m in q if m.sum() > 0))
    d = np.concatenate(d_all)
    print(f'  point dims: {sorted(dims)} (encoder ppdim={ppdim}; '
          f'3 = colour-free convention)')
    print(f'  GT vertex -> nearest LiDAR point over {n} train crops '
          f'(after 0.15 m grid sampling): median {np.median(d):.3f} m | '
          f'p90 {np.percentile(d, 90):.3f} | max {d.max():.3f}')
    print(f'  VERDICT: {"ALIGNED" if np.median(d) < 0.5 else "MISALIGNED"}')
    print(f'  all 4 quadrants populated: {sum(o == 4 for o in occ)}/{len(occ)}')
    print(f'  points/crop (grid-sampled, before padding): median '
          f'{int(np.median(npts)):,} min {min(npts):,} max {max(npts):,}; '
          f'crops hitting max_lidar_points={tr.max_lidar_points}: {trimmed}')
    print(f'  GT lines/crop: median {np.median(ngt):.1f} min {min(ngt)} '
          f'max {max(ngt)} empty: {sum(g == 0 for g in ngt)}')
    print(f'  crops with points outside +-{half} m (+noise slack): {bad_p}/{n}; '
          f'GT polylines outside: {bad_g}')

    print('\n' + '=' * 78 + '\n3. ROTATION\n' + '=' * 78)
    m = min(300, len(tr))
    Rs = [tr.crop_pose(int(i))[1] for i in rng.choice(len(tr), m, replace=True)]
    ang = np.array([np.arctan2(R[1, 0], R[0, 0]) for R in Rs]) % (2 * np.pi)
    h, _ = np.histogram(ang, bins=8, range=(0, 2 * np.pi))
    print(f'  yaw histogram over {m} draws (8 bins): {h.tolist()}')

    print('\n' + '=' * 78 + '\n4. EVAL DETERMINISM\n' + '=' * 78)
    k = min(5, len(te))
    a = [te[i] for i in range(k)]
    b = [te[i] for i in range(k)]
    same = all(
        np.array_equal(x['targets']['masks'], y['targets']['masks'])
        and np.array_equal(x['lidars'].numpy(), y['lidars'].numpy())
        and all(np.array_equal(x['targets']['points'][c],
                               y['targets']['points'][c]) for c in range(len(classes)))
        for x, y in zip(a, b))
    print(f'  val __getitem__ identical across calls: {same}')
    print(f'  val yaw is identity: {np.allclose(te.crop_pose(0)[1], np.eye(2))}')
    ta, tb = tr.crop_pose(0)[1], tr.crop_pose(0)[1]
    print(f'  train crop varies across calls: {not np.array_equal(ta, tb)}')
    assert same and not np.array_equal(ta, tb)

    if args.full_eval_scan:
        print('\n' + '=' * 78 + '\n5. FULL VAL SCAN (frozen crops)\n' + '=' * 78)
        empties, per_cls = 0, np.zeros(len(classes), dtype=int)
        for i in range(len(te)):
            _, R, sh = te.crop_pose(i)
            vecs = te.get_crop_vectors(te.infos[i], R, sh)
            if not vecs:
                empties += 1
            for v in vecs:
                per_cls[v['type']] += 1
        print(f'  val tiles: {len(te)}  GT-empty eval crops: {empties} '
              f'({100.0 * empties / len(te):.1f}%)')
        print('  GT instances per class: ' + ', '.join(
            f'{c}={int(k)}' for c, k in zip(classes, per_cls)))
    print('\nOK')


if __name__ == '__main__':
    main()
