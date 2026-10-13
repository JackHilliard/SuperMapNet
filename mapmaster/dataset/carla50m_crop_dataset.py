"""CARLA 50 m tiles -> rotated 30 m crops, augmented at load time.

Port of MapTRv2's ``CustomCarla50mCropDataset`` + ``LoadCarla50mCrop``
(MapTR ``projects/mmdet3d_plugin/datasets/carla50m_crop_dataset.py``) onto
``CarlaMapDataset``'s item contract, so the SAME crops the 50 m benchmark
feeds MapTRv2 reach PivotNet's rasteriser, pivot generator and pillar
encoder. The crop geometry is deliberately identical, step for step:

  1. random yaw about the tile centre + random centre shift, then a 30 m
     square CROP. A 30 m square rotated arbitrarily needs 30*sqrt(2) =
     42.43 m, which fits inside the 50 m tile (``tile_radius`` 25), so every
     crop is fully populated. Leftover slack (25 - 21.21 = 3.79 m) is the
     max centre shift. The crop is AIMED at a random GT vertex
     (``_draw_crop``) so it is non-empty.
  2. additive Gaussian noise on the point coordinates (train only)
  3. GT polylines rotated and clipped to the SAME 30 m box with the tiler's
     own ``clip_polyline_to_box`` (vendored, as in MapTR), stubs shorter
     than ``min_polyline_len`` dropped.

FRAMES (verified against the npz/json on disk):
  * block ``points``            world-absolute (N, 3)
  * block ``features[:, 3:6]``  rgb in [0, 1] -> BT.709 "strength", read
                                only when ``point_dim`` == 4
  * GT ``polylines[].points``   world-absolute (N, 3)
  * ``tile_center``             world centre of the 50 m tile
The augmentation is done in world frame about ``tile_center``; the
crop-local frame is what the network sees. The crop centre IS the origin,
so there is no ``lidar_recenter_shift`` here -- this replaces the pkl
datasets' tile_center frame. Points' z is re-zeroed per crop on its own
``z_ground_percentile`` ground (the pillar encoder's single z bin spans
[-72, 96], so this only matters for keeping every return in range).

EVAL DETERMINISM. The evaluator scores each sample against the GT masks
the dataset hands it in the same batch, so there is no cached-GT trap as
in MapTRv2 -- but a frozen val crop is still what makes two evaluations
comparable and the reported numbers reproducible. Val/test therefore run
with ``rotate=False``, ``noise_std=0.0`` and a ``crop_seed``: the pose is
drawn ONCE per index from a seeded rng at construction and stored. Train
draws a fresh pose every ``__getitem__``.

This is a manifest-driven dataset: ``ann_file`` is the export's
``<split>/manifest.json``, not a converter pkl; ``<data_root>/<split>/``
is derived from it. None of ``CarlaMapDataset``'s pkl bookkeeping
(metadata class check, in-range filter, ``lidar_recenter_shift``) applies.
"""

import json
import os
import warnings
from typing import List

import numpy as np
import torch

from .carla_dataset import (LUMA_BT709, CarlaMapDataset, grid_sample_points)
from .voxel import pad_or_trim_to_np


# ---------------------------------------------------------------------------
# Vendored (via MapTR) from Pointcept/data/grid_tile_export.py so the crop GT
# is clipped by EXACTLY the tiler's logic (boundary vertex placed on the edge,
# runs of <2 vertices dropped).
# ---------------------------------------------------------------------------
def _clip_segment_to_box(p0, p1, box):
    """Liang-Barsky. Returns (q0, q1, t0, t1) clipped to box, or None."""
    xmin, ymin, xmax, ymax = box
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin),
                 (dy, ymax - y0)):
        if abs(p) < 1e-12:
            if q < 0:
                return None
        else:
            r = q / p
            if p < 0:
                if r > t1:
                    return None
                if r > t0:
                    t0 = r
            else:
                if r < t0:
                    return None
                if r < t1:
                    t1 = r
    q0 = p0 + t0 * (p1 - p0)
    q1 = p0 + t1 * (p1 - p0)
    return q0, q1, t0, t1


def clip_polyline_to_box(points: np.ndarray, curved: np.ndarray,
                         box) -> List[dict]:
    """Clip a polyline to a square -> continuous sub-polylines reaching the edge."""
    out: List[dict] = []
    cur_pts: List[np.ndarray] = []
    cur_cur: List[bool] = []

    def flush():
        if len(cur_pts) >= 2:
            out.append({
                'points': np.asarray(cur_pts, dtype=np.float32),
                'curved': np.asarray(cur_cur, dtype=bool)
            })
        cur_pts.clear()
        cur_cur.clear()

    for i in range(len(points) - 1):
        res = _clip_segment_to_box(points[i], points[i + 1], box)
        if res is None:
            flush()
            continue
        q0, q1, t0, t1 = res
        c_i = bool(curved[i])
        if not cur_pts:
            cur_pts.append(q0)
            cur_cur.append(c_i)
        elif np.linalg.norm(cur_pts[-1] - q0) > 1e-4:
            flush()
            cur_pts.append(q0)
            cur_cur.append(c_i)
        cur_pts.append(q1)
        cur_cur.append(bool(curved[i + 1]) if t1 >= 1.0 - 1e-9 else c_i)
        if t1 < 1.0 - 1e-9:
            flush()
    flush()
    return out


class Carla50mCropDataset(CarlaMapDataset):
    """50 m CARLA tiles -> augmented 30 m crops, in ``CarlaMapDataset``'s contract.

    Args (beyond ``CarlaMapDataset``'s ``map_conf`` / ``point_conf`` /
    ``transforms`` / ``max_lidar_points`` / ``lidar_voxel_size`` /
    ``pivot_vm_thre``):
        ann_file (str): ``<data_root>/<split>/manifest.json``; ``blocks/``
            and ``<reference_dir>/`` are siblings of it.
        crop_size (float): side of the square crop, metres. Must equal the
            ``map_region`` extent (checked).
        rotate / translate / noise_std: the augmentation. Train: all on.
            Val/test: ``rotate=False``, ``noise_std=0.0`` and a ``crop_seed``.
        require_gt / gt_tries: aim each crop at a random GT vertex so it is
            non-empty (see ``_draw_crop``).
        min_polyline_len (float): drop clipped stubs shorter than this.
        gt_classes (tuple[str] | None): the export's class names, in the
            order they map onto ``map_classes`` (positional). ``None`` means
            the export uses the config's own names (the driving/curb case).
            Polylines carrying no class at all count as ``gt_classes[0]``.
        reference_dir (str): polyline directory under ``<split>/``.
        z_ground_percentile (float): points' z is re-zeroed on this
            percentile of the crop's own z.
        crop_seed (int | None): freeze the crop pose per index (val/test).
        point_dim (int): 3 -> xyz (the colour-free benchmark convention),
            4 -> xyz + BT.709 strength. Must equal the pillar encoder's
            ``ppdim``.
    """

    def __init__(self,
                 ann_file,
                 map_conf,
                 point_conf,
                 transforms=None,
                 data_split='train',
                 max_lidar_points=40000,
                 lidar_voxel_size=0.15,
                 pivot_vm_thre=0.2,
                 point_dim=3,
                 crop_size=30.0,
                 rotate=True,
                 translate=True,
                 noise_std=0.02,
                 noise_xy_only=False,
                 require_gt=True,
                 gt_tries=32,
                 min_polyline_len=2.0,
                 gt_classes=None,
                 reference_dir='reference_lines',
                 z_ground_percentile=2.0,
                 crop_seed=None):
        # torch Dataset init only -- CarlaMapDataset.__init__ would try to
        # unpickle ann_file. The shared members are set here by hand.
        torch.utils.data.Dataset.__init__(self)
        self.ann_file = ann_file
        self.map_conf = map_conf
        self.split_mode = data_split
        self.max_lidar_points = max_lidar_points
        self.lidar_voxel_size = lidar_voxel_size
        self.z_max = None          # no stored-z cut: z is re-zeroed per crop
        self.use_colour = point_dim == 4
        if point_dim not in (3, 4):
            raise ValueError(f'point_dim must be 3 or 4, got {point_dim}')
        self.point_dim = int(point_dim)

        self.map_region = map_conf['map_region']
        self.ego_size = map_conf['ego_size']
        self.map_classes = list(map_conf['map_classes'])
        if len(point_conf['max_pieces']) != len(self.map_classes):
            raise ValueError(
                f'pivot_conf["max_pieces"] has {len(point_conf["max_pieces"])} '
                f'entries for {len(self.map_classes)} map classes')

        patch_h = map_conf['map_region'][3] - map_conf['map_region'][2]
        patch_w = map_conf['map_region'][1] - map_conf['map_region'][0]
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (int(patch_h / map_conf['map_resolution']),
                            int(patch_w / map_conf['map_resolution']))
        self.half = float(crop_size) / 2.0
        if not (np.isclose(patch_h, crop_size) and np.isclose(patch_w, crop_size)):
            raise ValueError(
                f'crop_size={crop_size} must equal the map_region extent '
                f'({patch_w} x {patch_h}): the crop IS the patch.')

        from .generate_pivots import GenPivots
        from .rasterize import RasterizedLocalMap
        self.raster_map = RasterizedLocalMap(
            self.patch_size,
            self.canvas_size,
            num_degrees=[2] * len(self.map_classes),
            max_channel=len(self.map_classes),
            thickness=[1, map_conf['line_width']],
            bezier=False)
        self.pivot_gen = GenPivots(max_pts=point_conf['max_pieces'],
                                   map_region=map_conf['map_region'],
                                   vm_thre=pivot_vm_thre,
                                   resolution=map_conf['map_resolution'])
        self.transforms = transforms

        # --- crop machinery (mirrors MapTR's CustomCarla50mCropDataset) ---
        split_dir = os.path.dirname(os.path.abspath(ann_file))
        self.blocks_dir = os.path.join(split_dir, 'blocks')
        self.gt_dir = os.path.join(split_dir, reference_dir)
        self.rotate = bool(rotate)
        self.translate = bool(translate)
        self.noise_std = float(noise_std)
        self.noise_xy_only = bool(noise_xy_only)
        self.require_gt = bool(require_gt)
        self.gt_tries = int(gt_tries)
        self.min_polyline_len = float(min_polyline_len)
        self.z_ground_percentile = float(z_ground_percentile)
        self.crop_seed = crop_seed
        self.gt_classes = tuple(gt_classes) if gt_classes else tuple(
            self.map_classes)
        if len(self.gt_classes) != len(self.map_classes):
            raise ValueError(
                f'gt_classes {self.gt_classes} must pair positionally with '
                f'map_classes {self.map_classes}')
        self._src_keep = {n: i for i, n in enumerate(self.gt_classes)}
        self._need = self.half * (np.sqrt(2.0) if self.rotate else 1.0)
        self.max_shift = 0.0

        self.metadata = {}
        self.infos = self.load_annotations(ann_file)
        self.tokens = [info['token'] for info in self.infos]

    # ------------------------------------------------------------------
    def load_annotations(self, ann_file):
        with open(ann_file) as f:
            man = json.load(f)
        tiles = man.get('tiles', [])
        if not tiles:
            raise RuntimeError(f'no tiles in manifest {ann_file}')
        self.metadata = {k: v for k, v in man.items() if k != 'tiles'}

        name0 = tiles[0]['name'] if isinstance(tiles[0], dict) else tiles[0]
        with np.load(os.path.join(self.blocks_dir, f'{name0}.npz')) as d:
            radius = float(d['tile_radius'])
        if self._need > radius + 1e-6:
            raise ValueError(
                f'crop_size={2*self.half} with rotate={self.rotate} needs '
                f'tile_radius >= {self._need:.2f} m, but tiles have radius '
                f'{radius:.2f} m')
        self.max_shift = max(0.0, radius - self._need) if self.translate else 0.0

        infos = []
        for i, t in enumerate(tiles):
            name = t['name'] if isinstance(t, dict) else t
            infos.append(dict(
                token=name,
                sample_idx=name,
                lidar_path=os.path.join(self.blocks_dir, f'{name}.npz'),
                timestamp=i,
                tile_center=np.asarray(t['center'], dtype=np.float64)[:2]
                if isinstance(t, dict) and 'center' in t else None,
            ))

        # Polylines are needed on EVERY __getitem__ (fresh crop each time), so
        # read the json files once here.
        self._polys = {}
        self._tc_z = {}
        for info in infos:
            self._polys[info['token']] = self._load_polylines(info['token'])
            if info['tile_center'] is None:
                with np.load(info['lidar_path']) as d:
                    info['tile_center'] = np.asarray(
                        d['tile_center'], dtype=np.float64)[:2]

        # Frozen eval crops: see the module docstring.
        if self.crop_seed is not None:
            for idx, info in enumerate(infos):
                rng = np.random.default_rng(self.crop_seed + idx)
                th, R, sh = self._draw_crop(self._polys.get(info['token'], []),
                                            info['tile_center'], rng)
                info['crop_theta'] = th
                info['crop_R'] = R
                info['crop_shift'] = sh
        return infos

    def _load_polylines(self, name):
        f = os.path.join(self.gt_dir, f'{name}_reference_lines.json')
        if not os.path.exists(f):
            self._tc_z[name] = 0.0
            return []
        with open(f) as fh:
            gt = json.load(fh)
        lookup = gt.get('classes') or {}
        tc = gt.get('tile_center')
        self._tc_z[name] = float(tc[2]) if tc is not None and len(tc) > 2 else 0.0
        out = []
        for pl in gt.get('polylines', []):
            cname = pl.get('class')
            if cname is None and 'class_id' in pl:
                cname = lookup.get(str(pl['class_id']))
            if cname is None:
                cname = self.gt_classes[0]
            if cname not in self._src_keep:
                continue
            p = np.asarray(pl['points'], dtype=np.float32)
            if p.ndim != 2 or len(p) < 2:
                continue
            out.append({'points': p, 'class_id': self._src_keep[cname],
                        'is_arc': bool(pl.get('is_arc', False))})
        return out

    # ------------------------------------------------------------------
    # Augmentation geometry -- identical to MapTR's
    # ------------------------------------------------------------------
    @staticmethod
    def _transform(xy, centre, R, shift=None):
        out = (xy - centre) @ R.T
        return out if shift is None else out - shift

    def _crop_polylines(self, polys, centre, R, shift=None):
        """Rotate + shift into crop frame, then clip to the +-half box."""
        box = (-self.half, -self.half, self.half, self.half)
        pts_out, cls_out = [], []
        for pl in polys:
            p = pl['points']
            q = p.copy().astype(np.float64)
            q[:, :2] = self._transform(p[:, :2].astype(np.float64), centre, R,
                                       shift)
            curved = np.full(len(q), pl['is_arc'], dtype=bool)
            for sub in clip_polyline_to_box(q.astype(np.float32), curved, box):
                s = sub['points']
                if len(s) < 2:
                    continue
                if float(np.linalg.norm(np.diff(s[:, :2], axis=0),
                                        axis=1).sum()) < self.min_polyline_len:
                    continue
                pts_out.append(np.asarray(s, dtype=np.float32))
                cls_out.append(pl['class_id'])
        return pts_out, cls_out

    def _rand_pose(self, rng):
        th = float(rng.uniform(0.0, 2.0 * np.pi)) if self.rotate else 0.0
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        if self.max_shift > 0 and self.translate:
            rad = self.max_shift * np.sqrt(rng.random())
            ang = rng.uniform(0.0, 2.0 * np.pi)
            sh = np.array([rad * np.cos(ang), rad * np.sin(ang)],
                          dtype=np.float64)
        else:
            sh = np.zeros(2, dtype=np.float64)
        return th, R, sh

    def _draw_crop(self, polys, centre, rng):
        """Pick (yaw, shift), AIMING the crop at a random GT vertex."""
        if not (self.require_gt and polys):
            return self._rand_pose(rng)
        for _ in range(max(1, self.gt_tries)):
            pl = polys[int(rng.integers(len(polys)))]
            q = pl['points']
            p = q[int(rng.integers(len(q)))][:2].astype(np.float64)
            th, R, _ = self._rand_pose(rng)
            v = (p - centre) @ R.T
            n = float(np.hypot(v[0], v[1]))
            if n > 1e-9:
                pull = min(self.max_shift, max(0.0, n - 0.6 * self.half))
                sh = v * (pull / n)
            else:
                sh = np.zeros(2, dtype=np.float64)
            if self._crop_polylines(polys, centre, R, sh)[0]:
                return th, R, sh
        best = None
        for pl in polys:
            q = np.asarray(pl['points'], dtype=np.float64)[:, :2]
            d = np.hypot(q[:, 0] - centre[0], q[:, 1] - centre[1])
            j = int(np.argmin(d))
            if best is None or d[j] < best[0]:
                best = (float(d[j]), q[j])
        if best is not None:
            p = best[1]
            for th in (0.0,) + tuple(
                    np.linspace(0, 2 * np.pi, 8, endpoint=False)[1:]):
                c, s_ = np.cos(th), np.sin(th)
                R = np.array([[c, -s_], [s_, c]], dtype=np.float64)
                v = (p - centre) @ R.T
                n = float(np.hypot(v[0], v[1]))
                sh = v * (min(self.max_shift, max(0.0, n - 0.6 * self.half)) / n) \
                    if n > 1e-9 else np.zeros(2, dtype=np.float64)
                if self._crop_polylines(polys, centre, R, sh)[0]:
                    return float(th), R, sh
        return self._rand_pose(rng)

    def crop_pose(self, idx):
        """(theta, R, shift) for sample ``idx``: frozen if seeded, else fresh."""
        info = self.infos[idx]
        if 'crop_R' in info:
            return info['crop_theta'], info['crop_R'], info['crop_shift']
        rng = np.random.default_rng(
            None if self.crop_seed is None else self.crop_seed + idx)
        return self._draw_crop(self._polys.get(info['token'], []),
                               info['tile_center'], rng)

    # ------------------------------------------------------------------
    # per-sample work, in CarlaMapDataset's contract
    # ------------------------------------------------------------------
    def get_crop_vectors(self, info, R, shift):
        """Crop GT -> ``RasterizedLocalMap`` inputs (xy metres, crop frame)."""
        pts, cls = self._crop_polylines(self._polys.get(info['token'], []),
                                        info['tile_center'], R, shift)
        vectors = []
        for p, c in zip(pts, cls):
            if c >= len(self.map_classes):
                continue
            vectors.append({'pts': np.asarray(p[:, :2], dtype=np.float64),
                            'pts_num': len(p), 'type': int(c)})
        return vectors

    def get_crop_lidar(self, info, R, shift, noise_std=None):
        """``([P, point_dim], [P] mask)`` for one crop, same pose as the GT."""
        with np.load(info['lidar_path']) as d:
            world = d['points'].astype(np.float64)
            rgb = (d['features'][:, 3:6].astype(np.float32)
                   if self.use_colour else None)
        centre = info['tile_center']
        local = (world[:, :2] - centre) @ R.T - shift
        keep = (np.abs(local[:, 0]) <= self.half) & \
               (np.abs(local[:, 1]) <= self.half)
        n = int(keep.sum())
        xyz = np.empty((n, 3), dtype=np.float32)
        xyz[:, :2] = local[keep].astype(np.float32)
        xyz[:, 2] = world[keep, 2].astype(np.float32)

        std = self.noise_std if noise_std is None else float(noise_std)
        if std > 0 and n:
            noise = np.random.normal(
                0.0, std, size=(n, 2 if self.noise_xy_only else 3)
            ).astype(np.float32)
            xyz[:, :2] += noise[:, :2]
            if not self.noise_xy_only:
                xyz[:, 2] += noise[:, 2]
        if n:
            xyz[:, 2] -= np.percentile(xyz[:, 2], self.z_ground_percentile)

        if self.use_colour:
            strength = (rgb[keep] @ LUMA_BT709).reshape(-1, 1)
            points = np.concatenate([xyz, strength], axis=1)
        else:
            points = xyz

        # The pillar grid is [-half, half) per axis (points_to_voxels floors
        # into grid cells); a point exactly on +half would index one cell
        # past the grid, so keep the same half-open range CarlaMapDataset
        # applies.
        x_min, x_max, y_min, y_max = self.map_region
        in_range = ((points[:, 0] >= x_min) & (points[:, 0] < x_max)
                    & (points[:, 1] >= y_min) & (points[:, 1] < y_max))
        points = points[in_range]

        points = grid_sample_points(points, self.lidar_voxel_size)
        num_points = min(len(points), self.max_lidar_points)
        points = pad_or_trim_to_np(
            points, [self.max_lidar_points, self.point_dim]).astype(np.float32)
        mask = np.zeros(self.max_lidar_points, dtype=np.float32)
        mask[:num_points] = 1.0
        return torch.from_numpy(points), mask

    def __getitem__(self, idx):
        info = self.infos[idx]
        _, R, shift = self.crop_pose(idx)

        vectors = self.get_crop_vectors(info, R, shift)
        _, instance_masks, instance_vec_points, _ = \
            self.raster_map.convert_vec_to_mask(vectors)
        pivots, pivot_lengths = self.pivot_gen.pivots_generate(
            instance_vec_points)
        pivots = {k: np.asarray(v, dtype=np.float32)
                  for k, v in pivots.items()}
        pivot_lengths = {k: np.asarray(v, dtype=np.int64)
                         for k, v in pivot_lengths.items()}
        lidars, lidar_mask = self.get_crop_lidar(info, R, shift)

        targets = dict(masks=instance_masks[1], points=pivots,
                       valid_len=pivot_lengths)
        item = dict(lidars=lidars, lidar_mask=lidar_mask, targets=targets,
                    extra_infos=dict(token=info['token'],
                                     map_size=self.ego_size))
        if self.transforms is not None:
            item = self.transforms(item)
        return item
