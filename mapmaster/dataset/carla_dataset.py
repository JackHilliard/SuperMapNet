"""LiDAR-only CARLA tile dataset for SuperMapNet / PivotNet.

Reads the pkl written by ``tools/anno_converter/carla/convert.py`` and emits
the same item dict ``NuScenesMapDataset`` does, minus the camera keys: the
CARLA export ships a point cloud and reference polylines per tile and no
imagery at all, which is why the accompanying config drives the LiDAR
branch alone (see ``mapmaster/models/network.py``'s ``LiDARMapMaster``).

Everything downstream of the vectors is the repo's own machinery --
``RasterizedLocalMap`` for the instance/semantic masks and ``GenPivots``
for the pivot targets -- so CARLA GT arrives at the loss in exactly the
shape the nuScenes GT does. The only CARLA-specific steps are:

* reading the polylines out of the pkl and dropping them into the
  ``{'pts', 'pts_num', 'type'}`` form ``RasterizedLocalMap`` consumes,
* turning a tile's stored ``features`` (xyz + RGB) into the ``[x, y, z,
  strength]`` the pillar encoder wants, in the GT's own frame.

Frames: the converter records, per tile, the ``annotation_origin`` its
polylines are relative to and the ``lidar_recenter_shift`` that moves the
stored points into that same frame. Both are applied here; a pkl that
predates either key is rejected rather than loaded misaligned, because the
failure is otherwise invisible -- points and GT drift apart by a couple of
metres and the model simply learns less.
"""

import os
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from .generate_pivots import GenPivots
from .rasterize import RasterizedLocalMap
from .voxel import pad_or_trim_to_np

# ITU-R BT.709 luma weights. The CARLA blocks store per-point RGB in [0, 1]
# where a LiDAR dataset would store intensity; the sibling MapTRv2/GeMap
# loaders collapse it to this single "strength" scalar, and the pillar
# encoder's `ppdim=4` expects exactly one feature channel beside xyz. Hue
# and saturation are gone at that point, brightness is not.
LUMA_BT709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def load_carla_infos(ann_file):
    """Read a converter pkl, tolerating both the dict and bare-list forms."""
    import pickle

    with open(ann_file, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'infos' in data:
        return data['infos'], data.get('metadata', {})
    if isinstance(data, list):
        return data, {}
    raise ValueError(f'{ann_file}: not a CARLA annotation pkl '
                     f'(got a {type(data).__name__} without `infos`)')


def grid_sample_points(points, grid_size):
    """Keep one point per occupied grid cell, in xy.

    The pillar encoder collapses every point of an xy cell into one BEV cell
    anyway, so this throws away nothing it could have used, while bounding
    the per-tile point count: these tiles hold a median 78k points (max
    563k) but only ~21k occupied 0.15 m cells. The sibling MapTRv2 repo
    measured the same substitution at 26x faster voxelization with 100% of
    occupied cells recovered, against 9-22% for random subsampling -- which
    thins sparse structures like lane lines because it is
    density-proportional where this is density-uniform.

    The representative is the FIRST point of each cell by index (matching
    that implementation), which makes the result deterministic and identical
    between training and evaluation.
    """
    if len(points) == 0:
        return points
    cells = np.floor(points[:, :2] / grid_size).astype(np.int64)
    # A single int64 key per cell: cheaper than np.unique on a 2-column
    # array, and exact as long as the tile is far smaller than 2**31 cells.
    key = (cells[:, 0] - cells[:, 0].min()).astype(np.int64) * (2 ** 32) \
        + (cells[:, 1] - cells[:, 1].min()).astype(np.int64)
    _, first = np.unique(key, return_index=True)
    return points[np.sort(first)]


class CarlaMapDataset(Dataset):
    """One sample per CARLA tile.

    Args:
        ann_file (str): converter pkl for this split.
        map_conf (dict): the config's ``map_conf``; ``map_region``,
            ``map_resolution``, ``map_size``, ``ego_size``, ``line_width``
            and ``map_classes`` are read from it.
        point_conf (dict): the config's ``pivot_conf`` (``max_pieces``).
        transforms: applied to the item dict (``ToTensor_Pivot`` alone is
            enough here -- the image transforms have nothing to act on).
        data_split (str): informational only; the split is the pkl.
        max_lidar_points (int): fixed per-sample point budget. Tiles with
            more points after grid sampling are trimmed, tiles with fewer
            are zero-padded and masked -- the mask is what the encoder
            reads, so padding never contributes to a pillar.
        lidar_voxel_size (float): xy cell used by the grid sampling above.
            Keep it equal to the pillar size in the config's
            ``lidar_encoder`` or the sampling is either lossy or pointless.
        z_max (float): points above this stored z are dropped before the
            recentring shift, matching the converter's own count.
        pivot_vm_thre (float): Visvalingam-Whyatt simplification threshold
            (effective area, m^2) behind the pivot targets. nuScenes' 2.0 is
            tuned for 120 m lines; on these 30 m tiles it flattens CARLA's
            arcs, which are 11.5% of the polylines -- measured over 751 real
            polylines, thr 2.0 leaves an arc up to 0.68 m from its own
            geometry, i.e. past the 0.5 m chamfer threshold the eval scores
            at, while 0.2 caps that at 0.15 m (one pixel) and still fits
            every polyline inside `max_pieces` (0.00% truncated over 4250).
        use_colour (bool): keep the RGB-derived strength channel. False
            zeroes it, for an ablation against the geometry alone; the
            encoder's ``ppdim`` stays 4 either way.
    """

    def __init__(self,
                 ann_file,
                 map_conf,
                 point_conf,
                 transforms=None,
                 data_split='train',
                 max_lidar_points=40000,
                 lidar_voxel_size=0.15,
                 z_max=96.0,
                 min_lidar_points=1,
                 pivot_vm_thre=0.2,
                 use_colour=True):
        super().__init__()
        self.ann_file = ann_file
        self.map_conf = map_conf
        self.split_mode = data_split
        self.max_lidar_points = max_lidar_points
        self.lidar_voxel_size = lidar_voxel_size
        self.z_max = z_max
        self.use_colour = use_colour

        self.map_region = map_conf['map_region']
        self.ego_size = map_conf['ego_size']
        self.map_classes = list(map_conf['map_classes'])

        patch_h = map_conf['map_region'][3] - map_conf['map_region'][2]
        patch_w = map_conf['map_region'][1] - map_conf['map_region'][0]
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (int(patch_h / map_conf['map_resolution']),
                            int(patch_w / map_conf['map_resolution']))
        # `num_degrees` only matters with bezier=True (BeMapNet); the pivot
        # path never builds a curve here. `thickness` must keep 1 first --
        # RasterizedLocalMap asserts it -- and the second entry is the one
        # the targets use, so it is the config's own `line_width`.
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

        infos, metadata = load_carla_infos(ann_file)
        self.metadata = metadata
        self._check_metadata(metadata, point_conf)
        self.infos = self._filter(infos, min_lidar_points)
        self.tokens = [info['token'] for info in self.infos]

    # ------------------------------------------------------------------
    # loading-time checks
    # ------------------------------------------------------------------
    def _check_metadata(self, metadata, point_conf):
        """Fail loudly on the mismatches that otherwise train silently.

        A config naming a class the pkl lacks produces an annotation key of
        zero instances -- the model then learns a class that never appears,
        and nothing anywhere says so. A config listing the same names in a
        different ORDER is worse: order is label order, so every instance is
        relabelled.
        """
        pkl_classes = metadata.get('map_classes')
        if pkl_classes is not None and list(pkl_classes) != self.map_classes:
            missing = [c for c in self.map_classes if c not in pkl_classes]
            if missing:
                raise ValueError(
                    f'{self.ann_file} has no annotation for map class(es) '
                    f'{missing}; it carries {list(pkl_classes)}. Re-run the '
                    'converter with a matching --map-classes.')
            raise ValueError(
                f'{self.ann_file} lists its map classes as {list(pkl_classes)} '
                f'but the config says {self.map_classes}; the order IS the '
                'label order, so this would relabel every instance.')
        if len(point_conf['max_pieces']) != len(self.map_classes):
            raise ValueError(
                f'pivot_conf["max_pieces"] has {len(point_conf["max_pieces"])} '
                f'entries for {len(self.map_classes)} map classes')

        pkl_range = metadata.get('pc_range')
        if pkl_range is not None:
            want = [self.map_region[0], self.map_region[2],
                    self.map_region[1], self.map_region[3]]
            got = [pkl_range[0], pkl_range[1], pkl_range[3], pkl_range[4]]
            if not np.allclose(want, got, atol=1e-3):
                warnings.warn(
                    f'{self.ann_file} was converted for xy range {got} but '
                    f'the config maps {want}; the tile is being cropped or '
                    'padded relative to what the converter measured.')
        if metadata.get('gt_frame') == 'offset':
            warnings.warn(
                f'{self.ann_file} is in the `offset` frame. On this export '
                'the square point-cloud range is then centred away from the '
                'tile (median coverage 92.5%, 75 empty tiles); '
                '--gt-frame tile_center is what the config assumes.')

    def _filter(self, infos, min_lidar_points):
        """Drop tiles that would give the encoder nothing to look at.

        Done here rather than in ``__getitem__`` so ``len(self)`` and the
        index a result is reported under stay consistent -- skipping inside
        ``__getitem__`` only works while training and silently desynchronises
        an evaluation.
        """
        kept, dropped = [], 0
        for info in infos:
            count = info.get('num_lidar_points_in_range')
            if count is not None and count < min_lidar_points:
                dropped += 1
                continue
            kept.append(info)
        if dropped:
            warnings.warn(f'{self.ann_file}: dropped {dropped} tile(s) with '
                          f'<{min_lidar_points} in-range LiDAR points')
        if not kept:
            raise ValueError(f'{self.ann_file}: no usable tiles')
        return kept

    # ------------------------------------------------------------------
    # per-sample work
    # ------------------------------------------------------------------
    def get_vectors(self, info):
        """The tile's polylines as ``RasterizedLocalMap`` inputs.

        ``pts`` is xy in metres about the annotation origin, i.e. the same
        ego-style frame the nuScenes vectorizer emits (x longitudinal, y
        lateral, both centred on 0), and ``type`` is the class index --
        which is the position of that class name in ``map_classes``.
        """
        vectors = []
        for label, name in enumerate(self.map_classes):
            for pts in info['annotation'].get(name, []):
                pts = np.asarray(pts, dtype=np.float64)[:, :2]
                if len(pts) < 2:
                    continue
                vectors.append({'pts': pts, 'pts_num': len(pts),
                                'type': label})
        return vectors

    def get_lidar(self, info):
        """``([P, 4] xyz+strength, [P] mask)`` for one tile.

        Order matters and mirrors the converter's own count exactly: the
        ``z_max`` cut is defined on the STORED z, the recentring shift comes
        after it, and only then does the xy/z range filter apply.
        """
        with np.load(info['lidar_path']) as block:
            features = np.asarray(block['features'], dtype=np.float32)

        xyz = features[:, :3]
        if features.shape[1] >= 6:
            strength = features[:, 3:6] @ LUMA_BT709
        else:  # an export without colour: geometry only
            strength = np.zeros(len(features), dtype=np.float32)
        if not self.use_colour:
            strength = np.zeros(len(features), dtype=np.float32)

        if self.z_max is not None:
            keep = xyz[:, 2] <= self.z_max
            xyz, strength = xyz[keep], strength[keep]

        shift = info.get('lidar_recenter_shift')
        if shift is None:
            raise ValueError(
                f'{self.ann_file} predates `lidar_recenter_shift`; its points '
                'and GT would be offset from each other by up to ~100m on '
                'this export. Re-run the converter.')
        xyz = xyz + np.asarray(shift, dtype=np.float32)

        x_min, x_max, y_min, y_max = self.map_region
        in_range = ((xyz[:, 0] >= x_min) & (xyz[:, 0] < x_max)
                    & (xyz[:, 1] >= y_min) & (xyz[:, 1] < y_max))
        points = np.concatenate([xyz[in_range],
                                 strength[in_range, None]], axis=1)

        points = grid_sample_points(points, self.lidar_voxel_size)
        num_points = min(len(points), self.max_lidar_points)
        points = pad_or_trim_to_np(
            points, [self.max_lidar_points, 4]).astype(np.float32)
        mask = np.zeros(self.max_lidar_points, dtype=np.float32)
        mask[:num_points] = 1.0
        return torch.from_numpy(points), mask

    def __getitem__(self, idx):
        info = self.infos[idx]

        vectors = self.get_vectors(info)
        _, instance_masks, instance_vec_points, _ = \
            self.raster_map.convert_vec_to_mask(vectors)
        pivots, pivot_lengths = self.pivot_gen.pivots_generate(
            instance_vec_points)
        # float32 to match the model's own dtype: GenPivots works in float64
        # (numpy's default), and a float64 target reaches the losses as-is
        # through default collation.
        pivots = {k: np.asarray(v, dtype=np.float32)
                  for k, v in pivots.items()}
        pivot_lengths = {k: np.asarray(v, dtype=np.int64)
                         for k, v in pivot_lengths.items()}
        lidars, lidar_mask = self.get_lidar(info)

        # instance_masks is (thickness, class, X, Y); index 1 is the thick
        # rendering, the same one NuScenesMapDataset hands to the loss.
        targets = dict(masks=instance_masks[1], points=pivots,
                       valid_len=pivot_lengths)
        item = dict(lidars=lidars, lidar_mask=lidar_mask, targets=targets,
                    extra_infos=dict(token=info['token'],
                                     map_size=self.ego_size))
        if self.transforms is not None:
            item = self.transforms(item)
        return item

    def __len__(self):
        return len(self.infos)
