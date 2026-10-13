"""Convert the CARLA road-polyline tile dataset into a SuperMapNet map-
annotation pkl.

Ported from the sibling MapTRv2 / GeMap converters
(``MapTR/tools/maptrv2/custom_carla_map_converter.py``,
``GeMap/tools/gemap/custom_carla_map_converter.py``), which were both
verified against this exporter's output array-by-array. The geometry
decisions below -- which origin a tile's GT is expressed relative to, which
polyline directory to read, how a tile's own class taxonomy is discovered --
are theirs, and are kept identical on purpose: a pkl produced here holds the
same numbers as one produced there, so the two model families can be
compared on the same GT.

What differs is only the *container*: MapTR writes an mmdet3d-style pkl of
`sample` dicts; this writes a plain dict with an `infos` list that
``mapmaster/dataset/carla_dataset.py`` reads directly.

CARLA tiles are static square patches, not per-timestamp driving-log frames,
so there is no ego pose and no clipping against a moving patch -- this
script only shifts each tile's world-frame polylines into the same frame as
its point cloud.

Both export layouts are accepted::

    <data_root>/<split>/manifest.json          # split export (../carla_test)
    <data_root>/<split>/blocks/<tile>.npz
    <data_root>/<split>/reference_lines/<tile>_reference_lines.json

    <data_root>/grid_manifest.json             # single-town grid export
    <data_root>/blocks/<tile>.npz
    <data_root>/reference_lines/<tile>_reference_lines.json

An export may ship SEVERAL polyline directories holding DIFFERENT GT sets --
``../carla_test`` has an unclassified ``reference_lines/`` (2.1 polylines per
tile, no class) beside a ``reference_curb_driving_lines/`` (3.5 per tile,
every polyline carrying ``class_id``/``class``). ``--reference-dir`` picks
one explicitly; otherwise ``reference_lines/`` wins, except that
``--map-classes``/``--classes`` -- which *need* a taxonomy -- fall back to a
sibling that has one, printing what they picked.

Usage::

    # two-class GT, tile-centre frame, one pkl per town split
    python tools/anno_converter/carla/convert.py \\
        --data-root ../carla_test --split test --gt-frame tile_center \\
        --map-classes driving curb --holdout-towns town12 \\
        --out-dir data/carla --out-tag 30m_tc_2cls
"""

import argparse
import json
import os
import random

import mmcv
import numpy as np

# z half-range of the LiDAR branch. CARLA towns have long z tails (bridges,
# tunnels): the sibling repo measured z_min down to -97 on town03, so this is
# deliberately generous -- points outside it are simply never voxelized. The
# xy half is NOT a constant: it follows the tile size read from the manifest,
# because a range narrower than the tile silently crops it.
DEFAULT_Z_RANGE = (-72.0, 96.0)
DEFAULT_Z_MAX = 96.0
# Only used when a manifest carries no tile geometry at all; matches the
# original 25m export these converters were first written against.
FALLBACK_TILE_RADIUS = 12.5
# Slack allowed before a polyline is reported as leaving its own tile. The
# exporter samples arcs at a fixed arc_gap, so a vertex can land slightly
# past the boundary without anything being wrong.
BOUNDS_MARGIN = 1.0

MANIFEST_NAMES = ('manifest.json', 'grid_manifest.json')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--data-root', required=True,
                        help='CARLA export root (holds <split>/ or the '
                             'blocks/ + manifest directly)')
    parser.add_argument('--split', default='test',
                        help="split sub-directory to read ('' for a grid "
                             'export with no split level)')
    parser.add_argument('--out-dir', default='data/carla',
                        help='where the pkl(s) are written')
    parser.add_argument('--out-tag', default=None,
                        help='appended to the output filename '
                             '(carla_map_infos_<split>_<TAG>.pkl). Without '
                             'it the filename is a function of the split '
                             'name alone, so converting a second dataset '
                             'into the same --out-dir OVERWRITES the first')
    parser.add_argument('--gt-frame', default='tile_center',
                        choices=('offset', 'tile_center'),
                        help="origin the GT and the recentred point cloud "
                             "share. 'tile_center' is the default here: on "
                             "../carla_test the offset frame drops 75 tiles "
                             'outright and crops 1282 more, because a square '
                             'range centred on `offset` does not cover a tile '
                             'whose centre is up to 117m away from it')
    parser.add_argument('--reference-dir', default=None,
                        help='polyline directory to read (default: '
                             'reference_lines, or a class-carrying sibling '
                             'when --map-classes/--classes needs one)')
    parser.add_argument('--map-classes', nargs='+', default=None,
                        metavar='NAME[=EXPORT_CLASS[,EXPORT_CLASS...]]',
                        help='emit one annotation key per named map class, '
                             'in label order (e.g. `driving curb`, or '
                             '`curb=curb,road_edge`). Without it every kept '
                             'polyline becomes one `divider` instance')
    parser.add_argument('--classes', nargs='+', default=None,
                        help='keep only these export classes (ids or names) '
                             'and collapse them into one `divider` class')
    parser.add_argument('--lidar-point-cloud-range', nargs=6, type=float,
                        default=None,
                        help='x_min y_min z_min x_max y_max z_max used for '
                             'the in-range point count. Derived from the '
                             "manifest's own tile size when omitted -- which "
                             'is almost always what you want')
    parser.add_argument('--z-max', type=float, default=DEFAULT_Z_MAX,
                        help='drop points above this z before counting '
                             '(mirrors the dataset loader)')
    parser.add_argument('--min-lidar-points', type=int, default=1,
                        help='drop tiles with fewer in-range points than '
                             'this; they would produce an empty BEV')
    parser.add_argument('--no-lidar-check', dest='lidar_check',
                        action='store_false',
                        help='skip the per-tile point scan (fast, but no '
                             'empty-tile filtering and no coverage report)')
    parser.add_argument('--holdout-towns', nargs='+', default=None,
                        help='tiles of these towns become the val pkl and '
                             'the rest the train pkl. The strongest split '
                             'this export supports: towns share no geometry')
    parser.add_argument('--holdout-fraction', type=float, default=None,
                        help='alternative to --holdout-towns: this fraction '
                             'of tiles, chosen by a seeded shuffle, becomes '
                             'the val pkl')
    parser.add_argument('--holdout-seed', type=int, default=0)
    parser.add_argument('--holdout-names', nargs=2, default=('train', 'val'),
                        metavar=('TRAIN', 'VAL'),
                        help='split names used in the two output filenames')
    return parser.parse_args()


def load_manifest(data_root, split):
    """Locate and load the manifest describing one set of tiles.

    Accepts either export layout. Returns ``(tile_dir, manifest)``, where
    `tile_dir` is the directory holding `blocks/` and the polyline
    directories. Where both manifest names are present, `manifest.json`
    wins and `grid_manifest.json` only backfills keys it lacks --
    `manifest.json` is the curated view (in the Town10HD grid export it
    lists 30 tiles where grid_manifest lists 33, the 3 omitted having no
    driving centreline at all). Its dataset-level *counts* are not
    trustworthy, so nothing here reads them: everything is derived from the
    tile entries.
    """
    for candidate in ([os.path.join(data_root, split)] if split else
                      []) + [data_root]:
        present = [n for n in MANIFEST_NAMES
                   if os.path.isfile(os.path.join(candidate, n))]
        if not present:
            continue
        manifest = {}
        # reversed so the preferred name is applied last and its keys win. A
        # null value never overwrites a real one -- grid_manifest.json spells
        # absent options as `null` rather than omitting them.
        for name in reversed(present):
            with open(os.path.join(candidate, name), encoding='utf-8') as f:
                loaded = json.load(f)
            manifest.update({k: v for k, v in loaded.items()
                             if v is not None or k not in manifest})
        if not manifest.get('tiles'):
            raise ValueError(f'{candidate}: manifest lists no tiles')
        return candidate, manifest
    raise FileNotFoundError(
        f'no {" or ".join(MANIFEST_NAMES)} under {data_root!r}'
        f'{f" or {os.path.join(data_root, split)!r}" if split else ""}')


def _first_ref_json(tile_dir, ref_dir, tiles):
    """Parse the first tile's polyline json under ``ref_dir``, or None."""
    for tile in tiles:
        path = os.path.join(tile_dir, ref_dir,
                            f'{tile["name"]}_reference_lines.json')
        if os.path.isfile(path):
            with open(path, encoding='utf-8') as f:
                return json.load(f)
    return None


def reference_dir_candidates(tile_dir):
    """Directories that could hold per-tile polyline jsons, preferred first.

    ``reference_lines`` is the name every export has used for the
    unclassified set, so it stays first; anything else matching
    ``reference*lines*`` is a variant.
    """
    try:
        entries = sorted(os.listdir(tile_dir))
    except OSError:
        return []
    found = [d for d in entries
             if d.startswith('reference') and 'lines' in d
             and os.path.isdir(os.path.join(tile_dir, d))]
    found.sort(key=lambda d: (d != 'reference_lines', d))
    return found


def resolve_reference_dir(tile_dir, manifest, requested, need_classes):
    """Pick the polyline directory to convert, and its class lookup.

    Returns ``(ref_dir, class_lookup)``. The lookup comes from the manifest
    where an export states one, and otherwise from each polyline json's own
    ``classes`` key -- ../carla_test's manifest has no ``class_lookup`` at
    all and declares ``{"0": "driving", "1": "curb"}`` per tile instead.
    """
    tiles = manifest.get('tiles') or []
    candidates = reference_dir_candidates(tile_dir)
    if requested:
        path = os.path.join(tile_dir, requested)
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f'{path} does not exist; this export has: '
                f'{", ".join(candidates) or "no reference*lines* directory"}')
        candidates = [requested]
    if not candidates:
        raise FileNotFoundError(
            f'no reference*lines* directory under {tile_dir}')

    def lookup_for(ref_dir):
        stated = manifest.get('class_lookup')
        if stated:
            return {str(k): v for k, v in stated.items()}
        ref = _first_ref_json(tile_dir, ref_dir, tiles)
        classes = (ref or {}).get('classes') or {}
        return {str(k): v for k, v in classes.items()}

    chosen = candidates[0]
    lookup = lookup_for(chosen)
    if need_classes and not lookup:
        # Only now is it worth looking past the default directory.
        for alt in candidates[1:]:
            alt_lookup = lookup_for(alt)
            if alt_lookup:
                stated = ', '.join(f'{k}={v}'
                                   for k, v in sorted(alt_lookup.items()))
                print(f'  [ref] {chosen}/ carries no class taxonomy; using '
                      f'{alt}/ instead, which declares {stated}')
                return alt, alt_lookup
    print(f'  [ref] reading polylines from {chosen}/'
          + (f' (classes: {", ".join(sorted(lookup.values()))})'
             if lookup else ' (no class taxonomy)'))
    return chosen, lookup


def manifest_tile_radius(manifest):
    """Half a tile's side, in metres, from whichever key the export used."""
    if manifest.get('tile_radius') is not None:
        return float(manifest['tile_radius'])
    if manifest.get('tile_side') is not None:
        return float(manifest['tile_side']) / 2.0
    radii = []
    for tile in manifest.get('tiles', []):
        bounds = tile.get('bounds')
        if bounds is None:
            continue
        bounds = np.asarray(bounds, dtype=np.float64)
        if bounds.size == 4:
            radii.append(np.max(bounds[2:] - bounds[:2]) / 2.0)
        elif bounds.size == 6:
            radii.append(np.max(bounds[3:5] - bounds[:2]) / 2.0)
    return float(max(radii)) if radii else None


def tile_footprint(tile, ref, tile_radius):
    """World-frame xy footprint of one tile, as ``(lo, hi)`` arrays."""
    bounds = tile.get('bounds') or ref.get('tile_bounds')
    if bounds is not None:
        bounds = np.asarray(bounds, dtype=np.float64)
        if bounds.size == 4:      # [x_min, y_min, x_max, y_max]
            return bounds[:2], bounds[2:]
        if bounds.size == 6:      # [x_min, y_min, z_min, x_max, y_max, z_max]
            return bounds[:2], bounds[3:5]

    center = tile.get('center') or ref.get('tile_center')
    radius = ref.get('tile_radius')
    radius = float(radius) if radius is not None else tile_radius
    if center is None or radius is None:
        return None
    center = np.asarray(center, dtype=np.float64)[:2]
    return center - radius, center + radius


def default_pc_range(tile_radius):
    """The range a training config for this tile size would use: square in
    xy, matching the tile, with the config's z span."""
    radius = tile_radius if tile_radius is not None else FALLBACK_TILE_RADIUS
    z_min, z_max = DEFAULT_Z_RANGE
    return [-radius, -radius, z_min, radius, radius, z_max]


def resolve_classes(selected, manifest):
    """Map ``--classes`` entries (ids or names) onto class_lookup ids.

    Returns a set of int ids, or None for "keep everything". Raises rather
    than silently keeping nothing -- an empty pkl with no explanation is the
    worst possible outcome here.
    """
    if not selected:
        return None
    lookup = manifest.get('class_lookup') or {}
    if not lookup:
        raise ValueError(
            'this export has no class taxonomy, so its polylines cannot be '
            'filtered or split; re-run without --classes/--map-classes '
            '(every polyline then becomes one `divider` instance)')
    by_name = {name: int(cid) for cid, name in lookup.items()}
    out = set()
    for entry in selected:
        entry = str(entry)
        if entry in by_name:
            out.add(by_name[entry])
        elif entry.lstrip('-').isdigit() and str(int(entry)) in lookup:
            out.add(int(entry))
        else:
            known = ', '.join(f'{cid}={name}'
                              for cid, name in sorted(lookup.items()))
            raise ValueError(
                f'unknown class {entry!r}; this export has: {known}')
    return out


def resolve_map_classes(selected, manifest):
    """Parse ``--map-classes`` into ordered ``[(name, {class ids}), ...]``.

    The order is the label order the model trains against, so it is
    preserved exactly as given and must equal the config's ``map_classes``.
    A class appearing in two groups is an error, not a duplication: the same
    polyline would become two GT instances at the same coordinates with
    different labels, which no matcher can resolve.
    """
    if not selected:
        return None
    groups = []
    seen_names = set()
    owner = {}
    for entry in selected:
        name, _, members = entry.partition('=')
        name = name.strip()
        if not name:
            raise ValueError(f'--map-classes entry {entry!r} has no name')
        if name in seen_names:
            raise ValueError(f'--map-classes names must be unique; {name!r} '
                             'appears twice')
        seen_names.add(name)
        member_list = [m.strip() for m in members.split(',') if m.strip()]
        ids = resolve_classes(member_list or [name], manifest)
        for cid in sorted(ids):
            if cid in owner:
                raise ValueError(
                    f'export class id {cid} is listed under both '
                    f'{owner[cid]!r} and {name!r}; a polyline can only carry '
                    'one label')
            owner[cid] = name
        groups.append((name, ids))
    return groups


def polyline_class_id(poly):
    """The polyline's class id, or None on a class-free export.

    Deliberately does not fall back to `type`: that key exists in every
    export and holds a geometry kind ('arc'/'straight'), not a class.
    """
    if poly.get('class_id') is not None:
        return int(poly['class_id'])
    return None


def tile_center_origin(tile, offset):
    """The 3-vector to subtract from a tile's world-frame polylines under
    ``--gt-frame tile_center``.

    xy is the tile centre: ``center`` where the export states one, else the
    midpoint of ``bounds``.

    z is the awkward part: not every export gives its tile centres a z
    (../carla_test writes ``center`` as ``[x, y]``). Where a z is stated it
    is used; where it is not, it falls back to the block's own ``offset``,
    making the z component of the recentring shift exactly zero. That keeps
    polylines and points in ONE frame either way -- the shift is rigid and
    applied to both. This mirrors GeMap's ``tile_origin()`` exactly, which
    is the point: the two converters must agree, or a checkpoint trained
    under one is scored against GT built by the other.
    """
    center = tile.get('center')
    if center is None:
        bounds = tile.get('bounds')
        if bounds is None:
            raise ValueError(
                f'tile {tile.get("name")!r} states neither `center` nor '
                '`bounds`, so --gt-frame tile_center cannot place it')
        bounds = np.asarray(bounds, dtype=np.float64)
        if bounds.size == 4:
            center = (bounds[:2] + bounds[2:]) / 2.0
        elif bounds.size == 6:
            center = (bounds[:3] + bounds[3:]) / 2.0
        else:
            raise ValueError(
                f'tile {tile.get("name")!r}: cannot read a centre from '
                f'`bounds` of length {bounds.size}')
    center = np.asarray(center, dtype=np.float32).ravel()
    if center.size == 2:
        return np.array([center[0], center[1], offset[2]], dtype=np.float32)
    if center.size == 3:
        return center.astype(np.float32)
    raise ValueError(
        f'tile {tile.get("name")!r}: `center` has {center.size} components, '
        'expected 2 or 3')


def count_points_in_range(lidar_path, pc_range, z_max, shift=None):
    """Count a block's points that would survive the dataset's own filters.

    Mirrors ``CarlaMapDataset``: the ``z <= z_max`` cut on the STORED z,
    then the recentring shift, then the xyz range test. Grid sampling sits
    between the last two in the dataset but only merges co-located points,
    so it cannot change a zero/non-zero verdict.

    Returns ``(raw count, count surviving both filters)``.
    """
    with np.load(lidar_path) as block:
        xyz = np.asarray(block['features'][:, :3], dtype=np.float32)
    n_raw = int(xyz.shape[0])
    if z_max is not None:
        xyz = xyz[xyz[:, 2] <= z_max]
    if shift is not None:
        xyz = xyz + np.asarray(shift, dtype=np.float32)
    lo = np.asarray(pc_range[:3], dtype=np.float32)
    hi = np.asarray(pc_range[3:], dtype=np.float32)
    n_in_range = int(np.all((xyz >= lo) & (xyz < hi), axis=1).sum())
    return n_raw, n_in_range


def convert_carla_tiles(data_root,
                        split,
                        classes=None,
                        map_classes=None,
                        pc_range=None,
                        z_max=DEFAULT_Z_MAX,
                        min_lidar_points=1,
                        lidar_check=True,
                        gt_frame='tile_center',
                        reference_dir=None):
    assert gt_frame in ('offset', 'tile_center')
    tile_dir, manifest = load_manifest(data_root, split)
    ref_dir, class_lookup = resolve_reference_dir(
        tile_dir, manifest, reference_dir,
        need_classes=bool(classes or map_classes))
    # The lookup may have come from the polyline jsons rather than the
    # manifest; resolve_classes() only ever reads it off the manifest, so put
    # it there. A copy -- the caller's dict is not ours to edit.
    manifest = dict(manifest, class_lookup=class_lookup)
    tile_radius = manifest_tile_radius(manifest)
    if classes and map_classes:
        raise ValueError('--classes and --map-classes both filter the same '
                         'polylines; pass only one')
    keep_classes = resolve_classes(classes, manifest)
    groups = resolve_map_classes(map_classes, manifest) or [('divider',
                                                             keep_classes)]
    class_names = [name for name, _ in groups]
    if pc_range is None:
        pc_range = default_pc_range(tile_radius)
        if tile_radius is None:
            print(f'[warn] this export states no tile size; assuming '
                  f'tile_radius={FALLBACK_TILE_RADIUS}m for the point-cloud '
                  'range. Pass --lidar-point-cloud-range if that is wrong')
    print(f'{tile_dir}: {len(manifest["tiles"])} tiles, tile_radius='
          f'{tile_radius if tile_radius is not None else "unknown"}, '
          f'pc_range={[round(v, 2) for v in pc_range]}')

    samples, dropped, out_of_bounds, coverage = [], [], [], []
    class_conflicts = []
    total_instances = 0
    per_class_instances = {name: 0 for name in class_names}
    prog_bar = mmcv.ProgressBar(len(manifest['tiles']))
    for idx, tile in enumerate(manifest['tiles']):
        prog_bar.update()
        name = tile['name']

        lidar_path = os.path.abspath(
            os.path.join(tile_dir, 'blocks', f'{name}.npz'))
        if not os.path.isfile(lidar_path):
            raise FileNotFoundError(lidar_path)

        # The polylines below are in WORLD coordinates and must be shifted
        # into the same frame as the LiDAR points the model sees. The stored
        # `features[:, 0:3]` are relative to the block's `offset`
        # (`points - offset == features[:, :3]` holds exactly), NOT to
        # `tile_center`. np.load is lazy, so this reads only the small
        # `offset` array.
        with np.load(lidar_path) as block:
            block_offset = np.asarray(block['offset'], dtype=np.float32)

        # --gt-frame tile_center expresses everything -- polylines AND point
        # cloud -- relative to the tile's own centre instead. The stored
        # features are in the offset frame, so the dataset adds
        # `shift = offset - origin` to every point; it is recorded per
        # sample below so the two can never drift apart.
        origin = (tile_center_origin(tile, block_offset)
                  if gt_frame == 'tile_center' else block_offset)
        shift = (block_offset - origin).astype(np.float32)

        n_raw = n_in_range = None
        if lidar_check:
            # Counted in the frame the model will see, i.e. after the
            # recentring shift -- otherwise a tile_center run would measure a
            # square range against a cloud displaced from it, which is the
            # coverage problem this frame choice exists to remove.
            n_raw, n_in_range = count_points_in_range(
                lidar_path, pc_range, z_max, shift=shift)
            if n_in_range < min_lidar_points:
                # This tile would produce an all-zero BEV. Drop it before its
                # polylines are counted, so the reported instance totals
                # describe what training actually sees.
                dropped.append(dict(name=name, town=tile.get('town'),
                                    n_points=n_raw,
                                    n_points_in_range=n_in_range))
                continue
            if n_raw:
                coverage.append((n_in_range / n_raw, name))

        ref_path = os.path.join(tile_dir, ref_dir,
                                f'{name}_reference_lines.json')
        with open(ref_path, encoding='utf-8') as f:
            ref = json.load(f)

        # Every tile of this export restates the taxonomy, so check it rather
        # than trusting the first file: a tile whose ids mean something else
        # would be relabelled silently.
        stated = ref.get('classes')
        if stated and {str(k): v for k, v in stated.items()} != class_lookup:
            class_conflicts.append(dict(name=name, classes=stated))

        annotation = {c: [] for c in class_names}
        for poly in ref['polylines']:
            cid = polyline_class_id(poly)
            # Groups are disjoint by construction, so this appends at most
            # once. (`cls_name`, not `name` -- that is the TILE's name.)
            for cls_name, keep in groups:
                if keep is not None and cid not in keep:
                    continue
                pts = np.array(poly['points'], dtype=np.float32)
                if pts.shape[0] < 2:
                    break
                annotation[cls_name].append((pts - origin).astype(np.float32))
                per_class_instances[cls_name] += 1
                total_instances += 1
                break
        instances = [p for c in class_names for p in annotation[c]]

        # Sanity check only -- tiles are already the patch, so nothing is
        # clipped. Compared against the tile's own footprint shifted into the
        # GT frame, not against +/-tile_radius around zero: in the offset
        # frame the tile is not centred on the origin, and testing a bare
        # radius there fires on nearly every tile.
        footprint = tile_footprint(tile, ref, tile_radius)
        if footprint is not None and instances:
            lo, hi = (footprint[0] - origin[:2] - BOUNDS_MARGIN,
                      footprint[1] - origin[:2] + BOUNDS_MARGIN)
            pts = np.concatenate([p[:, :2] for p in instances])
            overshoot = float(
                np.max(np.maximum(lo - pts, pts - hi), initial=0.0))
            if overshoot > 0:
                out_of_bounds.append(dict(name=name, overshoot=overshoot))

        samples.append(dict(
            lidar_path=lidar_path,
            sample_idx=name,
            token=name,
            timestamp=idx,
            town=tile.get('town'),
            tile_center=tile['center'],
            # the origin the annotation is relative to -- recorded
            # explicitly so the frame is never ambiguous when reading the
            # pkl back (tile_center above is NOT it in the offset frame)
            annotation_origin=origin.tolist(),
            gt_frame=gt_frame,
            # the block's own frame, kept regardless of gt_frame so the
            # shift can be re-derived or undone later
            lidar_offset=block_offset.tolist(),
            # `offset - annotation_origin`: what the dataset adds to the
            # stored features to land in the GT frame. All-zero when
            # gt_frame == 'offset'.
            lidar_recenter_shift=shift.tolist(),
            tile_bounds=tile.get('bounds'),
            num_lidar_points=n_raw,
            num_lidar_points_in_range=n_in_range,
            annotation=annotation,
        ))

    n = max(len(samples), 1)
    print()
    if lidar_check:
        print(f'{split}: {len(samples)} tiles kept, {len(dropped)} dropped '
              f'(<{min_lidar_points} in-range LiDAR point'
              f'{"" if min_lidar_points == 1 else "s"}), {total_instances} '
              f'map instances ({total_instances / n:.1f} per tile)')
        report_class_counts(per_class_instances, n)
        for entry in dropped[:20]:
            print(f'  [drop] {entry["name"]}: {entry["n_points"]} raw points, '
                  f'{entry["n_points_in_range"]} in range')
        if len(dropped) > 20:
            print(f'  [drop] ... and {len(dropped) - 20} more')
    else:
        print(f'{split}: {len(samples)} tiles, {total_instances} map '
              f'instances ({total_instances / n:.1f} per tile) '
              '[LiDAR check skipped]')
        report_class_counts(per_class_instances, n)
    report_coverage(coverage, pc_range)
    report_out_of_bounds(out_of_bounds)
    report_class_conflicts(class_conflicts, class_lookup)
    return samples, dropped, dict(
        tile_dir=tile_dir,
        reference_dir=ref_dir,
        tile_radius=tile_radius,
        tile_side=manifest.get('tile_side'),
        pc_range=list(pc_range),
        z_max=z_max,
        gt_frame=gt_frame,
        class_lookup=manifest.get('class_lookup') or {},
        classes_kept=sorted(keep_classes) if keep_classes else None,
        map_classes=class_names,
        class_groups={name: (sorted(keep) if keep is not None else None)
                      for name, keep in groups},
        n_out_of_bounds=len(out_of_bounds))


def report_class_counts(per_class_instances, n_tiles):
    """Per-map-class instance totals, printed even for the single-class
    default so a run's output always states which annotation keys the pkl
    carries -- a config naming a class the pkl lacks trains on empty GT."""
    for name, count in per_class_instances.items():
        print(f'  [class] {name}: {count} instances '
              f'({count / n_tiles:.1f} per tile)')
    empty = [n for n, c in per_class_instances.items() if c == 0]
    if empty:
        print(f'  [warn] no instances at all for: {", ".join(empty)}')


def report_coverage(coverage, pc_range):
    """Fraction of each tile's points that survive the range filter.

    A low median means the range is wrong for this export (e.g. copied from
    a smaller tile size), which crops geometry silently.
    """
    if not coverage:
        return
    fracs = np.array([c for c, _ in coverage])
    low = sorted((c, n) for c, n in coverage if c < 0.9)
    print(f'  [range] median coverage {np.median(fracs) * 100:.1f}%, '
          f'{len(low)}/{len(coverage)} tiles below 90% '
          f'(xy +/-{pc_range[3]:g}, z {pc_range[2]:g}..{pc_range[5]:g})')
    for frac, name in low[:5]:
        print(f'  [range] {name}: {frac * 100:.1f}%')


def report_out_of_bounds(out_of_bounds):
    if not out_of_bounds:
        return
    ranked = sorted(out_of_bounds, key=lambda e: -e['overshoot'])
    print(f'  [warn] {len(ranked)} tile(s) have polyline vertices past their '
          f'own footprint (margin {BOUNDS_MARGIN}m):')
    for entry in ranked[:5]:
        print(f'  [warn]   {entry["name"]}: {entry["overshoot"]:.2f}m')


def report_class_conflicts(class_conflicts, class_lookup):
    """Tiles whose own class taxonomy disagrees with the one being used --
    relabelling those silently would corrupt the GT."""
    if not class_conflicts:
        return
    print(f'  [warn] {len(class_conflicts)} tile(s) declare a DIFFERENT class '
          f'taxonomy than the one in use ({class_lookup}); their polylines '
          'were labelled with the one in use:')
    for entry in class_conflicts[:5]:
        print(f'  [warn]   {entry["name"]}: {entry["classes"]}')


def partition(samples, args):
    """Split converted samples into (name, samples) pairs to write.

    Without a holdout this is one pkl named after --split. The export this
    was written for (../carla_test) ships a single `test` split of two
    towns, so a real train/val separation has to be carved out here.
    """
    if args.holdout_towns and args.holdout_fraction is not None:
        raise ValueError('pass only one of --holdout-towns / '
                         '--holdout-fraction')
    train_name, val_name = args.holdout_names
    if args.holdout_towns:
        held = set(args.holdout_towns)
        towns = {s.get('town') for s in samples}
        unknown = held - towns
        if unknown:
            raise ValueError(f'--holdout-towns names {sorted(unknown)}, which '
                             f'this export does not have: {sorted(towns)}')
        val = [s for s in samples if s.get('town') in held]
        train = [s for s in samples if s.get('town') not in held]
    elif args.holdout_fraction is not None:
        if not 0 < args.holdout_fraction < 1:
            raise ValueError('--holdout-fraction must be in (0, 1)')
        order = list(range(len(samples)))
        random.Random(args.holdout_seed).shuffle(order)
        n_val = max(1, int(round(len(samples) * args.holdout_fraction)))
        val_idx = set(order[:n_val])
        val = [s for i, s in enumerate(samples) if i in val_idx]
        train = [s for i, s in enumerate(samples) if i not in val_idx]
    else:
        return [(args.split or 'all', samples)]
    return [(train_name, train), (val_name, val)]


def out_path(out_dir, split, tag):
    tag = f'_{tag}' if tag else ''
    return os.path.join(out_dir, f'carla_map_infos_{split}{tag}.pkl')


def main():
    args = parse_args()
    samples, dropped, meta = convert_carla_tiles(
        data_root=args.data_root,
        split=args.split,
        classes=args.classes,
        map_classes=args.map_classes,
        pc_range=args.lidar_point_cloud_range,
        z_max=args.z_max,
        min_lidar_points=args.min_lidar_points,
        lidar_check=args.lidar_check,
        gt_frame=args.gt_frame,
        reference_dir=args.reference_dir)

    mmcv.mkdir_or_exist(args.out_dir)
    for split_name, split_samples in partition(samples, args):
        if not split_samples:
            raise ValueError(f'the {split_name!r} side of the holdout is '
                             'empty; check --holdout-towns/-fraction')
        counts = {c: sum(len(s['annotation'][c]) for s in split_samples)
                  for c in meta['map_classes']}
        path = out_path(args.out_dir, split_name, args.out_tag)
        mmcv.dump(dict(infos=split_samples,
                       metadata=dict(meta, split=split_name,
                                     source_split=args.split,
                                     data_root=os.path.abspath(
                                         args.data_root))), path)
        print(f'wrote {path}: {len(split_samples)} tiles, '
              + ', '.join(f'{n} {c}' for c, n in counts.items()))

    if dropped:
        drop_path = out_path(args.out_dir, args.split,
                             args.out_tag).replace('.pkl', '_dropped.json')
        with open(drop_path, 'w', encoding='utf-8') as f:
            json.dump(dropped, f, indent=1)
        print(f'wrote {drop_path}: {len(dropped)} dropped tiles')


if __name__ == '__main__':
    main()
