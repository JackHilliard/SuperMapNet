"""SuperMapNet/PivotNet on the CARLA 30 m tile export -- LiDAR only.

The CARLA export ships a point cloud and reference polylines per tile and
NO imagery, so this config drives ``LiDARMapMaster``: the PointPillars
encoder produces the BEV that the camera branch produces on nuScenes, and
the instance decoder, pivot head and losses are unchanged. See
``mapmaster/dataset/carla_dataset.py`` for the data side and
``tools/anno_converter/carla/convert.py`` for the annotation pkl.

Geometry, all of it derived from the export's own tile size (``tile_side``
30.0, i.e. ``tile_radius`` 15) rather than copied from nuScenes:

    map_region      (-15, 15, -15, 15) m      square, unlike nuScenes' 120x30
    map_resolution  0.15 m/px                 as nuScenes
    map_size        (200, 200) px             = region / resolution
    pillar grid     200 x 200 -> BEV 25 x 25  three stride-2 blocks

The class taxonomy is the export's own (``driving`` = its driving
centrelines, ``curb`` = its curbs), in label order; it must match the
``--map-classes`` the pkl was converted with, and the dataset raises if it
does not.

Prepare the data (../carla_test ships a single `test` split of two towns,
so the train/val separation is carved out by town -- train on town13, hold
out town12; they share no geometry)::

    python tools/anno_converter/carla/convert.py \\
        --data-root ../carla_test --split test --gt-frame tile_center \\
        --map-classes driving curb --holdout-towns town12 \\
        --out-dir data/carla --out-tag 30m_tc_2cls

Train / evaluate::

    bash run.sh train supermapnet_carla_lidar_30m_2cls 24
    bash run.sh test supermapnet_carla_lidar_30m_2cls <checkpoint.pth>
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torchvision.transforms import Compose
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data.distributed import DistributedSampler

from mapmaster.models.network import LiDARMapMaster
from mapmaster.engine.core import MapMasterCli
from mapmaster.engine.experiment import BaseExp
from mapmaster.dataset.carla_dataset import CarlaMapDataset
from mapmaster.dataset.transform import ToTensor_Pivot
from mapmaster.utils.misc import get_param_groups, is_distributed
from tools.evaluation.ap import instance_mask_ap as get_batch_ap


class EXPConfig:

    DATA_ROOT = './data/carla/'

    # Tile half-width in metres, from the export's manifest (`tile_side`
    # 30.0). Everything spatial below is derived from it, so a different
    # tile size needs this line changed and nothing else.
    tile_radius = 15.0
    map_resolution = 0.15

    map_conf = dict(
        dataset_name="carla",
        # Label order. Must equal the converter's --map-classes order: the
        # index IS the label, so a permutation relabels every instance.
        map_classes=["driving", "curb"],
        ego_size=(2 * tile_radius, 2 * tile_radius),
        map_region=(-tile_radius, tile_radius, -tile_radius, tile_radius),
        map_resolution=map_resolution,
        map_size=(int(2 * tile_radius / map_resolution),
                  int(2 * tile_radius / map_resolution)),
        # px, as nuScenes: the thickness the GT instance masks are drawn at
        # and the width the matcher re-rasterizes predictions to. 8 px is
        # 1.2 m here, comfortably below CARLA's ~3.5 m lane spacing.
        line_width=8,
        ann_file_train=os.path.join(DATA_ROOT,
                                    'carla_map_infos_train_30m_tc_2cls.pkl'),
        ann_file_val=os.path.join(DATA_ROOT,
                                  'carla_map_infos_val_30m_tc_2cls.pkl'),
        ann_file_test=os.path.join(DATA_ROOT,
                                   'carla_map_infos_val_30m_tc_2cls.pkl'),
    )

    # Pivot points kept per instance, per class. Both classes get the same
    # budget rather than nuScenes' lopsided (10, 2, 30): CARLA's reference
    # lines are one kind of thing, mostly straight inside a 30 m tile, and
    # 15 is what it takes for the simplifier below to never truncate one
    # (measured: max 13 pivots over 4250 real polylines, 0.00% past 15).
    pivot_conf = dict(
        max_pieces=(15, 15),
    )

    dataset_setup = dict(
        # Fixed per-sample point budget after grid sampling at the pillar
        # size. Measured over 40 random tiles: median 21k occupied 0.15 m
        # cells, max 37k -- so 40k trims nothing on this export while
        # bounding the batch for one that is denser.
        max_lidar_points=40000,
        lidar_voxel_size=map_resolution,
        # Matches the converter's own z cut, so the dataset keeps exactly
        # the points the pkl's in-range counts describe.
        z_max=96.0,
        # nuScenes' 2.0 is tuned for 120 m lines and flattens CARLA's arcs
        # (11.5% of polylines): measured over 751 of them, it leaves an arc
        # up to 0.68 m from its own geometry -- past the 0.5 m chamfer
        # threshold this is scored at. 0.2 caps that at 0.15 m, one pixel.
        pivot_vm_thre=0.2,
        use_colour=True,
    )

    model_setup = dict(
        # No im_backbone / bev_decoder: there are no cameras in this
        # dataset, and LiDARMapMaster rejects a config that declares them.
        lidar_encoder=dict(
            arch_name="pointpillar_bev_encoder",
            net_kwargs=dict(
                C=256,
                xbound=[-tile_radius, tile_radius, map_resolution],
                ybound=[-tile_radius, tile_radius, map_resolution],
                # ONE z bin, deliberately: the encoder reshapes its voxel
                # features to (B, C, grid_x, grid_y), which is only valid
                # when grid_z == 1 -- these are pillars, not voxels. The
                # span covers the whole export (measured z from -59 to +50
                # over 40 random tiles; CARLA towns have elevated highways),
                # because a point outside the range is masked out and lost.
                zbound=[-72.0, 96.0, 168.0],
                # [x, y, z, strength] -- CARLA blocks carry RGB where a real
                # LiDAR carries intensity; the dataset collapses it to one
                # luma channel.
                ppdim=4,
            )
        ),
        ins_decoder=dict(
            arch_name="line_aware_decoder",
            net_kwargs=dict(
                decoder_ids=[0, 1, 2, 3, 4, 5],
                in_channels=256,
                num_feature_levels=1,
                mask_classification=True,
                # binary fore/background per query group, as on nuScenes --
                # the MAP classes are the query split below, not this
                num_classes=1,
                hidden_dim=256,
                nheads=8,
                dim_feedforward=2048,
                dec_layers=6,
                pre_norm=False,
                mask_dim=256,
                enforce_input_project=False,
                # instance queries per class. This export averages 2.5
                # driving + 1.0 curb instances per tile, so 20 each is
                # generous headroom.
                query_split=(20, 20),
                max_pieces=pivot_conf["max_pieces"],
                dropout=0.0,
            ),
        ),
        output_head=dict(
            arch_name="pivot_point_predictor",
            net_kwargs=dict(
                in_channel=256,
                num_queries=[20, 20],
                tgt_shape=map_conf['map_size'],
                max_pieces=pivot_conf["max_pieces"],
                bev_channels=256,
                ins_channel=64,
            )
        ),
        post_processor=dict(
            arch_name="pivot_post_processor",
            net_kwargs=dict(
                criterion_conf=dict(
                    weight_dict=dict(
                        sem_msk_loss=3,
                        ins_obj_loss=2, ins_msk_loss=5,
                        pts_loss=30, collinear_pts_loss=10,
                        pt_logits_loss=2,
                    ),
                    decoder_weights=[0.4, 0.4, 0.4, 0.8, 1.2, 1.6]
                ),
                matcher_conf=dict(
                    cost_obj=2, cost_mask=5,
                    coe_endpts=5,
                    cost_pts=30,
                    mask_loss_conf=dict(
                        ce_weight=1,
                        dice_weight=1,
                    )
                ),
                pivot_conf=pivot_conf,
                map_conf=map_conf,
                # ONE weight, not nuScenes' four: the semantic branch is
                # applied to every entry of `bev_enc_features`, and the
                # LiDAR encoder publishes a single BEV level where the
                # camera transformer publishes one per decoder layer.
                # Indexing past the end of this list is what a copied
                # 4-entry weight list would do.
                sem_loss_conf=dict(
                    decoder_weights=[1.6],
                    mask_loss_conf=dict(ce_weight=1, dice_weight=1)),
                no_object_coe=0.5,
                collinear_pts_coe=0.2,
                coe_endpts=5,
            )
        )
    )

    optimizer_setup = dict(
        base_lr=2e-4,
        wd=1e-4,
        # No image backbone to hold back here, so `backb_names` matches
        # nothing and every parameter trains at base_lr; the keys stay so
        # `get_param_groups` sees the shape it expects.
        backb_names=["backbone"],
        backb_lr=5e-5,
        extra_names=[],
        extra_lr=5e-5,
        freeze_names=[],
    )

    scheduler_setup = dict(
        gamma=0.9,
    )

    metric_setup = dict(
        map_resolution=(map_resolution, map_resolution),
        iou_thicknesses=(1,),
        cd_thresholds=(0.2, 0.5, 1.0, 1.5, 5.0)
    )


import warnings

warnings.filterwarnings("ignore")


class Exp(BaseExp):
    def __init__(self, batch_size_per_device=1, total_devices=1, max_epoch=24,
                 **kwargs):
        super(Exp, self).__init__(batch_size_per_device, total_devices,
                                  max_epoch)

        self.exp_config = EXPConfig()
        self.data_loader_workers = 4
        self.print_interval = 50
        self.dump_interval = 1
        self.eval_interval = 1
        self.seed = 0
        self.num_keep_latest_ckpt = 1
        self.ckpt_oss_save_dir = None
        self.enable_tensorboard = True
        self.max_line_count = 100

        lr_ratio_dict = {32: 2, 16: 1.5, 8: 1, 4: 0.5, 2: 0.5, 1: 0.5}
        assert total_devices in lr_ratio_dict, "Please set normal devices!"
        for k in ['base_lr', 'backb_lr', 'extra_lr']:
            self.exp_config.optimizer_setup[k] = \
                self.exp_config.optimizer_setup[k] * lr_ratio_dict[total_devices]
        self.evaluation_save_dir = None

    def _configure_model(self):
        return LiDARMapMaster(self.exp_config.model_setup)

    def _build_dataset(self, ann_file, data_split):
        # ToTensor_Pivot alone: the Resize/Normalize transforms act on
        # `images`, and this dataset has none.
        return CarlaMapDataset(
            ann_file=ann_file,
            map_conf=self.exp_config.map_conf,
            point_conf=self.exp_config.pivot_conf,
            transforms=Compose([ToTensor_Pivot()]),
            data_split=data_split,
            **self.exp_config.dataset_setup)

    def _configure_train_dataloader(self):
        from mapmaster.dataset.sampler import InfiniteSampler

        train_set = self._build_dataset(
            self.exp_config.map_conf['ann_file_train'], "train")
        if is_distributed():
            sampler = InfiniteSampler(len(train_set),
                                      seed=self.seed if self.seed else 0)
        else:
            sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.batch_size_per_device,
            pin_memory=True,
            num_workers=self.data_loader_workers,
            shuffle=sampler is None,
            drop_last=True,
            sampler=sampler,
        )
        self.train_dataset_size = len(train_set)
        return train_loader

    def _configure_val_dataloader(self):
        val_set = self._build_dataset(
            self.exp_config.map_conf['ann_file_val'], "val")
        sampler = DistributedSampler(val_set, shuffle=False) \
            if is_distributed() else None
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=1,
            pin_memory=True,
            num_workers=self.data_loader_workers,
            shuffle=False,
            drop_last=False,
            sampler=sampler,
        )
        self.val_dataset_size = len(val_set)
        return val_loader

    def _configure_test_dataloader(self):
        test_set = self._build_dataset(
            self.exp_config.map_conf['ann_file_test'], "test")
        sampler = DistributedSampler(test_set, shuffle=False) \
            if is_distributed() else None
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=1,
            pin_memory=True,
            num_workers=self.data_loader_workers,
            shuffle=False,
            drop_last=False,
            sampler=sampler,
        )
        self.test_dataset_size = len(test_set)
        return test_loader

    def _configure_optimizer(self):
        return AdamW(get_param_groups(self.model,
                                      self.exp_config.optimizer_setup))

    def _configure_lr_scheduler(self):
        return ExponentialLR(
            optimizer=self.optimizer,
            gamma=self.exp_config.scheduler_setup["gamma"],
            last_epoch=-1,
        )

    def training_step(self, batch):
        batch["lidars"] = batch["lidars"].float().cuda()
        batch["lidar_mask"] = batch["lidar_mask"].float().cuda()
        outputs = self.model(batch)
        return self.model.module.post_processor(outputs["outputs"],
                                                batch["targets"])

    def test_step(self, batch, step, ap_matrix, ap_count_matrix):
        with torch.no_grad():
            batch["lidars"] = batch["lidars"].float().cuda()
            batch["lidar_mask"] = batch["lidar_mask"].float().cuda()

            outputs = self.model(batch)
            results, dt_masks = self.model.module.post_processor(
                outputs["outputs"])

            metric_setup = self.exp_config.metric_setup
            SAMPLED_RECALLS = torch.linspace(0.1, 1, 10).cuda()
            THRESHOLDS = [0.2, 0.5, 1.0, 1.5]
            dt_masks = np.asarray(dt_masks)
            dt_scores = results[0]["confidence_level"]
            dt_scores = np.array(list(dt_scores)
                                 + [-1] * (self.max_line_count - len(dt_scores)))

            ap_matrix, ap_count_matrix = get_batch_ap(
                ap_matrix.cuda(),
                ap_count_matrix.cuda(),
                torch.from_numpy(dt_masks).cuda(),
                batch['targets']['masks'].cuda(),
                *metric_setup["map_resolution"],
                torch.from_numpy(np.array(dt_scores)).unsqueeze(0).cuda(),
                THRESHOLDS,
                SAMPLED_RECALLS,
            )
        return ap_matrix, ap_count_matrix

    def save_results(self, tokens, results, dt_masks):
        if self.evaluation_save_dir is None:
            self.evaluation_save_dir = os.path.join(self.output_dir,
                                                    "evaluation", "results")
            os.makedirs(self.evaluation_save_dir, exist_ok=True)
        for (token, dt_res, dt_mask) in zip(tokens, results, dt_masks):
            save_path = os.path.join(self.evaluation_save_dir, f"{token}.npz")
            np.savez_compressed(save_path, dt_mask=dt_mask, dt_res=dt_res)


if __name__ == "__main__":
    MapMasterCli(Exp).run()
