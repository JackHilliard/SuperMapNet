import torch
import numpy as np
import torch.nn as nn
from mapmaster.models.lidar_encoder.pointpillar import PointPillarEncoder
import icecream as ic



class LiDARPointPillarEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(LiDARPointPillarEncoder, self).__init__()
        self.pp = PointPillarEncoder( **kwargs)  

    def forward(self, inputs):
        lidar_data = inputs["lidars"]
        lidar_mask = inputs["lidar_mask"]
        neck_feature = self.pp(lidar_data, lidar_mask)
        #ic(neck_feature.size())
        
        return {
            "lidar_enc_features": list(neck_feature),
        }


class LiDARPointPillarBEVEncoder(nn.Module):
    """PointPillars as the *only* BEV producer, for LiDAR-only datasets.

    Same encoder as ``LiDARPointPillarEncoder``, but it publishes its output
    under ``bev_enc_features`` -- the key ``ins_decoder`` and the semantic
    branch of ``output_head`` read -- so it can stand where the camera
    ``bev_decoder`` normally stands. Used by ``LiDARMapMaster`` for the
    CARLA tiles, which ship no imagery.

    The list is the decoder-level axis, exactly as ``TransformerBEVDecoder``
    uses it (one entry per intermediate decoder layer): a single-level BEV
    is a list of one. Note this deliberately does NOT do what
    ``LiDARPointPillarEncoder`` does -- ``list(neck_feature)`` there splits
    the BATCH, not the levels, which only works for the fusion path's
    ``[-1]`` at batch size 1.

    Output is ``(B, C, X, Y)``, X and Y being the pillar grid downsampled 8x
    by the encoder's three stride-2 blocks. That axis order matches the
    rasterized GT masks, which are ``(class, X, Y)`` as well.
    """

    def __init__(self, **kwargs):
        super(LiDARPointPillarBEVEncoder, self).__init__()
        self.pp = PointPillarEncoder(**kwargs)

    def forward(self, inputs):
        bev_feature = self.pp(inputs["lidars"], inputs["lidar_mask"])
        # Into the GT's pixel frame. The pillar grid indexes BEV[.., i, j]
        # with i increasing along +x and j along +y (points_to_voxels ravels
        # (x, y, z) in that order, and PointPillarEncoder reshapes to
        # (B, C, grid_x, grid_y)), but RasterizedLocalMap emits masks in
        # which i decreases along +x and j along +y -- it rot90s and flips
        # its canvas on the way out. Verified by rasterizing single lines at
        # known metres: a line at (x=-14, y=-14) on a 200x200 / 30 m canvas
        # lands at (i=179, j=193), not the naive (20, 7).
        #
        # So the two are 180 degrees apart. Unflipped, every mask target
        # asks the heads about the opposite corner of the tile, and nothing
        # would look broken: the losses still fall, and a 4-tile overfit
        # control reached the same loss either way (the pillar encoder's
        # conv stack has a wide enough receptive field to learn the rotation
        # when it is memorising four tiles). This is therefore a correctness
        # fix argued from geometry -- BEV cell and GT pixel agree to 0.17 m,
        # one pixel, in all four corners with the flip -- and NOT one with a
        # measured accuracy difference behind it.
        bev_feature = torch.flip(bev_feature, dims=(-2, -1))
        return {"bev_enc_features": [bev_feature]}
