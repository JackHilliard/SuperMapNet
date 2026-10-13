import torch.nn as nn
from mapmaster.models import backbone, bev_decoder, ins_decoder, output_head, lidar_encoder,fusionbev_encoder, cross_encoder
import os


#os.environ['TORCH_DISTRIBUTED_DEBUG'] = "INFO"
#warnings.filterwarnings('ignore')


class MapMaster(nn.Module):
    def __init__(self, model_config, *args, **kwargs):
        super(MapMaster, self).__init__()
        self.im_backbone = self.create_backbone(**model_config["im_backbone"])
        self.bev_decoder = self.create_bev_decoder(**model_config["bev_decoder"])
       
        #self.lidar_encoder = self.create_lidar_encoder(**model_config["lidar_encoder"])
        #self.cross_encoder = self.create_cross_encoder(**model_config["cross_encoder"])
        #self.fusion_encoder = self.create_fusion_encoder(**model_config["fusion_encoder"])
            
        self.ins_decoder = self.create_ins_decoder(**model_config["ins_decoder"])
        self.output_head = self.create_output_head(**model_config["output_head"])
        self.post_processor = self.create_post_processor(**model_config["post_processor"])

    def forward(self, inputs):
        outputs = {}
        outputs.update({k: inputs[k] for k in ["images", "extra_infos"]})
        outputs.update({k: inputs[k] for k in ["lidars", "lidar_mask"]})
        outputs.update({k: inputs[k].float() for k in ["extrinsic", "intrinsic"]})
        
        if "ida_mats" in inputs:
            outputs.update({"ida_mats": inputs["ida_mats"].float()})
        outputs.update(self.im_backbone(outputs))
        outputs.update(self.bev_decoder(outputs))
        #outputs.update(self.lidar_encoder(outputs))
        #outputs.update(self.cross_encoder(outputs))
        #outputs.update(self.fusion_encoder(outputs))
        outputs.update(self.ins_decoder(outputs))
        outputs.update(self.output_head(outputs))
        return outputs

    @staticmethod
    def create_backbone(arch_name, ret_layers, bkb_kwargs, fpn_kwargs, up_shape=None):
        __factory_dict__ = {
            "resnet": backbone.ResNetBackbone,
            "efficient_net": backbone.EfficientNetBackbone,
            "swin_transformer": backbone.SwinTRBackbone,
            "sam2_feature":backbone.SAM2Backbone,
        }
        return __factory_dict__[arch_name](bkb_kwargs, fpn_kwargs, up_shape, ret_layers)

    @staticmethod
    def create_bev_decoder(arch_name, net_kwargs):
        __factory_dict__ = {
            "transformer": bev_decoder.TransformerBEVDecoder,
            "ipm_deformable_transformer": bev_decoder.DeformTransformerBEVEncoder,
        }
        return __factory_dict__[arch_name](**net_kwargs)
    @staticmethod
    def create_lidar_encoder(arch_name, net_kwargs):
        __factory_dict__ = {
            "pointpillar_encoder": lidar_encoder.LiDARPointPillarEncoder,
        }
        return __factory_dict__[arch_name](**net_kwargs)
    @staticmethod
    def create_cross_encoder(arch_name, net_kwargs):
        __factory_dict__ = {
            "CrossEncoder": cross_encoder.CrossEncoder,
        }
        return __factory_dict__[arch_name](**net_kwargs)
    @staticmethod
    def create_fusion_encoder(arch_name, net_kwargs):
        __factory_dict__ = {
            "BevFusionEncoder": fusionbev_encoder.BevFusionEncoder,
            "ConcatBEV":fusionbev_encoder.ConcatBEV,
        }
        return __factory_dict__[arch_name](**net_kwargs)

    @staticmethod
    def create_ins_decoder(arch_name, net_kwargs):
        __factory_dict__ = {
            "mask2former": ins_decoder.Mask2formerINSDecoder,
            "line_aware_decoder": ins_decoder.PointMask2formerINSDecoder,
            "point_element_decoder":ins_decoder.PointElementMask2formerINSDecoder
        }

        return __factory_dict__[arch_name](**net_kwargs)

    @staticmethod
    def create_output_head(arch_name, net_kwargs):
        __factory_dict__ = {
            "bezier_output_head": output_head.PiecewiseBezierMapOutputHead,
            "pivot_point_predictor": output_head.PivotMapOutputHead,
        }
        return __factory_dict__[arch_name](**net_kwargs)

    @staticmethod
    def create_post_processor(arch_name, net_kwargs):
        __factory_dict__ = {
            "bezier_post_processor": output_head.PiecewiseBezierMapPostProcessor,
            "pivot_post_processor": output_head.PivotMapPostProcessor,
        }
        return __factory_dict__[arch_name](**net_kwargs)


class LiDARMapMaster(MapMaster):
    """MapMaster with the camera branch removed.

    ``MapMaster`` always builds an image backbone and an image-to-BEV
    decoder, because every nuScenes/Argoverse config has six cameras. The
    CARLA tile export has none -- it is a point cloud plus reference
    polylines -- so the BEV that ``ins_decoder`` and ``output_head`` consume
    has to come from the LiDAR encoder instead. That is the whole difference:
    the instance decoder, the pivot head and the post-processor are the same
    modules, configured the same way, and the loss never learns which branch
    filled ``bev_enc_features``.

    ``model_config`` therefore takes a ``lidar_encoder`` (whose arch must
    publish ``bev_enc_features`` -- i.e. ``pointpillar_bev_encoder``) and no
    ``im_backbone``/``bev_decoder``. Subclassing rather than making those
    optional on ``MapMaster`` keeps the existing configs untouched: they all
    *declare* a ``lidar_encoder`` in ``model_setup`` while leaving it unbuilt,
    so "is the key present" cannot be the switch.
    """

    def __init__(self, model_config, *args, **kwargs):
        # Deliberately skips MapMaster.__init__ -- it would build the camera
        # branch this class exists to omit.
        nn.Module.__init__(self)
        for absent in ("im_backbone", "bev_decoder"):
            if absent in model_config:
                raise ValueError(
                    f"LiDARMapMaster builds no camera branch, but model_setup "
                    f"declares {absent!r}; use MapMaster for a config with "
                    "cameras.")
        self.lidar_encoder = self.create_lidar_encoder(**model_config["lidar_encoder"])
        self.ins_decoder = self.create_ins_decoder(**model_config["ins_decoder"])
        self.output_head = self.create_output_head(**model_config["output_head"])
        self.post_processor = self.create_post_processor(**model_config["post_processor"])

    def forward(self, inputs):
        outputs = {}
        outputs.update({k: inputs[k] for k in ["extra_infos"]})
        outputs.update({k: inputs[k] for k in ["lidars", "lidar_mask"]})
        outputs.update(self.lidar_encoder(outputs))
        outputs.update(self.ins_decoder(outputs))
        outputs.update(self.output_head(outputs))
        return outputs

    @staticmethod
    def create_lidar_encoder(arch_name, net_kwargs):
        __factory_dict__ = {
            "pointpillar_encoder": lidar_encoder.LiDARPointPillarEncoder,
            "pointpillar_bev_encoder": lidar_encoder.LiDARPointPillarBEVEncoder,
        }
        return __factory_dict__[arch_name](**net_kwargs)
