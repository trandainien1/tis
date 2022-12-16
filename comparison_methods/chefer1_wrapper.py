import torch

import sys

sys.path.append("comparison_methods/chefer1")

from baselines.ViT.ViT_LRP import VisionTransformer, _conv_filter, _cfg
from baselines.ViT.helpers import load_pretrained
from baselines.ViT.ViT_explanation_generator import LRP

from timm.models.vision_transformer import default_cfgs as vit_cfgs
from timm.models.deit import default_cfgs as deit_cfgs


# Models

def vit_base_patch16_224(pretrained=False, model_name="vit_base_patch16_224", pretrained_cfg=None, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, **kwargs)
    cfg = _cfg(
        url=vit_cfgs[model_name].cfgs[pretrained_cfg].url,
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    )
    model.default_cfg = cfg
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model


def vit_large_patch16_224(pretrained=False, model_name="vit_large_patch16_224", pretrained_cfg=None, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True, **kwargs)
    cfg = _cfg(
        url=vit_cfgs[model_name].cfgs[pretrained_cfg].url,
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    )
    model.default_cfg = cfg
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


# Method computation

class Chefer1Wrapper():
    def __init__(self, model):
        # Check that model is a patched ViT
        assert isinstance(model, VisionTransformer), "Transformer architecture not recognised"

        self.model = model
        self.lrp = LRP(model)

    def __call__(self, x, class_idx=None):
        saliency_map = self.lrp.generate_LRP(x,  method="transformer_attribution", index=class_idx).detach()
        return saliency_map
