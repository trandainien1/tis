import torch

import numpy as np

import hydra
from hydra.utils import instantiate

from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm

import random

import os

import PIL
import Methods.AGCAM.ViT_for_AGCAM as ViT_Ours
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import gc

gc.collect()
torch.cuda.empty_cache()

# Try to import lovely_tensors
try:
    import lovely_tensors as lt
    lt.monkey_patch()
except ModuleNotFoundError:
    # But not mandatory, pass if lovely tensor is not available
    pass



# Define a function to seed everything
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_directory_if_not_exists(filepath):
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

# Use Hydra to allow easy configuration swap for comparison of methods
@hydra.main(version_base="1.3", config_path="config", config_name="generate")
def main(cfg: DictConfig):

    seed_everything(cfg.seed)

    # Get model
    # print("Loading model:", cfg.model.name, end="\n\n")
    # model = instantiate(cfg.model.init).cuda()
    # model.eval()

    MODEL = 'vit_base_patch16_224'
    class_num = 1000
    state_dict = model_zoo.load_url('https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth', progress=True, map_location='cuda')

    # explainer = RISE(model, (224, 224))
    model = ViT_Ours.create_model(MODEL, pretrained=True, num_classes=class_num).to('cuda')
    model.load_state_dict(state_dict, strict=True)
    model = model.eval()

    # Get method
    print("Initializing saliency method:", cfg.method.name, end="\n\n")
    method = instantiate(cfg.method.init, model)

    # Get dataset
    print("Loading dataset", end="\n\n")
    dataset = instantiate(cfg.dataset)

    # Keep saliency maps in a list
    saliency_maps_list = []

    count = 0
    # Loop over the dataset to generate the saliency maps
    for image, class_idx in tqdm(dataset, desc="Computing saliency maps"):
        count += 1
        if count > 1000:
            break
        image = image.unsqueeze(0).cuda()

        if cfg.no_target:
            class_idx = None

        # Compute current saliency map
        cur_map = method(image, class_idx=class_idx).detach().cpu()

        # Add the current map to the list of saliency maps
        saliency_maps_list.append(cur_map)

        

    # Stack into a single tensor
    saliency_maps = torch.stack(saliency_maps_list)

    # Save as a npz
    output_npz = cfg.output_npz
    if cfg.no_target:
        output_npz += ".notarget"
    print("\nSaving saliency maps to file:", cfg.output_npz)
    create_directory_if_not_exists(output_npz)
    np.savez(cfg.output_npz, saliency_maps.cpu().numpy())


if __name__ == "__main__":
    main()
