import torch

import numpy as np

import hydra
from hydra.utils import instantiate

from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm

import random

from torchvision.transforms import Resize

import pandas as pd

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


# Use Hydra to allow easy configuration swap for comparison of methods
@hydra.main(version_base="1.3", config_path="config", config_name="evaluate")
def main(cfg: DictConfig):

    seed_everything(cfg.seed)

    # Get model
    # print("Loading model:", cfg.model.name, end="\n\n")
    # model = instantiate(cfg.model.init).cuda()
    # model.eval()

    MODEL = 'vit_base_patch16_224'
    class_num = 1000
    state_dict = model_zoo.load_url('https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth', progress=True, map_location='cuda')
    model = ViT_Ours.create_model(MODEL, pretrained=True, num_classes=class_num).to('cuda')
    model.load_state_dict(state_dict, strict=True)
    model = model.eval()

    # Get dataset
    print("Loading dataset", end="\n\n")
    dataset = instantiate(cfg.dataset)

    if cfg.metric.npz_only:
        # Get saliencies from npz

        input_npz = cfg.input_npz

        print("Loading saliency maps from", cfg.input_npz, end="\n\n")
        saliency_maps = torch.tensor(np.load(cfg.input_npz)['arr_0'])

        # Get metric
        metric = instantiate(cfg.metric.init, model)

        # Set resize transformation for the saliency maps if upsampling is required
        upsampling_fn = Resize(dataset[0][0].shape[-2:])
        # print(len(dataset), len(saliency_maps))

        # assert len(dataset) == len(
        #     saliency_maps), "The saliency maps and the dataset don't have the same number of items"

    else:
        # Get method
        print("Initializing saliency method:", cfg.method.name, end="\n\n")
        method = instantiate(cfg.method.init, model)

        # Get metric
        metric = instantiate(cfg.metric.init, model, method)

    metric_scores = []

    # Limit to a subset of the dataset if needed
    start_idx = cfg.start_idx
    if cfg.end_idx == -1:
        end_idx = len(dataset) - 1
    else:
        end_idx = cfg.end_idx

    # Loop over the dataset to generate the saliency maps

    for idx in tqdm(range(start_idx, end_idx+1),
                    desc="Computing metric",
                    total=(end_idx - start_idx)):
        (image, target) = dataset[idx]
        image = image.unsqueeze(0).cuda()
        if cfg.no_target:
            target = torch.argmax(model(image)).item()

        # ------------- Nien: store target --------
        # import json

        # # Save the list to a file in JSON format
        # with open("target.json", "w") as file:
        #     json.dump(target, file)

        # print("Numbers saved in JSON format.")
        # -----------------------------------------

        if cfg.metric.npz_only:
            saliency_map = saliency_maps[idx - cfg.start_idx] # !Nien: subtract number of images here
            saliency_map = saliency_map.reshape((1, 1, *saliency_map.shape))
            if saliency_map.shape != image.shape:
                saliency_map = upsampling_fn(saliency_map)

            # score, heatmaps = metric(image, saliency_map, target=target)
            score = metric(image, saliency_map, target=target)

        else:
            score = metric(image, target=target)

        metric_scores.append(score)

    metric_scores = torch.stack(metric_scores).cpu().numpy()

    # Save as a csv
    csv_name = os.path.split(cfg.input_npz)[1].split(".npz")[0] + "_" + cfg.metric.name
    if cfg.start_idx != 0 or cfg.end_idx != -1:
        csv_name += "_subset" + str(start_idx) + "-" + str(end_idx)
    csv_name += ".csv"
    csv_path = os.path.join(cfg.output_csv_dir, csv_name)
    # Create dir if not exist
    os.makedirs(cfg.output_csv_dir, exist_ok=True)

    while os.path.exists(csv_path):
        print("WARNING: csv file already exists:", csv_path)
        csv_path += ".new"

    print("\nSaving scores to file:", csv_path)
    pd.DataFrame(metric_scores).to_csv(csv_path, header=False, index=False)

    # output_npz = f'npz/{cfg.model.name}_{cfg.method.name}_masks.npz'
    # print("\nSaving saliency maps to file: ", output_npz)
    # np.savez(output_npz, heatmaps.cpu().numpy())

if __name__ == "__main__":
    main()
