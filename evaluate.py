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
    print("Loading model:", cfg.model.name, end="\n\n")
    model = instantiate(cfg.model.init).cuda()
    model.eval()

    if not cfg.metric.npz_only:
        # Get method
        print("Initializing saliency method:", cfg.method.name, end="\n\n")
        method = instantiate(cfg.method.init, model)

    # Get saliencies from npz
    print("Loading saliency maps from", cfg.input_npz, end="\n\n")
    saliency_maps = torch.tensor(np.load(cfg.input_npz)['arr_0'])

    # Get dataset
    print("Loading dataset", end="\n\n")
    dataset = instantiate(cfg.dataset)

    # Get metric
    metric = instantiate(cfg.metric.init, model)

    # Set resize transformation for the saliency maps if upsampling is required
    upsampling_fn = Resize(dataset[0][0].shape[-2:])

    assert len(dataset) == len(saliency_maps), "The saliency maps and the dataset don't have the same number of items"

    metric_scores = []

    # Loop over the dataset to generate the saliency maps
    for (image, class_idx), saliency_map in tqdm(zip(dataset, saliency_maps),
                                                 desc="Computing saliency maps",
                                                 total=len(dataset)):
        image = image.unsqueeze(0).cuda()
        saliency_map = saliency_map.reshape((1, 1, *saliency_map.shape))

        if saliency_map.shape != image.shape:
            saliency_map = upsampling_fn(saliency_map)

        score = metric(image, saliency_map, class_idx=class_idx)
        metric_scores.append(score)

    metric_scores = torch.stack(metric_scores).cpu().numpy()

    # Save as a csv
    csv_name = os.path.split(cfg.input_npz)[1].split(".npz")[0] + "_" + cfg.metric.name + ".csv"
    csv_path = os.path.join(cfg.output_csv_dir, csv_name)
    # Create dir if not exist
    os.makedirs(cfg.output_csv_dir, exist_ok=True)

    while os.path.exists(csv_path):
        print("WARNING: csv file already exists:", csv_path)
        csv_path += ".new"

    print("\nSaving saliency maps to file:", csv_path)
    pd.DataFrame(metric_scores).to_csv(csv_path)


if __name__ == "__main__":
    main()