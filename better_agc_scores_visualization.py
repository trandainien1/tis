import hydra
import timm
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import torch
from torch.nn.functional import interpolate
import numpy as np
import random
import os

from datasets.ilsvrc2012 import classes

from PIL import Image

from matplotlib import pyplot as plt

import Methods.AGCAM.ViT_for_AGCAM as ViT_Ours
import torch.utils.model_zoo as model_zoo

import pandas as pd
import seaborn as sns

# Define a function to seed everything
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def overlay(image, saliency, alpha=0.7, output_file=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    image = image.permute(1, 2, 0)
    saliency = interpolate(saliency.reshape((1, 1, *saliency.shape)), size=image.shape[:2], mode='bilinear')
    saliency = saliency.squeeze()
    ax[0].imshow(image)
    ax[1].imshow(image)
    ax[1].imshow(saliency, alpha=alpha, cmap='jet')
    if output_file:
        # If 'output_file' path is absolute
        if os.path.isabs(output_file):
            print(f"Saving the output image under {output_file}")
        # Else print the current working directory path + 'output_file'
        else: 
            print(f"Saving the output image under {os.getcwd()}/{output_file}")
        # Save the explanation
        plt.savefig(output_file)
    else:
        plt.show()


@hydra.main(version_base="1.3", config_path="config", config_name="example")
def main(cfg: DictConfig):

    seed_everything(cfg.seed)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Get model
    print("Loading model:", cfg.model.name, end="\n\n")
    if cfg.method.name == 'agc' or cfg.method.name == 'better_agc':
        MODEL = 'vit_base_patch16_224'
        class_num = 1000
        # state_dict = model_zoo.load_url('https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth', progress=True, map_location='cuda')


        # explainer = RISE(model, (224, 224))
        timm_model = timm.create_model(MODEL, pretrained=True, num_classes=class_num).to(device)

        state_dict = timm_model.state_dict()

        model = ViT_Ours.create_model(MODEL, pretrained=True, num_classes=class_num).to('cuda')
        model.load_state_dict(state_dict, strict=True)
        model = model.eval()
    else:
        model = instantiate(cfg.model.init).cuda()
        model.eval()
    

    # Get method
    print("Initializing saliency method:", cfg.method.name, end="\n\n")
    method = instantiate(cfg.method.init, model)

    # Get transformations
    print("Setting transformations", end="\n\n")
    transform = instantiate(cfg.transform)

    # Get image
    print("Opening image:", cfg.input_file, end="\n\n")
    image_raw = Image.open(cfg.input_file).convert('RGB')
    image = transform(image_raw).to(device).unsqueeze(0)

    if not cfg.class_idx:
        class_idx = torch.argmax(model(image), dim=-1)[-1]
    else:
        class_idx = cfg.class_idx

    # Computing saliency map
    print("Computing saliency map using", cfg.method.name, "for class", classes[class_idx])
   
    saliency_map, scores = method(image, class_idx=class_idx)
    
    print('[SCORES SHAPE]')
    print(scores.shape)

    # print(f'shape of saliency map of {cfg.method.name}: ', saliency_map.shape)
    # image = image - image.min()
    # image = image/image.max()
    # overlay(image.squeeze(0).cpu(), saliency_map, output_file=cfg.output_file)
    df = pd.DataFrame(scores.numpy())
    print(df)

    # Plot the heatmap
    plt.figure(figsize=(8, 6))  # Optional: Adjust the figure size
    sns.heatmap(df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Heatmap of 12x12 Tensor")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.show()
    plt.savefig('scores heatmap')

if __name__ == "__main__":
    main()
