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


import torch
from einops.layers.torch import Reduce, Rearrange

class AGCAM:
    """ Implementation of our method."""
    def __init__(self, model, attention_matrix_layer = 'before_softmax', attention_grad_layer = 'after_softmax', head_fusion='sum', layer_fusion='sum'):
        """
        Args:
            model (nn.Module): the Vision Transformer model to be explained
            attention_matrix_layer (str): the name of the layer to set a forward hook to get the self-attention matrices
            attention_grad_layer (str): the name of the layer to set a backward hook to get the gradients
            head_fusion (str): type of head-wise aggregation (default: 'sum')
            layer_fusion (str): type of layer-wise aggregation (default: 'sum')
        """
        self.model = model
        self.head = None
        self.width = None
        self.head_fusion = head_fusion
        self.layer_fusion = layer_fusion
        self.attn_matrix = []
        self.grad_attn = []

        for layer_num, (name, module) in enumerate(self.model.named_modules()):
            if attention_matrix_layer in name:
                module.register_forward_hook(self.get_attn_matrix)
            if attention_grad_layer in name:
                module.register_full_backward_hook(self.get_grad_attn)
                
    def get_attn_matrix(self, module, input, output):
        # As stated in Methodology part, in ViT with [class] token, only the first row of the attention matrix is directly connected with the MLP head.
        self.attn_matrix.append(output[:, :, 0:1, :]) # shape: [batch, num_heads, 1, num_patches] 
        


    def get_grad_attn(self, module, grad_input, grad_output):
        # As stated in Methodology part, in ViT with [class] token, only the first row of the attention matrix is directly connected with the MLP head.
        self.grad_attn.append(grad_output[0][:, :, 0:1, :]) # shape: [batch, num_heads, 1, num_patches] 
        
    
    def generate(self, input_tensor, cls_idx=None):
        self.attn_matrix = []
        self.grad_attn = []

        # backpropagate the model from the classification output
        self.model.zero_grad()
        output = self.model(input_tensor)
        _, prediction = torch.max(output, 1)
        self.prediction = prediction  
        if cls_idx==None:                               # generate CAM for a certain class label
            loss = output[0, prediction[0]]
        else:                                           # generate CAM for the predicted class
            loss = output[0, cls_idx]
        loss.backward()

        b, h, n, d = self.attn_matrix[0].shape
        self.head=h
        self.width = int((d-1)**0.5)

        # put all matrices from each layer into one tensor
        self.attn_matrix.reverse()
        attn = self.attn_matrix[0]
        gradient = self.grad_attn[0]
        for i in range(1, len(self.attn_matrix)):
            attn = torch.concat((attn, self.attn_matrix[i]), dim=0)
            gradient = torch.concat((gradient, self.grad_attn[i]), dim=0)

        # As stated in Methodology, only positive gradients are used to reflect the positive contributions of each patch.
        # The self-attention score matrices are normalized with sigmoid and combined with the gradients.
        gradient = torch.nn.functional.relu(gradient) # Here, the variable gradient is the gradients alpha^{k,c}_h in Equation 7 in the methodology part.
        attn = torch.sigmoid(attn) # Here, the variable attn is the attention score matrices newly normalized with sigmoid, which are eqaul to the feature maps F^k_h in Equation 2 in the methodology part.
        mask = gradient * attn

        # aggregation of CAM of all heads and all layers and reshape the final CAM.
        mask = mask[:, :, :, 1:].unsqueeze(0)
        mask = Reduce('b l h z p -> b l z p', reduction=self.head_fusion)(mask)
        mask = Reduce('b l z p -> b z p', reduction=self.layer_fusion)(mask)
        mask = Rearrange('b z (h w) -> b z h w', h=self.width, w=self.width)(mask)
        
        mask = mask.unsqueeze(0)[0]
        # Reshape the mask to have the same size with the original input image (224 x 224)
        upsample = torch.nn.Upsample(224, mode = 'bilinear', align_corners=False)
        mask = upsample(mask)

        # Normalize the heatmap from 0 to 1
        mask = (mask-mask.min())/(mask.max()-mask.min())

        # mask = mask.detach().cpu().numpy()[0]
        mask = mask[0][0]

        return prediction, mask

    def __call__(self, x, class_idx=None):
        # Check that we get only one image
        assert x.dim() == 3 or (x.dim() == 4 and x.shape[0] == 1), "Only one image can be processed at a time"

        # Unsqueeze to get 4 dimensions if needed
        if x.dim() == 3:
            x = x.unsqueeze(dim=0)

        with torch.enable_grad():
            prediction, ours_heatmap = self.generate(x, class_idx)
        
        return ours_heatmap


# Use Hydra to allow easy configuration swap for comparison of methods
@hydra.main(version_base="1.3", config_path="config", config_name="generate")
def main(cfg: DictConfig):

    seed_everything(cfg.seed)

    # Get model
    print("Loading model:", cfg.model.name, end="\n\n")
    if cfg.method.name == 'agc' or cfg.method.name == 'better_agc':
        MODEL = 'vit_base_patch16_224'
        class_num = 1000
        state_dict = model_zoo.load_url('https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth', progress=True, map_location='cuda')

        # explainer = RISE(model, (224, 224))
        model = ViT_Ours.create_model(MODEL, pretrained=True, num_classes=class_num).to('cuda')
        model.load_state_dict(state_dict, strict=True)
        model = model.eval()
    else:
        model = instantiate(cfg.model.init).cuda()
        model.eval()

    # Get method
    print("Initializing saliency method:", cfg.method.name, end="\n\n")
    if cfg.method.name == 'agc':
        method = AGCAM(model)
    else:
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
        # if count < 4000:
        #     continue
        if count >= 1000:
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
