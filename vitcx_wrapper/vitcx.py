# Wrappers classes for comparison in benchmarks
import torch
import sys
import copy
import os
import sys

from vitcx.cam import get_feature_map
from vitcx.causal_score import causal_score
import numpy as np
import cv2
import copy
# from skimage.transform import resize
# from sklearn.cluster import AgglomerativeClustering
# from scipy.special import softmax
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision.transforms import Compose, Normalize, ToTensor
cudnn.benchmark = True

print('DEBUG: Init file')

def reshape_function_vit(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def ViT_CX(model,image,target_layer,target_category=None,distance_threshold=0.1,reshape_function=reshape_function_vit,gpu_batch=50):
    image=image.cuda()
    model_softmax=copy.deepcopy(model)
    model=model.eval()
    model=model.cuda()
    model_softmax = nn.Sequential(model_softmax, nn.Softmax(dim=1))
    model_softmax = model_softmax.eval()
    model_softmax = model_softmax.cuda()
    for p in model_softmax.parameters():
        p.requires_grad = False
    y_hat = model_softmax(image)
    y_hat_1=y_hat.detach().cpu().numpy()[0]
    if target_category==None:
        top_1=np.argsort(y_hat_1)[::-1][0]
        target_category = top_1
    class_p=y_hat_1[target_category]
    input_size=(image.shape[2],image.shape[3])
    transform_fp = transforms.Compose([transforms.Resize(input_size)])


    # Extract the ViT feature maps 
    GetFeatureMap= get_feature_map(model=model,target_layers=[target_layer],use_cuda=True,reshape_transform=reshape_function)
    _ = GetFeatureMap(input_tensor=image,target_category=int(target_category))
    feature_map=GetFeatureMap.featuremap_and_grads.featuremaps[0][0].cuda()

    # Reshape and normalize the ViT feature maps to get ViT masks
    feature_map=transform_fp(feature_map)
    mask=norm_matrix(torch.reshape(feature_map, (feature_map.shape[0],input_size[0]*input_size[1])))


    # Compute the pairwise cosine similarity and distance of the ViT masks
    similarity = get_cos_similar_matrix(mask,mask)
    distance = 1 - similarity

    # Apply the  AgglomerativeClustering with a given distance_threshold
    cluster = AgglomerativeClustering(n_clusters = None, distance_threshold=distance_threshold,metric='precomputed', linkage='complete') 
    cluster.fit(distance.cpu())
    cluster_num=len(set(cluster.labels_))
    # print('number of masks after the clustering:'+str(cluster_num))

    # Use the sum of a clustering as a representation of the cluster
    cluster_labels=cluster.labels_
    cluster_labels_set=set(cluster_labels)
    mask_clustering=torch.zeros((len(cluster_labels_set),input_size[0]*input_size[1])).cuda()
    for i in range(len(mask)):
        mask_clustering[cluster_labels[i]]+=mask[i]

    # normalize the masks
    mask_clustering_norm=norm_matrix(mask_clustering).reshape((len(cluster_labels_set),input_size[0],input_size[1]))
    
    # compute the causal impact score
    compute_causal_score = causal_score(model_softmax, (input_size[0], input_size[1]),gpu_batch=gpu_batch)
    sal = compute_causal_score(image,mask_clustering_norm, class_p)[target_category].cpu().numpy()

    # return sal, mask_clustering_norm
    # print('[DEBUG]', target_category)
    return target_category, sal


print('DEBUG: vitcx imported successfully')

from torchvision.models import VisionTransformer as VisionVIT
from timm.models.vision_transformer import VisionTransformer as TimmVIT

print('SUCCESS: ViT-CX loaded successfully')

class ViTCXWrapper:
    """
    Wrapper for ViT-CX: Wrap ViT-CX method to allow similar usage in scripts
    """
    def __init__(self, model, batch_size=2, **kwargs):
        """
        initialisation of the class
        :param model: model used for the maps computations
        """
        print('DEBUG: init')
        self.model = model
        self.batch_size = batch_size
        

    def exec_method(self, x, class_idx=None):
        """
        Call the saliency method
        :param x: input image tensor
        :param class_idx: index of the class to explain
        :return: a saliency map in shape (input_size, input_size)
        """
        with torch.enable_grad():
            model = copy.deepcopy(self.model)
            if isinstance(model, VisionVIT):
                target_layer = model.encoder.layers[-1].ln_1
            elif isinstance(model, TimmVIT):
                target_layer = model.blocks[-1].norm1
            else:
                raise NotImplementedError("Model not supported")
        
            prediction, saliency = ViT_CX(model,
                                           x,
                                           target_layer,
                                           class_idx,
                                           reshape_function=reshape_function_vit,
                                           gpu_batch=self.batch_size,
                                           )
                                    
            saliency = torch.Tensor(saliency).detach()
            del model
            return prediction, saliency
    def generate(self, x, target=None):
        target = target.to('cpu')
        with torch.enable_grad():
            prediction, saliency_map = self.exec_method(x, class_idx=target)
            return prediction, saliency_map.detach().cpu()

    def __call__(self, x, class_idx=None):
      # class_idx = class_idx.to('cpu')
      with torch.enable_grad():
          prediction, saliency_map = self.exec_method(x, class_idx=class_idx)
          return saliency_map.detach().cpu()