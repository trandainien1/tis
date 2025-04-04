import torch
from einops.layers.torch import Reduce, Rearrange
import torchvision.transforms as transforms
import numpy as np
import timm 
import torch.nn.functional as F

# update date: 17/3
# !default values is the best config to achieve the performance in the paper
class ScoreAGC:
    def __init__(self, model, attention_matrix_layer = 'before_softmax', attention_grad_layer = 'after_softmax', head_fusion='sum', layer_fusion='sum', 
                 normalize_cam_heads=True, 
                 score_normalization_formula="minmax",
                 add_noise=True, 
                 plus=0, 
                 vitcx_score_formula=False, 
                 is_head_fuse=False,
                 is_binarize_cam_of_heads=False,
                 handle_pixel_coverage_bias=False,
                 score_formula='increase_in_confidence',
                 ):
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
        self.normalize_cam_heads = normalize_cam_heads
        self.score_normalization_formula = score_normalization_formula
        self.add_noise = add_noise
        self.plus = plus
        self.vitcx_score_formula = vitcx_score_formula
        self.is_head_fuse = is_head_fuse
        self.is_binarize_cam_of_heads = is_binarize_cam_of_heads
        self.handle_pixel_coverage_bias = handle_pixel_coverage_bias
        self.score_formula = score_formula # ['softmax_logit', 'increase_in_confidence'] 

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


    def generate_cams_of_heads(self, input_tensor, cls_idx=None):
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
        # b, h, n, d = self.attn_matrix.shape
        self.head=h
        self.width = int((d-1)**0.5)

        # put all matrices from each layer into one tensor
        self.attn_matrix.reverse()
        attn = self.attn_matrix[0]
        # attn = self.attn_matrix
        gradient = self.grad_attn[0]
        # gradient = self.grad_attn
        # layer_index = 2
        for i in range(1, len(self.attn_matrix)):
        # for i in range(layer_index, layer_index+1):
            # print('hia')
            attn = torch.concat((attn, self.attn_matrix[i]), dim=0)
            gradient = torch.concat((gradient, self.grad_attn[i]), dim=0)

        # As stated in Methodology, only positive gradients are used to reflect the positive contributions of each patch.
        # The self-attention score matrices are normalized with sigmoid and combined with the gradients.
        gradient = torch.nn.functional.relu(gradient) # Here, the variable gradient is the gradients alpha^{k,c}_h in Equation 7 in the methodology part.
        attn = torch.sigmoid(attn) # Here, the variable attn is the attention score matrices newly normalized with sigmoid, which are eqaul to the feature maps F^k_h in Equation 2 in the methodology part.
        mask = gradient * attn

        # aggregation of CAM of all heads and all layers and reshape the final CAM.
        mask = mask[:, :, :, 1:].unsqueeze(0) # * niên: chỗ này thêm 1 ở đầu (ví dụ: (2) -> (1, 2)) và 1: là bỏ token class
        # print(mask.shape)

        # *Niên:Thay vì tính tổng theo blocks và theo head như công thức để ra 1 mask cuối cùng là CAM thì niên sẽ giữ lại tất cả các mask của các head ở mỗi block
        if self.is_head_fuse:
            mask = Reduce('b l h z p -> b l z p', reduction=self.head_fusion)(mask)
            mask = Rearrange('b l z (h w)  -> b l z h w', h=self.width, w=self.width)(mask)
        else:
            mask = Rearrange('b l hd z (h w)  -> b l hd z h w', h=self.width, w=self.width)(mask) # *Niên: chỗ này tách từng token (1, 196) thành từng patch (1, 14, 14)

        return prediction, mask, output

    def binarize_head_cams(self, head_cams):
        head_cams = torch.squeeze(head_cams) # (1, 12, 12,1, 14, 14) -> (12, 12, 14, 14)
        cam_1_indice_list = []
        bin_cam_list = []
        for i in range(head_cams.shape[0]):
            for j in range(head_cams.shape[1]):
                cam = head_cams[i][j]
                # Computer the number of tokens to keep based on the ratio
                n_tokens = int(0.5 * cam.flatten().shape[0]) # 196 // 2 = 98 tokens

                # Compute the indexes of the n_tokens with the highest values in the raw mask
                cam_1_indices = cam.flatten().topk(n_tokens)[1]  # indices where value will be 1

                # Create binary mask
                bin_cam_flatten = torch.zeros_like(cam.flatten())
                bin_cam_flatten[cam_1_indices] = 1

                # Append current mask to lists
                cam_1_indice_list.append(cam_1_indices)
                bin_cam_list.append(bin_cam_flatten.reshape(14, 14))
        return bin_cam_list 


    def generate_scores(self, head_cams, prediction, output_truth, image):
        with torch.no_grad():
            tensor_heatmaps = head_cams[0]
            

            if self.is_head_fuse:
                tensor_heatmaps = tensor_heatmaps.reshape(12, 1, 14, 14)
            else:
                tensor_heatmaps = tensor_heatmaps.reshape(144, 1, 14, 14)
            tensor_heatmaps = transforms.Resize((224, 224))(tensor_heatmaps)        

            if self.is_binarize_cam_of_heads:
                heatmaps_list = self.binarize_head_cams(head_cams)
                tensor_heatmaps = torch.stack(heatmaps_list).unsqueeze(1)
                
                tensor_heatmaps = transforms.Resize((224, 224))(tensor_heatmaps)   
                # tensor_heatmaps = F.interpolate(tensor_heatmaps, size=(image.shape[2], image.shape[3]), mode='nearest')

                      
            elif self.normalize_cam_heads:
                # Compute min and max along each image
                min_vals = tensor_heatmaps.amin(dim=(2, 3), keepdim=True)  # Min across width and height
                max_vals = tensor_heatmaps.amax(dim=(2, 3), keepdim=True)  # Max across width and height
                # Normalize using min-max scaling
                tensor_heatmaps = (tensor_heatmaps - min_vals) / (max_vals - min_vals + 1e-7)  # Add small value to avoid division by zero
            
            if self.add_noise:
                # -------------- add noise ------------
                N  = tensor_heatmaps.shape[0]
                H = tensor_heatmaps.shape[2]
                W = tensor_heatmaps.shape[3]
                # Generate the inverse of masks, i.e., 1-M_i
                masks_inverse=torch.from_numpy(np.repeat((1-tensor_heatmaps.cpu().numpy())[:, :, np.newaxis,:], 3, axis=2)).cuda()
                masks_inverse = masks_inverse.squeeze(1)

                random_whole=torch.randn([N]+list((3,H,W))).cuda()* 0.1
                noise_to_add = random_whole * masks_inverse

                mask = torch.mul(tensor_heatmaps, image)
                m = mask + noise_to_add
            else:
                m = torch.mul(tensor_heatmaps, image)

            with torch.no_grad():
                output_mask = self.model(m)

            if self.vitcx_score_formula:
                p_mask_with_noise = output_mask[:, prediction.item()]
                p_x_with_noise = self.model(image + noise_to_add)[0, prediction.item()]
                class_p = output_truth[0, prediction.item()]
                agc_scores = p_mask_with_noise - p_x_with_noise + class_p
            else:
                if  self.score_formula == 'softmax_logit':
                    softmax_tensor = F.softmax(output_mask, dim=1)
                    agc_scores = softmax_tensor[:, prediction.item()]

                elif self.score_formula == 'increase_in_confidence':
                    # increase in confidence
                    agc_scores = output_mask[:, prediction.item()] - output_truth[0, prediction.item()]
            
            if self.score_normalization_formula == 'minmax':   
                print('min max ne')
                agc_scores = (agc_scores - agc_scores.min() ) / (agc_scores.max() - agc_scores.min())
            elif self.score_normalization_formula == 'sigmoid':   
                print('sigmoid ne')
                agc_scores = torch.sigmoid(agc_scores)
            elif self.score_normalization_formula == 'softmax':   
                print('softmax ne')
                agc_scores = torch.sigmoid(agc_scores)
                agc_scores = torch.softmax(agc_scores, dim=0)
            
            agc_scores += self.plus

            agc_scores = agc_scores.reshape(head_cams[0].shape[0], head_cams[0].shape[1])

            del output_mask  # Delete unnecessary variables that are no longer needed
            torch.cuda.empty_cache()  # Clean up cache if necessary
            
            return agc_scores

    def generate_saliency(self, head_cams, agc_scores):
        if self.is_head_fuse:
            mask = (agc_scores.view(1, 12, 1, 1, 1) * head_cams[0]).sum(axis=(0, 1))
        else:
            mask = (agc_scores.view(12, 12, 1, 1, 1) * head_cams[0]).sum(axis=(0, 1))
            # mask = (agc_scores.view(12, 12, 1, 1, 1) * head_cams[0]) # use for getting head of cams for score-agc


        if self.handle_pixel_coverage_bias:
            raw_sal = head_cams.squeeze().sum(axis=(0, 1)).unsqueeze(0) # (1, 14, 14)
            mask /= raw_sal

        mask = mask.squeeze()
        return mask

    def generate(self, x, class_idx=None):
        return self(x, class_idx)
        
    def __call__(self, x, class_idx=None):

        # Check that we get only one image
        assert x.dim() == 3 or (x.dim() == 4 and x.shape[0] == 1), "Only one image can be processed at a time"

        # Unsqueeze to get 4 dimensions if needed
        if x.dim() == 3:
            x = x.unsqueeze(dim=0)

        with torch.enable_grad():
            # * Head cam shape: (1, 12, 12, 1, 14, 14) - 12 layers - 12 heads - 1 saliency of shape 14x14 = 196 tokens
            predicted_class, head_cams, output_truth = self.generate_cams_of_heads(x)

        # print("After generate cams: ")
        # print(torch.cuda.memory_allocated()/1024**2)
        # print()
        
        # Define the class to explain. If not explicit, use the class predicted by the model
     
        if class_idx is None:
            class_idx = predicted_class
            # print("class idx", class_idx)

        # Generate the saliency map for image x and class_idx
        scores = self.generate_scores(
            image=x,
            head_cams=head_cams,
            prediction=predicted_class, output_truth=output_truth
        )
        # print("After generate scores: ")
        # print(torch.cuda.memory_allocated()/1024**2)
        # print()
        
        saliency_map = self.generate_saliency(head_cams=head_cams, agc_scores=scores)
        # print("After generate saliency maps: ")
        # print(torch.cuda.memory_allocated()/1024**2)
        # print()

        return saliency_map

    def get_scores(self, x, class_idx=None):
        
        assert x.dim() == 3 or (x.dim() == 4 and x.shape[0] == 1), "Only one image can be processed at a time"

        
        if x.dim() == 3:
            x = x.unsqueeze(dim=0)

        with torch.enable_grad():
            # * Head cam shape: (1, 12, 12, 1, 14, 14) - 12 layers - 12 heads - 1 saliency of shape 14x14 = 196 tokens
            predicted_class, head_cams, output_truth = self.generate_cams_of_heads(x)
        if class_idx is None:
            class_idx = predicted_class
            
        scores = self.generate_scores(
            image=x,
            head_cams=head_cams,
            prediction=predicted_class, output_truth=output_truth
        )

        return scores
    
    def _get_head_cams_only(self, x, class_idx=None):
        assert x.dim() == 3 or (x.dim() == 4 and x.shape[0] == 1), "Only one image can be processed at a time"
        
        if x.dim() == 3:
            x = x.unsqueeze(dim=0)

        with torch.enable_grad():
            # * Head cam shape: (1, 12, 12, 1, 14, 14) - 12 layers - 12 heads - 1 saliency of shape 14x14 = 196 tokens
            predicted_class, head_cams, output_truth = self.generate_cams_of_heads(x)
        if class_idx is None:
            class_idx = predicted_class
        
        return head_cams
     