import torch


def rollout(attentions, discard_ratio, head_fusion, device='cuda'):
    result = torch.eye(attentions[0].size(-1)).to(device)
    with torch.no_grad():
        for attention in attentions: 
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1) #[1, 38809] = [1, (197x197)]
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            
            I = torch.eye(attention_heads_fused.size(-1)).to(device)
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)

    mask = result[0, 0 , 1 :]    
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width)

    # modified to have the same shape as the result of other methods. 
    mask = mask.unsqueeze(0)
    mask = mask.unsqueeze(0)
    return mask


class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean", discard_ratio=0.0, device='cpu'):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        # print('[DEBUG] Initialization')
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                # print('[DEBUG] Foudn attention layer')
                module.register_forward_hook(self.get_attention) 

        self.attentions = []
        self.device=device

    def get_attention(self, module, input, output):
        self.attentions.append(output)

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)
        _, prediction = torch.max(output, 1)


        return prediction, rollout(self.attentions, self.discard_ratio, self.head_fusion)
    
    def generate(self, input, label=None):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input)
        _, prediction = torch.max(output, 1)
        # print('[DEBUG] Attention shape', len(self.attentions))

        return prediction, rollout(self.attentions, self.discard_ratio, self.head_fusion, device=self.device)

class RolloutWrapper():
    """
    Wrapper for Attention Rollout
    """
    def __init__(self, model, discard_ratio=0.9, head_fusion='mean', **kwargs):
        """
        initialisation of the class
        :param model: model used for the maps computations
        :param n_masks: number of masks used
        :param input_size: input size in pixels
        :param batch_size: batch size for the perturbations
        """

        self.method = VITAttentionRollout(model, discard_ratio=discard_ratio, head_fusion=head_fusion)

    def __call__(self, x, class_idx=None):
        """
        Call the saliency method
        :param x: input image tensor
        :param class_idx: index of the class to explain
        :return: a saliency map in shape (input_size, input_size)
        """
        _, saliency_map = self.method(x)
        return torch.tensor(saliency_map)
