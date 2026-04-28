import torch
import cv2
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None

        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output.detach()   # detach here

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.eval()
        self.model.zero_grad()               # zero_grad BEFORE forward+backward

        output = self.model(input_tensor)

        # Handle multi-image batch (MVFF) — average logits
        if output.dim() == 2 and output.size(0) > 1:
            output = torch.mean(output, dim=0, keepdim=True)

        loss = output[0, class_idx]
        loss.backward()

        gradients  = self.gradients.cpu().numpy()[0]   # [C, H, W]
        activations = self.activations.cpu().numpy()[0] # [C, H, W]

        weights = np.mean(gradients, axis=(1, 2))      # [C]
        cam     = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam