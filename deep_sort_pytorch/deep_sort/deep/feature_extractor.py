import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging

class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        
        # Simple feature extractor using basic CNN features
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        # Use a simple feature extraction method
        logging.warning("Using simple feature extractor as DeepSORT model not available")

    def _preprocess(self, im_crops):
        """
        Preprocessing for feature extraction
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            # Simple feature extraction using mean pooling over spatial dimensions
            features = torch.mean(im_batch, dim=[2, 3])  # Global average pooling
        return features.cpu().numpy()