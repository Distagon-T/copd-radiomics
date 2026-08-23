# -*- coding: utf-8 -*-


from networks.airway_network import UNet3D
from util.normalize_ct import normalize_CT, lumTrans
from monai.inferers import sliding_window_inference
from configs.airway_config import config
from util.utils import InnerTransformer

import torch
import numpy as np
import os


class AirwayExtractionModel(object):
    def __init__(self):
        self.config = config
        self.device = []
        self.device.append(self.config['device'])
        self.net = UNet3D(
            in_channels=self.config['in_channels'],
            out_channels=self.config['out_channels'],
            finalsigmoid=self.config['finalsigmoid'],
            fmaps_degree=self.config['fmaps_degree'],
            fmaps_layer_number=self.config['fmaps_layer_number'],
            layer_order=self.config['layer_order'],
            GroupNormNumber=self.config['GroupNormNumber'],
            device=self.device
        )

        # Resolve weight path: prefer absolute, try several fallbacks and warn instead of crashing
        weight_path = self.config.get('weight_path')
        resolved = None
        if weight_path:
            if os.path.isabs(weight_path) and os.path.exists(weight_path):
                resolved = weight_path
            else:
                # try as given (relative to current working dir)
                if os.path.exists(weight_path):
                    resolved = weight_path
                else:
                    # try relative to project root (one level up from this file)
                    candidate = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(__file__)), weight_path))
                    if os.path.exists(candidate):
                        resolved = candidate

        if resolved:
            try:
                print(f'Loading model weights from: {resolved}')
                # Load only tensor weights to avoid arbitrary object unpickling
                self.net.load_state_dict(torch.load(resolved, map_location=lambda storage, loc: storage.cuda(0), weights_only=True))
            except Exception as e:
                print(f'Warning: failed to load weights from {resolved}: {e}')
        else:
            print(f'Warning: weight file not found (tried "{weight_path}"); continuing without pretrained weights')

    @torch.no_grad()
    def predict(self, image: np.ndarray):
        self.net.eval()
        if self.config['use_HU_window']:
            image = lumTrans(image)
        image = normalize_CT(image)
        image = InnerTransformer.ToTensor(image)
        image = InnerTransformer.AddChannel(image)
        image = InnerTransformer.AddChannel(image)
        # Ensure image is a torch tensor before moving to device
        if not isinstance(image, torch.Tensor):
            try:
                image = torch.as_tensor(image, dtype=torch.float)
            except Exception:
                image = torch.tensor(image, dtype=torch.float)
        image = image.to(self.device[0])

        pred = sliding_window_inference(
            inputs=image,
            roi_size=self.config['roi_size'],
            sw_batch_size=self.config['sw_batch_size'],
            predictor=self.net,
            overlap=self.config['overlap'],
            mode=self.config['mode'],
            sigma_scale=self.config['sigma_scale']
        )

        pred = InnerTransformer.AsDiscrete(pred[:, 1, ...])
        if self.config['KeepLargestConnectedComponent']:
            pred = InnerTransformer.KeepLargestConnectedComponent(pred)
        pred = InnerTransformer.ToNumpy(pred)
        pred = InnerTransformer.CastToNumpyUINT8(pred[0, ...])
        torch.cuda.empty_cache()
        return pred
