from __future__ import division

import os
import os.path as osp
from pathlib import Path
from .retinaface import RetinaFace
import numpy as np

__all__ = ['FaceAnalysis']

class FaceAnalysis:
    def __init__(self, root=osp.join(Path(__file__).parent.parent.absolute(), 'weights')):
        self.models = {}
        model_name = os.listdir(root)
        for mod in model_name:
            if mod == 'det_500m.onnx':
                mods = os.path.join(root, mod)
                self.models['detection'] = RetinaFace(mods)

        self.det_model = self.models['detection']

    def prepare(self, det_thresh=0.5, det_size=(640, 640)):
        self.det_thresh = det_thresh
        assert det_size is not None
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname=='detection':
                model.prepare(input_size=det_size, det_thresh=det_thresh)

    def get(self, img, max_num=0):
        bboxes, kpss = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric='default')
        if bboxes.shape[0] == 0:
            return np.array([])
        else:
            return bboxes