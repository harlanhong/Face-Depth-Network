from __future__ import absolute_import, division, print_function
import torch
import numpy 
import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import pdb
from .mono_dataset import MonoFaceDataset,MonoFlipFaceDataset

class CELEBDataset(MonoFaceDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(CELEBDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def get_color(self, folder, frame_index, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

class CELEBRAWDataset(CELEBDataset):
    def __init__(self, *args, **kwargs):
        super(CELEBRAWDataset, self).__init__(*args, **kwargs)
    def get_image_path(self, folder, frame_index):
        f_str = "{:08d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(folder, f_str)
        return image_path

class CELEBFlipDataset(MonoFlipFaceDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(CELEBFlipDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def get_color(self, folder, frame_index, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color
        
class CELEBFlipRAWDataset(CELEBFlipDataset):
    def __init__(self, *args, **kwargs):
        super(CELEBFlipRAWDataset, self).__init__(*args, **kwargs)
    def get_image_path(self, folder, frame_index):
        f_str = "{:08d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(folder, f_str)
        return image_path