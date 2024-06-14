import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class CustomDataSet(Dataset):
    def __init__(self, main_dir, ref_dir, transform, time_stamp, view_num):
        self.main_dir = main_dir
        self.ref_dir = ref_dir
        self.transform = transform
        self.tts = time_stamp
        self.v_num = view_num

    def __len__(self):
        return self.tts * self.v_num # len(self.frame_idx)

    def __getitem__(self, idx):
        frame_idx = torch.tensor(idx)
        time_idx = torch.tensor(idx % self.tts)
        view_idx = torch.tensor(idx // self.tts)

        img_name = os.path.join(self.main_dir, 'f%05d_v%02d.png' % (time_idx + 1, view_idx))
        image = Image.open(img_name).convert("RGB")
        tensor_image = self.transform(image)

        tensor_ref = tensor_image
        ref_name = os.path.join(self.ref_dir, 'f%05d_v%02d.png' % (time_idx + 1, view_idx))
        if os.path.exists(ref_name):
            ref = Image.open(ref_name).convert("RGB")
            tensor_ref = self.transform(ref)

        if 0:
            print(idx, time_idx, view_idx, img_name, ref_name)
            # sys.exit(0)

        return tensor_image, tensor_ref, time_idx, view_idx, frame_idx


def ReadCameraPara(camera_data_path):
    # read txt
    Position = []
    Rotation = []
    fp = open(camera_data_path, 'r')
    fp_lines = fp.readlines()
    for l_idx in range(len(fp_lines) // 2):
        Pos1, Pos2, Pos3 = fp_lines[2 * l_idx].strip().split(',')
        Position.append([float(Pos1), float(Pos2), float(Pos3)])
        Rot1, Rot2, Rot3 = fp_lines[2 * l_idx + 1].strip().split(',')
        Rotation.append([float(Rot1), float(Rot2), float(Rot3)])
    
    if 0:
        print(Position)
        print(Rotation)
        sys.exit(0)
    
    return Position, Rotation