import cv2
import numpy as np
from torch.utils.data import Dataset
import math


def crop_face_from_scene(image, face_name_full, scale):
    f=open(face_name_full,'r')
    lines=f.readlines()
    y1,x1,w,h=[float(ele) for ele in lines]
    f.close()
    y2=y1+w
    x2=x1+h
    y_mid=(y1+y2)/2.0
    x_mid=(x1+x2)/2.0
    h_img, w_img = image.shape[0], image.shape[1]
    w_scale=scale*w
    h_scale=scale*h
    y1=y_mid-w_scale/2.0
    x1=x_mid-h_scale/2.0
    y2=y_mid+w_scale/2.0
    x2=x_mid+h_scale/2.0
    y1=np.int16(max(math.floor(y1),0))
    x1=np.int16(max(math.floor(x1),0))
    y2=np.int16(min(math.floor(y2),w_img))
    x2=np.int16(min(math.floor(x2),h_img))
    region=image[x1:x2,y1:y2]
    return region


class Spoofing_valtest(Dataset):

    def __init__(self, info_list, root_dir, transform=None, face_scale=1.3, img_size=256, UUID=-1):
        with open(info_list, 'r') as file:
            info = file.readlines()
        self.frame = np.array([[i.replace('\n','')[:-2], i.replace('\n','')[-1]] for i in info])
        self.root_dir = root_dir
        self.transform = transform
        self.face_scale = face_scale
        self.img_size = img_size
        self.UUID = UUID

    def __len__(self):
        return len(self.frame)
    
    def __getitem__(self, idx):
        img_name = self.frame[idx, 0]
        image_path = self.root_dir+'/'+img_name
        spoofing_label = self.frame[idx, 1]
        if spoofing_label == '1':
            spoofing_label = 1
        else:
            spoofing_label = 0
        image_x = self.get_single_image_x(image_path)
        sample = {'image_x': image_x, 'label': spoofing_label, "UUID": self.UUID}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_single_image_x(self, image_path):
        bbx_path = image_path.replace(".jpg", ".txt")
        image_x_temp = cv2.imread(image_path)
        image_x = cv2.resize(crop_face_from_scene(image_x_temp, bbx_path, self.face_scale), (self.img_size, self.img_size))
        return image_x
