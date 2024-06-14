import os
import torch
from .load_train import Spoofing_train
from .load_valtest import Spoofing_valtest


class dataset_info(object):

    def __init__(self):
        self.root_dir = ""


class data_merge(object):

    def __init__(self, image_dir):
        self.dic = {}
        self.image_dir = image_dir

        # photo
        photo_info = dataset_info()
        photo_info.root_dir = self.image_dir+'/'+'photo'
        self.dic["photo"] = photo_info

        # replay_clear
        replay_clear_info = dataset_info()
        replay_clear_info.root_dir = self.image_dir+'/'+'replay_clear'
        self.dic["replay_clear"] = replay_clear_info

        # replay_notclear
        replay_notclear_info = dataset_info()
        replay_notclear_info.root_dir = self.image_dir+'/'+'replay_notclear'
        self.dic["replay_notclear"] = replay_notclear_info

        # zalo
        zalo_info = dataset_info()
        zalo_info.root_dir = self.image_dir+'/'+'zalo'
        self.dic["zalo"] = zalo_info

    def get_single_dataset(self, data_name="", train=True, img_size=256, transform=None, debug_subset_size=None, UUID=-1):
        if train:
            data_dir = self.dic[data_name].root_dir
            data_set = Spoofing_train(os.path.join(data_dir, "label.txt"), os.path.join(data_dir, "image"), transform=transform, img_size=img_size, UUID=UUID)
            if debug_subset_size is not None:
                data_set = torch.utils.data.Subset(data_set, range(0, debug_subset_size))
        else:
            data_dir = self.dic[data_name].root_dir
            data_set = Spoofing_valtest(os.path.join(data_dir, "label.txt"), os.path.join(data_dir, "image"), transform=transform, img_size=img_size, UUID=UUID)
            if debug_subset_size is not None:
                data_set = torch.utils.data.Subset(data_set, range(0, debug_subset_size))
        print("Loading {}, number: {}".format(data_name, len(data_set)))
        return data_set

    def get_datasets(self, train=True, protocol="1", img_size=256, transform=None, debug_subset_size=None):
        if protocol == "Z_P_C_to_N":
            data_name_list_train = ["zalo", "photo", "replay_clear"]
            data_name_list_test = ["replay_notclear"]
        elif protocol == "Z_N_C_to_P":
            data_name_list_train = ["zalo", "replay_notclear", "replay_clear"]
            data_name_list_test = ["photo"]
        elif protocol == "Z_P_N_to_C":
            data_name_list_train = ["zalo", "photo", "replay_notclear"]
            data_name_list_test = ["replay_clear"]
        elif protocol == "N_P_C_to_Z":
            data_name_list_train = ["replay_notclear", "photo", "replay_clear"]
            data_name_list_test = ["zalo"]
        elif protocol == "all":
            data_name_list_train = ["photo", "replay_clear", "replay_notclear", "zalo"]
            data_name_list_test = ["photo", "replay_clear", "replay_notclear", "zalo"]
        sum_n = 0
        if train:
            data_set_sum = self.get_single_dataset(data_name=data_name_list_train[0], train=True, img_size=img_size, transform=transform, debug_subset_size=debug_subset_size, UUID=0)
            sum_n = len(data_set_sum)
            for i in range(1, len(data_name_list_train)):
                data_tmp = self.get_single_dataset(data_name=data_name_list_train[i], train=True, img_size=img_size, transform=transform, debug_subset_size=debug_subset_size, UUID=i)
                data_set_sum += data_tmp
                sum_n += len(data_tmp)
        else:
            data_set_sum = {}
            for i in range(len(data_name_list_test)):
                data_tmp = self.get_single_dataset(data_name=data_name_list_test[i], train=False, img_size=img_size, transform=transform, debug_subset_size=debug_subset_size, UUID=i)
                data_set_sum[data_name_list_test[i]] = data_tmp
                sum_n += len(data_tmp)
        print("Total number: {}".format(sum_n))
        return data_set_sum
