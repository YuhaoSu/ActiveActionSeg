'''
    Adapted from https://github.com/yabufarha/ms-tcn
'''

import torch
import numpy as np
import random
# from ASFormerUnused.grid_sampler import GridSampler, TimeWarpLayer  # Not used (if_warp=False always)
from sklearn.model_selection import KFold
import os
from core import annot_to_framelabel, get_annotation_by_name

class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate, frames_per_seg, adding_ratio):
        self.train_index = 0
        self.test_index = 0
        self.selected_train_index = 0
        self.unselected_train_index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        # self.frame_gt_path = frame_gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.frames_per_seg = frames_per_seg
        self.adding_ratio = adding_ratio

        # self.timewarp_layer = TimeWarpLayer()  # Not used (if_warp=False always)

    
    def active_reset(self, if_train):
        if if_train is True:
            self.selected_train_index=0
        elif if_train is False:
            self.unselected_train_index=0

    def reset(self, if_train, randnum=None):
        if if_train is False:
            self.test_index = 0
            self.my_test_shuffle(randnum)
        elif if_train is True:
            self.train_index = 0
            # self.my_train_shuffle()
        else:
            raise Exception("Train or test data?")    
            
    def has_next(self, if_train):
        if if_train is False:
            if self.test_index < len(self.test_examples):
                return True
            return False
        elif if_train is True:
            if self.train_index < len(self.train_examples):
                return True
            return False
        else:
            raise Exception("Train or test data?") 

    def active_train_has_next(self, if_train, active_cycle):
        cycle_to_add = int(len(self.train_examples)*self.adding_ratio)
        if if_train is True:
            if self.selected_train_index < (active_cycle+1)*cycle_to_add:
                return True
            return False
        elif if_train is False:
            if self.unselected_train_index < len(self.unselected_train_examples):
                return True
            return False
    
    def my_active_read_data(self, randnum=None, remove_list=None):
        self.list_of_examples=[]
        # print(self.features_path)
        for _,dirs,name in os.walk(self.features_path):
            for _ in name:
                if  _ != '.DS_Store':
                    self.list_of_examples.append(_[:-21])
                # train test split
        if remove_list is not None:
            for _ in range(len(remove_list)):
                self.list_of_examples.remove(remove_list[_])
        kf = KFold(n_splits=4,random_state=randnum,shuffle=True)
        self.data_splits = []
        for i, (train_index, test_index) in enumerate(kf.split(self.list_of_examples)):
            self.data_splits.append([train_index,test_index])
        self.train_examples = [self.list_of_examples[_] for _ in self.data_splits[0][0]]
        self.test_examples = [self.list_of_examples[_] for _ in self.data_splits[0][1]]
        # self.train_gts = [self.gt_path + vid.split('.')[0] +".csv" for vid in self.train_examples]

        self.train_gts = [self.gt_path + vid.split('.')[0] +".npy" for vid in self.train_examples]
        # self.train_frame_gts = [self.frame_gt_path + vid.split('.')[0] +".txt" for vid in self.train_examples]
        self.train_features = [self.features_path + vid.split('.')[0] + '_video_embeddings.npy' for vid in self.train_examples]
        # self.test_gts = [self.gt_path + vid.split('.')[0] +".csv" for vid in self.test_examples]

        self.test_gts = [self.gt_path + vid.split('.')[0] +".npy" for vid in self.test_examples]
        # self.test_frame_gts = [self.frame_gt_path + vid.split('.')[0] +".txt" for vid in self.test_examples]
        self.test_features = [self.features_path + vid.split('.')[0] + '_video_embeddings.npy' for vid in self.test_examples]
        
        
        self.my_train_shuffle(randnum)
        self.my_test_shuffle(randnum)
        active_size = int(len(self.train_examples)*self.adding_ratio)
        self.selected_train_examples = self.train_examples[:active_size]
        self.unselected_train_examples = self.train_examples[active_size:]


    def my_train_shuffle(self, randnum=None):
        # shuffle list_of_examples, gts, features with the same order
        if randnum is None:
            randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(self.train_examples)
        random.seed(randnum)
        random.shuffle(self.train_gts)
        random.seed(randnum)
        random.shuffle(self.train_features)


    def my_test_shuffle(self, randnum=None):
        # shuffle list_of_examples, gts, features with the same order
        if randnum is None:
            randnum = random.randint(0, 100)        
        random.seed(randnum)
        random.shuffle(self.test_examples)
        random.seed(randnum)
        random.shuffle(self.test_gts)
        random.seed(randnum)
        random.shuffle(self.test_features)




    def warp_video(self, batch_input_tensor, batch_target_tensor):
        '''
        :param batch_input_tensor: (bs, C_in, L_in)
        :param batch_target_tensor: (bs, L_in)
        :return: warped input and target
        '''
        bs, _, T = batch_input_tensor.shape
        grid_sampler = GridSampler(T)
        grid = grid_sampler.sample(bs)
        grid = torch.from_numpy(grid).float()

        warped_batch_input_tensor = self.timewarp_layer(batch_input_tensor, grid, mode='bilinear')
        batch_target_tensor = batch_target_tensor.unsqueeze(1).float()
        warped_batch_target_tensor = self.timewarp_layer(batch_target_tensor, grid, mode='nearest')  # no bilinear for label!
        warped_batch_target_tensor = warped_batch_target_tensor.squeeze(1).long()  # obtain the same shape

        return warped_batch_input_tensor, warped_batch_target_tensor

    def merge(self, bg, suffix):
        '''
        merge two batch generator. I.E
        BatchGenerator a;
        BatchGenerator b;
        a.merge(b, suffix='@1')
        :param bg:
        :param suffix: identify the video
        :return:
        '''

        self.list_of_examples += [vid + suffix for vid in bg.list_of_examples]
        self.gts += bg.gts
        self.features += bg.features

        print('Merge! Dataset length:{}'.format(len(self.list_of_examples)))


    def my_active_next_batch(self, batch_size, if_train=True, if_warp=False): # if_warp=True is a strong data augmentation. See grid_sampler.py for details.
        if if_train is True:
            # print(self.selected_train_index)
            batch = self.selected_train_examples[self.selected_train_index:self.selected_train_index + batch_size]
            # batch_gts = [self.gt_path + batch[0].split('.')[0] +".csv" ]
            batch_gts = [self.gt_path + batch[0].split('.')[0] +".npy" ]
            # batch_frame_gts = [self.frame_gt_path + batch[0].split('.')[0] +".txt"]
            
            batch_features = [self.features_path + batch[0].split('.')[0] + '_video_embeddings.npy' ]
            self.selected_train_index += batch_size

        elif if_train is False:
            batch = self.unselected_train_examples[self.unselected_train_index:self.unselected_train_index + batch_size]
            # batch_gts = [self.gt_path + batch[0].split('.')[0] +".csv" ]  
            # batch_frame_gts = [self.frame_gt_path + batch[0].split('.')[0] +".txt"]
            batch_gts = [self.gt_path + batch[0].split('.')[0] +".npy" ]  
            batch_features = [self.features_path + batch[0].split('.')[0] + '_video_embeddings.npy' ]
            self.unselected_train_index += batch_size

        batch_input = []
        batch_target = []
        action_length = []

        for idx, vid in enumerate(batch):
            # clip features
            features = np.transpose(np.load(batch_features[idx]))

            # clip label
            # annon = get_annotation_by_name(batch_gts[idx])
            # frame_labels = annot_to_framelabel(annon,features.shape[1]*self.frames_per_seg, fps=10)
            frame_labels = np.load(batch_gts[idx])
            # majority voting
            # clip_label = np.zeros(features.shape[1], dtype=np.int32)
            # for j in range(len(clip_label)):
            #     temp = np.arange(start=j*self.frames_per_seg, stop=(j+1)*self.frames_per_seg)
            #     clip_labels = frame_labels[temp]
            #     values, counts = np.unique(clip_labels, return_counts=True)
            #     clip_label[j] = values[np.argmax(counts)]

            # use middle frame
            clip_label = np.zeros(features.shape[1], dtype=np.int32)
            for j in range(len(clip_label)):
                temp = np.arange(start=j*self.frames_per_seg, stop=(j+1)*self.frames_per_seg)
                clip_label[j] = frame_labels[temp][int(self.frames_per_seg/2)-1 ]
            feature = features[:, ::self.sample_rate]
            target = clip_label[::self.sample_rate]
            # f_target = classes[::self.sample_rate]

            batch_input.append(feature)
            batch_target.append(target)
        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)  # bs, C_in, L_in
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long) * (-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            if if_warp:
                warped_input, warped_target = self.warp_video(torch.from_numpy(batch_input[i]).unsqueeze(0), torch.from_numpy(batch_target[i]).unsqueeze(0))
                batch_input_tensor[i, :, :np.shape(batch_input[i])[1]], batch_target_tensor[i, :np.shape(batch_target[i])[0]] = warped_input.squeeze(0), warped_target.squeeze(0)
            else:
                batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
                batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])
        
        return batch_input_tensor, batch_target_tensor, mask, batch, frame_labels#, action_length
    
    def my_next_batch(self, batch_size, if_train=True, if_warp=False): # if_warp=True is a strong data augmentation. See grid_sampler.py for details.
        if if_train is True:
            batch = self.train_examples[self.train_index:self.train_index + batch_size]
            batch_gts = self.train_gts[self.train_index:self.train_index + batch_size]
            batch_features = self.train_features[self.train_index:self.train_index + batch_size]
            self.train_index += batch_size

        elif if_train is False:
            batch = self.test_examples[self.test_index:self.test_index + batch_size]
            batch_gts = self.test_gts[self.test_index:self.test_index + batch_size]
            batch_features = self.test_features[self.test_index:self.test_index + batch_size]
            self.test_index += batch_size

        batch_input = []
        batch_target = []
        action_length = []

        for idx, vid in enumerate(batch):
            # clip features
            features = np.transpose(np.load(batch_features[idx]))
            # clip label
            # annon = get_annotation_by_name(batch_gts[idx])
            # for _ in annon:
            #     action_length.append(_[2] - _[1])
            # if (idx %30) ==0:
            #     print(annon)
            #     print(action_length)
            # frame_labels = annot_to_framelabel(annon,features.shape[1]*self.frames_per_seg, fps=10)
            frame_labels = np.load(batch_gts[idx])
            clip_label = np.zeros(features.shape[1], dtype=np.int32)
            for j in range(len(clip_label)):
                temp = np.arange(start=j*self.frames_per_seg, stop=(j+1)*self.frames_per_seg)
                clip_labels = frame_labels[temp]
                values, counts = np.unique(clip_labels, return_counts=True)
                clip_label[j] = values[np.argmax(counts)]
 
            feature = features[:, ::self.sample_rate]
            target = clip_label[::self.sample_rate]
            batch_input.append(feature)
            batch_target.append(target)
        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)  # bs, C_in, L_in
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long) * (-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            if if_warp:
                warped_input, warped_target = self.warp_video(torch.from_numpy(batch_input[i]).unsqueeze(0), torch.from_numpy(batch_target[i]).unsqueeze(0))
                batch_input_tensor[i, :, :np.shape(batch_input[i])[1]], batch_target_tensor[i, :np.shape(batch_target[i])[0]] = warped_input.squeeze(0), warped_target.squeeze(0)
            else:
                batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
                batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])
        
        return batch_input_tensor, batch_target_tensor, mask, batch, frame_labels#, action_length





if __name__ == '__main__':
    pass