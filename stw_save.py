#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
note should take care of stw format
problem setting 
each iteration, select m videos and n clips in m videos to label
or 
each iteration, select m*n clips to label
"""
import argparse
import torch
import time
import os
from core import *
from optimizer import *
from datetime import date
from temp import *
import time
import torch
import pandas as pd
from trainer import Trainer
from active_batch_gen import *
import math
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate ')
parser.add_argument('--epochs', type=int, default=80, help='epochs ')
parser.add_argument('--num_active_cycle', type=int, default=4, help='num_active_cycle ')
parser.add_argument('--dataset', type=str, default="50salads", help='either procel or cross task')
parser.add_argument('--cut_start', type=int, default=0, help='start ind for slicing task list')
parser.add_argument('--cut_end', type=int, default=18, help='end inf for slicing task list')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--clip_active_method', type=str, default="stw", help='clip_active_method')
parser.add_argument('--video_active_method', type=str, default="drop_dtw", help='video_active_method')
parser.add_argument('--clip_size', type=float, default=0.25, help='video_active_method')
parser.add_argument('--loss_type', type=str, default="ce1_and_reg", help='video_active_method')
parser.add_argument('--adding_ratio', type=float, default=0.05, help='video added during each active cycle')
parser.add_argument('--device', type=str, default='cuda:1', help='video_active_method')
torch.manual_seed(0)

args = parser.parse_args()

np.set_printoptions(suppress=True)
torch.set_printoptions(precision=4)
today = date.today()
def main(args):
    results=[]

    # Dataset configuration (GTEA, 50Salads, Breakfast)
    task_list=[args.dataset]
    mother_data_path = args.data_dir + "/"
    mapping_file = mother_data_path+args.dataset+"/mapping.txt"
    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
    num_classes = len(actions_dict)
    input_dim=2048
    num_decoders=3
    if args.dataset == "50salads":
            num_layers=4
            channel_masking_rate=0.3
            frames_per_seg=32
            feat_file_list = mother_data_path+args.dataset+"/clip_features/"
            label_file_list = mother_data_path+args.dataset+"/clip_labels/"
            for key in actions_dict.keys():
                actions_dict[key]=actions_dict[key]+1
            for key in ["action_start", "action_end"]:
                actions_dict[key]=0
            index2label = dict()
            index2label[0] = 'background'
            for key in list(set(actions_dict.keys())-set(["action_start", "action_end"])):
                index2label[actions_dict[key]]=key

        elif args.dataset == "gtea":
            num_layers=2
            channel_masking_rate=0.75
            frames_per_seg=32
            feat_file_list = mother_data_path+args.dataset+"/clip_features/"
            label_file_list = mother_data_path+args.dataset+"/clip_labels/"
            for key in list(set(actions_dict.keys())-set(["background"])):
                actions_dict[key]=actions_dict[key]+1
            actions_dict['background']=0
            index2label = dict()
            for key in actions_dict.keys():
                index2label[actions_dict[key]]=key
            index2label[0]='background'

        elif args.dataset == "gtea_1s":
            num_layers=3
            channel_masking_rate=0.5
            frames_per_seg=10
            feat_file_list = mother_data_path+args.dataset+"/1s_clip_features/"
            label_file_list = mother_data_path+args.dataset+"/1s_clip_labels/"
            for key in list(set(actions_dict.keys())-set(["background"])):
                actions_dict[key]=actions_dict[key]+1
            actions_dict['background']=0
            index2label = dict()
            for key in actions_dict.keys():
                index2label[actions_dict[key]]=key
            index2label[0]='background'

        elif args.dataset == 'breakfast':
            num_layers=6
            channel_masking_rate=0.3
            frames_per_seg=32
            feat_file_list = mother_data_path+args.dataset+"/clip_features/"
            label_file_list = mother_data_path+args.dataset+"/clip_labels/"
            index2label = dict()
            for k,v in actions_dict.items():
                index2label[v] = k
            index2label[0]='background'

    for ta in range(len(task_list)):
        task = task_list[ta]
        if any(args.dataset == _ for _ in ["ProceL", "CrossTask"]):
            mapping = open(os.path.join(mother_path+args.dataset+"_dataset/", task)\
                                    +"/mapping.txt","rt")
            mapping = mapping.readlines()
            # print(mapping)
            index2label = dict()
            for a in mapping:
                index2label[int(a.split()[0])] = a.split()[1]
            index2label[0]='background'
            feat_file_list = data_default_path+task+"/video_embd/"  
            label_file_list = mother_path+"/clip_labels/"
            num_classes=len(mapping)+1
        
        # print all 
        print(args)
        print("num_classes", num_classes,
            "num_layers",num_layers,
            "channel_masking_rate",channel_masking_rate,
            "task_list:",task_list)
        print("start task:", task)
        # init batch generator
        my_batch_gen = BatchGenerator(
            num_classes=num_classes,
            actions_dict=index2label, 
            gt_path=label_file_list, 
            features_path=feat_file_list, 
            sample_rate=1,
            frames_per_seg=frames_per_seg,
            adding_ratio=args.adding_ratio)
        if task == "change_iphone_battery":
            remove_list=["change_iphone_battery_0002"]
        elif task == "change_toilet_seat":
            remove_list=["change_toilet_seat_0012", "change_toilet_seat_0034"]
        else:
            remove_list=None
        my_batch_gen.my_active_read_data(randnum=1, remove_list=remove_list)  

        stw_output=torch.load("./stw_cache/"+args.dataset+"_"+task+".pt")
        save_dict={}
        while my_batch_gen.has_next(if_train=True):
            batch_input, batch_target, mask, vids, _ = my_batch_gen.my_next_batch(batch_size=1,if_train=True)  
            save_dict[vids[0]]={}
        my_batch_gen.reset(if_train=True)
        
        while my_batch_gen.has_next(if_train=True):
            batch_input, batch_target, mask, vids, _ = my_batch_gen.my_next_batch(batch_size=1,if_train=True)  
            num_clip_to_label = max(1, int(0.25*batch_target.size()[1]))
            temp_num = min(batch_target.size()[1], num_clip_to_label)
            # convert stw to label
            gnd = batch_target.squeeze(0).numpy()
            stw_result=stw_output[vids[0]]['assign'][temp_num].numpy()
            stw_label=gnd[stw_result]
            save_dict[vids[0]]['stw']=stw_label #stw_output[vids[0]]['assign'][temp_num].numpy()
            gre_result=greedy_vectorized(batch_input.squeeze(0).t().cpu(), temp_num, device="cpu").numpy()
            gre_label=gnd[gre_result]
            save_dict[vids[0]]['gre']=gre_label#greedy_vectorized(batch_input.squeeze(0).t().cpu(), temp_num, device="cpu").numpy()
            save_dict[vids[0]]['gnd']=gnd
        torch.save(save_dict,"intra_label_result_"+args.dataset+"_"+str(task)+"_clip_size_"+str(args.clip_size)+".pt")
if __name__ == '__main__':


    time_start = time.time()
    main(args)
    time_end = time.time()
    print("total time used:", time_end-time_start)
    
    # time_start = time.time()
    # all_results = []
    # for dataset in ["ProceL","CrossTask"]:
    #     for clip_size in [0.1, 0.2, 0.3, 0.4, 0.5]:
    #         for clip_active_method in [ "stw", "adapt_stw"]:
    #             if clip_active_method == "stw":
    #                 for loss_type in ["info", "mil"]:
    #                     args.video_active_method="random"
    #                     args.clip_active_method=clip_active_method
    #                     args.loss_type=loss_type
    #                     args.dataset=dataset
    #                     args.clip_size=clip_size
    #                     results=main(args)    
    #                     all_results=all_results+results
    #             else:
    #                 args.loss_type="info"
    #                 args.video_active_method="random"
    #                 args.clip_active_method=clip_active_method
    #                 args.dataset=dataset
    #                 args.clip_size=clip_size
    #                 results=main(args)    
    #                 all_results=all_results+results

               

    #     time_end = time.time()
    # print("all finished, total time used:", time_end-time_start)
    # headers = ["dataset", "task", "video_active_method", "clip_active_method", "active_cycle", "loss_type", "clip_size",
    #                                     "epoch", "num_layers",
    #                                          "precision", "recall", "fscore", "acc", "acc_bg",
    #                                            "num_bg_pred", "num_bg", "num_frames", 
    #                                            "clip_edit_score", "f1@10", "f1@25", "f1@50"]
    # clip_csv_name="09_11_weak_data_stw_active.csv"
    # with open(clip_csv_name, 'w', encoding='UTF8', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(headers)
    #     writer.writerows(all_results)
    # print("total time used:", time_end-time_start)