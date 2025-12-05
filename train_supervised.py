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
parser.add_argument('--dataset', type=str, default="breakfast", help='either procel or cross task')
parser.add_argument('--cut_start', type=int, default=0, help='start ind for slicing task list')
parser.add_argument('--cut_end', type=int, default=18, help='end inf for slicing task list')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--loss_type', type=str, default="ce1_and_reg", help='video_active_method')
parser.add_argument('--adding_ratio', type=float, default=0.05, help='video added during each active cycle')
parser.add_argument('--device', type=str, default='cuda:6', help='video_active_method')
parser.add_argument('--data_dir', type=str, default='./data', help='path to dataset directory')
parser.add_argument('--output_dir', type=str, default='./results', help='path to save results')
torch.manual_seed(0)

args = parser.parse_args()


np.set_printoptions(suppress=True)
torch.set_printoptions(precision=4)
today = date.today()
def main(args):
    results=[]
    mother_result_path = args.output_dir + "/"
    if not torch.cuda.is_available():
        args.device = torch.device('cpu')

    # Dataset configuration (GTEA, 50Salads, Breakfast)
    if True:
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

        print("total train:", len(my_batch_gen.train_examples))
        # init trainer
        trainer = Trainer(num_decoders=num_decoders, 
                          num_layers=num_layers,
                          r1=2, 
                          r2=2, 
                          num_f_maps=64, 
                          input_dim=input_dim, 
                          num_classes=num_classes, 
                          channel_masking_rate=channel_masking_rate,
                          seed = 2,
                          loss_type=args.loss_type,
                          device = args.device)

        # trainer.train(save_dir=mother_result_path, 
        #               batch_gen=my_batch_gen, 
        #               batch_size=args.batch_size, 
        #               epochs=args.epochs,
        #               learning_rate=args.lr, 
        #               dataset=args.dataset,
        #               task=task)

        result = trainer.my_predict_eval(model_dir=mother_result_path, 
                        dataset=args.dataset,
                        task=task,
                        batch_gen=my_batch_gen, 
                        action_dict=index2label,
                        epoch=args.epochs,
                        if_train=False)
        results=results+result

    # if len(task_list) >1:
    #     headers=["dataset", "task",  "loss_type", 
    #                                     "epoch", "num_layers",
    #                                          "precision", "recall", "fscore", "acc", "acc_bg",
    #                                            "num_bg_pred", "num_bg", "num_frames", 
    #                                            "clip_edit_score", "f1@10", "f1@25", "f1@50"]
    #     temp_df = pd.DataFrame.from_records(results, columns=headers)
    #     temp_mean = temp_df.iloc[:,5:].mean().to_list()
    #     results.append([args.dataset, args.dataset, args.loss_type, args.epochs,num_layers]+temp_mean)
    # print(headers[5:])
    # print(temp_mean)
    # return results
if __name__ == '__main__':
    main(args)
    # time_start = time.time()
    # for dataset in ["50salads", "gtea_1s"]:
    #     for loss_type in ["ce1_and_var","ce1_and_mean"]:#["asf_only", "info", "ce1_and_reg", "ce2_and_reg", "ce_only", "ce1_only","ce2_only"]:
    #         args.loss_type=loss_type            # args.dataset= dataset
    #         main(args)
    # time_end = time.time()
    # print("all finished, total time used:", time_end-time_start)
    # time_start = time.time()
    # all_results = []
    # for dataset in ["50salads", "gtea_1s"]:  
    #     for loss_type in ["ce1_and_mean", "ce1_and_var"]:#["asf_only", "info", "ce1_and_reg","ce2_and_reg", "ce_only", "ce1_only","ce2_only"]:
    #         args.dataset=dataset
    #         args.loss_type=loss_type
    #         results=main(args)    
    #         all_results=all_results+results
    # headers = ["dataset", "task", "loss_type", "epoch", "num_layers",
    #                                          "precision", "recall", "fscore", "acc", "acc_bg",
    #                                            "num_bg_pred", "num_bg", "num_frames", 
    #                                            "clip_edit_score", "f1@10", "f1@25", "f1@50"]
    # clip_csv_name="10_30_sup_data_full_ablation_part2.csv"
    # with open(clip_csv_name, 'w', encoding='UTF8', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(headers)
    #     writer.writerows(all_results)
    # time_end = time.time()
    # print("total time used:", time_end-time_start)




    # mean_list = []
    # results_df = pd.read_csv("08_09_k+1_active_full.csv")
    # for dataset in ["ProceL", "CrossTask"]:
    #     data_df = results_df[results_df['dataset']==dataset]
    #     if  dataset == "CrossTask":
    #         for clip_active in ["stw", "random"]:
    #             data_ca_df = data_df[data_df['clip_active_method']==clip_active]
    #             for active_cycle in range(5):
    #                 active_cycle=active_cycle+1
    #                 data_ac_ca_df = data_ca_df[data_ca_df['active_cycle']==active_cycle]
    #             # for num_layers in [6]: # we only have 6 as num layers for crosstask
    #                 num_layers=6
    #                 data_layer_ac_ca_df = data_ac_ca_df[data_ac_ca_df['num_layers']==num_layers]
    #                 for epoch in [20,40,60,80]:
    #                     epoch_layer_data_epoch_df = data_layer_ac_ca_df[data_layer_ac_ca_df['epoch']==epoch]
    #                     temp_mean = epoch_layer_data_epoch_df.iloc[:,7:].mean().to_list()
    #                     if not math.isnan(temp_mean[0]):
    #                         mean_list.append([dataset,clip_active, active_cycle, epoch]+temp_mean)

    # headers = ["dataset", "clip_active", "active_cycle","epoch",
    #             "precision", "recall", "fscore", "acc", "acc_bg",
    #             "num_bg_pred", "num_bg", "num_frames", 
    #             "clip_edit_score", "f1@10", "f1@25", "f1@50"]
    # clip_csv_name="08_09_k+1_active_full_mean.csv"
    # with open(clip_csv_name, 'w', encoding='UTF8', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(headers)
    #     writer.writerows(mean_list)
