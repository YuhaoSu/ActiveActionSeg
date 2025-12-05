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
parser.add_argument('--lr', type=float, default=0.0005, help='earning rate ')
parser.add_argument('--epochs', type=int, default=80, help='epochs ')
parser.add_argument('--num_active_cycle', type=int, default=4, help='num_active_cycle ')
parser.add_argument('--dataset', type=str, default="50salads", help='either procel or cross task')
parser.add_argument('--cut_start', type=int, default=0, help='start ind for slicing task list')
parser.add_argument('--cut_end', type=int, default=18, help='end inf for slicing task list')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--clip_active_method', type=str, default="stw", help='clip_active_method')
parser.add_argument('--video_active_method', type=str, default="drop_dtw", help='video_active_method')
parser.add_argument('--clip_size', type=int, default=0.25, help='video_active_method')
parser.add_argument('--loss_type', type=str, default="ce1_and_reg", help='video_active_method') #ce1_and_reg
parser.add_argument('--adding_ratio', type=float, default=0.05, help='video added during each active cycle')
parser.add_argument('--device', type=str, default='cuda:2', help='video_active_method')
parser.add_argument('--data_dir', type=str, default='./data', help='path to dataset directory')
parser.add_argument('--output_dir', type=str, default='./results', help='path to save results')

torch.manual_seed(0)

args = parser.parse_args()

np.set_printoptions(suppress=True)
torch.set_printoptions(precision=4)
today = date.today()
def main(args):
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
            # avg_clip=119.86
            # avg_#_action=19
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
            # avg_clip=73.78
            # avg_#_action=32
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
            # avg_clip=43.2
            # avg_#_action=6
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
        seed = int(np.random.choice(100,1))
        print("curent seed:", seed)
        my_batch_gen.my_active_read_data(randnum=10, remove_list=remove_list)  

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
                          seed = seed,
                          loss_type=args.loss_type,
                          device = args.device)
        


        trainer.active_train(save_dir=mother_result_path, 
                      batch_gen=my_batch_gen, 
                      batch_size=args.batch_size, 
                      learning_rate=args.lr, 
                      dataset=args.dataset,
                      task=task,
                      epochs=args.epochs,
                      num_active_cycle=args.num_active_cycle,
                      clip_active_method=args.clip_active_method,
                      video_active_method=args.video_active_method,
                      clip_size=args.clip_size)

        # trainer.my_drop_dtw_predict_eval(model_dir=mother_result_path, 
        #                     dataset=args.dataset,
        #                     task=task,
        #                     batch_gen=my_batch_gen, 
        #                     if_train=True,
        #                     num_active_cycle=5,
        #                     clip_active_method="random",
        #                     video_active_method="random")
        # count = count +1
    #     dtw_all = dtw_recall + dtw_all
    #     asf_all = asf_recall + asf_all
    # print("crosstask dtw:", dtw_all/count)
    # print("crosstask asf:", asf_all/count)

            # results=results+result
            # return results

    #     extracted_features, labels = trainer.get_features(
    #         model_dir=mother_result_path, 
    #                         dataset=args.dataset,
    #                         task=task,
    #                         batch_gen=my_batch_gen, 
    #                         action_dict=id2class_map,
    #                         epoch=120,
    #                         if_train=True)
    #     model_features[task]=extracted_features
    #     feature_labels[task]=labels
    # torch.save(model_features,feature_result_path+ "/cl_p_asf_"+str(args.num_decoders)+"dec_"+str(args.num_layers)+"layers_"+args.dataset+"_train_features.pt")
    # torch.save(feature_labels, feature_result_path+ "/cl_p_asf_"+str(args.num_decoders)+"dec_"+str(args.num_layers)+"layers_"+args.dataset+"_train_features_labels.pt")
if __name__ == '__main__':
    time_start = time.time()
    # args.clip_size=0.5
    # args.num_active_cycle=2
    main(args)
    # main(args)  # Commented out for single run
    # main(args)  # Commented out for single run
    # main(args)
    # main(args)

    # args.clip_size=0.25
    # args.adding_ratio=0.1
    # args.num_active_cycle=2
    # main(args)
    # for loss in [  "ce1_and_var", "ce1_only"]:#"asf_only", "ce1_and_var","asf_only", "ce1_and_var"
    #     args.loss_type = loss 
    #     main(args)
    # for size in [0.25]:
    #     # args.clip_size=size
    #     # args.clip_active_method = "stw"
    #     # args.video_active_method="drop_dtw"
    #     # main(args)
    #     args.clip_active_method = "adapt_stw"
    #     args.video_active_method="random"
    #     main(args)
    # args.clip_active_method = "adapt_stw"
    # args.video_active_method="drop_dtw"
    # main(args)
    # for clip_size in [0.3,0.35, 0.4,0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
    #     args.clip_size=clip_size
    #     main(args)
    # args.clip_active_method = "greedy"
    # args.video_active_method="random"
    # main(args)

    #         args.video_active_method =video_method
    #         main(args)

    # for clip_method in ["equidistant", "split_rand", "greedy", "split_entropy"]:# ,equidistant split_entropy
    #     args.clip_active_method = clip_method
    #     args.video_active_method ="random"
    #     main(args)

    # args.clip_active_method = "greedy"
    # args.video_active_method ="random"
    # main(args)

    # for clip_method in ["adapt_stw"]:
    #     for video_method in ["drop_dtw"]:
    #         args.clip_active_method = clip_method
    #         args.video_active_method =video_method
    #         main(args)
    
    # args.clip_size=1
    # args.num_active_cycle=1
    # args.adding_ratio=1
    # main(args)
    
    time_end = time.time()
    print("all finished, total time used:", time_end-time_start)
    