import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import copy
import numpy as np
import math
import time
import networkx as nx
from metric import *
from core import *
from optimizer import *
import pandas as pd
from drop_dtw import *
# from thop import profile
from model import MyTransformer, p_loss, mil_p_loss
from new_eval import compute_IoU_IoD, compute_mof

torch.set_printoptions(precision=10)

class Trainer:
    def __init__(self, num_decoders, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, seed, loss_type, device):
        self.device = device
        self.num_decoders = num_decoders
        self.num_layers = num_layers
        self.num_classes = num_classes 
        self.num_f_maps=num_f_maps
        self.loss_type=loss_type
        self.model = MyTransformer(num_decoders, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, self.device)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.generator = torch.Generator()
        self.seed=seed
        self.generator.manual_seed(self.seed)
        self.mse = nn.MSELoss(reduction='none')
    
    
    def train(self, save_dir, batch_gen, epochs, batch_size, learning_rate, dataset, task):

        # pre compute some helpers
        p_dis_ind_  = torch.triu_indices(self.num_classes, self.num_classes)
        ind = torch.triu_indices(self.num_classes, self.num_classes).unbind()
        p_dis_ind = p_dis_ind_[:,ind[0] != ind[1]].unbind()
        device = self.device
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3,threshold=5e-3,verbose=False)
        self.model.train()
        for epoch in range(epochs):
            time_start = time.time()
            epoch_loss = 0
            epoch_cl_loss = 0
            correct = 0
            total = 0
            ce1_show = 0
            ce2_show = 0
            p_var_show = 0
            p_mean_show = 0
            asf_loss = 0
            macs_avg=0
            params_avg=0
            while batch_gen.has_next(if_train=True):
                batch_input, batch_target, mask, vids, _ = batch_gen.my_next_batch(batch_size,if_train=True)
                batch_input, batch_target, mask = batch_input.to(device),batch_target.to(device),  mask.to(device)
                # macs, params = profile(self.model, inputs=(batch_input,mask, ))
                # print("macs, params", macs, params)
                # macs_avg=macs+macs_avg
                # params_avg=params+params_avg
                # create helper, consider background as action 0
                label = batch_target.clone().view(-1)
                action_label = label[label>-100]
                arranged_prototype = self.model.prototype.t()[action_label]
                swapped_label = torch.arange(len(label)).to(device)[label>-100]
                optimizer.zero_grad()
                ps,fs = self.model(batch_input, mask)
                loss = 0
                cl_loss = 0
                for p in ps:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                        max=16) * mask[:, :, 1:])
                if self.loss_type == "asf_only":
                    cl_loss=torch.zeros(1).to(device)
                    ce1=torch.zeros(1).to(device)
                    ce2=torch.zeros(1).to(device)
                    var=torch.zeros(1).to(device)
                    mean=torch.zeros(1).to(device)
                else:
                    for f in fs:
                        ce1, ce2, var, mean = p_loss(f, label, swapped_label, arranged_prototype, self.model.prototype, p_dis_ind, device)
                        if self.loss_type == "info":
                            cl_loss = cl_loss + ce1 + ce2 +var + mean
                        elif self.loss_type == "ce1_and_reg":
                            ce2=torch.zeros(1).to(device)
                            cl_loss = cl_loss + ce1 +var + mean
                        elif self.loss_type == "ce2_and_reg":
                            ce1=torch.zeros(1).to(device)
                            cl_loss = cl_loss + ce2 +var + mean
                        elif self.loss_type == "ce_only":
                            var=torch.zeros(1).to(device)
                            mean=torch.zeros(1).to(device)
                            cl_loss = cl_loss + ce1 + ce2
                        elif self.loss_type == "ce1_only":
                            ce2=torch.zeros(1).to(device)
                            var=torch.zeros(1).to(device)
                            mean=torch.zeros(1).to(device)
                            cl_loss = cl_loss + ce1
                        elif self.loss_type == "ce2_only":
                            ce1=torch.zeros(1).to(device)
                            var=torch.zeros(1).to(device)
                            mean=torch.zeros(1).to(device)
                            cl_loss = cl_loss + ce2
                        elif self.loss_type == "ce1_and_mean":
                            ce2=torch.zeros(1).to(device)
                            var=torch.zeros(1).to(device)
                            cl_loss = cl_loss + ce1 + mean
                        elif self.loss_type == "ce1_and_var":
                            ce2=torch.zeros(1).to(device)
                            mean=torch.zeros(1).to(device)
                            cl_loss = cl_loss + ce1 + var
                loss=loss+0.1*cl_loss
                asf_loss += loss.item()
                epoch_cl_loss += 0.1*cl_loss.item()
                epoch_loss += loss.item()
                ce1_show += 0.1*ce1.item()
                ce2_show += 0.1*ce2.item()
                p_var_show += 0.1*var.item()
                p_mean_show += 0.1*mean.item()
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(ps.data[-1], 1)
                correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()
            scheduler.step(epoch_loss)
            num_trained = batch_gen.train_index
            batch_gen.reset(if_train=True)

            time_end = time.time()
            if (epoch + 1) % 20 == 0:
                print("[epoch %d]: epoch loss =%f, asf loss = %f,epoch cl loss = %f, ce1 = %f, ce2 = %f, p_var = %f, p_mean = %f, train acc = %f, seen %f videos, time = %f " %
                        ( epoch + 1,          epoch_loss / len(batch_gen.train_examples),
                                                            asf_loss / len(batch_gen.train_examples),
                                                            epoch_cl_loss / len(batch_gen.train_examples),
                                                            ce1_show /len(batch_gen.train_examples),
                                                            ce2_show / len(batch_gen.train_examples),
                                                            p_var_show /len(batch_gen.train_examples),
                                                            p_mean_show / len(batch_gen.train_examples),
                                                float(correct) / total,
                                                int(num_trained),
                                                (time_end-time_start)))
                # print("avg macs:", macs_avg/int(num_trained), "model size", params_avg/int(num_trained))
                # print("[epoch %d]: epoch loss =%f, asf loss = %f, train acc = %f, seen %f videos, time = %f " %
                #         ( epoch + 1,          epoch_loss / len(batch_gen.train_examples),
                #                                             asf_loss / len(batch_gen.train_examples),
                #                                 float(correct) / total,
                #                                 int(num_trained),
                #                                 (time_end-time_start)))

            if (epoch + 1) % 40 == 0:
                self.test(batch_gen, epoch, batch_gen.actions_dict, False)
            # if (epoch + 1) % 2 == 0:
            #     self.test(batch_gen, epoch, batch_gen.actions_dict, False)

                torch.save(self.model.state_dict(), save_dir + "/i3d_asf_cl_p_"+str(self.num_decoders)+"dec_"+str(self.num_layers)+"layers_"+dataset+"_"+\
                            str(task)+"_epoch-" + str(epoch + 1)+"_"+self.loss_type+".model")
                torch.save(optimizer.state_dict(), save_dir + "/i3d_asf_cl_p_"+str(self.num_decoders)+"dec_"+str(self.num_layers)+"layers_"+dataset+"_"+\
                            str(task)+"_epoch-" + str(epoch + 1) +"_"+self.loss_type+".opt")





    def active_train(self, save_dir, batch_gen, batch_size, learning_rate, dataset, task, epochs, num_active_cycle, clip_active_method, video_active_method, clip_size):

        # pre compute some helpers
        cycle_to_add = int(len(batch_gen.train_examples)*batch_gen.adding_ratio)
        p_dis_ind_  = torch.triu_indices(self.num_classes, self.num_classes)
        ind = torch.triu_indices(self.num_classes, self.num_classes).unbind()
        p_dis_ind = p_dis_ind_[:,ind[0] != ind[1]].unbind()
        device = self.device
        self.model.to(device)

        # Load STW cache if using STW method and cache exists
        stw_output = {}
        if clip_active_method == "stw":
            stw_cache_path = "./stw_cache/"+dataset+"_"+task+".pt"
            if os.path.exists(stw_cache_path):
                stw_output=torch.load(stw_cache_path)
        # pseudo_output=torch.load("./pseudo_cache/"+dataset+"_"+task+"_seg3.pt")
        # print('LR:{}'.format(learning_rate))
        saving_records={}
        label_dict={}
        pseudo_label_dict={} # only when use stw
        saving_label_dict = {}
        train_time=0
        for active_cycle in range(num_active_cycle):
            # active_cycle=active_cycle+4
            ac_start=time.time()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
            # saving_records[active_cycle]=[batch_gen.selected_train_examples]
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3,threshold=5e-3,verbose=False)
            prev_epoch = epochs 
            prev_model_path = save_dir + "/i3d_asf_cl_p_"+str(self.num_decoders)+"dec_"+str(self.num_layers)+"layers_"+dataset+"_"\
                +str(task)+"_ac_"+str(active_cycle)+"_epoch-"+str(prev_epoch) +"_clip_"+clip_active_method+"_video_"+video_active_method+"_label_size_"+str(clip_size)+"_"+self.loss_type+"_adding_ratio_"+str(batch_gen.adding_ratio)+".model"
            print("start", task, "active cycle",active_cycle+1, "clip size", clip_size)
            if os.path.isfile(prev_model_path):
                print("load model:", prev_model_path)
                self.model.load_state_dict(torch.load(prev_model_path))
            # start adding label to new selected videos for this cycle 
            self.model.eval()
            while batch_gen.active_train_has_next(if_train=True, active_cycle=active_cycle):
                batch_input, batch_target, mask, vids, _ = batch_gen.my_active_next_batch(batch_size,if_train=True)   
                batch_f=batch_input.squeeze(0).t().cpu()
                # print(batch_f.size())
                if not vids[0] in label_dict:
                    # print(clip_size,type(clip_size))
                    if type(clip_size) is float:
                        num_clip_to_label = max(1, int(clip_size*batch_target.size()[1]))
                        if clip_active_method == "random":
                            # print("working on",vids[0],"random labeling")
                            new_batch_target = torch.ones_like(batch_target)* -100
                            label_index = torch.randperm(new_batch_target.size()[1], generator=self.generator)[:num_clip_to_label]
                            new_batch_target[:,label_index] = batch_target[:,label_index]
                            label_dict[vids[0]] = new_batch_target
                        elif clip_active_method == "split_rand":
                            new_batch_target = torch.ones_like(batch_target)* -100
                            label_index = torch.ones(num_clip_to_label, dtype=torch.long)* -100
                            num_equal_seg = int(batch_target.size()[1]/num_clip_to_label)
                            for i in range(num_clip_to_label):
                                temp_num=torch.randperm(num_equal_seg)[0]
                                label_index[i]=torch.arange(i*num_equal_seg,(i+1)*num_equal_seg,step=1)[temp_num]
                            new_batch_target[:,label_index] = batch_target[:,label_index]
                            label_dict[vids[0]] = new_batch_target
                        elif clip_active_method == "split_entropy":
                            new_batch_target = torch.ones_like(batch_target)* -100
                            label_index = torch.ones(num_clip_to_label, dtype=torch.long)* -100
                            num_equal_seg = int(batch_target.size()[1]/num_clip_to_label)
                            if active_cycle == 0:
                                for i in range(num_clip_to_label):
                                    temp_num=torch.randperm(num_equal_seg)[0]
                                    label_index[i]=torch.arange(i*num_equal_seg,(i+1)*num_equal_seg)[temp_num]
                                new_batch_target[:,label_index] = batch_target[:,label_index]
                                label_dict[vids[0]] = new_batch_target
                            else:
                                batch_input, mask = batch_input.to(device), mask.to(device)
                                entropy = entropy_mc_helper_2(self.model, batch_input, mask, self.num_classes, num_clip_to_label, device, forward_passes=50)
                                for i in range(num_clip_to_label):
                                    temp_range = torch.arange(i*num_equal_seg,(i+1)*num_equal_seg)
                                    label_index[i]=temp_range[torch.argmax(entropy[temp_range])]
                                
                                new_batch_target[:,label_index] = batch_target[:,label_index.cpu()]
                                label_dict[vids[0]] = new_batch_target    

                        elif clip_active_method == "equidistant":
                            # print("working on",vids[0],"equidistant labeling")
                            new_batch_target = torch.ones_like(batch_target)* -100
                            num_equal_seg = int(batch_target.size()[1]/num_clip_to_label)
                            label_index = torch.arange(start=0, end=batch_target.size()[1]-1, step=num_equal_seg)
                            if len(label_index) > num_clip_to_label:
                                # print("cutting additional clip...", len(label_index), num_clip_to_label)
                                label_index=label_index[:num_clip_to_label]
                            new_batch_target[:,label_index] = batch_target[:,label_index]
                            label_dict[vids[0]] = new_batch_target
                            # print(vids[0], len(label_index), new_batch_target)
                        elif clip_active_method == "stw":     
                            # if clip_size <=0.5:
                            if vids[0] in stw_output.keys():
                                new_batch_target = torch.ones_like(batch_target)* -100
                                temp_num = min(batch_target.size()[1], num_clip_to_label)
                                if temp_num in stw_output[vids[0]]['summary'].keys():
                                    # print("found existing..")
                                    label_index = stw_output[vids[0]]['summary'][temp_num]
                                    label_assign = stw_output[vids[0]]['assign'][temp_num]
                                    # print("stw label index", label_index)
                                    new_batch_target[:,label_index] = batch_target[:,label_index]
                                    # print("new_batch_target", new_batch_target)

                                    label_dict[vids[0]] = new_batch_target
                                else:
                                    new_batch_target = torch.ones_like(batch_target)* -100
                                    label_index, ass = parallel_singular_summarizing(batch_input.squeeze(0).t().cpu(), temp_num) 
                                    new_batch_target[:,label_index] = batch_target[:,label_index]
                                    label_dict[vids[0]] = new_batch_target
                            else:
                                new_batch_target = torch.ones_like(batch_target)* -100
                                temp_num = min(batch_target.size()[1], num_clip_to_label)
                                label_index, ass = parallel_singular_summarizing(batch_input.squeeze(0).t().cpu(), temp_num) 
                                new_batch_target[:,label_index] = batch_target[:,label_index]
                                label_dict[vids[0]] = new_batch_target
                            # else:
                                # new_batch_target = torch.ones_like(batch_target)* -100
                                # temp_num = min(batch_target.size()[1], num_clip_to_label)
                                # label_index, ass = parallel_singular_summarizing(batch_input.data.squeeze(0).t().cpu(), temp_num) 
                                # new_batch_target[:,label_index] = batch_target[:,label_index]
                                # label_dict[vids[0]] = new_batch_target

                            if self.loss_type == "mil":
                                # pseduo_batch_target = stw_label_helper(new_batch_target, label_assign, num_neighbor=1)
                                # pseduo_batch_target=pseudo_output[vids[0]][temp_num]
                                pseudo_label_dict[vids[0]]=pseduo_batch_target
                        elif clip_active_method == "greedy":
                            new_batch_target = torch.ones_like(batch_target)* -100
                            temp_num = min(batch_target.size()[1], num_clip_to_label)
                            batch_input, mask = batch_input.to(device), mask.to(device)
                            ps,fs = self.model(batch_input, mask)
                            greedy_assign = greedy_vectorized(fs[-1].data.squeeze(0).t().cpu(), temp_num, device="cpu")
                            label_index=torch.unique(greedy_assign)
                            # print("greedy label index", label_index)
                            new_batch_target[:,label_index] = batch_target[:,label_index]
                            label_dict[vids[0]] = new_batch_target
                        elif clip_active_method == "adapt_stw":     
                            temp_num = min(batch_target.size()[1], num_clip_to_label)
                            if active_cycle == 0:
                                new_batch_target = torch.ones_like(batch_target)* -100
                                temp_num = min(batch_target.size()[1], num_clip_to_label)
                                label_index, ass = parallel_singular_summarizing(batch_input.squeeze(0).t().cpu(), temp_num) 
                                new_batch_target[:,label_index] = batch_target[:,label_index]
                                label_dict[vids[0]] = new_batch_target
                            else:
                                new_batch_target = torch.ones_like(batch_target)* -100
                                batch_input, mask = batch_input.to(device), mask.to(device)
                                ps,fs = self.model(batch_input, mask)
                                label_index, ass = parallel_singular_summarizing(fs[-1].data.squeeze(0).t().cpu(), temp_num) 
                                new_batch_target[:,label_index] = batch_target[:,label_index]
                                label_dict[vids[0]] = new_batch_target
                                pseduo_batch_target = stw_label_helper(new_batch_target, ass, num_neighbor=1)
                                pseudo_label_dict[vids[0]]=pseduo_batch_target
                            # else:
                            # label_index, ass = parallel_singular_summarizing(batch_input.squeeze(0).t(), temp_num) 
                        elif clip_active_method == "entropy_mc":
                            # use random at first iteration
                            new_batch_target = torch.ones_like(batch_target)* -100
                            if active_cycle == 0:
                                label_index = torch.randperm(new_batch_target.size()[1], generator=self.generator)[:num_clip_to_label]
                                new_batch_target[:,label_index] = batch_target[:,label_index]
                                label_dict[vids[0]] = new_batch_target
                            else:
                                batch_input, mask = batch_input.to(device), mask.to(device)
                                label_index = entropy_mc_helper(self.model, batch_input, mask, self.num_classes, num_clip_to_label, device, forward_passes=50)
                                new_batch_target[:,label_index.cpu()] = batch_target[:,label_index.cpu()]
                                label_dict[vids[0]] = new_batch_target    
                        else:
                            print("Please given correct labeling method")
 

                    else:
                        if clip_active_method == "adapt_stw":     
                            temp_num = min(batch_target.size()[1], clip_size)
                            if active_cycle == 0:
                                new_batch_target = torch.ones_like(batch_target)* -100
                                label_index, ass = parallel_singular_summarizing(batch_input.squeeze(0).t().cpu(), temp_num) 

                                # label_index = stw_output[vids[0]]['summary'][temp_num]
                                # label_assign = stw_output[vids[0]]['assign'][temp_num]
                                new_batch_target[:,label_index] = batch_target[:,label_index]
                                label_dict[vids[0]] = new_batch_target
                                # pseduo_batch_target = stw_label_helper(new_batch_target, label_assign, num_neighbor=1)
                                # pseudo_label_dict[vids[0]]=pseduo_batch_target
                            else:
                                new_batch_target = torch.ones_like(batch_target)* -100
                                batch_input, mask = batch_input.to(device), mask.to(device)
                                ps,fs = self.model(batch_input, mask)
                                label_index, ass = parallel_singular_summarizing(fs[-1].data.squeeze(0).t().cpu(), temp_num) 
                                new_batch_target[:,label_index] = batch_target[:,label_index]
                                label_dict[vids[0]] = new_batch_target
                                # pseduo_batch_target = stw_label_helper(new_batch_target, ass, num_neighbor=1)
                                # pseudo_label_dict[vids[0]]=pseduo_batch_target
                        elif clip_active_method == "stw":     
                            temp_num = min(batch_target.size()[1], clip_size)
                            
                            new_batch_target = torch.ones_like(batch_target)* -100
                            label_index, ass = parallel_singular_summarizing(batch_input.squeeze(0).t().cpu(), temp_num) 
                            new_batch_target[:,label_index] = batch_target[:,label_index]
                            label_dict[vids[0]] = new_batch_target



            if self.loss_type == "mil":
                saving_label_dict[active_cycle] = pseudo_label_dict
            else:
                saving_label_dict[active_cycle] = label_dict

            
            print("labeling finished, labeled %f videos... start training" % int(len(label_dict.keys())))
            # edge_lists=[]
            # for video in list(label_dict.keys()):
            #     edges,_ = graph_helper(label_dict[video].squeeze(0).numpy())
            #     edge_lists=edge_lists+edges
            # G1=nx.DiGraph()
            # # print(edge_lists)
            # G1.add_edges_from(edge_lists)
            # g_edges1 = G1.number_of_edges()
            # g_nodes1 = G1.number_of_nodes()
            # print("active cycle:", active_cycle)
            # print("g_edges", g_edges1, "g_nodes",g_nodes1)

            batch_gen.active_reset(if_train=True)
            self.model.train()
            for epoch in range(epochs):
                time_start = time.time()
                epoch_loss = 0
                epoch_cl_loss = 0
                correct = 0
                total = 0
                ce1_show = 0
                ce2_show = 0
                p_var_show = 0
                p_mean_show = 0
                
                while batch_gen.active_train_has_next(if_train=True, active_cycle=active_cycle):
                    batch_input, batch_target, mask, vids, _ = batch_gen.my_active_next_batch(batch_size,if_train=True)
                    new_batch_target = label_dict[vids[0]]
                    batch_input, batch_target, new_batch_target, mask = batch_input.to(device),batch_target.to(device), new_batch_target.to(device), mask.to(device)
                    # create helper 
                    new_label = new_batch_target.clone().view(-1)
                    selected_label = new_label[new_label>-100]
                    arranged_prototype = self.model.prototype.t()[selected_label]
                    swapped_label = torch.arange(len(selected_label)).to(device)[selected_label>-100]
                    optimizer.zero_grad()
                    ps,fs = self.model(batch_input, mask)
                    loss = 0
                    cl_loss = 0
                    for p in ps:
                        loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), new_batch_target.view(-1))
                        loss += 0.15 * torch.mean(torch.clamp(
                            self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                            max=16) * mask[:, :, 1:])    
                    # symmetric Info NCE
                    if self.loss_type =="asf_only":
                        cl_loss=torch.zeros(1).to(device)
                        ce1=torch.zeros(1).to(device)
                        ce2=torch.zeros(1).to(device)
                        var=torch.zeros(1).to(device)
                        mean=torch.zeros(1).to(device)
                    else:
                        for f in fs:
                            ce1, ce2, var, mean = p_loss(f, new_label, swapped_label, arranged_prototype, self.model.prototype, p_dis_ind, device)
                            if self.loss_type == "info":
                                cl_loss = cl_loss + ce1 + ce2 +var + mean
                            elif self.loss_type == "ce1_and_reg":
                                ce2=torch.zeros(1).to(device)
                                cl_loss = cl_loss + ce1 +var + mean
                            elif self.loss_type == "ce1_and_var":
                                ce2=torch.zeros(1).to(device)
                                mean=torch.zeros(1).to(device)
                                cl_loss = cl_loss + ce1 +var
                            elif self.loss_type == "ce1_and_mean":
                                ce2=torch.zeros(1).to(device)
                                var=torch.zeros(1).to(device)
                                cl_loss = cl_loss + ce1 +mean
                            elif self.loss_type == "ce2_and_reg":
                                ce1=torch.zeros(1).to(device)
                                cl_loss = cl_loss + ce2 +var + mean
                            elif self.loss_type == "ce_only":
                                var=torch.zeros(1).to(device)
                                mean=torch.zeros(1).to(device)
                                cl_loss = cl_loss + ce1 + ce2
                            elif self.loss_type == "ce1_only":
                                ce2=torch.zeros(1).to(device)
                                var=torch.zeros(1).to(device)
                                mean=torch.zeros(1).to(device)
                                cl_loss = cl_loss + ce1
                            elif self.loss_type == "ce2_only":
                                ce1=torch.zeros(1).to(device)
                                var=torch.zeros(1).to(device)
                                mean=torch.zeros(1).to(device)
                                cl_loss = cl_loss + ce2


                    loss=loss+0.1*cl_loss
                    asf_loss = loss.item()
                    epoch_cl_loss += 0.1*cl_loss.item()
                    epoch_loss += loss.item()
                    ce1_show += 0.1*ce1.item()
                    ce2_show += 0.1*ce2.item()
                    p_var_show += 0.1*var.item()
                    p_mean_show += 0.1*mean.item()
                    loss.backward()
                    optimizer.step()
                    _, predicted = torch.max(ps.data[-1], 1)
                    correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                    total += torch.sum(mask[:, 0, :]).item()
                scheduler.step(epoch_loss)
                num_trained = batch_gen.selected_train_index
                batch_gen.active_reset(if_train=True)

                time_end = time.time()
                # if (epoch + 1) % 20==0:
                #     print("[active cycle: %d epoch %d]: epoch loss =%f, asf loss = %f,epoch cl loss = %f, ce1 = %f, ce2 = %f, p_var = %f, p_mean = %f, train acc = %f, seen %s videos, time = %f " %
                #            (active_cycle+1, epoch + 1,          epoch_loss / len(batch_gen.selected_train_examples),
                #                                                 asf_loss / len(batch_gen.selected_train_examples),
                #                                                 epoch_cl_loss / len(batch_gen.selected_train_examples),
                #                                                 ce1_show /len(batch_gen.selected_train_examples),
                #                                                 ce2_show / len(batch_gen.selected_train_examples),
                #                                                 p_var_show /len(batch_gen.selected_train_examples),
                #                                                 p_mean_show / len(batch_gen.selected_train_examples),
                #                                     float(correct) / total,
                #                                     int(num_trained),
                #                                     (time_end-time_start)))
    
                # if (epoch + 1) % 2 == 0:
                    # self.test(batch_gen, epoch, batch_gen.actions_dict, False)
                if (epoch + 1) % 40 == 0:
                    self.test(batch_gen, epoch, batch_gen.actions_dict, False)
                    torch.save(self.model.state_dict(), save_dir + "/i3d_asf_cl_p_"+str(self.num_decoders)+"dec_"+str(self.num_layers)+"layers_"+dataset+"_"+\
                            str(task)+"_ac_"+str(active_cycle+1)+"_epoch-" + str(epoch + 1) +"_clip_"+clip_active_method+"_video_"+video_active_method+"_label_size_"+\
                                str(clip_size)+"_"+self.loss_type+"_adding_ratio_"+str(batch_gen.adding_ratio)+"_seed_"+str(self.seed)+".model")
                    # torch.save(optimizer.state_dict(), save_dir + "/i3d_asf_cl_p_"+str(self.num_decoders)+"dec_"+str(self.num_layers)+"layers_"+dataset+"_"+\
                            # str(task)+"_ac_"+str(active_cycle+1)+"_epoch-" + str(epoch + 1) +"_clip_"+clip_active_method+"_video_"+video_active_method+"_label_size_"+str(clip_size)+"_"+self.loss_type+"_adding_ratio_"+str(batch_gen.adding_ratio)+".opt")
            
                    # print("checking learning performance...")
                    # prototype = self.model.prototype.t().detach().cpu()
                    # print("prototype dis:", dis_sim_matrix(prototype, prototype))
                    # while batch_gen.active_train_has_next(if_train=True, active_cycle=active_cycle):
                    #     batch_input, batch_target, mask, vids, _ = batch_gen.my_active_next_batch(batch_size,if_train=True)
                    #     if vids[0] in label_dict.keys():
                    #         print(vids[0])
                    #         ps,fs = self.model(batch_input, mask)
                    #         ann_label = label_dict[vids[0]]
                    #         video_features = fs[-1].data.squeeze(0).t().cpu() # need to check dim
                    #         p_dis = dis_sim_matrix(video_features, prototype)
                    #         for _ in range(len(p_dis)):
                    #             if ann_label[_] >0:
                    #                 print(ann_label[_], p_dis[_])
                    # batch_gen.active_reset(if_train=True)


            
            if active_cycle +1 < num_active_cycle:
                print("training finished, selecting videos...")
                self.model.eval()
                label_dict_vids = list(label_dict.keys())
                clean_label_dict = {}
                cur_records={}
                for vid in label_dict_vids:
                        _, clean_label = graph_helper(label_dict[vid].squeeze(0))
                        if len(clean_label) > 0 and not any(clean_label==label for label in clean_label_dict.values()):
                            clean_label_dict[vid]=clean_label
                active_record=[]
                pred_records ={}
                if len(clean_label_dict.keys())>0:
                    if video_active_method == "drop_dtw":
                        if self.loss_type == "asf_only":
                            proto_dict ={}
                            for _ in range(self.num_classes):
                                proto_dict[_]=[]
                            prototype = torch.rand(self.num_classes, self.num_f_maps)
                            # using asformer features center as prototype
                            while batch_gen.active_train_has_next(if_train=True, active_cycle=active_cycle):
                                batch_input, batch_target, mask, vids, _ = batch_gen.my_active_next_batch(batch_size,if_train=True)
                                new_batch_target = label_dict[vids[0]].view(-1)
                                batch_input, batch_target, mask = batch_input.to(device),batch_target.to(device), mask.to(device)
                                ps,fs = self.model(batch_input, mask)
                                video_features = fs[-1].data.squeeze(0).t().cpu()
                                for _ in range(self.num_classes):
                                    for cur_clip in range(len(new_batch_target)):
                                        if new_batch_target[cur_clip] == _:
                                            proto_dict[_].append(video_features[cur_clip].view(1,self.num_f_maps))
                            batch_gen.active_reset(if_train=True)
                            # build prototype 
                            for _ in range(self.num_classes):
                                if len(proto_dict[_])>0:
                                    prototype_conbine = torch.cat(proto_dict[_],dim=0)
                                    prototype[_] = torch.mean(prototype_conbine,dim=0)

                        else:
                            prototype = self.model.prototype.t().detach().cpu()



                        step_feature_dict = {}
                        for key in clean_label_dict.keys():
                            clean_label=clean_label_dict[key]
                            proto_seq = []
                            for label in clean_label:
                                proto_seq.append(prototype[label].view(1,prototype.size()[1]))
                            proto_seq = torch.cat(proto_seq, dim=0)
                            step_feature_dict[key]=proto_seq 

                        while batch_gen.active_train_has_next(if_train=False, active_cycle=None):
                            batch_input, batch_target, mask, vids, _ = batch_gen.my_active_next_batch(batch_size,if_train=False)
                            batch_input, batch_target, mask = batch_input.to(device),batch_target.to(device), mask.to(device)
                            ps,fs = self.model(batch_input, mask)
                            video_features = fs[-1].data.squeeze(0).t().cpu() # need to check dim
                            # print("feature size", video_feature.size())
                            temp_score = 10000
                            dropdtw_records = {}
                            for key in step_feature_dict.keys():
                                step_features=step_feature_dict[key]
                                p_clip_dis = -torch.log(F.softmax(sim_matrix(step_features, video_features,gamma=0.07),dim=0))
                                drop_ = torch.quantile(p_clip_dis, q=0.3)
                                drop_cost = torch.ones(len(video_features))*drop_
                                min_cost, path, x_dropped = drop_dtw(p_clip_dis.numpy(), drop_cost.numpy())
                                dropdtw_records[key] = [ min_cost, path, x_dropped]
                                # print(min_cost)
                                if min_cost < temp_score:
                                    temp_score=min_cost # keep lowest score, so if the lowest score is high, then it has minimal match with current all seqs.
                            active_record.append((vids[0], temp_score/len(video_features)))
                            cur_records[vids[0]]=dropdtw_records
                        saving_records[active_cycle]=cur_records
                        batch_gen.active_reset(if_train=False)
                        active_record_df = pd.DataFrame.from_records(active_record, columns=['vid', 'min_score'])
                        sorted_df = active_record_df.sort_values(by=['min_score'], ascending=False)
                        batch_gen.selected_train_examples = batch_gen.selected_train_examples+list(sorted_df.values[:cycle_to_add,0])
                        batch_gen.unselected_train_examples = list(set(batch_gen.unselected_train_examples)-set(list(sorted_df.values[:cycle_to_add,0])))
                    elif video_active_method == "random":
                        print("Use random pick next videos")
                        saving_records[active_cycle]=["Use random"]
                        # since we already shuffled the train examples
                        batch_gen.selected_train_examples = batch_gen.selected_train_examples+batch_gen.unselected_train_examples[:cycle_to_add]
                        batch_gen.unselected_train_examples = batch_gen.unselected_train_examples[cycle_to_add:]
                else:
                    print("No valid label seqs. at this active cycle..., randomly pick next two videos")
                    saving_records[active_cycle]=["No valid label seqs"]
                    # since we already shuffled the train examples
                    batch_gen.selected_train_examples = batch_gen.selected_train_examples+batch_gen.unselected_train_examples[:cycle_to_add]
                    batch_gen.unselected_train_examples = batch_gen.unselected_train_examples[cycle_to_add:]
            ac_end=time.time()
            train_time=train_time+ac_end-ac_start
            print("this active training is finished, time used=",ac_end-ac_start, "total time used:",train_time)


        # torch.save(saving_label_dict, save_dir + "/label_i3d_selected_asf_cl_p_"+str(self.num_decoders)+"dec_"+str(self.num_layers)+"layers_"+dataset+"_"+str(task)+"_"+clip_active_method+"_"+video_active_method+"_clip_size_"+str(clip_size)+"_"+self.loss_type+".pt")
        # torch.save(saving_records, save_dir + "/saving_records_i3d_selected_asf_cl_p_"+str(self.num_decoders)+"dec_"+str(self.num_layers)+"layers_"+dataset+"_"+str(task)+"_"+clip_active_method+"_"+video_active_method+"_clip_size_"+str(clip_size)+"_"+self.loss_type+".pt")

        print("this task active info saved!")

                
    def test(self, batch_gen, epoch, actions_dict, if_train=False):
        device = self.device
        self.model.eval()
        clip_scores = ScoreMeter(id2class_map=actions_dict)
        frame_pred_list = []
        frame_label_list = []
        if_warp = False  # When testing, always false
        infer_avg=0
        with torch.no_grad():
            while batch_gen.has_next(if_train=False):
                batch_input, batch_target, mask, vids, frame_labels = batch_gen.my_next_batch(1, if_train=False, if_warp=False)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                infer_start=time.time()
                ps,fs = self.model(batch_input, mask)
                _, predicted = torch.max(ps.data[-1], 1)
                infer_finish=time.time()
                infer_avg=infer_avg+(infer_finish-infer_start)
                predicted = predicted.squeeze(0).cpu().numpy()
                clip_pred_to_frame = np.ones([len(predicted)*batch_gen.frames_per_seg], dtype=np.int32) 
                for _ in range(len(predicted)):
                    temp1 = np.arange(start=_*batch_gen.frames_per_seg, stop=(_+1)*batch_gen.frames_per_seg)
                    clip_pred_to_frame[temp1]=predicted[_]
                frame_pred_list.append(clip_pred_to_frame)
                frame_label_list.append(frame_labels)
                clip_scores.update(clip_pred_to_frame, frame_labels)
        [precision, recall, fscore, acc, acc_bg, num_bg_pred, num_bg, num_frames] = framewise_eval_pure(frame_pred_list,frame_label_list )
        clip_acc, clip_edit_score, clip_f1s = clip_scores.get_scores()
        print("testing...Acc: %.4f  precision: %.4f recall: %.4f fscore: %.4f Edit: %4f  F1@10: %4f, F1@25: %4f, F1@50: %4f" % (clip_acc, precision, recall, fscore, clip_edit_score, clip_f1s[0],clip_f1s[1],clip_f1s[2] ))
        # print("videos infered:",len(batch_gen.test_examples),"avg infer speed", infer_avg/len(batch_gen.test_examples))
        batch_gen.reset(if_train=False)
        self.model.train()

    

    def my_predict_eval(self, model_dir, dataset, task, batch_gen, action_dict, epoch, if_train):
        device = self.device
        self.model.eval()
        clip_scores = ScoreMeter(id2class_map=action_dict)
        frame_pred_list = []
        frame_label_list = []
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(
                            model_dir + "/i3d_asf_cl_p_"+str(self.num_decoders)+"dec_"+str(self.num_layers)+"layers_"+dataset+"_"+\
                            str(task)+"_epoch-" + str(epoch)+"_"+self.loss_type+".model"))
            time_start = time.time()
            while batch_gen.has_next(if_train):
                batch_input, batch_target, mask, vids, frame_labels = batch_gen.my_next_batch( batch_size=1, if_train=if_train, if_warp=False)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                ps,fs = self.model(batch_input, mask)
                _, predicted = torch.max(ps.data[-1], 1)
                predicted = predicted.squeeze(0).cpu().numpy()
                clip_pred_to_frame = np.ones([len(predicted)*batch_gen.frames_per_seg], dtype=np.int32) 
                for _ in range(len(predicted)):
                    temp1 = np.arange(start=_*batch_gen.frames_per_seg, stop=(_+1)*batch_gen.frames_per_seg)
                    clip_pred_to_frame[temp1]=predicted[_]
                frame_pred_list.append(clip_pred_to_frame)
                frame_label_list.append(frame_labels)
                clip_scores.update(clip_pred_to_frame, frame_labels)
        [precision, recall, fscore, acc, acc_bg, num_bg_pred, num_bg, num_frames] = framewise_eval_pure(frame_pred_list,frame_label_list )
        clip_acc, clip_edit_score, clip_f1s = clip_scores.get_scores()
        print("Acc: %.4f  Edit: %4f  F1@10: %4f, F1@25: %4f, F1@50: %4f" % (clip_acc, clip_edit_score, clip_f1s[0],clip_f1s[1],clip_f1s[2] ))
        batch_gen.reset(if_train)
        return [[dataset, task, self.loss_type, epoch, self.num_layers]+[precision, recall, fscore, acc, acc_bg, num_bg_pred, num_bg, num_frames]+[clip_edit_score] + clip_f1s]
        

    def my_drop_dtw_predict_eval(self, model_dir, dataset, task, batch_gen, if_train, num_active_cycle, clip_active_method, video_active_method):
        device = self.device
        self.model.eval()
        print("clip selection:",clip_active_method)
        print("model prototype:", self.model.prototype.size())
        dis_record_dict = {}
        for active_cycle in range(num_active_cycle):
            # clip_scores = ScoreMeter(id2class_map=action_dict)
            active_cycle=active_cycle+1
            dis_record_dict[active_cycle]={}
            epoch_list = [80]
            for epoch in epoch_list:
                model_path = model_dir + "/i3d_asf_cl_p_"+str(self.num_decoders)+"dec_"+str(self.num_layers)+"layers_"+dataset+"_"+\
                                str(task)+"_ac_"+str(active_cycle)+"_epoch-" + str(epoch) +"_"+clip_active_method+"_"+video_active_method+".model"
                stw_seq_path = model_dir+ "/label_i3d_selected_asf_cl_p_"+str(self.num_decoders)+"dec_"+str(self.num_layers)+"layers_"+dataset+"_"+str(task)+"_"+clip_active_method+"_"+video_active_method+".pt"
                stw_seq_dict = torch.load(stw_seq_path)
                clean_label_list = {}
                print("active_cycle",active_cycle,"epoch", epoch)
                # for vid in stw_seq_dict.keys():
                #     # print(vid)
                #     _, clean_label = graph_helper(stw_seq_dict[vid].squeeze(0))
                #     if len(clean_label) > 0:
                #         clean_label_list[vid]=clean_label
                        
                if os.path.isfile(model_path):
                    with torch.no_grad():
                        self.model.to(device)
                        self.model.load_state_dict(torch.load(model_path))
                        prototype = self.model.prototype.t().detach().cpu()
                        # print("prototype dis:", dis_sim_matrix(prototype, prototype))
                        step_feature_list = {}
                        for vid in clean_label_list.keys():
                            proto_seq = []
                            clean_label = clean_label_list[vid]
                            for label in clean_label:
                                proto_seq.append(prototype[label-1].view(1,prototype.size()[1]))
                            proto_seq = torch.cat(proto_seq, dim=0)
                            step_feature_list[vid]=proto_seq
                        batch_gen.selected_train_examples=batch_gen.train_examples[:active_cycle*2]
                        while batch_gen.active_train_has_next(if_train, active_cycle-1):
                            batch_input, batch_target, mask, vids, frame_labels = batch_gen.my_active_next_batch( batch_size=1, if_train=if_train, if_warp=False)
                            batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)                
                            new_label = batch_target.squeeze(0).cpu()
                            non_bg_ind = new_label!=0
                            print(vids[0])
                            dis_record_dict[active_cycle][vids[0]]=[]
                            ann_label = stw_seq_dict[vids[0]].squeeze(0).cpu()
                            labeled_action = {}
                            labeled_bg ={}
                            unlabeled = {}
                            ps,fs = self.model(batch_input, mask)
            #               # asf classification head pred
                            # _, asf_predicted = torch.max(ps.data[-1], 1)
                            # asf_predicted = asf_predicted.squeeze(0).cpu()
                            # print("asf:", asf_predicted)
            #                # drop dtw pred
                            video_features = fs[-1].data.squeeze(0).t().cpu() # need to check dim
                            p_dis = dis_sim_matrix(video_features, prototype, gamma=1)
                            print("labeled action:")
                            temp_labeled_action_p = {}
                            temp_labeled_action_others = {}
                            temp_proto_ind = torch.arange(p_dis.size()[1])
                            for _ in range(len(p_dis)):
                                if ann_label[_]>0:
                                    # print(ann_label[_], p_dis[_])
                                    if not ann_label[_].item() in temp_labeled_action_p.keys():
                                        temp_labeled_action_p[ann_label[_].item()] = []
                                        temp_labeled_action_others[ann_label[_].item()] =[]
                                        # print(temp_labeled_action[ann_label[_].item()])
                                    temp_labeled_action_p[ann_label[_].item()].append(p_dis[_, ann_label[_]-1])
                                    temp_labeled_action_others[ann_label[_].item()].append(torch.min(p_dis[_,temp_proto_ind!=ann_label[_]-1]))

                            for key in temp_labeled_action_p.keys():
                                print("action", key, "to proto:", sum(temp_labeled_action_p[key])/len(temp_labeled_action_p[key]),
                                    "to others:", sum(temp_labeled_action_others[key])/len(temp_labeled_action_others[key]))

                            # print("labeled background:")
                            temp_labeled_background = []
                            for _ in range(len(p_dis)):
                                if ann_label[_]==0:
                                    temp_labeled_background.append(torch.min(p_dis[_]))
                            print("labeled background:", sum(temp_labeled_background)/len(temp_labeled_background))

                            print("non labeled clips:")
                            temp_unlabeled_action_p = {}
                            temp_unlabeled_action_others = {}
                            for _ in range(len(p_dis)):
                                if ann_label[_]==-100 and new_label[_]>0:
                                    if not new_label[_].item() in temp_unlabeled_action_p.keys():
                                        temp_unlabeled_action_p[new_label[_].item()] = []
                                        temp_unlabeled_action_others[new_label[_].item()] =[]
                                        # print(temp_labeled_action[ann_label[_].item()])
                                    temp_unlabeled_action_p[new_label[_].item()].append(p_dis[_, new_label[_]-1])
                                    temp_unlabeled_action_others[new_label[_].item()].append(torch.min(p_dis[_,temp_proto_ind!=new_label[_]-1]))
                            for key in temp_unlabeled_action_p.keys():
                                print("action", key, "to proto:", sum(temp_unlabeled_action_p[key])/len(temp_unlabeled_action_p[key]),
                                    "to others:", sum(temp_unlabeled_action_others[key])/len(temp_unlabeled_action_others[key]))
                        batch_gen.active_reset(if_train=True)



                            # print("labeled background:")
                            # for _ in range(len(p_dis)):
                            #     if ann_label[_] >0: 
                            #         print(ann_label[_], p_dis[_])
                            # print("non labeled:")
                            # for _ in range(len(p_dis)):
                            #     if ann_label[_] == -100:
                            #         print(p_dis[_])

                                # if not ann_label[_] in labeled_action.keys():
                                #     labeled_action[ ann_label[_]]=[]
                                # labeled_action[ ann_label[_]].append(p_dis[_][ann_label[_]-1])

                        # print("action label:", ann_label[ann_label>0])
                        # print("clip to p dis:", dis_sim_matrix(video_features, prototype)[ann_label>0])
                        # for seq in step_feature_list.keys():
                        #     clean_label = clean_label_list[seq]
                        #     cur_seq_feat = step_feature_list[seq]
                        #     p_clip_dis = dis_sim_matrix(cur_seq_feat, video_features)
                        #     # _min, median, max = torch.min(p_clip_dis), torch.median(p_clip_dis), torch.max(p_clip_dis)
                        #     # print(_min, median, max)
                        #     drop_ = torch.quantile(p_clip_dis, q=0.05)
                        #     drop_cost = torch.ones(len(video_features))*drop_
                        #     min_cost, path, x_dropped = drop_dtw(p_clip_dis.numpy(), drop_cost.numpy())
                        #     # remove drop in path
                        #     to_rm = []
                        #     for pair_ind in range(len(path)):
                        #         cur_pair = path[pair_ind]
                        #         for ind in x_dropped:
                        #             if cur_pair[1] == ind:
                        #                 to_rm.append(cur_pair)
                        #     new_path = list(set(path)-set(to_rm))
                        #     pred = torch.zeros(len(video_features))
                        #     for new_pair in new_path:
                        #         if new_pair[0] >0:
                        #             pred[new_pair[1]-1]=clean_label[new_pair[0]-1]
                        #     print("cur seq:",seq, clean_label_list[seq], "drop cost", drop_, "min cost", min_cost)
                        #     print("pred:", pred.to(torch.long))
                        #     print("c_mat:", p_clip_dis[:,pred>0].t())

                            # for _ in range(video_features):
                            #     if ann_label[_] >0:
                            #         print(ann_label[_], p_clip_dis)
        # dtw_pred = torch.zeros(len(new_label),self.num_classes)
        #                 dtw_buffer = torch.zeros_like(dtw_pred)
        #                 helper_index = torch.arange(len(new_label))
        #                 for step_seq_ind in range(len(step_feature_list)):
        #                     clean_label = clean_label_list[step_seq_ind]
        #                     step_features = step_feature_list[step_seq_ind]
        #                     p_clip_dis = dis_sim_matrix(step_features, video_features)
        #                     drop_cost = np.ones(len(video_features))*0.8
        #                     min_cost, path, x_dropped = drop_dtw(p_clip_dis.numpy(), drop_cost)
        #                     # remove drop in path
        #                     to_rm = []
        #                     for pair_ind in range(len(path)):
        #                         cur_pair = path[pair_ind]
        #                         for ind in x_dropped:
        #                             if cur_pair[1] == ind:
        #                                 to_rm.append(cur_pair)
        #                     new_path = list(set(path)-set(to_rm))
        #                     # print(step_seq_ind,"th stw seq.,", clean_label )
        #                     pred = torch.zeros(len(video_features))
        #                     for new_pair in new_path:
        #                         if new_pair[0] >0:
        #                             pred[new_pair[1]-1]=clean_label[new_pair[0]-1]
        #                     dtw_buffer[helper_index,pred.long()]=1
        #                     dtw_pred=dtw_pred+dtw_buffer
        #                     # print(" match ratio", len(new_path)/len(path), "recall",[sum(pred[non_bg_ind]==new_label[non_bg_ind]).float().item()/sum(non_bg_ind.float()).item(),],",")
                        
        #                 # print(dtw_pred)
        #                 dtw_final_pred = torch.zeros(len(new_label)).long()
                        # p_dis = dis_sim_matrix(video_features, prototype)
                        # for _ in range(len(p_dis)):
                        #     if ann_label[_] >0:
                        #         print(ann_label[_], p_dis[_])

        #                 temp_proto_ind =torch.arange(self.num_classes-1)
        #                 for _ in range(len(new_label)):
        #                     cur_row = dtw_pred[_,:]
        #                     cur_ind =temp_proto_ind[cur_row[1:]>2]
        #                     if len(cur_ind)>0:
        #                         min_val = torch.min(p_dis[_,cur_ind])
        #                         min_ind = torch.argmin(p_dis[_,cur_ind])
        #                         # print(min_val, min_ind)
        #                         if min_val < 0.65:
        #                             dtw_final_pred[_]= cur_ind[min_ind]+1
        #                 # dtw_final_pred = torch.argmax(dtw_pred,dim=1)
        #                 # print("dtw")
        #                 # print(dtw_final_pred)
        #                 # print("asf")
        #                 # print(asf_predicted)
        #                 # print("asf recall", [sum(asf_predicted[asf_predicted!=0]==new_label[asf_predicted!=0]).float().item()/sum(non_bg_ind.float()).item(),])
        #                 # print("dtw recall", [sum(dtw_final_pred[dtw_final_pred!=0]==new_label[dtw_final_pred!=0]).float().item()/sum(non_bg_ind.float()).item(),])
        #                 if torch.count_nonzero(new_label) >0:
        #                     asf_recall = asf_recall+sum(asf_predicted[asf_predicted!=0]==new_label[asf_predicted!=0]).float().item()/sum(non_bg_ind.float()).item()
        #                     dtw_recall = dtw_recall+sum(dtw_final_pred[dtw_final_pred!=0]==new_label[dtw_final_pred!=0]).float().item()/sum(non_bg_ind.float()).item()
        #                     count =count+1
        #             dtw_overall = dtw_recall/count
        #             asf_overall = asf_recall/count
        #             print("dtw overall:", dtw_overall)
        #             print("asf overall:", asf_overall)
        # return dtw_overall, asf_overall
                            # # compute original cost
                            # o_cost = 0
                            # count = 0
                            # for pair_ind in range(len(path)):
                            #     cur_pair = path[pair_ind]
                            # for cur_pair in new_path:
                            #     if new_pair[0] >0:
                            #     print(p_clip_dis.size())
                            #     if cur_pair[0] >0:
                            #         print(cur_pair, p_clip_dis[cur_pair[0]-1, cur_pair[1]-1])
                            #         o_cost = o_cost +p_clip_dis[cur_pair[0]-1, cur_pair[1]-1]
                            #         count=count+1
                            #     else:
                            #         o_cost =o_cost+0.75
                            # o_cost = torch.exp(o_cost)
                            # print("dis size:", p_clip_dis.size())
                            # print("pred:",pred)
                            # print("labe:", new_label)
                            # num_actions = sum(non_bg_ind.float())
                            # print((num_actions/len(new_label)).item(), count/len(new_label), [sum(pred[non_bg_ind]==new_label[non_bg_ind]).float().item()/num_actions.item(),o_cost.item()/count],",")
                            # print([sum(pred[non_bg_ind]==new_label[non_bg_ind]).float().item()/num_actions.item(),o_cost.item()/count],",")


                        # _, predicted = torch.max(ps.data[-1], 1)
                        # predicted = predicted.squeeze(0).cpu().numpy()
                #         clip_pred_to_frame = np.ones([len(predicted)*32], dtype=np.int32) 
                #         for _ in range(len(predicted)):
                #             temp1 = np.arange(start=_*32, stop=(_+1)*32)
                #             clip_pred_to_frame[temp1]=predicted[_]
                #         frame_pred_list.append(clip_pred_to_frame)
                #         frame_label_list.append(frame_labels)
                #         clip_scores.update(clip_pred_to_frame, frame_labels)
                # [precision, recall, fscore, acc, acc_bg, num_bg_pred, num_bg, num_frames] = framewise_eval_pure(frame_pred_list,frame_label_list )
                # clip_acc, clip_edit_score, clip_f1s = clip_scores.get_scores()
                # print("Acc: %.4f  Edit: %4f  F1@10: %4f, F1@25: %4f, F1@50: %4f" % (clip_acc, clip_edit_score, clip_f1s[0],clip_f1s[1],clip_f1s[2] ))
                # batch_gen.reset(if_train)
                # return [[dataset, task, epoch, self.num_layers]+[precision, recall, fscore, acc, acc_bg, num_bg_pred, num_bg, num_frames]+[clip_acc, clip_edit_score] + clip_f1s]

    
    def get_features(self, model_dir, dataset, task, batch_gen,action_dict, epoch, if_train):
        device=self.device
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            # "/asf_"+str(self.num_decoders)+"dec_"+str(self.num_layers)+"layers_"+dataset+"_"+str(task)+"_epoch-" + str(epoch + 1) + ".model")
            self.model.load_state_dict(torch.load(model_dir +"/asf_cl_p_"+str(self.num_decoders)+"dec_"+str(self.num_layers)+"layers_"+dataset+"_"+str(task)+"_epoch-" + str(epoch) + ".model"))
            extracted_feature = []
            labels = []
            while batch_gen.has_next(if_train):
                batch_input, batch_target, mask, vids, frame_labels = batch_gen.my_next_batch( batch_size=1, if_train=if_train, if_warp=False)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                # feed into data and get features from asformer
                ps,fs = self.model(batch_input, mask)       
                feature=fs[-1].squeeze(0).t()
                batch_target=batch_target.squeeze(0)
                extracted_feature.append(feature)
                labels.append(batch_target)
            extracted_feature.append(self.model.prototype.t())
            extracted_feature=torch.cat(extracted_feature,dim=0)

            label=torch.cat(labels)
            print(extracted_feature.size())
            print(label.size())
        return extracted_feature, label #, self.model.prototype.t()
    
    def my_active_predict_eval(self, model_dir, dataset, task, batch_gen, action_dict, if_train, epoch, num_active_cycle, clip_active_method, video_active_method, clip_size):
        device = self.device
        # self.model.eval()
        all_results = []
        for active_cycle in range(num_active_cycle):
            clip_scores = ScoreMeter(id2class_map=action_dict)
            active_cycle=active_cycle+1
            frame_pred_list = []
            frame_label_list = []
            # previous name
            model_path1 = model_dir + "/i3d_asf_cl_p_"+str(self.num_decoders)+"dec_"+str(self.num_layers)+"layers_"+dataset+"_"+\
                            str(task)+"_ac_"+str(active_cycle)+"_epoch-" + str(epoch) +"_clip_"+clip_active_method+"_video_"+video_active_method+"_label_size_"+str(clip_size)+"_"+self.loss_type+".model"
            
            model_path2 = model_dir + "/i3d_asf_cl_p_"+str(self.num_decoders)+"dec_"+str(self.num_layers)+"layers_"+dataset+"_"+\
                            str(task)+"_ac_"+str(active_cycle)+"_epoch-" + str(epoch) +"_clip_"+clip_active_method+"_video_"+video_active_method+"_label_size_"+\
                                str(clip_size)+"_"+self.loss_type+"_adding_ratio_"+str(batch_gen.adding_ratio)+".model"

            # curent name
            # model_path2 = model_dir + "/i3d_asf_cl_p_"+str(self.num_decoders)+"dec_"+str(self.num_layers)+"layers_"+dataset+"_"+\
            #                 str(task)+"_ac_"+str(active_cycle)+"_epoch-" + str(epoch) +"_clip_"+clip_active_method+"_video_"+video_active_method+"_label_size_"+\
            #                     str(clip_size)+"_"+self.loss_type+"_adding_ratio_"+str(batch_gen.adding_ratio)+"_seed_"+str(self.seed)+".model"
            if os.path.isfile(model_path1):
                model_path=model_path1
            elif os.path.isfile(model_path2):
                model_path=model_path2
            else:
                model_path="no model found"
                print(model_path)
            if os.path.isfile(model_path):
                with torch.no_grad():
                    self.model.to(device)
                    self.model.load_state_dict(torch.load(model_path,map_location="cpu"))
                    self.model.eval()
                    while batch_gen.has_next(if_train):
                        batch_input, batch_target, mask, vids, frame_labels = batch_gen.my_next_batch( batch_size=1, if_train=if_train, if_warp=False)
                        batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                        ps,fs = self.model(batch_input, mask)              
                        _, predicted = torch.max(ps.data[-1], 1)
                        predicted = predicted.squeeze(0).cpu().numpy()
                        clip_pred_to_frame = np.ones([len(predicted)*batch_gen.frames_per_seg], dtype=np.int32) 
                        for _ in range(len(predicted)):
                            temp1 = np.arange(start=_*batch_gen.frames_per_seg, stop=(_+1)*batch_gen.frames_per_seg)
                            clip_pred_to_frame[temp1]=predicted[_]
                        frame_pred_list.append(clip_pred_to_frame)
                        frame_label_list.append(frame_labels)
                        clip_scores.update(clip_pred_to_frame, frame_labels)
                [precision, recall, fscore, acc, acc_bg, num_bg_pred, num_bg, num_frames] = framewise_eval_pure(frame_pred_list,frame_label_list )
                clip_acc, clip_edit_score, clip_f1s = clip_scores.get_scores()
                print("epoch",epoch, "active cycle:", active_cycle, "fscore:",fscore, "acc_bg:", acc_bg,"Acc: %.4f  Edit: %4f  F1@10: %4f, F1@25: %4f, F1@50: %4f" % (clip_acc, clip_edit_score, clip_f1s[0],clip_f1s[1],clip_f1s[2] ))
                batch_gen.reset(if_train)
                # print(clip_scores.n_videos)
                # print(len(frame_pred_list))

                all_results.append([self.seed, dataset, task, video_active_method, clip_active_method, active_cycle, self.loss_type, clip_size, epoch, self.num_layers]+[precision, recall, fscore, acc, acc_bg, num_bg_pred, num_bg, num_frames]+[clip_edit_score] + clip_f1s)
        return all_results


    def my_active_new_eval(self, model_dir, dataset, task, batch_gen, action_dict, if_train, epoch, num_active_cycle, clip_active_method, video_active_method, clip_size):
        device = self.device
        # self.model.eval()
        all_results = []
        for active_cycle in range(num_active_cycle):
            # clip_scores = ScoreMeter(id2class_map=action_dict)
            active_cycle=active_cycle+1
            frame_pred_list = []
            frame_label_list = []
            # previous name
            model_path1 = model_dir + "/i3d_asf_cl_p_"+str(self.num_decoders)+"dec_"+str(self.num_layers)+"layers_"+dataset+"_"+\
                            str(task)+"_ac_"+str(active_cycle)+"_epoch-" + str(epoch) +"_clip_"+clip_active_method+"_video_"+video_active_method+"_label_size_"+str(clip_size)+"_"+self.loss_type+".model"
            # curent name
            model_path2 = model_dir + "/i3d_asf_cl_p_"+str(self.num_decoders)+"dec_"+str(self.num_layers)+"layers_"+dataset+"_"+\
                            str(task)+"_ac_"+str(active_cycle)+"_epoch-" + str(epoch) +"_clip_"+clip_active_method+"_video_"+video_active_method+"_label_size_"+str(clip_size)+"_"+self.loss_type+"_adding_ratio_"+str(batch_gen.adding_ratio)+".model"
            if os.path.isfile(model_path1):
                model_path=model_path1
            elif os.path.isfile(model_path2):
                model_path=model_path2
            else:
                model_path="no model found"
                print(model_path)
            if os.path.isfile(model_path):
                with torch.no_grad():
                    self.model.to(device)
                    self.model.load_state_dict(torch.load(model_path,map_location="cpu"))
                    self.model.eval()
                    while batch_gen.has_next(if_train):
                        batch_input, batch_target, mask, vids, frame_labels = batch_gen.my_next_batch( batch_size=1, if_train=if_train, if_warp=False)
                        batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                        ps,fs = self.model(batch_input, mask)              
                        _, predicted = torch.max(ps.data[-1], 1)
                        predicted = predicted.squeeze(0).cpu().numpy()
                        clip_pred_to_frame = np.ones([len(predicted)*batch_gen.frames_per_seg], dtype=np.int32) 
                        for _ in range(len(predicted)):
                            temp1 = np.arange(start=_*batch_gen.frames_per_seg, stop=(_+1)*batch_gen.frames_per_seg)
                            clip_pred_to_frame[temp1]=predicted[_]
                        frame_pred_list.append(clip_pred_to_frame)
                        frame_label_list.append(frame_labels)
                    mof = compute_mof(frame_label_list, frame_pred_list)
                    iou, iou_nb, iod, iod_nb = compute_IoU_IoD(frame_label_list, frame_pred_list)
                    [precision, recall, fscore, acc, acc_nb, acc_bg, _, _, _] = new_framewise_eval_pure(frame_pred_list, frame_label_list) 
                        # clip_scores.update(clip_pred_to_frame, frame_labels)
                # [precision, recall, fscore, acc, acc_bg, num_bg_pred, num_bg, num_frames] = framewise_eval_pure(frame_pred_list,frame_label_list )
                # clip_acc, clip_edit_score, clip_f1s = clip_scores.get_scores()
                print("epoch",epoch, "active cycle:", active_cycle, "mof:",mof," iou: %.4f  iou_nb: %4f  iod: %4f, iod_nb: %4f " % (iou, iou_nb, iod, iod_nb))
                batch_gen.reset(if_train)
                # print(clip_scores.n_videos)
                # print(len(frame_pred_list))

                all_results.append([dataset, task, video_active_method, clip_active_method, active_cycle, self.loss_type, clip_size, epoch, self.num_layers]+[mof, acc_nb]+[iou, iou_nb, iod, iod_nb])
        return all_results
    

    def my_active_predict_save(self, model_dir, dataset, task, batch_gen, action_dict, if_train, epoch, num_active_cycle, clip_active_method, video_active_method, clip_size):
        device = self.device
        # self.model.eval()
        all_results = []
        saving_dict={}

        while batch_gen.has_next(if_train):
            batch_input, batch_target, mask, vids, frame_labels = batch_gen.my_next_batch( batch_size=1, if_train=if_train, if_warp=False)
            saving_dict[vids[0]]={}
        batch_gen.reset(if_train)

        for active_cycle in range(num_active_cycle):
            active_cycle=active_cycle+1                    
            # previous name
            model_path1 = model_dir + "/i3d_asf_cl_p_"+str(self.num_decoders)+"dec_"+str(self.num_layers)+"layers_"+dataset+"_"+\
                            str(task)+"_ac_"+str(active_cycle)+"_epoch-" + str(epoch) +"_clip_"+clip_active_method+"_video_"+video_active_method+"_label_size_"+str(clip_size)+"_"+self.loss_type+".model"
            # curent name
            model_path2 = model_dir + "/i3d_asf_cl_p_"+str(self.num_decoders)+"dec_"+str(self.num_layers)+"layers_"+dataset+"_"+\
                            str(task)+"_ac_"+str(active_cycle)+"_epoch-" + str(epoch) +"_clip_"+clip_active_method+"_video_"+video_active_method+"_label_size_"+str(clip_size)+"_"+self.loss_type+"_adding_ratio_"+str(batch_gen.adding_ratio)+".model"
            if os.path.isfile(model_path1):
                model_path=model_path1
            elif os.path.isfile(model_path2):
                model_path=model_path2
            else:
                model_path="no model found"
                print(model_path)
            if os.path.isfile(model_path):
                with torch.no_grad():
                    self.model.to(device)
                    self.model.load_state_dict(torch.load(model_path,map_location="cpu"))
                    self.model.eval()
                    while batch_gen.has_next(if_train):
                        batch_input, batch_target, mask, vids, frame_labels = batch_gen.my_next_batch( batch_size=1, if_train=if_train, if_warp=False)
                        batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                        ps,fs = self.model(batch_input, mask)              
                        _, predicted = torch.max(ps.data[-1], 1)
                        predicted = predicted.squeeze(0).cpu().numpy()
                        gnd = batch_target.squeeze(0).cpu().numpy()
                        saving_dict[vids[0]][active_cycle]=predicted
                        if active_cycle == num_active_cycle:
                            saving_dict[vids[0]]['gnd']=gnd
                batch_gen.reset(if_train)
                print(saving_dict)
        torch.save(saving_dict,"infer_result_"+dataset+"_"+str(task)+"_"+clip_active_method+"_"+video_active_method+"_clip_size_"+str(clip_size)+"_"+self.loss_type+".pt")


        return None
    
    

if __name__ == '__main__':
    pass