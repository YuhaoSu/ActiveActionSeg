import torch
import torch.nn.functional as F
import numpy as np
from core import *
import torch.optim as optim
import copy

def summary_time_warping(S_x, features):
    if len (S_x) == 0:
        return 0, 0
    else:
        S_x_feature = features[S_x]
        matrix = torch.zeros((len(S_x_feature) + 1, len(features) + 1))
        matrix[0, :] = np.inf
        matrix[:, 0] = np.inf
        matrix[0, 0] = 0
        # distance = torch.norm(S_x_feature[:, None] - features, dim=2)
        normalized_distance = dis_sim_matrix(S_x_feature, features)
        # normalized_distance = F.softmax(normalized_distance, -1)
        for i, vec1 in enumerate(S_x_feature):
            for j, vec2 in enumerate(features):
                matrix[i + 1, j + 1] = normalized_distance[i,j] + \
                                        min(matrix[i + 1, j], matrix[i, j])
        matrix = matrix[1:, 1:]
    return matrix, matrix[-1, -1]


def summary_matching(summary, prototype, bg_cost, device='cpu'):
    """
    summary match, using torch
    """
    matrix = torch.zeros((len(summary) + 1, len(prototype) + 3)).to(device)
    matrix[0, :] = np.inf
    matrix[:, 0] = np.inf
    matrix[:, -2] = np.inf
    matrix[0, 0:3] = 0
    matrix[0, 0:3] = 0
    num_clips, num_prototype = torch.tensor(len(summary)),torch.tensor(len(prototype))
    normalized_distance = dis_sim_matrix(summary, prototype).to(device)
    # print(torch.norm(normalized_distance,dim=0))
    for i, vec1 in enumerate(summary):
            for j, vec2 in enumerate(prototype):
                matrix[i + 1, j + 1] = normalized_distance[i,j] + \
                                    min(torch.cat([matrix[i, j:j+3].view(-1,1), matrix[i,-1].view(1,1)])) #+\
                                   # 1*torch.abs(i/num_clips-j/num_prototype)/torch.sqrt(i/(num_clips^2)+j/(num_prototype^2)+ 1e-10)
                                    # min(matrix[i, j], matrix[i, j+1], matrix[i, j+2], matrix[i,-1])
                                        # min(torch.cat([matrix[i, j:j+3].view(-1,1), matrix[i,-1].view(1,1)]))
            temp = matrix[i,:][~torch.isinf(matrix[i,:])]
            matrix[i+1,-1] =torch.quantile(temp, 0.1)+ bg_cost[i]
    matrix = matrix[1:,1:]
    matrix[:,-2]=matrix[:,-1]
    matrix=matrix[:,:-1]
    # temp_ind = torch.arange(len(prototype)+2)
    # temp_ind !=-2
    # matrix = matrix[:,temp_ind]
    return matrix


def summary_matching_test(summary, prototype, bg_cost, device='cpu'):
    """
    summary match, using torch
    """
    matrix = torch.zeros((len(summary) + 1, len(prototype) + 4)).to(device)
    matrix[0, :] = np.inf
    matrix[:, 0] = np.inf
    matrix[:, -2] = np.inf
    matrix[0, 0:4] = 0
    matrix[0, 0:4] = 0
    num_clips, num_prototype = torch.tensor(len(summary)),torch.tensor(len(prototype))
    normalized_distance = dis_sim_matrix(summary, prototype).to(device)
    # print(torch.norm(normalized_distance,dim=0))
    for i, vec1 in enumerate(summary):
            for j, vec2 in enumerate(prototype):
                matrix[i + 1, j + 2] = normalized_distance[i,j] + \
                                    min(torch.cat([matrix[i, j-1:j+4].view(-1,1), matrix[i,-1].view(1,1)])) #+\
                                   # 1*torch.abs(i/num_clips-j/num_prototype)/torch.sqrt(i/(num_clips^2)+j/(num_prototype^2)+ 1e-10)
                                    # min(matrix[i, j], matrix[i, j+1], matrix[i, j+2], matrix[i,-1])
                                        # min(torch.cat([matrix[i, j:j+3].view(-1,1), matrix[i,-1].view(1,1)]))
            temp = matrix[i,:][~torch.isinf(matrix[i,:])]
            matrix[i+1,-1] =torch.quantile(temp, 0.1)+ bg_cost[i]
    matrix = matrix[1:,2:]
    matrix[:,-2]=matrix[:,-1]
    matrix=matrix[:,:-1]
    # temp_ind = torch.arange(len(prototype)+2)
    # temp_ind !=-2
    # matrix = matrix[:,temp_ind]
    return matrix

def action_matching(summary, prototype, device='cpu'):
    """
    summary match, using torch
    """
    matrix = torch.zeros((len(summary) + 1, len(prototype) + 2)).to(device)
    matrix[0, :] = np.inf
    matrix[:, 0] = np.inf
    matrix[:, -1] = np.inf
    matrix[0,0:3] = 0
    num_clips, num_prototype = torch.tensor(len(summary)),torch.tensor(len(prototype))
        # dis_sim_mt = torch.norm(a[:, None] - b, dim=2)
    normalized_distance = dis_sim_matrix(summary, prototype)
    # print(normalized_distance)
    for i, vec1 in enumerate(summary):
        for j, vec2 in enumerate(prototype):
            matrix[i + 1, j + 1] = normalized_distance[i,j] + \
                                min(torch.cat([matrix[i, j:j+3].view(-1,1)]))# +\
                             #   torch.abs(i/num_clips-j/num_prototype)/torch.sqrt(i/(num_clips^2)+j/(num_prototype^2)+ 1e-10)
    matrix = matrix[1:,1:]
    matrix=matrix[:,:-1]
    return matrix

def gpu_action_matching(matrix, summary, prototype, prior=0, temperature=0):
    """
    summary match, using torch
    reduced first row
    """
    matrix[:, 0] = np.inf
    matrix[:, -1] = np.inf
    num_clips, num_prototype = torch.tensor(len(summary)),torch.tensor(len(prototype))
    with torch.no_grad():
        # normalized_distance = torch.norm(summary[:, None] - prototype, dim=2)
        normalized_distance = dis_sim_matrix(summary, prototype)
        for j in [0,1,2]:
            matrix[0,j+1]=normalized_distance[0,j]# +torch.abs(j/num_prototype)/torch.sqrt(j/(num_prototype^2)+ 1e-10)
        matrix[0,4:]=float('inf')
        for i in range(len(summary)-1):
            i=i+1
            for j, vec2 in enumerate(prototype):
                matrix[i , j + 1] = normalized_distance[i,j] + \
                                    min(torch.cat([matrix[i-1, j:j+3].view(-1,1)])) #+\
                                   # (1-temperature)*torch.abs(i/num_clips-j/num_prototype)/torch.sqrt(i/(num_clips^2)+j/(num_prototype^2)+ 1e-10)
        matrix = matrix[:,1:]
        matrix=matrix[:,:-1]
    return matrix

def summary_backtrack(matrix):
    num_of_clips = matrix.size()[0]
    num_of_class = matrix.size()[1]-1
    # print(num_of_clips,num_of_class)
    assign = torch.ones(num_of_clips, dtype=torch.long) * -1 
    last_ind = num_of_class
    for i in range(num_of_clips, 0, -1):
        if last_ind != num_of_class:
            if torch.min(matrix[i-1,[last_ind-1, last_ind, last_ind+1]]) <= matrix[i-1,-1]:
                assign[i-1] = last_ind + (torch.argmin(matrix[i-1,[last_ind-1, last_ind, last_ind+1]]) -1)
                last_ind = assign[i-1]
            else:
                assign[i-1]=num_of_class
                last_ind=assign[i-1]
        else:
            assign[i-1] = torch.argmin(matrix[i-1,:])
            last_ind = assign[i-1]
        # print(last_ind)
    assign[assign==num_of_class]=-1
    return assign

def action_backtrack(matrix, device='cpu'):
    num_of_clips = matrix.size()[0]
    num_of_class = matrix.size()[1]
    # print(num_of_clips,num_of_class)
    assign = torch.zeros(num_of_clips, dtype=torch.long).to(device)
    # we assume num_of_class >=3
    assign[-1] = torch.argmin(matrix[-1,num_of_class-3:])
    last_ind = assign[-1]
    for i in range(num_of_clips-1, 0, -1):
        # print(i, last_ind)
        if last_ind ==0:
            assign[i-1] = last_ind + (torch.argmin(matrix[i-1,[last_ind, last_ind+1, last_ind+2]]) )
            last_ind = assign[i-1]
        elif last_ind  ==(num_of_class-1):
            assign[i-1] = last_ind + (torch.argmin(matrix[i-1,[last_ind-2, last_ind-1, last_ind]]) -2)
            last_ind = assign[i-1]
        else:
            assign[i-1] = last_ind + (torch.argmin(matrix[i-1,[last_ind-1, last_ind, last_ind+1]]) -1)
            last_ind = assign[i-1]
        
    return assign

def gpu_action_backtrack(matrix, assign):
    num_of_clips = matrix.size()[0]
    num_of_class = matrix.size()[1]
    # print(num_of_clips,num_of_class)
    # we assume num_of_class >=3
    with torch.no_grad():
        assign[-1] =num_of_class-3+torch.argmin(matrix[-1,num_of_class-3:])
        last_ind = assign[-1]
        for i in range(num_of_clips-1, 0, -1):
            # print(i, last_ind)
            if last_ind ==0:
                assign[i-1] = last_ind + (torch.argmin(matrix[i-1,[last_ind, last_ind+1, last_ind+2]]) )
                last_ind = assign[i-1]
            elif last_ind  ==(num_of_class-1):
                assign[i-1] = last_ind + (torch.argmin(matrix[i-1,[last_ind-2, last_ind-1, last_ind]]) -2)
                last_ind = assign[i-1]
            else:
                assign[i-1] = last_ind + (torch.argmin(matrix[i-1,[last_ind-1, last_ind, last_ind+1]]) -1)
                last_ind = assign[i-1]
    return assign

def video_confidence(stw_ass, gre_check_ass, avg_num_clips):
    # number of matches
    acc_match= torch.sum(stw_ass==gre_check_ass)/len(stw_ass)
    # segment quality, measure of variablility
    # we do not want video with too large segment
    # this will prefer short videos, not sure this is right
    counts = torch.bincount(stw_ass)
    seg_vol = 1-torch.max(counts[counts.nonzero()])/len(stw_ass)
    # num_avg_clips_per_seg
    return acc_match+avg_num_clips+seg_vol


# def update_confidence(feat, stw_ass):
#     normalized_feat = F.normalize(feat)
    


def sum_feat_dis(feat, stw_ass):
    feat = F.normalize(feat)
    S_x_feat = feat[stw_ass]
    return torch.norm(feat-S_x_feat)



def make_center_test(features_list_dict, k, subset_ind):
    features = []
    orders = []
    p_data = []
    avail_videos = list(features_list_dict.keys())
    avail_videos = [avail_videos[_] for _ in subset_ind]
    for i in range(k):
        feat_list = []
        # for j in range(len(avail_videos)):
        for key in avail_videos:
            # key = avail_videos[j]
            feat = F.normalize(features_list_dict[key])
            ind = np.arange(feat.size()[0])
            split_ind = np.array_split(ind, k)
            feat_list.append(feat[split_ind[i]])
        feat_list =torch.cat(feat_list)
        ind_temp = torch.argsort(torch.mean(dis_sim_matrix(feat_list, feat_list),dim=0))
        feat_sublist = feat_list[ind_temp[:int(0.75*len(ind_temp))]].numpy()
        # print("full list:", torch.mean(dis_sim_matrix(feat_list, feat_list)))
        # print("sub list", torch.mean(torch.mean(dis_sim_matrix(feat_list, feat_list),dim=0)[ind_temp[:int(0.5*len(ind_temp))]]))
        # print()
        p_data.append(np.mean(feat_sublist, axis=0))
    p_data = np.stack(p_data)
    return p_data

# temp_tensor = torch.rand(10)
# print(temp_tensor)
# temp_ind = torch.argsort(temp_tensor)
# print(temp_tensor[temp_ind[:int(0.5*len(temp_ind))]])

# summary = torch.tensor([
#     [0.,0,0,1],
#     [0.,0,0,1],
#     [0.,0,0,1],
#     [0,0,0,1],
#     [0.,0,0,1],
#     [1,0,0,0],
#     [0,0,0,1],
#     [0,1,0,0],
#     [0,1,0,0]
#     ])

# prototype =torch.tensor([
#     [1,0,0,0],
#     [0,0,0,1],
#     [0,1,0,0],
#     ])

# normalized_distance = dis_sim_matrix(summary, prototype)
# # print(normalized_distance)
# # bg_cost = torch.ones(len(summary))*0.5
# # print(summary)
# # print(prototype)
# acc_m = action_matching(summary, prototype)
# print(acc_m)
# # assign = action_backtrack(acc_m)
# # print(assign)
# # print(assign)
# matrix = torch.zeros((len(summary), len(prototype) + 2))
# acc_m = gpu_action_matching(matrix, summary, prototype)
# print(acc_m)
