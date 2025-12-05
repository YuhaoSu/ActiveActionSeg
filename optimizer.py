import torch
import os
import csv
import scipy.optimize
import numpy as np
import torch.nn.functional as F
from core import dis_sim_matrix, find_summary_assignment, find_match_index, set_difference_tensor
from joblib import Parallel, delayed
from sklearn.cluster import KMeans



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

def parallel_helper_first(feat, ind, element):
    index_to_check = torch.tensor([element], dtype=torch.long)
    _, value =  summary_time_warping(index_to_check, feat)
    return torch.tensor([ind, value])

def parallel_helper_others(feat, S_x, ind, element):
    index_to_check = torch.cat(
        [S_x, torch.tensor([element], dtype=torch.long)])
    sorted_, _  = torch.sort(index_to_check)
    _, value = summary_time_warping(sorted_, feat)
    return torch.tensor([ind, value])

def singular_summarizing(features, summary_size):
    torch.set_grad_enabled(False)
    S_x = []
    index1 = torch.arange(len(features))
    for i in range(summary_size):
        if i == 0:
            temp_result = torch.zeros([len(index1)])
            for j in range(len(index1)):
                index_to_check = torch.tensor([index1[j]], dtype=torch.long)
                _, temp_result[j] =  summary_time_warping(index_to_check, features)
            S_x = torch.tensor([index1[torch.argmin(temp_result)]],
                               dtype=torch.long) 
        else:
            index1 = set_difference_tensor(
                torch.arange(len(features), dtype=torch.long), S_x)
            temp_result = torch.zeros([len(index1)])
            for j in range(len(index1)):
                index_to_check = torch.cat(
                    [S_x, torch.tensor([index1[j]], dtype=torch.long)])
                sorted_, _  = torch.sort(index_to_check)
                _,temp_result[j] = summary_time_warping(sorted_, features)
            # temp_result = torch.abs(temp_result - summary_time_warping(S_x, features)[1])
            S_x = torch.cat(
                [S_x, torch.tensor([index1[torch.argmin(temp_result)]], dtype=torch.long)])
            S_x,_ = torch.sort(S_x)
    return S_x

def parallel_singular_summarizing(features, summary_size):
    '''
    '''
    S_x = []
    result = []
    index1 = torch.arange(len(features))

    if isinstance(summary_size, int):
        with torch.no_grad():
            for i in range(summary_size):
                if i == 0:
                    result = Parallel(n_jobs=6)(delayed(parallel_helper_first)(features, ind, element) 
                                                for ind, element in enumerate(index1))
                    result = torch.cat(result).view(len(result),2)
                    index_from_result = result[torch.argmin(result[:,1]),0].to(torch.long)
                    S_x = torch.tensor([index1[index_from_result]], dtype=torch.long) 
                else:
                    index1 = set_difference_tensor(
                        torch.arange(len(features), dtype=torch.long), S_x)
                    result = Parallel(n_jobs=6)(delayed(parallel_helper_others)(features, S_x, ind, element) 
                                                for ind, element in enumerate(index1))
                    result = torch.cat(result).view(len(result),2)
                    index_from_result = result[torch.argmin(result[:,1]),0].to(torch.long)
                    S_x = torch.cat(
                        [S_x, torch.tensor([index1[index_from_result]], dtype=torch.long)])
                    S_x,_ = torch.sort(S_x)
            matrix, final_value = summary_time_warping(S_x, features)
            assign =  find_summary_assignment(S_x, matrix)
        return S_x, assign
    
    if isinstance(summary_size, list):
        # assume the last one is the largest one
        # use dict to save assignment
        S_output = {}
        assign_output = {}
        with torch.no_grad():
            cur_ind = 0
            for i in range(summary_size[-1]):
                if i == 0:
                    result = Parallel(n_jobs=6)(delayed(parallel_helper_first)(features, ind, element) 
                                                for ind, element in enumerate(index1))
                    result = torch.cat(result).view(len(result),2)
                    index_from_result = result[torch.argmin(result[:,1]),0].to(torch.long)
                    S_x = torch.tensor([index1[index_from_result]], dtype=torch.long) 
                else:
                    index1 = set_difference_tensor(
                        torch.arange(len(features), dtype=torch.long), S_x)
                    result = Parallel(n_jobs=6)(delayed(parallel_helper_others)(features, S_x, ind, element) 
                                                for ind, element in enumerate(index1))
                    result = torch.cat(result).view(len(result),2)
                    index_from_result = result[torch.argmin(result[:,1]),0].to(torch.long)
                    S_x = torch.cat(
                        [S_x, torch.tensor([index1[index_from_result]], dtype=torch.long)])
                    S_x,_ = torch.sort(S_x)
                if len(S_x) == summary_size[cur_ind]:
                    # save S_x and assignment, move cur_ind to next
                    matrix, final_value = summary_time_warping(S_x, features)
                    assign =  find_summary_assignment(S_x, matrix)
                    S_output[summary_size[cur_ind]] = S_x
                    assign_output[summary_size[cur_ind]] = assign
                    cur_ind = cur_ind+1
                    # print("summary size is", len(S_x), S_x)
            out = {}
            out['summary']=S_output
            out["assign"]=assign_output
            return out

    


'''
def p_jsd(S_x, features1, S_y_features, threshold=1, softmax_type="row", gamma=0.001):
    if  len(S_x)== 0 or len(S_y_features)==0:
        return 0, 0
    else:
        S_x_features = features1[S_x]
        M, N = S_x_features.shape[0], S_y_features.shape[0]
        # compute cost matrix
        S_x_features = S_x_features / torch.sqrt(
            torch.sum(S_x_features ** 2, axis=-1, keepdims=True) + 1e-10)
        S_y_features = S_y_features / torch.sqrt(
            torch.sum(S_y_features ** 2, axis=-1, keepdims=True) + 1e-10)
        # print(sorted_centers_a,sorted_centers_b)
        cost = 1 - torch.matmul(S_x_features, S_y_features.t())

        cost = torch.cat((cost, torch.ones_like(cost) * threshold))
        cost = cost.t().reshape(2 * N, M).t()
        cost = torch.cat((threshold * torch.ones(M, 1, dtype=cost.dtype)
                          .to(cost.device), cost), dim=1)
        if softmax_type == 'row':
            # cost = F.softmax(cost, -1)
            cost = cost
        elif softmax_type == 'col':
            cost = F.softmax(cost, 0)
        elif softmax_type == 'all':
            cost = F.softmax(cost.view(-1), 0).view(M, -1)
        cost = cost.detach().numpy()
        M, N = cost.shape
        D = np.ones((M, N)) * 100
        # print("size of D:", D.shape)
        D[0, :] = cost[0, :]

        for i in range(1, M):
            for j in range(0, N):
                if j % 2 == 0:
                    last_row = D[i - 1, :j + 1]
                else:
                    last_row = D[i - 1, :j]
                rmin = np.min(last_row)
                rsum = np.sum(np.exp(- (last_row - rmin) / gamma))
                softmin = -gamma * np.log(rsum) + rmin
                D[i, j] = softmin + cost[i, j]

        last_row = D[-1, :]
        rmin = np.min(last_row)
        rsum = np.sum(np.exp(- (last_row - rmin) / gamma))
        softmin = -gamma * np.log(rsum) + rmin
        return D, torch.tensor(softmin)

def computing_c(x_features, y_features):
    C = torch.zeros(x_features.size()[0], y_features.size()[0])
    return C

def compute_soft_restricted_editdistance(C, bg_cost=1, swap_cost=1, gamma=0.1):
    M, N = C.shape
    #print('C:', C)
    D = np.zeros((M+1, N+1))
    D[0] = np.arange(0, N+1) * bg_cost
    D[1:, 0] = np.arange(1, M+1) * bg_cost

    for i in range(1, M+1):
        for j in range(1, N+1):
            if i >= 2 and j >= 2:
                min_val = min([D[i-1, j-1] + C[i-1, j-1],
                              D[i-1, j] + bg_cost,
                              D[i, j-1] + bg_cost,
                              D[i-2, j-2] + C[i-2, j-1] + C[i-1, j-2] + bg_cost
                              ]
                             )
                sum_val = np.exp(- (D[i-1, j-1] + C[i-1, j-1] - min_val) / gamma
                           ) + np.exp(- (D[i-1, j] + bg_cost - min_val) / gamma
                           ) + np.exp(- (D[i, j-1] + bg_cost - min_val) / gamma
                           ) + np.exp(- (D[i-2, j-2] + C[i-2, j-1] + C[i-1, j-2]  + swap_cost - min_val) / gamma
                           )
            else:
                min_val = min([D[i-1, j-1] + C[i-1, j-1],
                              D[i-1, j] + bg_cost,
                              D[i, j-1] + bg_cost,]
                             )
                sum_val = np.exp(- (D[i-1, j-1] + C[i-1, j-1] - min_val) / gamma
                           ) + np.exp(- (D[i-1, j] + bg_cost - min_val) / gamma
                           ) + np.exp(- (D[i, j-1] + bg_cost - min_val) / gamma)
            soft_min_val = -gamma * np.log(sum_val) + min_val
            D[i, j] = soft_min_val
    return D, D[-1, -1]

def compute_soft_restricted_editdistance_backward(C, D, bg_cost=1, swap_cost=1, gamma=0.1):
    M, N = C.shape
    grad_d = np.zeros((M+1, N+1))   ### partial L / partial d_(i, j)
    grad_d_c1 = np.zeros((M , N))   ### partial d_(i+1, j+1) / partial c_(i, j)
    grad_d_c2 = np.zeros((M , N))   ### partial d_(i+1, j+1) / partial c_(i, j-1)
                                    ### = partial d_(i+1, j+1) / partial c_(i-1, j)
    grad_d[-1, -1] = 1
    i = -1
    for j in range(N-1, -1, -1):
        b = np.exp(- (D[i, j] + bg_cost - D[i, j+1]) / gamma)
        grad_d[-1, j] = grad_d[-1, j+1] * b
    j = -1
    for i in range(M-1, -1, -1):
        x = np.exp(- (D[i, j] + bg_cost - D[i+1, j]) / gamma)
        grad_d[i, -1] = grad_d[i+1, -1] * x

    for i in range(M-1, -1, -1):
        for j in range(N-1, -1, -1):
            a = np.exp(- (D[i, j] + C[i, j] - D[i+1, j+1]) / gamma)
            b = np.exp(- (D[i, j] + bg_cost - D[i, j+1]) / gamma)
            x = np.exp(- (D[i, j] + bg_cost - D[i+1, j]) / gamma)
            if i < M-1 and j < N-1:
                y = np.exp(- (D[i, j] + C[i, j+1] + C[i+1, j]
                              + swap_cost - D[i+2, j+2]) / gamma)
                grad_d_c2[i+1, j+1] = y
                grad_d[i, j] = grad_d[i+1, j+1] * a + grad_d[i, j+1] * b + \
                    grad_d[i+1, j] * x + grad_d[i+2, j+2] * y
            else:
                grad_d[i, j] = grad_d[i+1, j+1] * a + grad_d[i, j+1] * b + grad_d[i+1, j] * x

            grad_d_c1[i, j] = a
    #print(grad_d)
    grad_c2 = np.zeros((M, N))
    grad_c3 = np.zeros((M, N))
    ### (partial L  * / partial d_(i+2, j+1)) * (partial d_(i+2, j+1)  * / partial d_(i, j))
    grad_c2[:-1, :]  = grad_d[2:, 1:] * grad_d_c2[1:, :]
    ### (partial L  * / partial d_(i+1, j+2)) * (partial d_(i+1, j+2)  * / partial d_(i, j))
    grad_c3[:, :-1] =  grad_d[1:, 2:] * grad_d_c2[:, 1:]

    grad_c = grad_d[1:, 1:] * grad_d_c1 + grad_c2 + grad_c3

    return grad_c

def parallel_jsd_helper_first(feat, S_y_features, ind, element,  bg_cost, swap_cost, gamma):
    index_to_check = torch.tensor([element], dtype=torch.long)
    C = dis_sim_matrix(feat[index_to_check], S_y_features)
    _, value = compute_soft_restricted_editdistance(C, bg_cost, swap_cost, gamma)
    # _, value =  p_jsd(index_to_check, feat, S_y_features, threshold=jsd_t)
    return torch.tensor([ind, value])

def parallel_jsd_helper_others(feat, S_x, S_y_features, ind, element,  bg_cost, swap_cost, gamma):
    index_to_check = torch.cat(
        [S_x, torch.tensor([element], dtype=torch.long)])
    sorted_, _  = torch.sort(index_to_check)
    C = dis_sim_matrix(feat[sorted_], S_y_features)
    _, value = compute_soft_restricted_editdistance(C, bg_cost, swap_cost, gamma)
    # _, value = p_jsd(sorted_, feat, S_y_features, threshold=jsd_t)
    return torch.tensor([ind, value])

def get_enhanced_predict(transcript, pred, bg_cost=1, swap_cost=1, adapt_threshold=False):
    if torch.is_tensor(transcript):
        transcript = transcript.detach().cpu().numpy() 
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    matching =  - np.matmul(transcript, pred.T)
    if adapt_threshold:
        per_threshold = np.quantile(matching, 1 / matching.shape[1])
        bg_cost, swap_cost = per_threshold, per_threshold

    D, loss = compute_soft_restricted_editdistance(matching, bg_cost, swap_cost, gamma=0.001)
    grad = compute_soft_restricted_editdistance_backward(matching, D, bg_cost, swap_cost, gamma=0.001)
    alignment = (grad > 0.5).astype(float)
    enhanced_pred = np.matmul(alignment.T, transcript)
    return enhanced_pred

def parallel_prototype_assisted_singular_summarizing(features, summary_size, prototype, bg_cost, swap_cost, gamma, gamma_1):
    feat = features.data
    prototype = prototype.data
    S_x = []
    index1 = torch.arange(len(feat))
    for i in range(summary_size):
        # print("trying to find", i,"th item in S...")
        if i == 0:
            result_stw = Parallel(n_jobs=6)(delayed(parallel_helper_first)(feat, ind, element) 
                                        for ind, element in enumerate(index1))
            result_stw = torch.cat(result_stw).view(len(result_stw),2)
            result_jsd = Parallel(n_jobs=6)(delayed(parallel_jsd_helper_first)(feat, prototype, ind, element, bg_cost, swap_cost, gamma) 
                                        for ind, element in enumerate(index1))
            result_jsd = torch.cat(result_jsd).view(len(result_jsd),2)           
            result = result_stw + gamma_1*result_jsd
            S_x = torch.tensor([index1[torch.argmin(result[:,1])]],
                               dtype=torch.long) 
            # print(i,"th,",(result_stw[:,1]))
            # print(i,"th,",(result_jsd[:,1]))

            # print("max m_gain:", torch.min(result_stw[:,1]), torch.min(result_jsd[:,1]))


        else:
            index1 = set_difference_tensor(
                torch.arange(len(features), dtype=torch.long), S_x)
            result_stw = Parallel(n_jobs=6)(delayed(parallel_helper_others)(feat, S_x, ind, element) 
                                        for ind, element in enumerate(index1))
            result_stw = torch.cat(result_stw).view(len(result_stw),2)
            result_jsd = Parallel(n_jobs=6)(delayed(parallel_jsd_helper_others)(feat, S_x, prototype, ind, element, bg_cost, swap_cost, gamma) 
                                        for ind, element in enumerate(index1))
            result_jsd = torch.cat(result_jsd).view(len(result_jsd),2)
            result = result_stw+gamma_1*result_jsd
            result[:,0] = result_stw[:,0]
            index_from_result = result[torch.argmin(result[:,1]),0].to(torch.long) 
            temp_, stw = summary_time_warping(S_x, features)
            # C = dis_sim_matrix(feat[S_x], prototype)
            # D, softmin= compute_soft_restricted_editdistance(C, bg_cost, swap_cost, gamma)
            S_x = torch.cat(
                [S_x,  torch.tensor([index1[index_from_result]], dtype=torch.long)])
            S_x,_ = torch.sort(S_x)
            # print(i,"th,",(result_stw[:,1]))
            # print(i,"th,",(result_jsd[:,1]))
            # print("max m_gain:", torch.abs(stw-torch.min(result_stw[:,1])), torch.abs(p_-torch.min(result_jsd[:,1])))
            # print(result_jsd[:,1])  

    matrix, _ = summary_time_warping(S_x, feat)
    assign =  find_summary_assignment(S_x, matrix)
    C = dis_sim_matrix(feat[S_x], prototype)
    D, softmin= compute_soft_restricted_editdistance(C, bg_cost, swap_cost, gamma)
    grad = compute_soft_restricted_editdistance_backward(C, D,  bg_cost, swap_cost, gamma)
    alignment = (grad > 0.5).astype(float)

    # enhanced_pred = get_enhanced_predict(feat[S_x], prototype, bg_cost, swap_cost, adapt_threshold=False)
    # print(enhanced_pred.shape)
    # obtain summary prediction from alignment
    s_pred={}
    for i in range(len(S_x)):
        if np.max(alignment[i,:]) == 1:
            s_pred[S_x[i].item()]=np.argmax((alignment[i,:]))
        else:
            s_pred[S_x[i].item()]= -1
    clip_pred = np.zeros(len(assign),dtype=int)
    for j in range(len(assign)):
        clip_pred[j] = s_pred[assign[j].item()]
    return grad, assign, clip_pred

def prototype_based_time_warping(S_x_feature, features):
    if len (S_x_feature) == 0:
        return 0, 0
    else:
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

def subset_sorter(S_ind, add, prototype):
    """
    make sure prototype are sorted 
    """
    p_S =  [rep for rep in S_ind if rep != -1] # remove bg
    cand = add[0]
    if cand < p_S[0]: # insert at the beginning
        if S_ind[0] == -1:
            newS= [-1]+add+S_ind[1:] 
        else:
            newS = add+S_ind
    elif cand >p_S[-1]: # insert at the end
        newS = S_ind +add
    else:
        newP = p_S + [cand]
        newP.sort()
        cand_ind = newP.index(cand)
        newS = S_ind[:cand_ind+1]+add+S_ind[cand_ind+1:]
    # make S features 
    feats = []

    for i in range(len(newS)):
        if newS[i] == -1:
            # temp = torch.zeros(prototype.size()[1])
            temp = prototype[-1]

        else:
            temp = prototype[newS[i]]
        feats.append(temp)
    newS_feat = torch.cat(feats).view(len(newS), prototype.size()[1])
    return newS, newS_feat

def prototype_based_singular_summarizing(features, prototype, m_gain_t, cutting):
    "at this moment, we only allow partial 1-to-1 from prototype summarizing"
    if len(prototype) == 0:
        return 0,0
    else:
        k, f_dim = prototype.size()
        prototype_ind = torch.arange(k, dtype=torch.long)
        # use list preserve order
        """
        consider case: S = [p1]*k, [0,p1]*k, [0,p1,0]*k
        """
        temp = torch.zeros(3*len(prototype_ind))
        for j in range(len(prototype_ind)):
            temp[3*j] = prototype_based_time_warping(prototype[j].view(1,f_dim), features)[1]
        for j in range(len(prototype_ind)):
            s_temp = torch.cat([prototype[j].view(1,f_dim), torch.zeros(1,f_dim)],dim=0)
            temp[3*j+1] = prototype_based_time_warping(s_temp, features)[1]
        for j in range(len(prototype_ind)):
            s_temp = torch.cat([torch.zeros(1,f_dim), prototype[j].view(1,f_dim), torch.zeros(1,f_dim)],dim=0)
            temp[3*j+2] = prototype_based_time_warping(s_temp, features)[1]
        new_gain, ind = torch.min(temp, dim=0)
        q, m = divmod(ind.item(), 3)
        if m  == 0:
            S=prototype[q].view(1,f_dim)
            S_ind = [q]
        elif m == 1:
            S=torch.cat([prototype[q].view(1,f_dim), torch.zeros(1,f_dim)],dim=0)
            S_ind = [q, -1]
        else:
            S=torch.cat([torch.zeros(1,f_dim), prototype[q].view(1,f_dim), torch.zeros(1,f_dim)],dim=0)
            S_ind = [-1,q,-1]
        prototype_ind = set_difference_tensor(prototype_ind, torch.tensor([q], dtype=torch.long))
        print(S_ind)
        old_gain = new_gain # initialze
        gain_diff = torch.abs(old_gain - torch.tensor(3*len(features))) 
        while gain_diff > m_gain_t:
            if len(prototype_ind) == 0:
                break
            """
            consider case: S+[p2]*(k-1), S+[p2,0]*(k-1) 
            """
            print("start new iteration")
            temp = torch.zeros(2*len(prototype_ind))
            for j in range(len(prototype_ind)): # S+ [p_i]
                add = [prototype_ind[j].item()]
                _, s_temp = subset_sorter(S_ind, add, prototype)
                temp[2*j] = prototype_based_time_warping(s_temp, features)[1]
                print(_,temp[2*j].item())
            print("start trying to add -1")
            for j in range(len(prototype_ind)): # S+ [p_i, -1] 
                add = [prototype_ind[j].item(), -1]
                _, s_temp = subset_sorter(S_ind, add, prototype)                    
                temp[2*j+1] = prototype_based_time_warping(s_temp, features)[1]
                print(_,temp[2*j+1].item())
            new_gain, ind = torch.min(temp, dim=0)
            q, m = divmod(ind.item(), 2)
            if m == 0:
                add = [prototype_ind[q].item()]
                S_ind, S= subset_sorter(S_ind, add, prototype)
            else:
                add = [prototype_ind[q].item(), -1]
                S_ind, S = subset_sorter(S_ind, add, prototype)
            prototype_ind = set_difference_tensor(prototype_ind, torch.tensor([prototype_ind[q]], dtype=torch.long))
            
            gain_diff = torch.abs(new_gain - old_gain)
            old_gain = new_gain
    print("summary:", S_ind)
    matrix,_ = prototype_based_time_warping(S, features)
    clip_pred = find_summary_assignment(torch.tensor(S_ind,dtype=torch.long), matrix.detach())
    # print("assignment:", clip_pred)

    # for t in range(k):
    #     temp = []
    #     for te in range(len(clip_pred)):
    #         if clip_pred[te] == t:
    #             temp.append(te)
    #     temp = torch.tensor(temp, dtype=torch.long)
    #     if len(temp) > len(clip_pred)/k: 
    #         new_dis = dis_sim_matrix(features[temp], prototype[t].view(1, f_dim), eps=1e-8)
    #         _, new_inds = torch.sort(new_dis.view(-1))
    #         cutting_point = int(cutting*len(temp))
    #         clip_pred[temp[new_inds[cutting_point:].view(-1)]] = -1            
    return clip_pred

def prototype_based_singular_summarizing(features, prototype, m_gain_t, cutting):
    "at this moment, we only allow partial 1-to-1 from prototype summarizing"
    if len(prototype) == 0:
        return 0,0
    else:
        k, f_dim = prototype.size()
        prototype_ind = torch.arange(k, dtype=torch.long)
        # use list preserve order
        """
        consider case: S = [p1]*k, [0,p1]*k, [0,p1,0]*k
        """
        temp = torch.zeros(3*len(prototype_ind))
        for j in range(len(prototype_ind)):
            temp[3*j] = prototype_based_time_warping(prototype[j].view(1,f_dim), features)[1]
            # temp[3*j] = 10000
        for j in range(len(prototype_ind)):
            s_temp = torch.cat([prototype[j].view(1,f_dim), torch.zeros(1,f_dim)],dim=0)
            temp[3*j+1] = prototype_based_time_warping(s_temp, features)[1]
            # temp[3*j+1] = 10000
        for j in range(len(prototype_ind)):
            s_temp = torch.cat([torch.zeros(1,f_dim), prototype[j].view(1,f_dim), torch.zeros(1,f_dim)],dim=0)
            temp[3*j+2] = prototype_based_time_warping(s_temp, features)[1]
        new_gain, ind = torch.min(temp, dim=0)
        q, m = divmod(ind.item(), 3)
        if m  == 0:
            S=prototype[q].view(1,f_dim)
            S_ind = [q]
        elif m == 1:
            S=torch.cat([prototype[q].view(1,f_dim), torch.zeros(1,f_dim)],dim=0)
            S_ind = [q, -1]
        else:
            S=torch.cat([torch.zeros(1,f_dim), prototype[q].view(1,f_dim), torch.zeros(1,f_dim)],dim=0)
            S_ind = [-1,q,-1]
        # prototype_ind = set_difference_tensor(prototype_ind, torch.tensor([q], dtype=torch.long))
        # print(S_ind)
        old_gain = new_gain # initialze
        gain_diff = torch.abs(old_gain - torch.tensor(3*len(features))) 
        while gain_diff > m_gain_t:# and  len(prototype_ind) >= int(0.4*k):
        # while len(prototype_ind) >= int(0.4*k):
            if len(prototype_ind) == 0:
                break
            """
            consider case: S+[p2]*(k-1), S+[p2,0]*(k-1) 
            """
            # print("start new iteration")
            temp = torch.zeros(2*len(prototype_ind))
            for j in range(len(prototype_ind)): # S+ [p_i]
                add = [prototype_ind[j].item()]
                _, s_temp = subset_sorter(S_ind, add, prototype)
                temp[2*j] = prototype_based_time_warping(s_temp, features)[1]
                # temp[2*j] =10000
                # print(temp[2*j].item())
            # print("start trying to add -1")
            for j in range(len(prototype_ind)): # S+ [p_i, -1] 
                add = [prototype_ind[j].item(), -1]
                _, s_temp = subset_sorter(S_ind, add, prototype)                    
                temp[2*j+1] = prototype_based_time_warping(s_temp, features)[1]
                # print(_,temp[2*j+1].item())
            new_gain, ind = torch.min(temp, dim=0)
            q, m = divmod(ind.item(), 2)
            if m == 0:
                add = [prototype_ind[q].item()]
                S_ind, S= subset_sorter(S_ind, add, prototype)
            else:
                add = [prototype_ind[q].item(), -1]
                S_ind, S = subset_sorter(S_ind, add, prototype)
            # prototype_ind = set_difference_tensor(prototype_ind, torch.tensor([prototype_ind[q]], dtype=torch.long))
            
            gain_diff = torch.abs(new_gain - old_gain)
            print(gain_diff.data, add)
            old_gain = new_gain
    print("summary:", S_ind)
    matrix,_ = prototype_based_time_warping(S, features)
    clip_pred = find_summary_assignment(torch.tensor(S_ind,dtype=torch.long), matrix.detach())
    # print("num_of_action_selected:", len(np.unique(clip_pred))-1)

    for t in range(k):
        temp = []
        for te in range(len(clip_pred)):
            if clip_pred[te] == t:
                temp.append(te)
        temp = torch.tensor(temp, dtype=torch.long)
        if len(temp) > len(clip_pred)/k: 
            new_dis = dis_sim_matrix(features[temp], prototype[t].view(1, f_dim), eps=1e-8)
            _, new_inds = torch.sort(new_dis.view(-1))
            cutting_point = int(cutting*len(temp))
            clip_pred[temp[new_inds[cutting_point:].view(-1)]] = -1            
    return clip_pred






if __name__ == '__main__':
    # S =[3,-1,4,-1]
    # add = [2]
    # p = torch.rand(6, 20)
    # print(p)

    # p = torch.cat([p, torch.randn(1, (p.size()[1]))],dim=0)
    # print(p)
    # k_ind = torch.arange(8)
    # ind= torch.cat((k_ind, torch.tensor([1])))
    # print(ind, torch.typename(ind))
    x = torch.tensor([
                    [0,0,1,0],
                    [1,1,2,3],
                    [10,9,-2,3]])
    y = torch.tensor([[1,1,2,3],
                    [-1,-10,2,0],
                    [1,1,2,3],
                    [10,9,-2,3]])
    b=0.1
    s=1
    g=0.01
    S_x = torch.LongTensor([0,2])
    feat = y
    matrix, _ = summary_time_warping(S_x, feat)
    assign =  find_summary_assignment(S_x, matrix)
    print(assign)
    print(feat[S_x], x)
    C = dis_sim_matrix(feat[S_x], x)
    
    D, softmin= compute_soft_restricted_editdistance(C, b, s, g)
    grad = compute_soft_restricted_editdistance_backward(C, D,  b, s, g)
    print(grad)
    alignment = (grad > 0.5).astype(float)
    print(alignment)
    s_pred={}
    for i in range(len(S_x)):
        if np.max(alignment[i,:]) == 1:
            s_pred[S_x[i].item()]=np.argmax((alignment[i,:]))
        else:
            s_pred[S_x[i].item()]= -1
    print(s_pred)
    clip_pred = np.zeros(len(assign), dtype=int)
    for j in range(len(assign)):
        clip_pred[j] = s_pred[assign[j].item()]
    print(clip_pred)

'''