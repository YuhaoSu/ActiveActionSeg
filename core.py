import torch
import os
import csv
import scipy.optimize
import numpy as np
import torch.nn.functional as F
# from gluoncv.torch.data.transforms.videotransforms import video_transforms, volume_transforms
# from gluoncv.torch.engine.config import get_cfg_defaults
# from gluoncv.torch.model_zoo import get_model
from joblib import Parallel, delayed
from sklearn.cluster import KMeans


def sim_matrix(a, b, gamma=1):
    """
    added eps for numerical stability
    """
    # a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    # a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    # b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    # sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    # dis_sim_mt = torch.ones_like(sim_mt) - sim_mt
    a = a / torch.sqrt(
            torch.sum(a ** 2, axis=-1, keepdims=True) + 1e-10)
    b = b / torch.sqrt(
            torch.sum(b ** 2, axis=-1, keepdims=True) + 1e-10)
     # print(sorted_centers_a,sorted_centers_b)
    sim_mt = (torch.matmul(a, b.t())+1)/gamma 
    # dis_sim_mt = torch.norm(a[:, None] - b, dim=2)
    return sim_mt
def dis_sim_matrix(a, b, gamma = 1):
    """
    added eps for numerical stability
    """
    # a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    # a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    # b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    # sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    # dis_sim_mt = torch.ones_like(sim_mt) - sim_mt
    a = a / torch.sqrt(
            torch.sum(a ** 2, axis=-1, keepdims=True) + 1e-10)
    b = b / torch.sqrt(
            torch.sum(b ** 2, axis=-1, keepdims=True) + 1e-10)
    # print("check check cehck")
    dis_sim_mt = (1 - torch.matmul(a, b.t()))/gamma
    # dis_sim_mt = torch.norm(a[:, None] - b, dim=2)
    return dis_sim_mt
    
def find_summary_assignment(sub_ind, matrix):
    i = matrix.shape[0] - 1  # row
    j = matrix.shape[1] - 1  # col
    assignment = []
    assignment.append([j, i])
    while j > 0:
        option_diag = matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf
        option_left = matrix[i, j - 1] if j > 0 else np.inf
        move = np.argmin([option_diag, option_left])
        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            j -= 1
        assignment.append([j,i])
    assignment = sorted([ item[1] for item in assignment])
    assignment = sub_ind[assignment]
    # dict_assignment = dict(assignment)
    # print(dict_assignment)
    return assignment
'''
def obtain_video_features(video):
    
    # model name is i3d default, can make it into argument later
    # if clip size less than 4, then error?
    
    # config_file = '../../../scripts/action-recognition/configuration/i3d_resnet50_v1_kinetics400.yaml'

    feature_output = []
    # config_file = './configuration/i3d_resnet50_v1_feat.yaml'  # Update this path if needed
    config_file = './configuration/i3d_resnet50_v1_feat.yaml'
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_file)
    model = get_model(cfg)
    model.eval()
    for i in range(video.shape[0]):
        video_data_per_clip = video[i]
        transform_fn = video_transforms.Compose([
                                             volume_transforms.ClipToTensor(),
                                             video_transforms.Normalize(
                                                 mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])])
        clip_input = transform_fn(video_data_per_clip)

        with torch.no_grad():
            feature = model(torch.unsqueeze(clip_input, dim=0))

            feature_output.append(feature)
    return torch.cat(feature_output)
'''
def set_difference_tensor(t1, t2):
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1]
    return difference
   
def greedy_vectorized(X, k, device):
    # removed to args.device
    # device = args.device
    num_datapoints = X.size()[0]
    assignment = torch.zeros(num_datapoints, dtype=torch.long, requires_grad=False)
    # index = torch.arange(num_datapoints)
    index = torch.arange(num_datapoints).to(device)

    distance = torch.norm(X[:, None] - X, dim=2)
    # max_value = torch.max(distance)+1

    for i in range(k):
        if i == 0:
            temp_index = torch.argmin(torch.sum(distance, dim=0))
            assignment = temp_index
            # distance[temp_index,:] = max_value
        if i >= 1:
            # print(i)
            index_temp = set_difference_tensor(index, torch.unique(assignment))
            # print(torch.unique(assignment))
            temp_value = torch.zeros([num_datapoints, len(index_temp)])
            for j in range(len(index_temp)):
                index_to_check = torch.cat(
                    [torch.unique(assignment), torch.tensor([index_temp[j]], 
                                                            # dtype=torch.long)])
                                                            dtype=torch.long).to(device)])

                temp_value[:, j] = torch.min(distance[:, index_to_check], 
                                             dim=1).values  # memorize all marginal gain
            index_i = index_temp[torch.argmin(torch.sum(temp_value, dim=0))]
            index_to_build = torch.cat(
                [torch.unique(assignment), torch.tensor([index_i], 
                                                        # dtype=torch.long)])
                                                        dtype=torch.long).to(device)])

            assignment = index_to_build[torch.argmin(distance[:, index_to_build],
                                                     dim=1)]
    # print("checking point")

    return assignment

def greedy_vectorized_check(X, SortedS):
    num_datapoints = X.size()[0]
    assignment = torch.zeros(num_datapoints, dtype=torch.long, requires_grad=False)
    distance = torch.norm(X[:, None] - X, dim=2)
    assignment = SortedS[torch.argmin(distance[:, SortedS],
                                                     dim=1)]
    return assignment


# def get_raw_video(video_path, framerate=10, n_frames=32):
#     if os.path.isfile(video_path):
#         height, width = 224, 224
#         #framerate = 10
#         #centercrop = True # To check
#         cmd = (
#             ffmpeg
#             .input(video_path)
#             .filter('fps', fps=framerate)
#             .filter('scale', width, height)
#         )
#         #if centercrop:
#         #    x = int((width - size) / 2.0)
#         #    y = int((height - size) / 2.0)
#         #    cmd = cmd.crop(x, y, size, size)
#         out, _ = (
#             cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
#             .run(capture_stdout=True, quiet=True)
#         )
#         video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
#         t = video.shape[0]
#         rmd = t % n_frames
#         #print(video.shape)
#         if not rmd == 0:
#             video = np.concatenate([video, np.zeros([n_frames-rmd, height, width, 3], dtype=np.uint8)])
#         #print(video.shape)

#         video = video.reshape([-1, n_frames, height, width, 3])
#         video = video.astype('float32')
#         # video = (video) / 255
#     return video

'''
def preprocessing(root_path, task_name):
    # obtain video info
    video_list_path = root_path+'videos/'+task_name+"/"
    temp_video_list = []
    for _,dirs,name in os.walk(video_list_path):
        for _ in name:
            if  _ != '.DS_Store':
                temp_video_list.append(_)
    print("task:", task_name, "num of video: ", len(temp_video_list))
    
    feature_list_dict = {}
    header = ['video_name', 'num_clips']
    video_info = []
    for video_name in temp_video_list:
        # try:
            video_path = video_list_path+video_name
            video =get_raw_video(video_path)
            features =  obtain_video_features(video)
            feature_list_dict[video_name[:-4]] = features
            video_info.append([video_name[:-4], len(features)])
            print(video_name,'contains %d clips' % len(features), "finished!")
        # except:
        #     print(video_name, "loading has error, passed")
        #     pass
    with open(task_name+'_info.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(video_info)
    torch.save(feature_list_dict, task_name+"_features_dict.pt")
    return feature_list_dict
'''
def get_annotation_by_fid(fid, annot_dir):
    annot = []
    annot_fname = os.path.join(annot_dir, fid+'.csv')
    if not os.path.exists(annot_fname):
        print('{} not exists'.format(annot_fname))
        return None

    with open(annot_fname, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')#, quotechar='|')
        for row in reader:
            step_id = int(row[0])
            begin_time = float(row[1])
            end_time = float(row[2])
            annot.append([step_id, begin_time, end_time])
    return annot            

def get_annotation_by_name(annot_fname):
    annot=[]
    if not os.path.exists(annot_fname):
        print('{} not exists'.format(annot_fname))
        return None

    with open(annot_fname, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')#, quotechar='|')
        for row in reader:
            step_id = int(row[0])
            begin_time = float(row[1])
            end_time = float(row[2])
            annot.append([step_id, begin_time, end_time])
    return annot   

def get_annotations(fid_list, annot_dir):
    annotations = {}
    for fid in fid_list:
        annot = get_annotation_by_fid(fid, annot_dir)
        annotations[fid] = annot
    return annotations            
                        
def annot_to_framelabel(annot, num_frames, fps):
    """
       Get framewise labels
    """
    # print(annot)
    label = np.zeros([num_frames], dtype=np.int32)
    for step_annot in annot:
        step, begin_time, end_time = step_annot
        begin_frame = int(np.floor(begin_time * fps))
        end_frame = int(np.ceil(end_time * fps))
        if end_frame > num_frames:
            label[begin_frame:] = step 
            continue
        label[begin_frame:end_frame] = step 
    return label

def get_framewise_label_list(annot_dir, fid_list, num_clip_list):
    """
       Load labels from annotations
       fps is set by default as 10
    """
    annotations = get_annotations(fid_list, annot_dir)
    label_list = []
    for i, fid in enumerate(fid_list):
        label = annot_to_framelabel(annotations[fid], num_clip_list[i]*32, fps=10)
        label_list.append(label)
    return label_list



def get_clipwise_label_list(framelabel_list, clip_num_list):
    clipwise_label_list = []
    for i in range(len(clip_num_list)):
        clipwise_label = np.ones([clip_num_list[i]], dtype=np.int32) * -1
        for j in range(len(clipwise_label)):
            temp = np.arange(start=j*32, stop=(j+1)*32)
            clip_labels = framelabel_list[i][temp]
            values, counts = np.unique(clip_labels, return_counts=True)
            clipwise_label[j] = values[np.argmax(counts)]
            # print(np.asarray([values,counts]).T, clipwise_label[j])
        clipwise_label_list.append(clipwise_label)
    return clipwise_label_list

def framewise_eval(pred_list, label_list):
    if isinstance(pred_list, list):
        preds = np.concatenate(pred_list)
    elif isinstance(pred_list, np.ndarray):
        preds=pred_list
    labels = np.concatenate(label_list)

    k_pred = int(preds.max()) + 1
    k_label = int(labels.max()) + 1

    overlap = np.zeros([k_pred, k_label])
    for i in range(k_pred):
        for j in range(k_label):
            overlap[i, j] = np.sum((preds==i) * (labels==j))
    row_ind, col_ind = scipy.optimize\
    .linear_sum_assignment(-overlap / preds.shape[0])
    K = max(k_pred, k_label)
    
    bg_row_ind = np.concatenate([row_ind, -np.ones(K+1-row_ind.shape[0], 
                                                   dtype=np.int32)]) # for label
    bg_col_ind = np.concatenate([col_ind, -np.ones(K+1-col_ind.shape[0], 
                                                   dtype=np.int32)]) # for preds
    acc = np.mean(bg_col_ind[preds]==bg_row_ind[labels])
    acc_steps = np.mean(bg_col_ind[preds[labels>=0]]==bg_row_ind[labels[labels>=0]])
    acc_bg = np.mean(bg_col_ind[preds[labels==-1]]==bg_row_ind[labels[labels==-1]])
    
    results = []
    for i, p in enumerate(row_ind):
        correct = preds[labels==col_ind[i]] == p
        if correct.shape[0] == 0:
            num_correct = 0
        else:
            num_correct = np.sum(correct)
        num_label = np.sum(labels==col_ind[i])
        num_pred = np.sum(preds==p)
        results.append([num_correct, num_label, num_pred])
    
    _results = np.array(results)
    # print("print(_results)",_results)

    for i in range(k_pred):
        if i not in row_ind:
            num_correct = 0
            num_label = 0
            num_pred = np.sum(preds==i)
            results.append([num_correct, num_label, num_pred])

    for j in range(k_label):
        if j not in col_ind:
            num_correct = 0
            num_label = np.sum(labels==j)
            num_pred = 0
            results.append([num_correct, num_label, num_pred])

    results = np.array(results)
    precision = np.sum(results[:, 0]) / (np.sum(results[:, 2]) + 1e-10)
    recall = np.sum(results[:, 0]) / (np.sum(results[:, 1]) + 1e-10)
    fscore = 2 * precision * recall / (precision + recall + 1e-10)

    return [precision, recall, fscore, acc, acc_bg, np.count_nonzero(preds==-1), np.count_nonzero(labels==-1), len(labels)]


def framewise_eval_pure(pred_list, label_list):
    if isinstance(pred_list, list):
        preds = np.concatenate(pred_list)
    elif isinstance(pred_list, np.ndarray):
        preds=pred_list
    labels = np.concatenate(label_list)

    acc = np.mean(preds==labels)
    acc_bg = np.mean(preds[labels==0]==labels[labels==0])
    # acc_nb = np.mean(preds[labels!=0]==labels[labels!=0])

    precision = np.sum(preds[labels>=1]==labels[labels>=1])/(np.sum(preds>=1)+ 1e-10)
    recall = np.sum(preds[labels>=1]==labels[labels>=1])/(np.sum(labels>=1)+1e-10)
    fscore = 2 * precision * recall / (precision + recall + 1e-10)
    return [precision, recall, fscore, acc, acc_bg, np.count_nonzero(preds==0), np.count_nonzero(labels==0), len(labels)]

def new_framewise_eval_pure(pred_list, label_list):
    if isinstance(pred_list, list):
        preds = np.concatenate(pred_list)
    elif isinstance(pred_list, np.ndarray):
        preds=pred_list
    labels = np.concatenate(label_list)

    acc = np.mean(preds==labels)
    acc_bg = np.mean(preds[labels==0]==labels[labels==0])
    acc_nb = np.mean(preds[labels!=0]==labels[labels!=0])

    precision = np.sum(preds[labels>=1]==labels[labels>=1])/(np.sum(preds>=1)+ 1e-10)
    recall = np.sum(preds[labels>=1]==labels[labels>=1])/(np.sum(labels>=1)+1e-10)
    fscore = 2 * precision * recall / (precision + recall + 1e-10)
    return [precision, recall, fscore, acc, acc_nb, acc_bg, np.count_nonzero(preds==0), np.count_nonzero(labels==0), len(labels)]



'''

def framewise_eval(pred_list, label_list):
    preds = np.concatenate(pred_list)
    labels = np.concatenate(label_list)

    k_pred = int(preds.max()) + 1
    k_label = int(labels.max()) + 1

    overlap = np.zeros([k_pred, k_label])
    for i in range(k_pred):
        for j in range(k_label):
            overlap[i, j] = np.sum((preds==i) * (labels==j))
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(-overlap / preds.shape[0])
    K = max(k_pred, k_label)
    
    bg_row_ind = np.concatenate([row_ind, -np.ones(K+1-row_ind.shape[0], dtype=np.int32)])
    bg_col_ind = np.concatenate([col_ind, -np.ones(K+1-col_ind.shape[0], dtype=np.int32)])
    acc = np.mean(bg_col_ind[preds]==bg_row_ind[labels])
    acc_steps = np.mean(bg_col_ind[preds[labels>=0]]==bg_row_ind[labels[labels>=0]])
    
    results = []
    for i, p in enumerate(row_ind):
        correct = preds[labels==col_ind[i]] == p
        if correct.shape[0] == 0:
            num_correct = 0
        else:
            num_correct = np.sum(correct)
        num_label = np.sum(labels==col_ind[i])
        num_pred = np.sum(preds==p)
        results.append([num_correct, num_label, num_pred])

    for i in range(k_pred):
        if i not in row_ind:
            num_correct = 0
            num_label = 0
            num_pred = np.sum(preds==i)
            results.append([num_correct, num_label, num_pred])

    for j in range(k_label):
        if j not in col_ind:
            num_correct = 0
            num_label = np.sum(labels==j)
            num_pred = 0
            results.append([num_correct, num_label, num_pred])

    results = np.array(results)

    precision = np.sum(results[:, 0]) / (np.sum(results[:, 2]) + 1e-10)
    recall = np.sum(results[:, 0]) / (np.sum(results[:, 1]) + 1e-10)
    fscore = 2 * precision * recall / (precision + recall + 1e-10)

    return [precision, recall, fscore, acc, acc_steps, np.count_nonzero(preds==-1)]
'''
def find_match_index(S_x, S_y, D):
    S_x = S_x.numpy()
    S_y = S_y.numpy()
    q = D.shape[0]-1 # 1st list, for indexing we minus 1
    j = D.shape[1]-1  # 2nd list, for indexing we minus 1 
    matched_index = []
    s_x_key = []
    s_y_key = []
    for i in range(q,-1,-1):
        if (j+1) % 2 ==1: # if j is odd
            j = np.argmin(D[i,:j+1])
        else:
            j = np.argmin(D[i,:j])
        if (j+1) %2 ==0: 
            index = int((j+1)/2 - 1)
            matched_index.insert(0, (S_y[index], S_x[i]))
            s_x_key.append(S_x[i])
            s_y_key.append(S_y[index])
    return s_x_key, s_y_key, matched_index

def gnd_matrix_builder(assign, clipwise_label, k):
    temp = torch.zeros(len(assign), dtype=torch.long)
    # print(temp)
    gnd_k_assign = torch.zeros([k, len(assign)], dtype=torch.long)
    gnd_bg_assign = torch.zeros(len(assign), dtype=torch.long)
    for i in range(len(assign)):
        temp[i] = clipwise_label[assign[i]]
        if temp[i] != -1:
            gnd_k_assign[temp[i], i] = 1
        elif temp[i] == -1:
            gnd_bg_assign[i] = 1 
    return gnd_k_assign, gnd_bg_assign, temp

def assign_matrix_builder(assign, matched_index, k):
    k_assign_m = torch.zeros([k, len(assign)])
    index_list = {}
    for i in range(k):
        for pair in matched_index:
            if pair[0] == i:
                index_list[i]=[]
                for j in range(len(assign)):
                    if assign[j] == pair[1]:
                        index_list[i].append(j)
    # build keystep assign matrix
    for key in list(index_list.keys()):
        k_assign_m[key,index_list[key]]=1
    # build background assign vector
    bg_assign = torch.zeros(len(assign), dtype=torch.long)
    for i in range(len(bg_assign)):
        if torch.sum(k_assign_m[:,i]) == 0:
            bg_assign[i] = 1

    # make clip pred list
    clip_pred = torch.ones(len(assign)) *-1
    clip_pred =clip_pred.to(torch.long)
    for i in range(len(k_assign_m)):
        if torch.sum(k_assign_m[i,:]) == 0:
            continue
        else:
            ind_list = torch.nonzero(k_assign_m[i,:],as_tuple=False).view(-1)
            clip_pred[ind_list] = i
    return k_assign_m, bg_assign, clip_pred

def make_ordered_center(features_list_dict, k):
    features = []
    orders = []
    avail_videos = list(features_list_dict.keys())
    for key in avail_videos:
        feat = features_list_dict[key]
        temp_order = np.arange(len(feat))/len(feat)
        orders.append(temp_order)
        features.append(feat)
    orders = np.concatenate(orders, axis=0)
    features = torch.cat(features).numpy()
    kmeans = KMeans(n_clusters=k, random_state=0).fit(features)
    preds = kmeans.labels_
    order_collection = {}
    prototype_order = np.zeros(k)
    for i in range(k):
        order_collection[i] = []
        for j in range(len(preds)):
            if preds[j] == i:
                order_collection[i].append(orders[j])
        prototype_order[i] = sum(order_collection[i])/len(order_collection[i])
    protoype_data = kmeans.cluster_centers_
    print(prototype_order[np.argsort(prototype_order)])
    sorted_ = protoype_data[np.argsort(prototype_order)]
    return sorted_


def make_new_ordered_center(features_list_dict,train_indices, k):
    features = []
    orders = []
    p_data = []
    avail_videos = list(features_list_dict.keys())
    # use train only
    avail_videos = [avail_videos[_] for _ in train_indices]
    for i in range(k):
        feat_list = []
        # for j in range(len(avail_videos)):
        for key in avail_videos:
            # key = avail_videos[j]
            feat = F.normalize(features_list_dict[key])
            ind = np.arange(feat.size()[0])
            split_ind = np.array_split(ind, k)
            feat_list.append(feat[split_ind[i]])
        feat_list =torch.cat(feat_list).numpy()
        p_data.append(np.mean(feat_list, axis=0))
    p_data = np.stack(p_data)
    # add background
    for key in avail_videos:
        feat = F.normalize(features_list_dict[key])
        features.append(feat)
    features =torch.cat(features).numpy()
    p_bg = np.mean(features, axis=0).reshape((1,512))
 
    # p_data=np.concatenate([p_bg, p_data])
    # print(p_data.shape)
    return p_data

def temp_make_ordered_center(features_list_dict, clip_label_list, k):
    orders = []
    avail_videos = list(features_list_dict.keys())
    prototype_data = torch.zeros([k, features_list_dict[avail_videos[0]].size()[1]])
    prototype_order = np.zeros(k)
    p_order = {}
    for i in range(k):
        features = []
        p_order[i] = []
        for j in range(len(avail_videos)):
            key = avail_videos[j]
            feat = F.normalize(features_list_dict[key])
            for _ in range(len(feat)):
                if clip_label_list[j][_] == i:
                    features.append(feat)
                    p_order[i].append(_/len(feat))
        
        features = torch.cat(features)
        prototype_data[i] = torch.mean(features, dim=0)
        prototype_order[i] = sum(p_order[i])/len(p_order[i])
    # print(prototype_order[np.argsort(prototype_order)])
    sorted_ = prototype_data[np.argsort(prototype_order)]            
    return sorted_

def make_pred(k_assign_m, bg_assign, num_clip):
    clip_pred = bg_assign.detach().numpy() * -1
    for i in range(len(k_assign_m)):
        if torch.sum(k_assign_m[i,:]) == 0:
            continue
        else:
            ind_list = torch.nonzero(k_assign_m[i,:],as_tuple=False).view(-1)
            clip_pred[ind_list] = i
    # print(num_clip)
    frame_pred = np.ones([num_clip*32], dtype=np.int32) * -1
    for _ in range(len(clip_pred)):
        # print(_)
        temp = np.arange(start=_*32, stop=(_+1)*32)
        frame_pred[temp]=clip_pred[_]
    return frame_pred, clip_pred

def stw_match_cost(pairwise_cost, ass):
    '''
    to obtain the match quality, so we can start with high confident videos
    '''
    temp_ind = torch.arange(len(pairwise_cost),dtype=torch.long)
    return torch.sum(pairwise_cost[temp_ind, ass])#/(len(pairwise_cost)-len(torch.unique(ass)))


def pred(clip_output, sum_ass):
    clip_pred = clip_output.data.max(1)[1].squeeze(0).cpu().numpy()
    sum_pred = np.copy(clip_pred)
    for s in torch.unique(sum_ass).numpy():
        sum_pred[sum_ass.numpy()==s]=clip_pred[s]
    clip_pred_to_frame = np.ones([len(clip_pred)*32], dtype=np.int32) * -1
    sum_pred_to_frame = np.ones([len(clip_pred)*32], dtype=np.int32) * -1
                # clip-level tp frame-level
    for _ in range(len(clip_pred)):
        temp1 = np.arange(start=_*32, stop=(_+1)*32)
        clip_pred_to_frame[temp1]=clip_pred[_]
        sum_pred_to_frame[temp1]=sum_pred[_]
    return clip_pred_to_frame, sum_pred_to_frame

def graph_helper(label):
    # remove background and ignored index and repeat 
    new_label = [0]
    for _ in label:
        if _ != 0 and  _ != -100 and _ != new_label[-1]:
            new_label.append(_)
    # remove first element
    # print(new_label)
    new_label.pop(0)
    # print(new_label)
    # create edges
    edge_list = []
    for i in range(len(new_label)-1):
        edge_list.append((new_label[i],new_label[i+1]))

    return edge_list, new_label

def new_graph_helper(label):
    # remove background and ignored index and repeat 
    new_label = [0]
    for _ in label:
        if _ != -100 and _ != new_label[-1]:
            new_label.append(_)
    # remove first element
    # print(new_label)
    new_label.pop(0)
    # print(new_label)
    # create edges
    edge_list = []
    for i in range(len(new_label)-1):
        edge_list.append((new_label[i],new_label[i+1]))

    return edge_list, new_label

def stw_label_helper(new_batch_target, stw_assignment, num_neighbor=1):
    pseduo_batch_target = torch.ones_like(new_batch_target) *-100
    summaries = torch.unique(stw_assignment)
    for summary in summaries:
        # locate segs 
        cur_seg_length = sum(stw_assignment==summary)
        # print("summary", summary, cur_seg_length)
        if cur_seg_length <= 1+num_neighbor*2:
            # label entire seg
            pseduo_batch_target[:,stw_assignment==summary]=\
            new_batch_target[:,summary]
        else:
            # pick the summary and label its neighbors
            # print("current summary", summary, "current label", new_batch_target[:,summary],[summary-1, summary+2] )
            temp_pseudo_range = torch.arange(start=max(0,summary-num_neighbor), 
                                             end=min(summary+num_neighbor+1, pseduo_batch_target.size()[1]),
                                             step=1)
            pseduo_batch_target[:,temp_pseudo_range]=\
            new_batch_target[:,summary]
            # print(pseduo_batch_target[:,temp_pseudo_range])

    return pseduo_batch_target

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def entropy_mc_helper(model, batch_input, mask, num_classes, num_clip_to_label, device, forward_passes=50):
    # support only 1 video, perform clip selection
    dropout_predictions = torch.zeros((forward_passes,batch_input.size()[2], num_classes)).to(device)
    for i in range(forward_passes):
        model.eval()
        enable_dropout(model)
        with torch.no_grad():
            ps,fs = model(batch_input, mask)
            dropout_predictions[i,:,:] = F.softmax(ps[-1].squeeze(0).t(),dim=1)
    # computer mean
    mean_dropout_pred = torch.mean(dropout_predictions, dim=0) # after shape: num_clip * num_classes
    # computer entropy: -sum(p_i * lg(p)i))
    entropy = -torch.sum(torch.mul(mean_dropout_pred, torch.log(mean_dropout_pred)),dim=1)
    label_index = torch.argsort(entropy, descending=True)[:num_clip_to_label]
    return label_index

def entropy_mc_helper_2(model, batch_input, mask, num_classes, num_clip_to_label, device, forward_passes=50):
    # support only 1 video, perform clip selection
    dropout_predictions = torch.zeros((forward_passes,batch_input.size()[2], num_classes)).to(device)
    for i in range(forward_passes):
        model.eval()
        enable_dropout(model)
        with torch.no_grad():
            ps,fs = model(batch_input, mask)
            dropout_predictions[i,:,:] = F.softmax(ps[-1].squeeze(0).t(),dim=1)
    # computer mean
    mean_dropout_pred = torch.mean(dropout_predictions, dim=0) # after shape: num_clip * num_classes
    # computer entropy: -sum(p_i * lg(p)i))
    entropy = -torch.sum(torch.mul(mean_dropout_pred, torch.log(mean_dropout_pred)),dim=1)
    # label_index = torch.argsort(entropy, descending=True)[:num_clip_to_label]
    return entropy

if __name__ == '__main__':
        
        pass

