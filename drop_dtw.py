import numpy as np
import torch
import torch.nn.functional as F
from core import *

# def compute_all_costs(step_features, frame_features, distractor, gamma_xz, drop_cost_type, keep_percentile, l2_nomalize=False):
#     """This function computes pairwise match and individual drop costs used in Drop-DTW

#     Parameters
#     __________

#     sample: dict
#         sample dictionary
#     distractor: torch.tensor of size [d] or None
#         Background class prototype. Only used if the drop cost is learnable.
#     distractor: torch.tensor of size [d] or None
#         Background class prototype. Only used if the drop cost is learnable.
#     drop_cost_type: str
#         The type of drop cost definition, i.g., learnable or logits percentile.
#     keep_percentile: float in [0, 1]
#         if drop_cost_type == 'logit', defines drop (keep) cost threshold as logits percentile
#     l2_normalize: bool
#         wheather to normalize clip and step features before computing the costs
#     """

#     # labels = sample['step_ids']
#     # step_features, frame_features = sample['step_features'], sample['frame_features']
#     if l2_nomalize:
#         frame_features = F.normalize(frame_features, p=2, dim=1)
#         step_features = F.normalize(step_features, p=2, dim=1)
#     sim = step_features @ frame_features.T

#     # unique_labels, unique_index, unique_inverse_index = np.unique(
#     #     labels.detach().cpu().numpy(), return_index=True, return_inverse=True)
#     # unique_sim = sim[unique_index]
#     unique_sim = len(frame_features)
#     if drop_cost_type == 'logit':
#         k = max([1, int(torch.numel(unique_sim) * keep_percentile)])
#         baseline_logit = torch.topk(unique_sim.reshape([-1]), k).values[-1].detach()
#         baseline_logits = baseline_logit.repeat([1, unique_sim.shape[1]])  # making it of shape [1, N]
#         sims_ext = torch.cat([unique_sim, baseline_logits], dim=0)
#     elif drop_cost_type == 'learn':
#         distractor_sim = frame_features @ distractor
#         sims_ext = torch.cat([unique_sim, distractor_sim[None, :]], dim=0)
#     else:
#         assert False, f"No such drop mode {drop_cost_type}"

#     unique_softmax_sims = torch.nn.functional.softmax(sims_ext / gamma_xz, dim=0)
#     unique_softmax_sim, drop_probs = unique_softmax_sims[:-1], unique_softmax_sims[-1]
#     matching_probs = unique_softmax_sim[unique_inverse_index]
#     zx_costs = -torch.log(matching_probs + 1e-5)
#     drop_costs = -torch.log(drop_probs + 1e-5)
#     return zx_costs, drop_costs, drop_probs
def drop_dtw(zx_costs, drop_costs, exclusive=True, contiguous=False, return_labels=False):
    """Drop-DTW algorithm that allows drop only from one (video) side. See Algorithm 1 in the paper.

    Parameters
    ----------
    zx_costs: np.ndarray [K, N] 
        pairwise match costs between K steps and N video clips
    drop_costs: np.ndarray [N]
        drop costs for each clip
    exclusive: bool
        If True any clip can be matched with only one step, not many.
    contiguous: bool
        if True, can only match a contiguous sequence of clips to a step
        (i.e. no drops in between the clips)
    return_label: bool
        if True, returns output directly useful for segmentation computation (made for convenience)
    """
    K, N = zx_costs.shape
    
    # initialize solutin matrices
    D = np.zeros([K + 1, N + 1, 2]) # the 2 last dimensions correspond to different states.
                                    # State (dim) 0 - x is matched; State 1 - x is dropped
    D[1:, 0, :] = np.inf  # no drops in z in any state
    D[0, 1:, 0] = np.inf  # no drops in x in state 0, i.e. state where x is matched
    D[0, 1:, 1] = np.cumsum(drop_costs)  # drop costs initizlization in state 1

    # initialize path tracking info for each state
    P = np.zeros([K + 1, N + 1, 2, 3], dtype=int) 
    for xi in range(1, N + 1):
        P[0, xi, 1] = 0, xi - 1, 1
    
    # filling in the dynamic tables
    for zi in range(1, K + 1):
        for xi in range(1, N + 1):
            # define frequently met neighbors here
            diag_neigh_states = [0, 1] 
            diag_neigh_coords = [(zi - 1, xi - 1) for _ in diag_neigh_states]
            diag_neigh_costs = [D[zi - 1, xi - 1, s] for s in diag_neigh_states]

            left_neigh_states = [0, 1]
            left_neigh_coords = [(zi, xi - 1) for _ in left_neigh_states]
            left_neigh_costs = [D[zi, xi - 1, s] for s in left_neigh_states]

            left_pos_neigh_states = [0] if contiguous else left_neigh_states
            left_pos_neigh_coords = [(zi, xi - 1) for _ in left_pos_neigh_states]
            left_pos_neigh_costs = [D[zi, xi - 1, s] for s in left_pos_neigh_states]

            top_pos_neigh_states = [0]
            top_pos_neigh_coords = [(zi - 1, xi) for _ in left_pos_neigh_states]
            top_pos_neigh_costs = [D[zi - 1, xi, s] for s in left_pos_neigh_states]

            z_cost_ind, x_cost_ind = zi - 1, xi - 1  # indexind in costs is shifted by 1

            # state 0: matching x to z
            if exclusive:
                neigh_states_pos = diag_neigh_states + left_pos_neigh_states
                neigh_coords_pos = diag_neigh_coords + left_pos_neigh_coords
                neigh_costs_pos = diag_neigh_costs + left_pos_neigh_costs
            else:
                neigh_states_pos = diag_neigh_states + left_pos_neigh_states + top_pos_neigh_states
                neigh_coords_pos = diag_neigh_coords + left_pos_neigh_coords + top_pos_neigh_coords
                neigh_costs_pos = diag_neigh_costs + left_pos_neigh_costs + top_pos_neigh_costs
            costs_pos = np.array(neigh_costs_pos) + zx_costs[z_cost_ind, x_cost_ind] 
            opt_ind_pos = np.argmin(costs_pos)
            P[zi, xi, 0] = *neigh_coords_pos[opt_ind_pos], neigh_states_pos[opt_ind_pos]
            D[zi, xi, 0] = costs_pos[opt_ind_pos]

            # state 1: x is dropped
            costs_neg = np.array(left_neigh_costs) + drop_costs[x_cost_ind] 
            opt_ind_neg = np.argmin(costs_neg)
            P[zi, xi, 1] = *left_neigh_coords[opt_ind_neg], left_neigh_states[opt_ind_neg]
            D[zi, xi, 1] = costs_neg[opt_ind_neg]

    cur_state = D[K, N, :].argmin()
    min_cost = D[K, N, cur_state]
    
    # backtracking the solution
    zi, xi = K, N
    path, labels = [], np.zeros(N)
    x_dropped = [] if cur_state == 1 else [N]

    while not (zi == 0 and xi == 0):
        # print("zi,xi", zi,xi)
        # print("path", path)
        path.append((zi, xi))
        zi_prev, xi_prev, prev_state = P[zi, xi, cur_state]
        if xi > 0:
            labels[xi - 1] = zi * (cur_state == 0)  # either zi or 0
        if prev_state == 1:
            x_dropped.append(xi_prev)
        zi, xi, cur_state = zi_prev, xi_prev, prev_state
        # print("after")
        # print("zi,xi", zi,xi)
        # print("path", path)
    
    if not return_labels:
        return min_cost, path, x_dropped
    else:
        return labels


if __name__ == '__main__':
    step_features = torch.tensor([[0,0,0,0,0,1.0],
                                  [0,0,0,0,0,1.0],
                                [0,0,0,0,1,0],
                                [0,0,0,1.,0,0]])

    frame_features = torch.tensor([[0,0,0,0,0,1],
                                [0,0,0,0,0,1],
                                # [0,0,0,0,0,1],
                                [0,0,0,0,1,0],
                                [0,0,0,0,1,0],
                                [0,0,0,0,0,0],
                                [0,0,0,1.,0,0]])


    p_clip_dis = dis_sim_matrix(step_features, frame_features)
    # print(p_clip_dis)
    drop_cost = np.ones(len(frame_features))*0.5
    min_cost, path, x_dropped = drop_dtw(p_clip_dis.numpy(), drop_cost)
    print("min_cost, path, x_dropped")
    print(min_cost, path, x_dropped)