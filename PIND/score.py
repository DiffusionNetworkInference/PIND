from time import time
import numpy as np
import time
import matplotlib.pyplot as plt 


def cal_F1(ground_truth_network, inferred_network):
    TP = np.sum(ground_truth_network + inferred_network == 2)
    FP = np.sum(ground_truth_network - inferred_network == -1)
    FN = np.sum(ground_truth_network - inferred_network == 1)
    epsilon = 1e-20
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f_score = 2 * precision * recall / (precision + recall + epsilon)

    return precision, recall, f_score


def cal_mae(groundtruth_network, p, infer_p):
    gt_p = groundtruth_network*p
    edges_num = np.sum(groundtruth_network)
    temp = gt_p.copy()
    temp[temp==0]=1
    temp_infer_p=infer_p.copy()
    temp_infer_p = groundtruth_network*temp_infer_p

    mae = np.sum(abs(temp_infer_p-gt_p)/temp)/edges_num

    return mae


def cal_mae_v2(groundtruth_network, p, infer_p):
    gt_p = groundtruth_network*p
    edges_num = np.sum(groundtruth_network)
    temp_infer_p = groundtruth_network*infer_p

    mae_v2 = np.sum(abs(temp_infer_p-gt_p))/edges_num
    
    return mae_v2


def cal_mse(p, infer_p):
    mse = np.mean(np.square(p-infer_p))

    return mse


def cal_mse_v2(groundtruth_network, p, infer_p):
    gt_p = groundtruth_network*p
    edges_num = np.sum(groundtruth_network)
    mse_v2=np.sum(np.square(groundtruth_network*infer_p-gt_p))/edges_num

    return mse_v2


def modify_p(groundtruth_network, infer_p):
    edges_num=np.sum(groundtruth_network)
    temp=infer_p*groundtruth_network
    mean_p=np.sum(temp)/edges_num
    modified_p=infer_p/mean_p*0.3

    return modified_p
    
def cal_mae_all(p, infer_p):
    return np.mean(abs(p-infer_p))


def draw_values(ground_truth_network, value_matrix):
    temp1=np.squeeze(ground_truth_network.reshape((1,-1)))
    temp2=np.squeeze(value_matrix.reshape((1,-1)))
    edge_list=[]
    for i in range(temp1.size):
        edge_list.append((temp2[i],temp1[i]))
    sorted_edge=sorted(edge_list, key=lambda x: x[0])
    green_index = []
    green_value = []
    red_index = []
    red_value = []
    for i in range(len(sorted_edge)):
        if sorted_edge[i][1]==0:
            red_index.append(i)
            red_value.append(sorted_edge[i][0])
        else:
            green_index.append(i)
            green_value.append(sorted_edge[i][0])

    plt.scatter(red_index,red_value,color='r')
    plt.scatter(green_index,green_value,color='g')
    plt.show()