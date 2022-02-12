from cmath import nan
from re import T
import numpy as np
from scipy.optimize import fsolve
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math
import time
import re
from score import *

def load_data(graph_path, result_path):
    with open(result_path, 'r') as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
        diffusion_result = np.array([[int(state) for state in line] for line in lines])

    nodes_num = diffusion_result.shape[1]

    with open(graph_path, 'r') as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
        data = np.array([[float(node) for node in line] for line in lines])
        ground_truth_network = np.zeros((nodes_num, nodes_num))
        ground_truth_p = np.zeros((nodes_num, nodes_num))
        edges_num = data.shape[0]
        for i in range(edges_num):
            ground_truth_network[int(data[i, 0]) - 1, int(data[i, 1]) - 1] = 1
            ground_truth_p[data[i, 0] - 1, data[i, 1] - 1] = data[i,2]
        

    return ground_truth_network, diffusion_result, ground_truth_p


def generate_prob_result(diffusion_result,mean,scale,prob_result_path,read_flag=False):
    if read_flag:
        prob_result=np.loadtxt(prob_result_path,delimiter=' ')
    elif not read_flag:
        beta,nodes_num=diffusion_result.shape
        bias=np.random.normal(mean,scale,(beta,nodes_num))
        bias[np.where(diffusion_result==1)]*=-1
        prob_result=diffusion_result+bias
        prob_result[np.where(prob_result>1)]=1
        prob_result[np.where(prob_result<0)]=0
        np.savetxt(prob_result_path,prob_result,fmt='%f',delimiter=' ')

    return prob_result


def weighted_mi_prune(record_states, prune_choice):
    # prune_choice = 0  mi   ;  prune_choice = 1 imi

    results_num, nodes_num = record_states.shape
    print("results_num = ", results_num)
    MI = np.zeros((nodes_num, nodes_num))

    for j in range(nodes_num):
        for k in range(nodes_num):
            if j >= k:
                continue
            state_mat = np.zeros((2, 2))
            for result_index in range(results_num):
                state_mat[0,0]+=(1-record_states[result_index,j])*(1-record_states[result_index,k])
                state_mat[0,1]+=(1-record_states[result_index,j])*record_states[result_index,k]
                state_mat[1,0]+=record_states[result_index,j]*(1-record_states[result_index,k])
                state_mat[1,1]+=record_states[result_index,j]*record_states[result_index,k]

            epsilon = 1e-5
            M00 = state_mat[0, 0] / results_num * math.log(
                state_mat[0, 0] * results_num / (state_mat[0, 0] + state_mat[0, 1]) / (
                        state_mat[0, 0] + state_mat[1, 0]) + epsilon, 2)
            M01 = state_mat[0, 1] / results_num * math.log(
                state_mat[0, 1] * results_num / (state_mat[0, 0] + state_mat[0, 1]) / (
                        state_mat[0, 1] + state_mat[1, 1]) + epsilon, 2)
            M10 = state_mat[1, 0] / results_num * math.log(
                state_mat[1, 0] * results_num / (state_mat[1, 0] + state_mat[1, 1]) / (
                        state_mat[0, 0] + state_mat[1, 0]) + epsilon, 2)
            M11 = state_mat[1, 1] / results_num * math.log(
                state_mat[1, 1] * results_num / (state_mat[1, 0] + state_mat[1, 1]) / (
                        state_mat[0, 1] + state_mat[1, 1]) + epsilon, 2)

            if prune_choice==0:
                MI[j, k] = M00 + M11 +M10 + M01
            else:
                MI[j, k] = M00 + M11 - abs(M10) - abs(M01)

            MI[k, j] = MI[j, k]

    # Kmeans cluster
    MI[np.where(MI<0)] = 0
    tmp_MI = MI.reshape((-1, 1))
    estimator = KMeans(n_clusters=2)
    estimator.fit(tmp_MI)
    label_pred = estimator.labels_
    temp_0 = tmp_MI[label_pred == 0]
    temp_1 = tmp_MI[label_pred == 1]

    if np.max(temp_0) > np.max(temp_1):
        tau = np.max(temp_1)
    else:
        tau = np.max(temp_0)

    prune_network = np.zeros((nodes_num, nodes_num))
    prune_network[np.where(MI>tau)] = 1


    for i in range(nodes_num):
        prune_network[i,i]=0

    return prune_network


def sample_x(x_edge_matrix, sample_times):
    # sampel x_edge_matrix
    x_matrix_list = []
    for i in range(sample_times):
        cur_x=np.zeros(x_edge_matrix.shape)
        sample = np.random.rand(*x_edge_matrix.shape)
        one_index = np.where(sample < x_edge_matrix)
        cur_x[one_index] = 1
        x_matrix_list.append(cur_x.copy())

    return x_matrix_list



def cal_loss(x_edge_matrix, p_matrix, prob_result):
    first_item=1-prob_result
    second_item=np.zeros(prob_result.shape)

    epsilon=1e-10
    beta,nodes_num=prob_result.shape
    for record_index in range(beta):
        temp_third = x_edge_matrix.copy()
        prob_l=prob_result[record_index,:].copy()
        for i in range(nodes_num):
            temp_third[i, i] = 0

        temp_third = temp_third * np.log(1 - prob_l.reshape((-1, 1)) * p_matrix + epsilon)
        sum_item = np.sum(temp_third, axis=0).reshape((1, -1))
        second_item[record_index,:]=sum_item.copy()

    loss=np.sum(np.square(first_item-second_item))

    return loss



def show_result(x_edge_matrix_list, p_matrix, prob_result, ground_truth_network):
    loss_list=[]
    for i in range(len(x_edge_matrix_list)):
        cur_matrix=x_edge_matrix_list[i].copy()
        cur_loss=cal_loss(cur_matrix,p_matrix,prob_result)
        loss_list.append(cur_loss)
    max_index=np.argmax(np.array(loss_list))
    max_edge=x_edge_matrix_list[max_index]
    precision, recall, f_score=cal_F1(ground_truth_network,max_edge)

    return precision, recall, f_score



def with_x_likelihood_update_x(p_matrix, s_matrix, prior_network, x_matrix, initial_epsilon, iter_cnt, epsilon_change, update_x_sum_threshold, groundtruth_network,sample_times,max_delta_x_threshold):
    # cal gradient of x, and update x
    inner_x_cnt=0
    small_value=1e-20
    pre_x_matrix=x_matrix.copy()
    while True:
        inner_x_cnt+=1
        # step 1: cal gradient
        beta, nodes_num = s_matrix.shape
        x_gradient_matrix = np.zeros(x_matrix.shape)
    
        for i in range(beta):
            for j in range(nodes_num):
                # parents = np.where(prior_network[:,j]==1)[0]

                # si=0 term
                x_j=x_matrix[:,j].copy()
                gradient_j_zero = np.zeros(nodes_num)
                temp = 1-p_matrix[:,j]*s_matrix[i]
                gradient_j_zero = prior_network[:,j]*(1-s_matrix[i,j])*np.log(temp+small_value)
                x_gradient_matrix[:,j]+=gradient_j_zero

                # si=1 term
                gradient_j_one = np.zeros(nodes_num)
                p = p_matrix[:, j].copy()
                s = s_matrix[i].copy()
                A = 1-np.prod((1-p*s+small_value)**x_j)
                orig_A=A.copy()
                if A==0:
                    A=np.inf
                    
                temp_gradient=prior_network[:,j]*(orig_A-1)*np.log(1-p_matrix[:,j]*s_matrix[i]+small_value)/A*s_matrix[i,j]
                gradient_j_one = temp_gradient.copy()
                x_gradient_matrix[:,j]+=gradient_j_one

        # step 2: update x
        if epsilon_change:
            epsilon = initial_epsilon/np.sqrt(iter_cnt)
        else:
            epsilon = initial_epsilon

        x_matrix+=epsilon*x_gradient_matrix

        # bound x_matrix
        x_matrix[np.where(x_matrix<0)]=0
        x_matrix[np.where(x_matrix>1)]=1

        # show x
        show_update_x(groundtruth_network, x_matrix, p_matrix, s_matrix, sample_times, inner_x_cnt)
        max_delta_x=np.max(abs(pre_x_matrix-x_matrix))
        delta_x_sum=np.sum(abs(pre_x_matrix-x_matrix))
        if delta_x_sum<update_x_sum_threshold or max_delta_x<max_delta_x_threshold or inner_x_cnt>30:
            break
    
        pre_x_matrix=x_matrix.copy()

    return x_matrix




def with_x_likelihood_update_p(p_matrix, s_matrix, prior_network, x_matrix, initial_epsilon, iter_cnt, epsilon_change, update_p_sum_threshold, groundtruth_network, groundtruth_p, max_delta_p_threshold, p_max_iteration):
    # cal gradient of p, and update p
    inner_p_cnt=0
    small_value=1e-20
    pre_p_matrix=p_matrix.copy()
    while True:
        begin_1=time.time()
        inner_p_cnt+=1
        # step 1: cal gradient
        beta, nodes_num = s_matrix.shape
        p_gradient_matrix = np.zeros(p_matrix.shape)

        for i in range(beta):
            for j in range(nodes_num):
                # parents = np.where(prior_network[:,j]==1)[0]

                # si=0 term
                x_j=x_matrix[:,j].copy()
                gradient_j_zero = np.zeros(nodes_num)
                temp = 1-p_matrix[:,j]*s_matrix[i]
                temp[np.where(temp==0)]=np.inf
                gradient_j_zero = -1*s_matrix[i]/temp*prior_network[:,j]*(1-s_matrix[i,j])*x_j
                p_gradient_matrix[:,j]+=gradient_j_zero

                # si=1 term
                gradient_j_one = np.zeros(nodes_num)
                p = p_matrix[:, j].copy()
                s = s_matrix[i].copy()
                A = 1-np.prod((1-p*s+small_value)**x_j)
                
                down=A*(1-s*p)
                down[np.where(down==0)]=np.inf
                up=(-A+1)*x_j*s*s_matrix[i,j]*prior_network[:,j]
                gradient_j_one=up/down
                p_gradient_matrix[:,j]+=gradient_j_one

                # if A==0:
                #     A=np.inf

                # x_ij=x_j[parents].copy()
                # s_il=s[parents].copy()
                # p_ij=p_matrix[parents,j].copy()
                # p = np.repeat(p[:,np.newaxis], parents.size, axis=1)
                # temp = np.arange(parents.size)
                # p[parents,temp] = 0     # Fi\vj
                # temp_gradient = np.prod((1-p*s[:,np.newaxis]+small_value)**x_j.reshape((-1,1)), axis=0)
                # temp_gradient=temp_gradient*x_ij*((1-s_il*p_ij+small_value)**(x_ij-1))*s[parents]/A*s_matrix[i,j]
                # gradient_j_one[parents] = temp_gradient.copy()
                # p_gradient_matrix[:,j]+=gradient_j_one

        # step 2: update p
        if epsilon_change:
            epsilon = initial_epsilon/np.sqrt(iter_cnt)
        else:
            epsilon = initial_epsilon

        p_matrix+=epsilon*p_gradient_matrix

        # bound p_matrix
        p_matrix[np.where(p_matrix<0)]=0
        p_matrix[np.where(p_matrix>1)]=1

        end_1=time.time()
        print("this iteration time cost=%f"%(end_1-begin_1))

        # show p
        show_update_p(groundtruth_network, groundtruth_p, p_matrix, inner_p_cnt)
        max_delta_p=np.max(abs(pre_p_matrix-p_matrix))
        delta_p_sum=np.sum(abs(pre_p_matrix-p_matrix))
        if delta_p_sum<update_p_sum_threshold or max_delta_p<max_delta_p_threshold or inner_p_cnt>=p_max_iteration:
            break
    
        pre_p_matrix=p_matrix.copy()

    return p_matrix



def show_update_p(ground_truth_network, ground_truth_p, p_matrix, iter_cnt):
    print("inner_p_cnt:%d"%(iter_cnt))
    mae = cal_mae(ground_truth_network, ground_truth_p, p_matrix)
    mse = cal_mse(ground_truth_p, p_matrix)
    mae_v2=cal_mae_v2(ground_truth_network, ground_truth_p, p_matrix)
    mse_v2=cal_mse_v2(ground_truth_network, ground_truth_p, p_matrix)
    print("MAE=%f, MSE=%f, MAE_v2=%f, MSE_v2=%f" % (mae, mse, mae_v2, mse_v2))

    modified_p=modify_p(ground_truth_network, p_matrix)
    modified_mae = cal_mae(ground_truth_network, ground_truth_p, modified_p)
    modified_mse = cal_mse(ground_truth_p, modified_p)
    modified_mae_v2=cal_mae_v2(ground_truth_network, ground_truth_p, modified_p)
    modified_mse_v2=cal_mse_v2(ground_truth_network, ground_truth_p, modified_p)
    print("modified_MAE=%f, modified_MSE=%f, modified_MAE_v2=%f, modified_MSE_v2=%f" % (modified_mae, modified_mse, modified_mae_v2, modified_mse_v2))

    mae_all=cal_mae_all(ground_truth_p, p_matrix)
    print("mae_all=%f"%(mae_all))
 

def show_update_x(ground_truth_network, x_matrix, p_matrix, prob_result, sample_times, iter_cnt):
    print("inner_x_cnt:%d"%(iter_cnt))

    mae = cal_mae(ground_truth_network, ground_truth_network, x_matrix)
    mse = cal_mse(ground_truth_network, x_matrix)
    mae_v2=cal_mae_v2(ground_truth_network, ground_truth_network, x_matrix)
    mse_v2=cal_mse_v2(ground_truth_network, ground_truth_network, x_matrix)
    print("x_MAE=%f, x_MSE=%f, x_MAE_v2=%f, x_MSE_v2=%f" % (mae, mse, mae_v2, mse_v2))
    mae_all=cal_mae_all(ground_truth_network, x_matrix)
    print("x_mae_all=%f"%(mae_all))

    x_matrix_list= sample_x(x_matrix, sample_times)
    precision, recall, f1=show_result(x_matrix_list, p_matrix, prob_result, ground_truth_network)
    print("precision=%f,recall=%f,f1=%f"%(precision,recall,f1))



def combine_network(x_edge_matrix_list, p_matrix, prob_result, ground_truth_network, comb_k):
    begin_1=time.time()
    loss_list=[]
    for i in range(len(x_edge_matrix_list)):
        cur_matrix=x_edge_matrix_list[i].copy()
        cur_loss=cal_loss(cur_matrix,p_matrix,prob_result)
        loss_list.append(cur_loss)
    loss_list=np.array(loss_list)
    comb_network=np.zeros(x_edge_matrix_list[0].shape)
    sorted_index=np.argsort(loss_list)
    last_val=np.inf
    comb_cnt=0
    end_1=time.time()
    print("cal x_matrix list score time cost:%f"%(end_1-begin_1))

    sorted_edge_list=[]
    for i in range(len(loss_list)):
        sorted_edge_list.append(x_edge_matrix_list[sorted_index[i]].copy())

    comb_network_list=[] 
    for i in range(loss_list.size-1,-1,-1):
        if loss_list[sorted_index[i]]<last_val:
            last_val=loss_list[sorted_index[i]]
            comb_network+=x_edge_matrix_list[sorted_index[i]]
            comb_network_list.append(comb_network.copy())
            comb_cnt+=1
            if comb_cnt>=comb_k:
                break
    
    for i in range(len(comb_network_list)):
        cur_combine_network=comb_network_list[i]
        cur_combine_network[np.where(cur_combine_network>0)]=1
        comb_network_list[i]=cur_combine_network.copy()
        precision, recall, f_score=cal_F1(ground_truth_network,cur_combine_network)
        print("%d networks:precision=%f,recall=%f,f1=%f"%(i+1,precision,recall,f_score))
        print("edge_num=%d"%(np.sum(comb_network_list[i])))



def show_update_x_combine(ground_truth_network, x_matrix, p_matrix, prob_result, sample_times, iter_cnt, comb_k):
    print("inner_x_cnt:%d"%(iter_cnt))

    mae = cal_mae(ground_truth_network, ground_truth_network, x_matrix)
    mse = cal_mse(ground_truth_network, x_matrix)
    mae_v2=cal_mae_v2(ground_truth_network, ground_truth_network, x_matrix)
    mse_v2=cal_mse_v2(ground_truth_network, ground_truth_network, x_matrix)
    print("x_MAE=%f, x_MSE=%f, x_MAE_v2=%f, x_MSE_v2=%f" % (mae, mse, mae_v2, mse_v2))
    mae_all=cal_mae_all(ground_truth_network, x_matrix)
    print("x_mae_all=%f"%(mae_all))

    begin_time=time.time()
    x_matrix_list= sample_x(x_matrix, sample_times)
    end_time=time.time()

    print("sample x_matrix time cost=%f"%(end_time-begin_time))

    begin_2=time.time()
    combine_network(x_matrix_list, p_matrix, prob_result, ground_truth_network,comb_k)
    end_2=time.time()
    print("select combine network time cost=%f"%(end_2-begin_2))


def with_x_likelihood_update_x_combine(p_matrix, s_matrix, prior_network, x_matrix, initial_epsilon, iter_cnt, epsilon_change, update_x_sum_threshold, groundtruth_network,sample_times, comb_k,max_delta_x_threshold, x_max_iteration):
    # cal gradient of x, and update x
    inner_x_cnt=0
    small_value=1e-20
    pre_x_matrix=x_matrix.copy()
    while True:
        begin_1=time.time()
        inner_x_cnt+=1
        # step 1: cal gradient
        beta, nodes_num = s_matrix.shape
        x_gradient_matrix = np.zeros(x_matrix.shape)
    
        for i in range(beta):
            for j in range(nodes_num):
                # parents = np.where(prior_network[:,j]==1)[0]

                # si=0 term
                x_j=x_matrix[:,j].copy()
                gradient_j_zero = np.zeros(nodes_num)
                temp = 1-p_matrix[:,j]*s_matrix[i]
                gradient_j_zero = prior_network[:,j]*(1-s_matrix[i,j])*np.log(temp+small_value)
                x_gradient_matrix[:,j]+=gradient_j_zero

                # si=1 term
                gradient_j_one = np.zeros(nodes_num)
                p = p_matrix[:, j].copy()
                s = s_matrix[i].copy()
                A = 1-np.prod((1-p*s+small_value)**x_j)
                orig_A=A.copy()
                if A==0:
                    A=np.inf
                    
                temp_gradient=prior_network[:,j]*(orig_A-1)*np.log(1-p_matrix[:,j]*s_matrix[i]+small_value)/A*s_matrix[i,j]
                gradient_j_one = temp_gradient.copy()
                x_gradient_matrix[:,j]+=gradient_j_one

        # step 2: update x
        if epsilon_change:
            epsilon = initial_epsilon/np.sqrt(iter_cnt)
        else:
            epsilon = initial_epsilon

        x_matrix+=epsilon*x_gradient_matrix

        # bound x_matrix
        x_matrix[np.where(x_matrix<0)]=0
        x_matrix[np.where(x_matrix>1)]=1

        end_1=time.time()
        print("this iteration time cost=%f"%(end_1-begin_1))

        # show x
        show_update_x_combine(groundtruth_network, x_matrix, p_matrix, s_matrix, sample_times, inner_x_cnt, comb_k)
        max_delta_x=np.max(abs(pre_x_matrix-x_matrix))
        delta_x_sum=np.sum(abs(pre_x_matrix-x_matrix))
        if delta_x_sum<update_x_sum_threshold or max_delta_x<max_delta_x_threshold or inner_x_cnt>=x_max_iteration:
            break
    
        pre_x_matrix=x_matrix.copy()

    return x_matrix


def show_init_x(ground_truth_network, x_matrix):

    mae = cal_mae(ground_truth_network, ground_truth_network, x_matrix)
    mse = cal_mse(ground_truth_network, x_matrix)
    mae_v2=cal_mae_v2(ground_truth_network, ground_truth_network, x_matrix)
    mse_v2=cal_mse_v2(ground_truth_network, ground_truth_network, x_matrix)
    print("init_x_MAE=%f, init_x_MSE=%f, init_x_MAE_v2=%f, init_x_MSE_v2=%f" % (mae, mse, mae_v2, mse_v2))
    mae_all=cal_mae_all(ground_truth_network, x_matrix)
    print("init_x_mae_all=%f"%(mae_all))



def kmeans_zero(data):
    data_num=data.size
    
    center_0=0
    for i in range(data_num):
        if data[i]>0:
            center_1=data[i]
            break
    
    max_iteration=300
    stop_threshold=1e-5
    cur_iter=0
    
    label_distribution = -1*np.ones(data_num)
    pre_center_1=center_1
    while True:
        cur_iter+=1
        for i in range(data_num):
            dist_0=abs(data[i]-center_0)
            dist_1=abs(data[i]-center_1)
            if dist_0<dist_1:
                label_distribution[i]=0
            else:
                label_distribution[i]=1
        
        # update center
        center_1=np.mean(data[np.where(label_distribution==1)])
        if abs(center_1-pre_center_1)<stop_threshold or cur_iter>max_iteration:
            break
        pre_center_1=center_1
        
    return label_distribution


def weighted_mi_prune_zero(record_states, prune_choice):
    # prune_choice = 0  mi   ;  prune_choice = 1 imi

    results_num, nodes_num = record_states.shape
    print("results_num = ", results_num)
    MI = np.zeros((nodes_num, nodes_num))

    for j in range(nodes_num):
        for k in range(nodes_num):
            if j >= k:
                continue
            state_mat = np.zeros((2, 2))
            for result_index in range(results_num):
                state_mat[0,0]+=(1-record_states[result_index,j])*(1-record_states[result_index,k])
                state_mat[0,1]+=(1-record_states[result_index,j])*record_states[result_index,k]
                state_mat[1,0]+=record_states[result_index,j]*(1-record_states[result_index,k])
                state_mat[1,1]+=record_states[result_index,j]*record_states[result_index,k]

            epsilon = 1e-5
            M00 = state_mat[0, 0] / results_num * math.log(
                state_mat[0, 0] * results_num / (state_mat[0, 0] + state_mat[0, 1]) / (
                        state_mat[0, 0] + state_mat[1, 0]) + epsilon, 2)
            M01 = state_mat[0, 1] / results_num * math.log(
                state_mat[0, 1] * results_num / (state_mat[0, 0] + state_mat[0, 1]) / (
                        state_mat[0, 1] + state_mat[1, 1]) + epsilon, 2)
            M10 = state_mat[1, 0] / results_num * math.log(
                state_mat[1, 0] * results_num / (state_mat[1, 0] + state_mat[1, 1]) / (
                        state_mat[0, 0] + state_mat[1, 0]) + epsilon, 2)
            M11 = state_mat[1, 1] / results_num * math.log(
                state_mat[1, 1] * results_num / (state_mat[1, 0] + state_mat[1, 1]) / (
                        state_mat[0, 1] + state_mat[1, 1]) + epsilon, 2)

            if prune_choice==0:
                MI[j, k] = M00 + M11 +M10 + M01
            else:
                MI[j, k] = M00 + M11 - abs(M10) - abs(M01)

            MI[k, j] = MI[j, k]

    # Kmeans cluster
    MI[np.where(MI<0)] = 0
    tmp_MI = MI.reshape((-1, 1))
    label_pred = kmeans_zero(tmp_MI)
    temp_0 = tmp_MI[label_pred == 0]
    temp_1 = tmp_MI[label_pred == 1]

    if np.max(temp_0) > np.max(temp_1):
        tau = np.max(temp_1)
    else:
        tau = np.max(temp_0)

    prune_network = np.zeros((nodes_num, nodes_num))
    prune_network[np.where(MI>tau)] = 1


    for i in range(nodes_num):
        prune_network[i,i]=0

    return prune_network
