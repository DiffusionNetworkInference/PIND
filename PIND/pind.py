from utils import *
import time


if __name__=='__main__':

    size=3000     # diffusion network size
    degree=4      # average degree for synthetic network 
    beta=300      # amount of diffusion processes
    mean=0.3      # mean of uncertainty factor
    scale=0.1     # scale of uncertainty factor
    read_flag=False  # read prob result from existing file or not
    update_x_sum_threshold=5
    update_p_sum_threshold=5
    max_delta_x_threshold=0.01
    max_delta_p_threshold=0.01
    learning_rate_p=0.001
    learning_rate_x=0.005
    sample_times=100
    do_prune=True
    epsilon_change=False
    ourter_delta_x_threshold=0.001
    comb_k=3
    x_max_iteration=15
    p_max_iteration=15
    outer_max_iteration=6


    begin=time.time()

    graph_path='./test_network.dat'
    result_path='./test_diffusion_results.txt'
    prob_result_path='./test_prob_results.txt'



    print("graph_path=%s,result_path=%s,mean=%f,scale=%f,prob_result_path=%s,read_flag=%r,update_x_sum_threshold=%f,"
        "update_p_sum_threshold=%f,learning_rate_p=%f,learning_rate_x=%f,do_prune=%r,degree=%d, sample_times=%d,epsilon_change=%r,"
        "x_max_iteration=%d, p_max_iteration=%d,outer_max_iteration=%d"%(graph_path,result_path,mean,scale,prob_result_path,read_flag,update_x_sum_threshold,update_p_sum_threshold,
                                                                learning_rate_p,learning_rate_x,do_prune,degree,sample_times,epsilon_change,
                                                                x_max_iteration,p_max_iteration,outer_max_iteration))

    overall_begin= time.time()


    # load data 
    ground_truth_network, diffusion_result, ground_truth_p=load_data(graph_path, result_path)
    print("load data done.")

    # generate probability result 
    prob_result=generate_prob_result(diffusion_result, mean, scale, prob_result_path, read_flag)
    print("generate prob result done.")

    # prune 
    prune_network=weighted_mi_prune(prob_result, prune_choice=1)

    # initialization
    x_coe=1e-5
    p_coe=1e-5
    print("soft initialization,x_coe=%f,p_coe=%f"%(x_coe,p_coe))

    x_matrix=np.random.rand(size,size)*x_coe
    x_matrix[np.where(prune_network==1)]=1
    show_init_x(ground_truth_network, x_matrix)

    p_matrix=np.random.rand(size,size)
    p_matrix[np.where(prune_network==0)]*=p_coe
    show_update_p(ground_truth_network, ground_truth_p, p_matrix, 0)

    prior_network=np.ones((size,size))


    it_cnt=0
    pre_x=x_matrix.copy()
    while True:
        it_cnt+=1
        print("--------------------------------------------------------------")
        print("outer %d th iteration begin."%(it_cnt))
        outer_begin = time.time()

        # update p 
        p_matrix=with_x_likelihood_update_p(p_matrix, prob_result, prior_network, x_matrix, learning_rate_p, it_cnt, epsilon_change, update_p_sum_threshold, ground_truth_network, ground_truth_p,max_delta_p_threshold,p_max_iteration)
        p_end=time.time()
        print("update p done! time cost=%f"%(p_end-outer_begin))
        
        print("...........................")

        # update x
        x_matrix=with_x_likelihood_update_x_combine(p_matrix, prob_result, prior_network, x_matrix, learning_rate_x, it_cnt, epsilon_change, update_x_sum_threshold, ground_truth_network,sample_times,comb_k,max_delta_x_threshold,x_max_iteration)
        x_end=time.time()
        print("update x done! time cost=%f"%(x_end-p_end))

        outer_end=time.time()
        print("%dth ourter iteration done, time cost=%f, until_now_time=%f"%(it_cnt, outer_end-outer_begin, outer_end-overall_begin))

        max_delta_x=np.max(abs(pre_x-x_matrix))
        pre_x=x_matrix.copy()
        if max_delta_x<ourter_delta_x_threshold or it_cnt>=outer_max_iteration:
            print("algorithm done!")
            break
