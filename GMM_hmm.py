import numpy as np
import math

# 计算GMM-HMM参数更新时所需要的的参数    
def getparam(model,observations,B_map,B_map_mix):
    o = observations
    N_samples = np.shape(o)[0]
    N_state = np.shape(model["pi"])[0]
    N_mix = len(model["S"][0]["ws"])

    # 计算前向概率
    # alpha 初始化
    alpha = np.zeros([N_samples,N_state])
    c    = np.zeros(N_samples) # 正则项
    
    # 计算第0个样本属于第i个状态的概率
    alpha[0] = model["pi"]*B_map[0]
    c[0] = 1/np.sum(alpha[0])
    alpha[0] = alpha[0]*c[0]
    
    # 计算其他时刻的样本属第i个状态的概率
    for t in range(1,N_samples):
        s_current = np.dot(alpha[t-1],model["A"])
        # alpha[t] = s_current*model["B"](model,o[t])
        alpha[t] = s_current*B_map[t]
        alpha[t] = alpha[t]
        c[t] =1.0/(np.sum(alpha[t]))
        
        alpha[t] = alpha[t]*c[t]
    
    # 计算后向概率
    beta = np.zeros([N_samples,N_state])
    
    # 反向初始值
    beta[-1] = c[-1]
    
    for t in range(N_samples-2,-1,-1):
        # 由t+1时刻的beta以及t+1时刻的观测值计算
        # t+1时刻的状态值
        # s_next = beta[t+1]*model["B"](model,o[t+1])
        s_next = beta[t+1]*B_map[t+1]
        beta[t] = np.dot(s_next,model["A"].T)
        beta[t] = beta[t]*c[t]
       
    
    # 计算状态间的转移概率 xi 
    xi = np.zeros([N_samples-1,N_state,N_state])

    for t in range(N_samples-1):
        denom = np.sum(alpha[t]*beta[t,:])
        temp = np.zeros([N_state,N_state])
        
        t_alpha = np.tile( np.expand_dims(alpha[t,:],axis=1),(1,N_state))
        t_beta = np.tile(beta[t+1,:],(N_state,1))
        # t_b = np.tile(model["B"](model,o[t+1]),(N_stats,1))
        t_b = np.tile(B_map[t+1],(N_state,1))
        temp = t_alpha*model["A"]*t_beta*t_b
        temp = temp/(denom+np.finfo(np.float64).eps)
 
        xi[t]=c[t]*temp
        
    
    # 计算每个样本在每个状态的每个mix上的概率
    gamma_mix = np.zeros([N_samples,N_state,N_mix])
    
    for t in range(N_samples):
        # 样本在状态上的概率
        pab = alpha[t]*beta[t]  # S
        
        sum_pab = np.sum(pab)
        if sum_pab==0:
            sum_pab = np.finfo(np.float64).eps
            
        for s in range(N_state):
            prob = B_map_mix[t,s]   # M
            sum_prob = np.sum(prob)
            
            if sum_prob ==0:
                sum_prob = np.finfo(np.float64).eps
            
            temp = pab[s]/sum_pab # 1
            prob = prob/sum_prob# M
            gamma_mix[t,s,:] = temp*prob #M

    return c,alpha,beta,xi,gamma_mix
    

# 维特比译码  为了避免数据过长的问题
# 这里用log 替代乘法
def decoder(model,observations,B_map):
    o = observations
    N_samples = np.shape(o)[0]
    N_state = np.shape(model["pi"])[0]
    
    pi = model['pi']
    log_pi = np.zeros(N_state)
    for i in range(N_state):
        if pi[i] == 0:
            log_pi[i] = -np.inf
        else:
            log_pi[i] = np.log(pi[i])
     
    A = model["A"]
    log_A = np.zeros([N_state,N_state])
    for i in range(N_state):
        for j in range(N_state):
            if A[i,j]==0:
                log_A[i,j] = -np.inf
            else:
                log_A[i,j] = np.log(A[i,j])
    
    # 记录了从t-1 到 t时刻，状态i
    # 最可能从哪个状态（假设为j）转移来的
    psi = np.zeros([N_samples,N_state])
    
    # 从t-1 到 t 时刻状态 状态j到状态i的最大的转移概率
    delta = np.zeros([N_samples,N_state])
    
    # 初始化
    # delta[0] = model["pi"]*model["B"](model,o[0])
    delta[0] = log_pi + np.log(B_map[0])
    psi[0]=0
    
    # 递推填充 delta 与 psi
    for t in range(1,N_samples):
        for i in range(N_state):
            states_prev2current = delta[t-1] + log_A[:,i]
            delta[t][i] = np.max(states_prev2current)
            psi[t][i] = np.argmax(states_prev2current)
        
        # delta[t] = delta[t]*model["B"](model,o[t])
        delta[t] = delta[t]+ np.log(B_map[t]) 
    # 反向回溯寻找最佳路径
    path = np.zeros(N_samples)
    path[-1] = np.argmax(delta[-1])
    prob_max = np.max(delta[-1])
    
    for t in range(N_samples-2,-1,-1):
        path[t] = psi[t+1][int(path[t+1])]
    
    return prob_max, path 
           
def update_A(A,collect_xi):
    N_state = np.shape(A)[-1]
    new_A = A.copy()
    
    collect_xi = np.concatenate(collect_xi,axis=0)
    sum_xi = np.sum(collect_xi,axis=0)
    
    for i in range(N_state):
        for j in range(N_state):
            # 只对A[i,j]>0的部分参数进行更新
            if A[i,j]>0:
                nom = sum_xi[i,j]
                denom = np.sum(sum_xi[i])
                new_A[i,j] = nom/(denom+np.finfo(np.float64).eps)
            
    return new_A

def update_GMM_in_States(model,train_datas,collect_gamma_mix):
    
    # 数据进行拼接
    train_datas = np.concatenate(train_datas,axis=0)
    collect_gamma_mix= np.concatenate(collect_gamma_mix,axis=0)
    
    # 获取数据的长度
    T,D = np.shape(train_datas)
    
    # 获取状态数N_state和每个状态中混合成分数N_mix
    N_state = len(model["S"])
    N_mix = np.shape(model["S"][0]["ws"])[0]
    
    for s in range(N_state):
        
        nommean = np.zeros(D); 
        nomvar  = np.zeros(D); 
        for m in range(N_mix):
            # 每个样本属于状态s第m个成分的概率(权重)
            weight = collect_gamma_mix[:,s,m]
            weight = np.expand_dims(weight,axis=1)
            
            # 加权的均值            
            nom_mean = np.sum(train_datas*weight, axis=0)
            
            # 加权的方差
            mu = np.expand_dims(model["S"][s]["mus"][m],axis=0)
            nom_var  = np.sum((train_datas - mu)*(train_datas - mu) *weight ,axis=0 )
            
            # 权重求和
            denom = np.sum(weight)
            if denom ==0:
                denom = np.finfo(np.float64).eps
            
            # 取平均 获得新的mu
            model["S"][s]["mus"][m] = nom_mean/denom
            
            # 取平均的 获得新的sigma
            sigma = nom_var/denom
            model["S"][s]["sigmas"][m] = sigma
            
            # 取平均 获得新的w
            nom_w = np.sum(weight)
            denom_w = np.sum(collect_gamma_mix[:,s,:]+np.finfo(np.float64).eps)
            model["S"][s]["ws"][m] = nom_w/denom_w
            
    return model

    
# 训练数据是一个numpy的列表
def train_step_GMM_HMM(train_datas,model,collect_B_map,collect_B_map_mix): 
    
    collect_gamma_mix = []
    collect_xi = []
    for data,B_map,B_map_mix in zip(train_datas,collect_B_map,collect_B_map_mix):
        c,alpha,beta,xi,gamma_mix=getparam(model,data,B_map,B_map_mix)
        collect_gamma_mix.append(gamma_mix)
        collect_xi.append(xi)
    
    new_A = update_A(model["A"],collect_xi)
    model = update_GMM_in_States(model,train_datas,collect_gamma_mix)
    model["A"] = new_A
    
    return model

        
# 计算一个高斯的pdf
# x: 数据 [D]
# sigma 方差 [D]
# mu 均值 [D]
def getPdf(x,mu,sigma):
    
    D = np.shape(x)[0]
    # 防止sigma 过小
    sigma[sigma<0.0001]=0.0001
    
    # 计算行列式的值,元素连乘
    covar_det = np.prod(sigma)
 
    # 计算pdf
    c = 1.0 / ( (2.0*np.pi)**(float(D/2.0)) * (covar_det)**(0.5))
    temp = np.dot( (x-mu)*(x-mu) , 1.0/sigma)
    pdfval = c* np.exp(-0.5*temp)
    
    return pdfval        
       
    
def compute_B_map(datas,model):
    # 计算 B_map
    T,D = np.shape(datas)
    N_mix = np.shape(model["S"][0]["ws"])[0]
    N_state = len(model["S"])
    
    B_map_mix = np.zeros([T,N_state,N_mix])
    B_map = np.zeros([T,N_state])
    
    for t in range(T):
    # 样本在状态上的概率
        for s in range(N_state):
            # o 在状态 s 的每个 mixture上的概率
            for m in range(N_mix):
                mu =model["S"][s]["mus"][m]
                sigma = model["S"][s]["sigmas"][m]
                w = model["S"][s]["ws"][m]
                B_map_mix[t,s,m] = w*getPdf(datas[t],mu,sigma)
            
            # 计算 o 在 每个状态 s 上的概率
            B_map[t,s] = np.sum(B_map_mix[t,s,:])
            if B_map[t,s] ==0:
                B_map[t,s] = np.finfo(np.float64).eps
  
    return B_map, B_map_mix
    

def compute_prob_viterbi(model,datas,collect_B_map):
    
    result = 0
    
    for o,B_map in zip(datas,collect_B_map):
        prob_max,_ = decoder(model,o,B_map)
        result = result+prob_max
    return result
    
    
def train_GMM_HMM(train_datas,model,n_iteration):
    
    probs = np.zeros(n_iteration+1)
    collect_B_map = []
    collect_B_map_mix = []
    for datas in train_datas:
        B_map,B_map_mix = compute_B_map(datas,model)
        collect_B_map.append(B_map)
        collect_B_map_mix.append(B_map_mix)
        
        
    prob_old = compute_prob_viterbi(model,train_datas,collect_B_map)
    probs[0] = prob_old
    print("Prob_first",prob_old)
    
    for i in range(n_iteration):
        
        # 一步训练获取一个新的模型
        model_old = model.copy()
        model=train_step_GMM_HMM(train_datas,model,collect_B_map,collect_B_map_mix)
        
        # 重新计算map_B
        collect_B_map = []
        collect_B_map_mix = []
        for datas in train_datas:
            B_map,B_map_mix = compute_B_map(datas,model)
            collect_B_map.append(B_map)
            collect_B_map_mix.append(B_map_mix)
        
        prob_new = compute_prob_viterbi(model,train_datas,collect_B_map)
        probs[i+1] = prob_new
        
        print("it %d prob %f"%(i,prob_new))
      
        if i>2:
            if np.abs((probs[i+1]-probs[i])/probs[i+1])< 5e-4:
                break
        
        if np.isnan(prob_new):
            model = model_old
            break
            
    return model  
    






   
    
    
                   