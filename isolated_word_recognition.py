import librosa
import numpy as np
import os
from GMM_hmm import train_GMM_HMM,compute_B_map,decoder
from sklearn.cluster import KMeans


def run_kmeans(dataset,K, m = 20):
    labs = KMeans(n_clusters=K, random_state=9).fit_predict(dataset)
    return labs  

def gen_para_GMM(fea_collect,N_mix):
    
    # 首先对特征进行kmeans 聚类
    feas = np.concatenate(fea_collect,axis=0)
    N,D = np.shape(feas)
    # print("sub_fea_shape",feas.shape)
    # 初始化聚类中心
    labs = run_kmeans(feas,N_mix, m = 20)
    mus = np.zeros([N_mix,D])
    sigmas = np.zeros([N_mix,D])
    ws = np.zeros(N_mix)
    for m in range(N_mix):
        index = np.where(labs == m)[0]
        # print("----index---------",index)
        sub_feas = feas[index]
        mu = np.mean(sub_feas,axis=0)
        sigma = np.var(sub_feas,axis=0)
        sigma = sigma+0.0001
        mus[m] = mu
        sigmas[m] = sigma
        
        # print("------N  D-------",N,np.shape(index)[0])
        ws[m] = np.shape(index)[0]/N
    ws = (ws+0.01)/np.sum(ws+0.01) 
    return ws,mus,sigmas
            
def creat_GMM(mus,sigmas,ws):
    gmm = dict()
    gmm['mus'] = mus
    gmm['sigmas'] = sigmas
    gmm['ws'] = ws
    return gmm    
    
def init_hmm(collect_fea,N_state,N_mix):
    
    model_GMM_hmm = dict()

    # 初始 一定从 state 0 开始
    pi = np.zeros(N_state)
    pi[0] = 1
    model_GMM_hmm["pi"] = pi

    #  当前状态 转移概率0.5 下一状态 转移概率0.5
    #  进入最后一个状态后不再跳出
    A = np.zeros([N_state,N_state])
    for i in range(N_state-1):
        A[i,i] = 0.5
        A[i,i+1] = 0.5
    A[-1,-1] =1
    model_GMM_hmm["A"] = A
    
    feas = collect_fea
    len_feas = []
    for fea in feas:
        len_feas.append(np.shape(fea)[0])
        
    states = []
    for s in range(N_state):
        print("STATE ----------------",s)
        sub_fea_collect = []
        # 初始化时 先为每个状态平均分配特征
        for fea,T in zip(feas,len_feas):
            
            T_s = int(T/N_state)*s
            T_e = (int(T/N_state))*(s+1)
            
            sub_fea_collect.append(fea[T_s:T_e])
        ws, mus, sigmas = gen_para_GMM(sub_fea_collect,N_mix)
        gmm = creat_GMM(mus,sigmas,ws)
        states.append(gmm)
    
    model_GMM_hmm["S"] = states
    
    return model_GMM_hmm
  
def extract_MFCC(wav_file):
    # 读取音频数据
    y,sr = librosa.load(wav_file,sr=8000)
    # 提取特征
    fea = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=12,n_mels=24,n_fft = 256, win_length=256,hop_length=80,lifter=12)
    # 进行正则化
    mean = np.mean(fea,axis=1,keepdims=True)
    std = np.std(fea,axis =1,keepdims=True)
    fea = (fea-mean)/std
    # 添加1阶差分
    fea_d = librosa.feature.delta(fea)
    fea = np.concatenate([fea.T, fea_d.T],axis=1)
   
    return fea            

if __name__ == "__main__":
    
    models = []
    train_path = "train\\train"
    for i in range(1,15):
        
        # 进入孤立词i所在的文件夹
        wav_path = os.path.join(train_path,str(i))
        collect_fea = []
        len_feas = []
        dirs = os.listdir(wav_path)
        for file in dirs:
            # 找到 .wav 文件并提取特征
            if file.split(".")[-1]=="wav":
                wav_file = os.path.join(wav_path,file)
                fea = extract_MFCC(wav_file)
                collect_fea.append(fea)
        
        # 模型参数初始化   
        GMM_HMM_1 = init_hmm(collect_fea,N_state=4,N_mix=3)
        print("train model %d"%i)
        model = train_GMM_HMM(collect_fea,GMM_HMM_1,40)
        models.append(model)
       
    np.save("models.npy",models)
    
    # 测试部分
    test_dir ="test\\test"
    count = 0
    models = np.load("models.npy",allow_pickle=True)
    
    for i in range(98):
        # 读取wav文件
        wav_file = os.path.join(test_dir,str(i+1)+".wav")
        fea = extract_MFCC(wav_file)
       
        # 获取测试语音的标签，每条孤立词有7条测试
        lab_true = int(i//7)+1
        
        # 在每个模型上测试
        scores = []
        for m in range(14):
            model = models[m]
            B_map,_=compute_B_map(fea,model)
            prob_max,_ = decoder(model,fea,B_map)
            scores.append(prob_max)
            
        lab_det = np.argmax(scores)+1
        if lab_det == lab_true:
            count = count+1

        print("true lab  %d det lab %d"%(lab_true,lab_det))
    print("decode  %.2f   "%(count*100/98))
   
    
    
    
    
    
    
    
    
    

    
    
    