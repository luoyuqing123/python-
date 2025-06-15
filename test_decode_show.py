import librosa
import numpy as np
import os
from GMM_hmm import compute_B_map, decoder
from isolated_word_recognition import extract_MFCC
import matplotlib.pyplot as plt

if __name__ == "__main__":

    ''' viterbi译码测试和孤立词识别 '''
    # 加载模型
    models = np.load("models.npy", allow_pickle=True)
    model = models[1]
    test_dir = "test\\test"

    # 假设识别词是孤立词的形式
    # 提供单词列表，可以根据实际情况调整
    word_list = ["煲汤", "做饭", "吃饭", "喝水"]

    # 读取测试 wav 并提取特征
    # wav_file = os.path.join(test_dir, str(2)+".wav")
    wav_file = "train\\train\\2\\1.wav"
    fea = extract_MFCC(wav_file)

    # 进行viterbi译码
    B_map, _ = compute_B_map(fea, model)
    prob_max, states = decoder(model, fea, B_map)

    # 显示Viterbi解码结果中的状态
    print("识别的状态序列：", states)

    # 确保 states 中的每个元素都是整数，避免类型错误
    states = [int(state) for state in states]

    # 根据状态序列输出识别的词
    # 假设模型与音频匹配的状态已经可以确定是哪一个词
    # 这里只是简单地将第一个状态映射到词
    recognized_word = word_list[states[0]]  # 假设状态映射到词列表的索引
    print(f"识别结果: {recognized_word}")

    # 读取音频文件并计算频谱
    y, sr = librosa.load(wav_file, sr=8000)
    S = librosa.stft(y, n_fft=256, hop_length=80, win_length=256)
    S = np.abs(S)
    Spec = librosa.amplitude_to_db(S, ref=np.max)

    # 绘制谱图
    fig, ax = plt.subplots()
    ax.imshow(Spec, origin='lower')

    # 找到状态变化的位置并画线
    for i in range(1, len(states)):
        if states[i] != states[i - 1]:
            plt.vlines(i - 1, 0, 128, colors="c", linestyles="dashed")

    plt.show()
