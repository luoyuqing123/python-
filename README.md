项目简介
本项目实现了基于隐马尔可夫模型（HMM）和Viterbi算法的孤立词语音识别。代码通过提取音频的MFCC特征，利用训练好的模型对测试语音进行Viterbi译码，识别对应的词语，并对音频的频谱进行可视化展示，标注出状态变化的位置。

功能特点
特征提取：从音频文件中提取MFCC特征。
Viterbi译码：根据训练的HMM模型对特征序列进行译码，得到最优状态路径。
词语识别：根据译码的状态序列映射到预定义的词语列表，实现识别。
频谱可视化：展示语音信号的频谱图，并在状态变化处用虚线标记。
依赖环境
请确保安装以下Python库：
pip install librosa numpy matplotlib

此外，项目中需要以下文件：
models.npy：预训练好的HMM模型参数文件
GMM_hmm.py：包含compute_B_map和decoder函数的脚本
isolated_word_recognition.py：包含extract_MFCC函数的脚本
测试音频文件（WAV格式）

使用说明
准备好测试音频文件（wav格式），并修改 wav_file 变量指向正确路径。
确认 models.npy 模型文件存在于项目根目录。
运行脚本：
python main.py

常见问题
找不到模型文件
请确保models.npy在当前目录，且格式正确。音频文件路径错误确认wav_file路径正确，且文件格式为WAV。缺少依赖库使用pip install安装缺失的库。

版权声明
本项目代码开源，欢迎学习和交流。
