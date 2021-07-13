import numpy as np
import math
from collections import Counter
from process_data import load_and_process_data
from evaluation import get_micro_F1,get_macro_F1,get_acc

class NaiveBayes:
    '''参数初始化
    Pc: P(c) 每个类别c的概率分布
    Pxc: P(c|x) 每个特征的条件概率
    '''
    def __init__(self):
        self.Pc={}
        self.Pxc={}
        self.attr_upper={}
        self.attr_lower={}
        self.d={}
        self.N=19
        self.label_num=3
    '''
    通过训练集计算先验概率分布p(c)和条件概率分布p(x|c)
    建议全部取log，避免相乘为0
    '''
    def fit(self,traindata,trainlabel,featuretype):
        '''
        需要你实现的部分
        '''
        attr_num = traindata.shape[1]
        data_num = traindata.shape[0]
        self.Pxc = np.zeros((self.label_num+1,attr_num,self.N+1),dtype=float)
        for i in range(attr_num):
            x_upper = -2147483648
            x_lower = 2147483647
            for j in range(data_num):
                if traindata[j][i] > x_upper:
                    x_upper = traindata[j][i]
                if traindata[j][i] < x_lower:
                    x_lower = traindata[j][i]
            self.attr_lower[i] = x_lower
            self.attr_upper[i] = x_upper
            self.d[i]=(x_upper-x_lower)/self.N
        for i in range(1,self.label_num+1):
            Dc = 0
            Dxc = np.zeros((attr_num,self.N+1),dtype=np.int)
            for j in range(data_num):
                if trainlabel[j][0] == i:
                    Dc = Dc + 1
                    for k in range(attr_num):
                        if featuretype[k] == 0:
                            x_label = int(traindata[j][k])
                            Dxc[k][x_label] = Dxc[k][x_label] + 1
                        else:
                            x_label = int(round((traindata[j][k]-self.attr_lower[k])/self.d[k]))
                            Dxc[k][x_label] = Dxc[k][x_label] + 1
            self.Pc[i] = math.log((Dc+1)/(data_num+self.label_num))
            for j in range(attr_num):
                if featuretype[j] == 1:
                    Ni = self.N + 1
                    for k in range(Ni):
                        self.Pxc[i][j][k] = math.log((Dxc[j][k]+1)/(Dc+Ni))
                else:
                    Ni = 3
                    for k in range(1,Ni+1):
                        self.Pxc[i][j][k] = math.log((Dxc[j][k]+1)/(Dc+Ni))
        print(self.Pxc)
    '''
    根据先验概率分布p(c)和条件概率分布p(x|c)对新样本进行预测
    返回预测结果,预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目
    feature_type为0-1数组，表示特征的数据类型，0表示离散型，1表示连续型
    '''
    def predict(self,features,featuretype):
        '''
        需要你实现的部分
        '''
        test_num = features.shape[0]
        attr_num = features.shape[1]
        pred = np.zeros((test_num,1),dtype=np.int)
        for i in range(test_num):
            c = 1
            max_cut = -2147483648
            for j in range(1,self.label_num+1):
                now_cut = self.Pc[j]
                for k in range(attr_num):
                    if featuretype[k] == 0:
                        now_cut = now_cut + self.Pxc[j][k][int(features[i][k])]
                    else:
                        x_label = int(round((features[i][k]-self.attr_lower[k])/self.d[k]))
                        now_cut = now_cut + self.Pxc[j][k][x_label]
                if now_cut > max_cut:
                    max_cut = now_cut
                    c = j
            pred[i][0] = c
        return pred


def main():
    # 加载训练集和测试集
    train_data,train_label,test_data,test_label=load_and_process_data()
    feature_type=[0,1,1,1,1,1,1,1] #表示特征的数据类型，0表示离散型，1表示连续型

    Nayes=NaiveBayes()
    Nayes.fit(train_data,train_label,feature_type) # 在训练集上计算先验概率和条件概率

    pred=Nayes.predict(test_data,feature_type)  # 得到测试集上的预测结果
    # 计算准确率Acc及多分类的F1-score
    print("Acc: "+str(get_acc(test_label,pred)))
    print("macro-F1: "+str(get_macro_F1(test_label,pred)))
    print("micro-F1: "+str(get_micro_F1(test_label,pred)))

main()