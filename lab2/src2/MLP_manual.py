import numpy as np
import math
import matplotlib.pyplot as plt
from random import randint

class MLP_munual:

    def __init__(self,lr,epoch):
        self.y1 = np.zeros((5, 1), dtype=float) #input
        self.w2 = np.zeros((4, 5), dtype=float)
        self.b2 = np.zeros((4, 1), dtype=float)
        self.y2 = np.zeros((4, 1), dtype=float)
        self.w3 = np.zeros((4, 4), dtype=float)
        self.b3 = np.zeros((4, 1), dtype=float)
        self.y3 = np.zeros((4, 1), dtype=float)
        self.w4 = np.zeros((3, 4), dtype=float)
        self.b4 = np.zeros((3, 1), dtype=float)
        self.y4 = np.zeros((3, 1), dtype=float) #output
        self.lr = lr
        self.epoch = epoch

    def sigmiod(self,mat):

        e = math.e

        for i in range(mat.shape[0]):
            mat[i][0] = 1/(1 + e**(-mat[i][0]))

        return mat

    def MLP(self,train_data,train_label):

        label = np.zeros((train_label.shape[0],3),dtype=float)
        for i in range(train_label.shape[0]):
            for j in range(3):
                label[i][j] = 0
            label[i][int(train_label[i][0])-1] = 1

        loss_array = np.zeros(self.epoch, dtype=float)

        epoch = 0
        while epoch < self.epoch:

            for i in range(train_data.shape[0]):

                self.y1 = train_data[i].reshape((-1,1))
                self.y2 = self.sigmiod(np.dot(self.w2,self.y1)+self.b2)
                self.y3 = self.sigmiod(np.dot(self.w3,self.y2)+self.b3)
                self.y4 = self.sigmiod(np.dot(self.w4,self.y3)+self.b4)

                eta2 = np.zeros((4, 1), dtype=float)
                eta3 = np.zeros((4, 1), dtype=float)
                eta4 = np.zeros((3, 1), dtype=float)

                for j in range(3):
                    eta4[j][0] = label[i][j]*(self.y4[j][0]-1)

                self.w4 -= self.lr * np.dot(eta4,(self.y3).reshape(1,-1))
                self.b4 -= self.lr * eta4

                for j in range(4):
                    for k in range(3):
                        eta3[j][0] += eta4[k][0]*self.w4[k][j]
                    eta3[j][0] *= self.y3[j][0]*(1-self.y3[j][0])

                self.w3 -= self.lr * np.dot(eta3, (self.y2).reshape(1, -1))
                self.b3 -= self.lr * eta3

                for j in range(4):
                    for k in range(4):
                        eta2[j][0] += eta3[k][0] * self.w3[k][j]
                    eta2[j][0] *= self.y2[j][0] * (1 - self.y2[j][0])

                self.w2 -= self.lr * np.dot(eta2, (self.y1).reshape(1, -1))
                self.b2 -= self.lr * eta2

            #calc loss

            loss = 0

            for i in range(train_data.shape[0]):

                self.y1 = train_data[i].reshape((-1, 1))
                self.y2 = self.sigmiod(np.dot(self.w2, self.y1) + self.b2)
                self.y3 = self.sigmiod(np.dot(self.w3, self.y2) + self.b3)
                self.y4 = self.sigmiod(np.dot(self.w4, self.y3) + self.b4)

                for j in range(3):
                    loss -= label[i][j]*math.log(self.y4[j][0],math.e)

            loss_array[epoch] = loss
            epoch += 1

        plt.plot(loss_array)
        plt.show()


def main():

    train_data = np.random.randn(100,5)
    train_label = np.zeros((100,1), dtype=int)
    for i in range(100):
        train_label[i][0] = randint(1,3)

    lr = 0.01
    epoch = 100

    mlp_m = MLP_munual(lr,epoch)
    mlp_m.MLP(train_data,train_label)

main()