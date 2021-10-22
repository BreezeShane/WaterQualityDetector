from os import mkdir
from os.path import exists
import scipy.special as ssp
import numpy as np


class NeuralNetwork:
    def __init__(self, inputnode, hidennode, outputnode, trained=False):
        self.inodes = inputnode
        self.hnodes = hidennode
        self.onodes = outputnode
        # 以上是设置神经网络节点的前期准备
        if trained:
            self.load_model()
        else:
            self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
            self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        # 初始化权重数组为训练的前期准备做好铺垫，并采取较优的初始值（隐藏节点数）^（-0.5)作为标准差
        # self.lr = 0.000310557090
        self.lr = 0.0003105570
        # 设定初始的学习率
        self.actfun = lambda x: ssp.expit(x)
        # 定义激活函数，也许后期这个可以换成其它更好的激活函数，比如tanh(x)

    def name(self):
        return 'Neural Network'

    def train(self, inputlist, targetlist, iteration, total_iteration, optimize_learning_rate=False):  # 此处开始定义神经网咯的训练过程
        inputs = np.array(inputlist, ndmin=2).T
        targets = np.array(targetlist, ndmin=2).T
        hideninputs = np.dot(self.wih, inputs)
        hidenoutputs = self.actfun(hideninputs)
        # 这两行表示输入层加权后输入给输入层的激活函数，并将结果作为隐藏层的输入。
        finalinputs = np.dot(self.who, hidenoutputs)
        finaloutputs = self.actfun(finalinputs)
        # 这两行表示隐藏层加权后输入给输出层的激活函数，并将结果作为最终输出
        self.backward(targets, finaloutputs, hidenoutputs, inputs)
        if optimize_learning_rate:
            self.optimize_learning_rate('', step=iteration, total_steps=total_iteration)

    def predict(self, inputlist):  # 从此处开始定义测试过程
        inputs = np.array(inputlist, ndmin=2).T
        hiddeninputs = np.dot(self.wih, inputs)
        hiddenoutputs = self.actfun(hiddeninputs)
        finalinputs = np.dot(self.who, hiddenoutputs)
        finaloutputs = self.actfun(finalinputs)
        return finaloutputs

    def backward(self, targets, finaloutputs, hidenoutputs, inputs):
        oerror = targets - finaloutputs
        herror = np.dot(self.who.T, oerror)
        # 更新权重
        self.who += self.lr * np.dot((oerror * finaloutputs * (1 - finaloutputs)), np.transpose(hidenoutputs))
        self.wih += self.lr * np.dot((herror * hidenoutputs * (1 - hidenoutputs)), np.transpose(inputs))

    def load_model(self, iteration):
        if exists("./model"):
            if exists(f"./model/wihNNmodel_{iteration}.npy") and exists(f"./model/whoNNmodel_{iteration}.npy"):
                self.wih = np.load("./model/wihNNmodel.npy")
                self.who = np.load("./model/whoNNmodel.npy")
            else:
                print("Failed to load model!\nPlease make sure whether the models exist!")
        else:
            print("Failed to open folder model!\nPlease make sure the folder exists!")

    def save_model(self, step):
        if not exists("./model"):
            mkdir("./model")
        np.save(f"./model/wihNNmodel_{ step }.npy", self.wih)
        np.save(f"./model/whoNNmodel_{ step }.npy", self.who)

    # https://www.jianshu.com/p/7311e7151661
    def optimize_learning_rate(self, decay_type, step=1, total_steps=10000):
        if decay_type == 'piecewise_constant' or decay_type == 'step':
            iter_step = 60
            constant_rate = 0.5
            if step == iter_step:
                self.lr *= constant_rate
        elif decay_type == 'invers_time':
            decay_rate = 0.1
            self.lr *= (1. / 1 + decay_rate * step)
        elif decay_type == 'exponential':
            decay_rate = 0.96
            self.lr *= pow(decay_rate, step)
        elif decay_type == 'nature_exponential':
            decay_ratio = 0.005
            self.lr *= np.exp(-1 * decay_ratio * step)
        elif decay_type == 'cosine':
            self.lr *= 0.5 * (1 + np.cos(step * np.pi / total_steps))
        elif decay_type == 'gradual_warmup':
            warm_steps = 50
            if step <= warm_steps:
                self.lr *= step / warm_steps
        elif decay_type == 'triangular_cyclic':
            pass
        elif decay_type == 'sgdr' or decay_type == 'stochastic_gradient_descent_with_warm_restarts':
            pass
        elif decay_type == 'exponential':
            pass
        else:
            print(f"There is no such mode {decay_type}!")