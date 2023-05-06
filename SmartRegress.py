import numpy as np

class Regression:
    def __init__(self,X,y,mode = 'auto',lr = 0.01,lm = 5,EPOACH = 100,lm2 = 5):
        '''
        多元线性回归分类器的numpy实现
        :param X: 特征矩阵(不要自己补一列1，我会给你补的)
        :param y: 真实的标签值
        :param mode: 有三种取值
        :param lr:如果使用Ridge或者Lasso回归所使用的学习率
        :param lm:如果使用Ridge或者Lasso回归，请手动设置罚函数的lambda值
        :param lm2:如果使用弹性网络回归，请设置lm2的值
        :param EPOACH:Lasso回归的迭代次数,因为lasso回归的梯度是常数因此必须要有epoach
        :parameter mode:
        mode = 'auto':自动推断矩阵存不存在多重共线性，选在合适的回归方法
        mode = 'Ridge':使用岭回归的手段,通过lambda趋近正无穷的方式逼迫函数取值全局极小
        modde = 'Lasso':lasso回归模式，由于lasso经常过拟合，所以你可以用这个做特征选择再普通OLS
        mode = 'elasticnetregression':弹性网络回归
        '''
        self.X = np.insert(X,0,1,axis=1)
        self.y = y
        self.mode = mode
        self.lr = lr
        self.lm = lm
        self.lm2 = lm2
        self.ep = EPOACH
        self.beta = None

    def _OLS(self):
        '''
        OLS具有解析解
        :return: 权重向量
        '''
        X = self.X
        y = self.y
        reverse = np.linalg.inv(np.dot(X.T, X))
        ytx = np.dot(y.T, X)
        beta = np.dot(reverse, ytx)
        self.beta = beta

    def _Lasso(self):
        '''
        lasso回归的实现
        :return: 权重向量
        '''
        #压缩系数成为0的阈值
        thresold = 1e-6
        X = self.X
        y = self.y
        #梯度下降的初始点用高斯分布随机数随机
        beta = np.random.randn(X.shape[1])
        #计算beta的梯度
        m = beta.shape[0]
        g = -(1.0/m) * np.dot(X.T,y) + (1.0/m) * np.dot(np.dot(X.T,X),beta) + self.lm * np.sign(beta)
        #由于lasso回归是以常数速率收敛，所以迭代固定的次数
        for i in range(self.ep):
            beta = beta - self.lr * g
            #更新罚函数和梯度的取值
            g = -(1.0 / m) * np.dot(X.T, y) + (1.0 / m) * np.dot(np.dot(X.T, X), beta) + self.lm * np.sign(beta)

        beta[beta < thresold] = 0
        self.beta = beta

    def _Ridge(self):
        '''
        Lasso回归的实现
        lasso回归具有解析解
        :return: 权重向量
        '''
        X = self.X
        y = self.y
        # 梯度下降的初始点用高斯分布随机数随机
        beta = np.random.randn(X.shape[1])
        dim = beta.shape[0]
        left = np.dot(X.T,X) + 2 * self.lm * np.eye(dim)
        right = np.dot(X.T,y)
        beta = np.dot(np.linalg.inv(left),right)
        self.beta = beta

    def _elasticNetRegression(self):
        '''
        弹性网络回归
        :return:权重系数向量
        '''
        X = self.X
        y = self.y

        thershold = 1e-6
        #服从高斯分布的随机数初始化
        beta = np.random.randn(X.shape[1])
        m = beta.shape[0]
        #计算初始点的损失函数梯度
        gradiant = (-1/m) * np.dot(X.T,(y - np.dot(X,beta))) + self.lm * np.sign(beta) + 2 * self.lm2 * beta
        for i in range(self.ep):
            beta -= self.lr * gradiant
            gradiant = (-1/m) * np.dot(X.T,(y - np.dot(X,beta))) + self.lm * np.sign(beta) + 2 * self.lm2 * beta

        beta[beta < thershold] = 0
        self.beta = beta

    def train(self):
        assert self.mode in ('auto','Ridge','Lasso','elasticnetregression'),r'模式必须是auto,Ridge,lasso之一'
        if self.mode == 'auto':
            #自动推断模式
            a = np.linalg.matrix_rank(self.X)
            if np.linalg.matrix_rank(self.X) < self.X.shape[1]:
                #存在多重共线性
                print("存在多重共线性，默认使用岭回归，如果想用其他回归自行指定模式")
                self._Ridge()

            else:
                #不存在多重共线性进行古典回归
                print("不存在多重共线性，使用OLS")
                self._OLS()


        elif self.mode == 'Lasso':
            self._Lasso()


        elif self.mode == 'Ridge':
            self._Ridge()

        elif self.mode == 'elasticnetregression':
            self._elasticNetRegression()

    def batch_predict(self,X0):
        '''
        批量预测数据
        :param X0:特征矩阵
        :param beta: 权值向量
        :return: 预测值
        '''
        #自动补一列1
        X0 = np.insert(X0,0,1,axis=1)
        y_pred = np.zeros(X0.shape[0])
        for i in range(y_pred.shape[0]):
            y_pred[i] = np.dot(self.beta, X0[i])

        return y_pred
