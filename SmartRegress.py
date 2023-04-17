import numpy as np

class Regression:
    def __init__(self,X,y,mode = 'auto',lr = 0.01,lm = 5,EPOACH = 100):
        '''
        多元线性回归分类器的numpy实现
        :param X: 特征矩阵(不要自己补一列1，我会给你补的)
        :param y: 真实的标签值
        :param mode: 有三种取值
        :param lr:如果使用Ridge或者Lasso回归所使用的学习率
        :param lmd:如果使用Ridge或者Lasso回归，请手动设置罚函数的lambda值
        :param EPOACH:Lasso回归的迭代次数,因为lasso回归的梯度是常数因此必须要有epoach

        :parameter mode:
        mode = 'auto':自动推断矩阵存不存在多重共线性，选在合适的回归方法
        mode = 'Ridge':使用岭回归的手段,通过lambda趋近正无穷的方式逼迫函数取值全局极小
        modde = 'Lasso':lasso回归模式，由于lasso经常过拟合，所以你可以用这个做特征选择再普通OLS
        '''
        self.X = np.insert(X,0,1,axis=1)
        self.y = y
        self.mode = mode
        self.lr = lr
        self.lm = lm
        self.ep = EPOACH

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
        return beta

    def _Lasso(self):
        '''
        lasso回归的实现
        :return: 权重向量
        '''
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

        return beta

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
        return beta

    def train(self):
        assert self.mode in ('auto','Ridge','Lasso'),r'模式必须是auto,Ridge,lasso之一'
        if self.mode == 'auto':
            #自动推断模式
            if np.linalg.matrix_rank(self.X)<self.X[1]:
                #存在多重共线性
                res = self._Ridge()
                return res
            else:
                #不存在多重共线性进行古典回归
                res = self._OLS()
                return res
        elif self.mode == 'Lasso':
            res = self._Lasso()
            return res

        elif self.mode == 'Ridge':
            res = self._Ridge()
            return res

    @classmethod
    def batch_predict(self,X0,beta):
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
            y_pred[i] = np.dot(beta, X0[i])

        return y_pred