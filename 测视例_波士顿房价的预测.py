import numpy as np
import pandas as pd
from SmartRegress import Regression

def load_data():
    # 从文件导入数据
    datafile = 'boston_house_prices.csv'
    data = pd.read_csv(datafile, header=0)
    data = np.array(data)

    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                     'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    feature_num = len(feature_names)

    # 将原数据集拆分成训练集和测试集
    # 这里使用80%的数据做训练，20%的数据做测试
    # 测试集和训练集必须是没有交集的
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]

    # 计算train数据集的最大值，最小值
    maximums, minimums = training_data.max(axis=0), training_data.min(axis=0)

    # 记录数据的归一化参数，在预测时对数据做归一化
    global max_values
    global min_values

    max_values = maximums
    min_values = minimums

    # 对数据进行归一化处理
    for i in range(feature_num):
        data[:, i] = (data[:, i] - min_values[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data

def main():
    train, val = load_data()
    X, y = train[:, :-1], train[:, -1]
    X_test, y_test = val[:, :-1], val[:, -1]

    #测试Lasso回归模式
    regressor = Regression(X,y,mode='Lasso',EPOACH=100,lm=0.1)
    beta = regressor.train()
    print("权重向量为： ",beta)
    y_pred = Regression.batch_predict(X_test,beta)
    #用预测误差的平均值简单衡量一下效果
    print("Lasso回归预测误差的平均值为： ",np.mean(np.abs(y_pred-y_test)))

    #测试Ridge回归模式
    regressor2 = Regression(X, y, mode='Ridge', lm=10,lr=0.01)
    beta = regressor2.train()
    print("权重向量为: ",beta)
    y_pred = Regression.batch_predict(X_test, beta)
    print("Ridge回归预测误差平均值为: ",np.mean(np.abs(y_pred - y_test)))

    #测试auto(自动推断多重共线性)回归模式
    regressor3 = Regression(X, y, mode='auto')
    beta = regressor3.train()
    print("权重向量为: ", beta)
    y_pred = Regression.batch_predict(X_test, beta)
    print("自动推断回归预测误差平均值为: ", np.mean(np.abs(y_pred - y_test)))

main()