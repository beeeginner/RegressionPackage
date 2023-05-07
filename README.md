## Update Log
## 更新日志2023-05-06
Major update! Added the Elastic Net Regression module. Furthermore, to address the sparsity of Lasso Regression and Elastic Net Regression, unused coefficients are compressed to zero. This feature can be utilized for feature selection.
The usage method has also undergone substantial changes. I will provide test examples once I have completed the experimental report.
大更新！加入了弹性网络回归模块。并且针对Lasso回归和弹性网络回归的稀疏性，将没用的系数压缩成0。可以供应于特征选择
使用方法也发生了本质性的改变。测试例等我写完实验报告给。

# description
# 算法包说明
Four regressions have been encapsulated: OLS, Ridge Regression, Lasso Regression, and Elastic Net Regression. They support automatic inference for multicollinearity.

Both Ridge Regression and OLS have closed-form solutions, which are solved directly without the need to set the "epoch" parameter. For Ridge Regression, you only need to customize an appropriate lambda (coefficient of the penalty function).

Lasso Regression and Elastic Net Regression require setting a fixed "epoch" because batch gradient descent is currently used. In the future, I may update it to use a more suitable coordinate descent algorithm similar to Ridge Regression. The initial point for Lasso Regression iterations is a random number following a Gaussian distribution. If you want consistent results, please set a fixed seed for the numpy.random() module to ensure result consistency.

The selection of lambda for the algorithms with penalty terms needs to be done manually for now. I may update it in the future to provide a lambda selection method.
封装了四种回归：OLS，岭回归，lasso回归算法,弹性网络回归算法。支持多重共线性的自动推断。
岭回归回归和OLS都有解析解，是直接解析解出，无需设置epoach参数，对于岭回归只需要自定义合适的lambda(罚函数前面的系数)。
Lasso回归和弹性网络回归需要设置一个固定的epoach，因为暂时使用的是批量梯度下降算法，有空我会更新成更合适岭回归的坐标轴下降算法进行实现。Lasso回归的迭代初始点是服从高斯分布的随机数，如果想要固定结果，请给numpy.random()模块设置一个确定的种子以保证结果的一致性。
带罚函数项的算法部分的lambda的选取，需要自行选取，暂时没做lambda的只能选取，以后，我可能会更新。

# how to use
# 使用的方法

Download this package and then import it in the same directory. After that, you can use the Regression class.
When initializing the regressor, please note that you don't need to add an additional column of ones to the feature matrix because it will be added automatically in the algorithm package. You just need to pass the original feature matrix and label vector.
The usage process is as follows:
Create a regressor object and feed it with data and parameters:
下载这个包，然后在同文件目录下import本包然后就可以使用Regression类了。
初始化回归器的时候请注意，特征矩阵不需要补一列全1列，因为我在算法包里会给你补全，传入原始特征矩阵和标签向量就可以。
使用流程为
创建回归器对象并喂入数据和参数:

reg = Regression(X,y,mode,lr,lm,EPOACH,lm2)

no parameters for train the model
训练模型不需要传入参数
reg.train()

batch prediction
批量预测数据
y_pred = reg.batch_predict(X)

all members are public,if you want to get weights and bias vector,using reg.beta,the first dimension of vector reg.beta is bias
所有的成员变量均为共有
如果想要获取系数访问reg.beta即可，其中beta的第一个数字是截距项。

I commented meaning of parameters when defining Constructor of Regression,here I only list the parameter "mode" 
Regression的构造函数定义和参数说明注释里有写。我这里只介绍构造参数中的mode。
1. mode == 'auto':
    Automatically determines whether the dataset has multicollinearity. If not, it uses OLS; otherwise, it uses Ridge Regression for a closed-form solution and provides informative messages.
    自动判别数据集有没有多重共线性，如果没有就使用OLS;否则使用岭回归进行解析。并且给出提示信息。
2. mode == 'Ridge':
Directly uses the Ridge Regression algorithm.
    直接采用岭回归算法
3. mode == ‘Lasso’
Directly uses the Lasso Regression algorithm.
    直接采用Lasso回归算法
4. mode == 'elasticnetregression'
   Directly uses the Elastic Net Regression algorithm.
   直接采用弹性网络回归算法
    


