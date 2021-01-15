import numpy as np

def training(X_train, y_train, alpha=0.1, max_cycles=100, w_fit=[1,1,1,1]):
    n_sample = np.shape(y_train)[0]
    # 迭代100次，走100步看看能否到达山脚下
    for index in range(max_cycles):
        z = np.dot(X_train,w_fit)  # 矩阵乘法
        yhat = 1.0/(1+np.exp(-z))   # 计算预测的y 
        loss = (-1/n_sample)*(np.sum((y_train*np.log(yhat)) + ((1-y_train)*(np.log(1-yhat)))))  # 损失函数
        delta_loss = (1/n_sample)*(np.dot(X_train.T, (yhat-y_train).T)) # 当前loss在w上的导数， 即对w求导
#         print(f"{index}:loss={loss:.5f}, w_fit={w_fit}")
        w_fit = w_fit - alpha * delta_loss.T  # 更新w_fit
        
    return w_fit


    if __name__ == "__main__":
        print("a")