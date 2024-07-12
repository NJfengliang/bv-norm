import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import deepxde as dde
import numpy as np
import scipy.io as sio
from utilities3 import get_parameter_number, UnitGaussianNormalizer
import time

import pandas as pd

start = time.perf_counter()


def train_net(device, data_path, epoch, batch_sizes, l_r, part,npc):
    Data = sio.loadmat(data_path)
    dataX = torch.tensor(Data['f_bc'].astype(np.float32))

    dataY = torch.tensor(Data['u_field'].astype(np.float32))
    dataO = torch.tensor(Data['MeshNodes'].astype(np.float32)).T

    num = dataX.shape[0]
    n_t = dataX.shape[1]
    ntrain = int(npc * num)
    ntest = int((1-npc) * num)

    x_train = dataX[:ntrain, :]
    y_train = dataY[:ntrain, :]
    x_test = dataX[-ntest:, :]
    y_test = dataY[-ntest:, :]

    # regulazaition
    norm_y = UnitGaussianNormalizer(y_train)
    y_train = norm_y.encode(y_train).cpu().numpy()
    y_test = norm_y.encode(y_test).cpu().numpy()
    norm_x = UnitGaussianNormalizer(x_train)
    x_train = norm_x.encode(x_train).cpu().numpy()
    x_test = norm_x.encode(x_test).cpu().numpy()
    norm_o = UnitGaussianNormalizer(dataO)
    O_input = norm_o.encode(dataO).cpu().numpy()

    x_train = (x_train, O_input)
    y_train = y_train
    x_test = (x_test, O_input)
    y_test = y_test

    data = dde.data.TripleCartesianProd(X_train=x_train, y_train=y_train, X_test=x_test,
                                        y_test=y_test)

    net = dde.nn.pytorch.deeponet.DeepONetCartesianProd(
        [n_t, 512, 512, 128], [2, 512, 512,512, 128, ], "relu", "Glorot normal"
    )

    model = dde.Model(data, net)
    model.compile(
        "adam",
        lr=l_r,
        loss="mean l2 relative error",
        decay=("step", 2000, 0.5),
        metrics=["MSE"],
    )
    losshistory, train_state = model.train(epochs=epoch, batch_size=batch_sizes, display_every=100)

    y_predtest = model.predict(data.test_x).astype(np.float32)
    y_predtrain = model.predict(data.train_x).astype(np.float32)
    y_predtest = norm_y.decode(torch.from_numpy(y_predtest).to(device=device)).cpu().numpy()
    y_predtrain = norm_y.decode(torch.from_numpy(y_predtrain).to(device=device)).cpu().numpy()
    Y_test = norm_y.decode(torch.from_numpy(y_test).to(device=device)).cpu().numpy()
    Y_train = norm_y.decode(torch.from_numpy(y_train).to(device=device)).cpu().numpy()
    X_train = norm_x.decode(torch.from_numpy(x_train[0]).to(device=device)).cpu().numpy()
    X_test = norm_x.decode(torch.from_numpy(x_test[0]).to(device=device)).cpu().numpy()
    #
    Test_error = np.abs(y_predtest - Y_test)
    Train_error = np.abs(y_predtrain - Y_train)
    ##############################################
    num_paras = get_parameter_number(net)
    end = time.perf_counter()
    # 计算运行时间
    runTime = end - start
    ##############################################
    E_max_t = np.max(Test_error)
    E_max = np.max(Test_error[:, :], axis=(1))
    E_mean = np.mean(Test_error[:, :], axis=(1))

    E_max_a = np.max(Train_error)
    E_max_ta = np.max(Train_error[:, :], axis=(1))
    E_mean_ta = np.mean(Train_error[:, :], axis=(1))
    #
    print('Runtime:', runTime)
    print('num_paras:', num_paras)
    print("============================================\n")
    #
    # In[]
    # ================ Save Data ====================
    sava_path = "../Results/02.DeepONet/" + part + "/"
    if not os.path.exists(sava_path):
        os.makedirs(sava_path)

    dataframe = pd.DataFrame({
        'num_paras': [num_paras],
        'train_time': [runTime]})
    dataframe.to_csv(sava_path + 'log.csv', index=False, sep=',')
    # ---------------------save model parameters----------------------------
    dataframe2 = pd.DataFrame({
        'epochs': [epochs],
        'lr': [lr],
        'ntarin': [ntrain],
        'ntest': [ntest],
        'batch_size': [batch_size]})
    dataframe2.to_csv("../Results/02.DeepONet/" + part + 'paraments.csv', index=False, sep=',')

    filename = sava_path + 'DeepONet_pre' + '.mat'
    filenameloss = sava_path + 'DeepONet_loss' + '.mat'
    dde.saveplot(losshistory, train_state, issave=False, isplot=False)  # True

    loss_dict = {
        'L2_train': np.array(losshistory.loss_train),
        'L2_test': np.array(losshistory.loss_test),
        'MSE_test': np.array(losshistory.metrics_test),

    }

    out_data_dict = {

        'pre_train': y_predtrain,
        'pre_test': y_predtest,
        'x_train': X_train,
        'x_test': X_test,
        'y_train': Y_train,
        'y_test': Y_test,

    }

    sio.savemat(filename, mdict=out_data_dict)
    sio.savemat(filenameloss, mdict=loss_dict)
    #-----------------------------------------------------------------------------

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = "../Data/mult_holes.mat"

    epochs = 20000
    batch_size = 10
    lr = 1e-3
    npc = 0.7
    for i in range(5):
        print('====================================')
        print('NO' + str(i + 1) + 'train......')
        print('====================================')

        part = "_dacy_m_20000_0711/" + str(i)  # composite_part.mat
        train_net(device, data_path, epochs, batch_size, lr, part,npc)
