import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import scipy.io as sio
import pandas as pd
from utilsNORM import MeshNO
from utilsNORM2 import MeshNO2
from utilities3 import LpLoss, UnitGaussianNormalizer, get_parameter_number
from utilsBVNORM import BVNORM
from Adam import Adam


def train_net(device, data_path, modes, epochs, batch_size, lr, part, width):
    Data = sio.loadmat(data_path)

    dataX = torch.tensor(Data['input'].astype(np.float32))
    dataY = torch.tensor(Data['output'].astype(np.float32))
    dataO = torch.tensor(Data['output_points'].astype(np.float32)).T
    num = dataX.shape[0]
    n_in = dataX.shape[1]

    ntrain = int(0.7 * num)
    ntest = int(0.3 * num)

    # xx = Data['x'].astype(np.float32)
    # yy = Data['y'].astype(np.float32)
    # zz = Data['z'].astype(np.float32)
    # MeshNodes = np.concatenate((xx, yy, zz), axis=0).T

    x_train = dataX[:ntrain, :]
    y_train = dataY[:ntrain, :]
    x_test = dataX[-ntest:, :]
    y_test = dataY[-ntest:, :]

    # regulazaition
    norm_y = UnitGaussianNormalizer(y_train)
    y_train = norm_y.encode(y_train)
    y_test = norm_y.encode(y_test)
    norm_x = UnitGaussianNormalizer(x_train)
    x_train = norm_x.encode(x_train)
    x_test = norm_x.encode(x_test)
    norm_o = UnitGaussianNormalizer(dataO)
    O_input = norm_o.encode(dataO).to(device=device).T

    #################
    ##lbo basis

    LBO_Output = sio.loadmat('../Data/lbo_basis_3/lbe_ev_output.mat')['Eigenvectors']

    BASE_Output = LBO_Output[:, :modes]
    MATRIX_Output = torch.Tensor(BASE_Output).cuda()
    INVERSE_Output = (MATRIX_Output.T @ MATRIX_Output).inverse() @ MATRIX_Output.T

    print('BASE_MATRIX:', MATRIX_Output.shape, 'BASE_INVERSE:', INVERSE_Output.shape)

    LBO_Input = sio.loadmat('../Data/lbo_basis_3/lbe_ev_input.mat')['Eigenvectors']

    BASE_Input = LBO_Input[:, :modes]
    MATRIX_Input = torch.Tensor(BASE_Input).cuda()
    INVERSE_Input = (MATRIX_Input.T @ MATRIX_Input).inverse() @ MATRIX_Input.T

    basis_name = "lbo"

    # BC_NET = FNO1d(modes)
    # BC_NET = FNN([n_in, 256, 256, modes])
    BC_NET = MeshNO2(modes,MATRIX_Input,INVERSE_Input, width,n_in)
    BC_NET.apply(weigth_init)
    BC_NET.to(device=device)

    # Geo_NET = UNet(1, 1, modes)
    # Geo_NET = CNN([1, 64, 64, 64, 1], kernel_size=3, padding=1)
    Geo_NET = MeshNO(modes, MATRIX_Output, INVERSE_Output, width,3)
    Geo_NET.apply(weigth_init)
    Geo_NET.to(device=device)

    net = BVNORM(BC_NET, Geo_NET).to(device=device)  # basis，BC_NET,Geo_NET

    # train
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                               batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                                              batch_size=batch_size, shuffle=False)

    optimizer = Adam(net.parameters(), lr=lr, weight_decay=1e-4, betas=(0.99, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    myloss = LpLoss(size_average=False)

    time_start = time.perf_counter()
    time_step = time.perf_counter()

    train_error = np.zeros((epochs))
    test_error = np.zeros((epochs))
    ET_list = np.zeros((epochs))
    for epoch in range(epochs):

        net.train()
        train_mse = 0
        train_l2 = 0

        for x, label in train_loader:
            x = x.to(device=device, dtype=torch.float32)
            input = (x, O_input)
            label = label.to(device=device, dtype=torch.float32)

            optimizer.zero_grad()

            pred = net(input)

            mse = F.mse_loss(pred.view(batch_size, -1), label.view(batch_size, -1), reduction='mean')
            l2 = myloss(pred.view(batch_size, -1), label.view(batch_size, -1))
            l2.backward()  # use the l2 relative loss

            # inverse normalization
            pred_real = norm_y.decode(pred.cpu())
            label_real = norm_y.decode(label.cpu())

            train_l2 += myloss(pred_real,label_real ).item()
            train_e = (torch.abs(label_real - pred_real)).cpu().detach().numpy()
            # loss_max_train = torch.tensor(np.max(train_e[:, :, :], axis=(1, 2))).mean()

            optimizer.step()
            train_mse += mse.item()

        scheduler.step()
        net.eval()
        test_l2 = 0.0

        with torch.no_grad():
            for x, label in test_loader:
                x = x.to(device=device, dtype=torch.float32)
                input = (x, O_input)

                label = label.to(device=device, dtype=torch.float32)
                pred = net(input)

                # inverse normalization
                pred_real = norm_y.decode(pred.cpu())
                label_real = norm_y.decode(label.cpu())

                test_l2 += myloss(pred_real, label_real).item()
                test_e = (torch.abs(label_real - pred_real)).cpu().detach().numpy()
                # loss_max_test = torch.tensor(np.max(test_e[:, :, :], axis=(1, 2))).mean()

        train_mse /= len(train_loader)
        train_l2 /= ntrain
        test_l2 /= ntest
        train_error[epoch] = train_l2
        test_error[epoch] = test_l2

        # ET_list[epoch] = loss_max_test
        # print("Idx: %u L_tr: %2.3f L_te: %2.3f Emax_tr: %2.3f Emax_te: %2.3f"  % (ep, train_l2, test_l2, loss_max_train, loss_max_test))
        time_step_end = time.perf_counter()
        T = time_step_end - time_step

        # print('Step: %d, Train L2: %.5f, Test L2: %.5f, Emax_tr: %.5f, Emax_te: %.5f, Time: %.3fs' % (
        # epoch, train_l2, test_l2, loss_max_train, loss_max_test, T))
        if epoch % 10 == 0:
            print('Step: %d, Train L2: %.5f, Test L2: %.5f, Time: %.3fs' % (
                epoch, train_l2, test_l2, T))

        time_step = time.perf_counter()

    print("\n=============================")
    print("Training done...")
    print("=============================\n")
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                               batch_size=1, shuffle=False)
    pre_train = torch.zeros(y_train.shape)
    y_train = torch.zeros(y_train.shape)
    x_train = torch.zeros(x_train.shape)

    # x_test    = torch.zeros(x_train.shape[0:2])

    index = 0
    with torch.no_grad():
        for x, label in train_loader:
            x = x.to(device=device, dtype=torch.float32)
            input = (x, O_input)
            label = label.to(device=device, dtype=torch.float32)

            pred = net(input)

            pred_real = norm_y.decode(pred.cpu())
            # pred_real = pred_real.cpu()
            y_real = norm_y.decode(label.cpu())
            x_real = norm_x.decode(x.cpu())

            pre_train[index, :] = pred_real
            y_train[index, :] = y_real

            x_train[index] = x_real
            index = index + 1

    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                                              batch_size=1, shuffle=False)
    pre_test = torch.zeros(y_test.shape)
    y_test = torch.zeros(y_test.shape)
    x_test = torch.zeros(x_test.shape[0:2])

    index = 0
    with torch.no_grad():
        for x, label in test_loader:
            x = x.to(device=device, dtype=torch.float32)
            input = (x, O_input)
            label = label.to(device=device, dtype=torch.float32)

            pred = net(input)

            pred_real = norm_y.decode(pred.cpu())
            # pred_real = pred_real.cpu()
            y_real = norm_y.decode(label.cpu())
            x_real = norm_x.decode(x.cpu())

            pre_test[index, :] = pred_real
            y_test[index, :] = y_real
            x_test[index, :] = x_real

            index = index + 1

    pre_test = pre_test.cpu().detach().numpy()
    x_test = x_test.cpu().detach().numpy()
    y_test = y_test.cpu().detach().numpy()
    pre_train = pre_train.cpu().detach().numpy()
    y_train = y_train.cpu().detach().numpy()
    x_train = x_train.cpu().detach().numpy()

    Eest_error = np.abs(pre_test - y_test)
    E_max_t = np.max(Eest_error)
    E_max = np.max(Eest_error[:, :], axis=(1))
    E_mean = np.mean(Eest_error[:, :], axis=(1))

    # ================ Save Data ====================
    sava_path = "../Results/01.BV-NORM/" + part + "/"
    if not os.path.exists(sava_path):
        os.makedirs(sava_path)

    dataframe = pd.DataFrame({'Test_loss': [test_l2],
                              'Train_loss': [train_l2],
                              'num_paras': [get_parameter_number(net)],
                              'train_time': [time_step_end - time_start]})
    dataframe.to_csv(sava_path + 'log.csv', index=False, sep=',')

    dataframe2 = pd.DataFrame({'width': [width],
                               'modes': [modes],
                               'epochs': [epochs],
                               'lr': [lr],
                               'ntarin': [ntrain],
                               'ntest': [ntest],
                               'batch_size': [batch_size]})
    dataframe2.to_csv("../Results/01.BV-NORM/" + part + 'paraments.csv', index=False, sep=',')

    # model_output = model.state_dict()
    loss_dict = {'train_error': train_error,
                 'test_error': test_error}

    pred_dict = {'pre_test': pre_test,
                 'pre_train': pre_train,
                 'x_test': x_test,
                 'x_train': x_train,
                 'y_test': y_test,
                 'y_train': y_train,
                 }

    # torch.save(model_output, sava_path + '_MeshNO_net_params.pkl')
    sio.savemat(sava_path + 'DNORM_loss' + '.mat', mdict=loss_dict)
    sio.savemat(sava_path + 'DNORM_pre' + '.mat', mdict=pred_dict)

    print('\nTesting error: %.3e' % (test_l2))
    print('Training time: %.3f' % (time_step_end - time_start))
    print('Num of paras : %d' % (get_parameter_number(net)))
    print('Num of paras BC_NET: %d' % (get_parameter_number(BC_NET)))
    print('Num of paras Geo_NET: %d' % (get_parameter_number(Geo_NET)))


def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.001)
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.001)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(0.001)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(0.001)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        m.bias.data.zero_()


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = "../Data/data_heat_3.mat"
    width = 32
    modes = 128
    epochs = 2000
    batch_size = 10
    lr = 1e-2
    for i in range(5):
        print('====================================')
        print('NO ' + str(i + 1) + ' train......')
        print('====================================')

        part = "data_heat_2000_1205/" + str(i)  # composite_part.mat
        train_net(device, data_path, modes, epochs, batch_size, lr, part, width)
