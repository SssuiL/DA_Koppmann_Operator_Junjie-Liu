import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

torch.manual_seed(923)
np.random.seed(924)

X_train = np.loadtxt("/Users/ketong/Documents/Phd Vodafone Chair/NetUAVs Collision Avoidance/Deep Learning for Control/python script/control_dataset/X_train.csv", dtype=np.float64, delimiter=",", skiprows=0)
Y_label = np.loadtxt("/Users/ketong/Documents/Phd Vodafone Chair/NetUAVs Collision Avoidance/Deep Learning for Control/python script/control_dataset/Y_label.csv", dtype=np.float64, delimiter=",", skiprows=0)
rbf_points = np.loadtxt("/Users/ketong/Documents/Phd Vodafone Chair/NetUAVs Collision Avoidance/Deep Learning for Control/python script/control_dataset/rbf_points.csv", dtype=np.float64, delimiter=",", skiprows=0)
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_label, test_size=0.2, random_state=923)
X_train, X_test, Y_train, Y_test, rbf_points = torch.from_numpy(X_train), torch.from_numpy(X_test), torch.from_numpy(Y_train), torch.from_numpy(Y_test), torch.from_numpy(rbf_points)

Ix, Iy, Iz = 7.5e-3, 7.5e-3, 1.3e-2
Ts = 0.01
Bd1 = np.zeros([3,3])
bd2_v = np.array([Ts/Ix, Ts/Iy, Ts/Iz])
Bd2 = np.diag(bd2_v)
Bd3 = np.zeros([150, 3])
Bd = np.concatenate((Bd1, Bd2, Bd3), 0)


class KoopmanDNN(nn.Module):

    def __init__(self):
        super(KoopmanDNN, self).__init__()
        # layer 0:
        self.linear_0 = nn.Linear(6, 150)
        self.activ_0 = nn.Tanh()
        # layer 1:
        self.linear_1 = nn.Linear(150, 150)
        self.activ_1 = nn.Tanh()
        # layer 2:
        self.linear_2 = nn.Linear(150, 150)
        self.activ_2 = nn.Tanh()
        # layer 3:
        self.linear_3 = nn.Linear(150, 150)
        self.activ_3 = nn.Tanh()

    def forward(self, x):
        out = self.activ_0(self.linear_0(x)) # output: layer 0
        out = self.activ_1(self.linear_1(out)) # output: layer 1
        out = self.activ_2(self.linear_2(out)) # output: layer 2
        out = self.activ_3(self.linear_3(out)) # output: layer 3
        return out


class KoopmanDataset(Dataset):

    def __init__(self, x, y):
        self.X_train = x
        self.Y_train = y
        self.n_samples = x.shape[0]

    def __getitem__(self, index):
        return self.X_train[index], self.Y_train[index]

    def __len__(self):
        return self.n_samples


def my_loss(x_train_in, x_train_out, y_train_in, y_train_out, A):
    x_train_aug = torch.cat((x_train_in, x_train_out), 1).transpose_(0, 1)
    y_train_aug = torch.cat((y_train_in, y_train_out), 1).transpose_(0, 1)
    error = y_train_aug - torch.matmul(A, x_train_aug)
    error_square = torch.square(error)
    loss = torch.sum(error_square)
    loss_true = torch.sum(error_square[0:6, :])
    return loss, loss_true


def data_augment_batch(x_in, add_dim):
    ori_dim = rbf_points.shape[0]
    n_samples = x_in.shape[0]
    data_out = torch.zeros([n_samples, ori_dim + add_dim])
    for aug_idx in range(ori_dim+add_dim):
        if aug_idx + 1 <= ori_dim:
            data_out[:, aug_idx] = x_in[:, aug_idx]
        else:
            point = rbf_points[:, aug_idx - ori_dim]
            two_norm = torch.norm(x_in-point, dim=1)
            sudo_state = two_norm**2 * torch.log(two_norm)
            data_out[:, aug_idx] = sudo_state
    return data_out


def augment_single_state(x, add_dim):
    ori_dim = rbf_points.shape[0]
    x_aug = torch.zeros(ori_dim + add_dim)
    x_aug[0:x.shape[0]] = x
    error = torch.transpose(rbf_points[:, 0:add_dim], 0, 1) - x
    two_norm = torch.norm(error, dim=1)
    sudo_state = two_norm**2 * torch.log(two_norm)
    x_aug[ori_dim:] = sudo_state
    return x_aug


def get_test_traj_dict():
    test_traj_BT2500 = np.loadtxt("/Users/ketong/Documents/Phd Vodafone Chair/NetUAVs Collision Avoidance/Deep Learning for Control/python script/control_dataset/test_traj_BT2500.csv", dtype=np.float64, delimiter=",", skiprows=0)
    test_traj_dict = {}
    for i in range(100):
        test_traj_dict['traj'+str(i)] = test_traj_BT2500[501*i:501*(i+1), :]
    return test_traj_dict


def get_predict_traj_dict(test_traj_dict, A, aug_method, model=None):
    predict_traj_dict = {}
    prediction_step = 300
    for key in test_traj_dict:
        current_predict_traj = np.zeros([prediction_step, 6])
        current_predict_traj[0, :] = test_traj_dict[key][0, 0:6]
        for step in range(1, prediction_step):
            uc = np.transpose(test_traj_dict[key][step-1, 6:9])
            uc_tensor = torch.from_numpy(uc).double()
            xc = np.transpose(current_predict_traj[step-1, :])
            xc_tensor = torch.from_numpy(xc).double()
            Bd_tensor = torch.from_numpy(Bd).double()
            if aug_method == 'edmd':
                A_tensor = torch.from_numpy(A).double()
                xc_aug = augment_single_state(xc_tensor, 150).double()
                x_next = torch.matmul(A_tensor, xc_aug) + torch.matmul(Bd_tensor, uc_tensor)
                x_next = x_next.numpy()
            elif aug_method == 'linear':
                A_tensor = torch.from_numpy(A).double()
                x_next = torch.matmul(A_tensor, xc_tensor) + torch.matmul(Bd_tensor[0:6, :], uc_tensor)
                x_next = x_next.numpy()
            else:
                model_out = model(xc_tensor)
                xc_aug = torch.cat((xc_tensor, model_out), 0)
                x_next = torch.matmul(A, xc_aug) + torch.matmul(Bd_tensor, uc_tensor)
                x_next = x_next.numpy()
            x_next = x_next[0:6]
            current_predict_traj[step, :] = x_next
        predict_traj_dict[key] = current_predict_traj
    return predict_traj_dict


def get_stepwise_rmse(test_traj_dict, predict_traj_dict):
    stepwise_rmse = np.zeros(300)
    for rmse_idx in range(300):
        square_error = 0
        for key in test_traj_dict:
            square_error += np.linalg.norm(test_traj_dict[key][rmse_idx, 0:3] - predict_traj_dict[key][rmse_idx, 0:3])**2
        square_error /= len(test_traj_dict)
        rmse_step = np.sqrt(square_error)
        stepwise_rmse[rmse_idx] = rmse_step
    return stepwise_rmse


#eDMD setting
# with torch.no_grad():
#     print('edmd start')
#     X_train_aug = data_augment_batch(X_train, 150)  # samples * dim
#     Y_train_aug = data_augment_batch(Y_train, 150)  # samples * dim
#     A_edmd = np.loadtxt("A_edmd.csv", dtype=np.float64, delimiter=",", skiprows=1, usecols=range(1, 157))
#     A_edmd = torch.from_numpy(A_edmd)
#     print(f'edmd end, A_edmd type:{A_edmd.dtype}')
#     loss_edmd, loss_edmd_T = my_loss(X_train, X_train_aug[:, 6:], Y_train, Y_train_aug[:, 6:], A_edmd)


# hyper parameters
num_epochs = 1000
learning_rate = 0.001
rl_rate = 1
batch_size = 320000
model = KoopmanDNN()
model.double()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
train_dataset = KoopmanDataset(X_train, Y_train)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

# continuous trajectories
test_traj_dict = get_test_traj_dict() # keys = 'traj0' --> 'traj99', value = 501*9
# predict_traj_dict_edmd = get_predict_traj_dict(test_traj_dict, A_edmd.numpy(), 'edmd')
# stepwise_rmse_edmd = get_stepwise_rmse(test_traj_dict, predict_traj_dict_edmd)
# with torch.no_grad():
#     predict_traj_dict = get_predict_traj_dict(test_traj_dict, A, 'DNN', model)

# training
A = torch.randn(156, 156, dtype=torch.float64)
epoch = 0
valid_flag = True
current_best_total_loss = 1000000
current_best_rmse = 20
loss_info = np.zeros([num_epochs * np.int_(np.round(320000/batch_size)), 3])
while epoch < num_epochs and valid_flag:
    update_flag = 1
    for i, (inputs, labels) in enumerate(train_dataloader):
        if update_flag: # Update A
            with torch.no_grad():
                x_predict = model(X_train)
                y_predict = model(Y_train)
                x_current = torch.cat((X_train[:, 0:6], x_predict), 1).transpose_(0, 1)
                x_next = torch.cat((Y_train[:, 0:6], y_predict), 1).transpose_(0, 1)
                x_current_pseduo = torch.matmul(x_current.transpose(0, 1), torch.inverse(torch.matmul(x_current, x_current.transpose(0, 1))))
                A_tar = torch.matmul(x_next, x_current_pseduo)
                x_pred_rank = torch.matrix_rank(x_current)
                y_pred_rank = torch.matrix_rank(x_next)
                print(f'--------- new epoch({epoch}): start A update-----------')
                print(f': A_old(rank {torch.matrix_rank(A)}) -> A_tar(rank {torch.matrix_rank(A_tar)})')
                print(f': x_pred(rank {x_pred_rank})|y_pred(rank {y_pred_rank})')
                if x_pred_rank != 156 or y_pred_rank != 156:
                    print('--------- Model prediction rank error, update failed! ---------')
                    valid_flag = False
                    break
                else:
                    loss_mini, loss_mini_T = my_loss(inputs, model(inputs), labels, model(labels), A)
                    loss_mini_tar, loss_mini_T_tar = my_loss(inputs, model(inputs), labels, model(labels), A_tar)
                    loss_total, loss_total_T = my_loss(X_train, model(X_train), Y_train, model(Y_train), A)
                    loss_total_tar, loss_total_T_tar = my_loss(X_train, model(X_train), Y_train, model(Y_train), A_tar)
                    if loss_total_T_tar.item() < current_best_total_loss:
                        current_best_total_loss = loss_total_T_tar.item()
                    print(f': update A|loss_mini({loss_mini_T.item():.2f}->{loss_mini_T_tar.item():.2f})|loss_total({loss_total_T.item():.2f}->{loss_total_T_tar.item():.2f})')
                    # print(f': eDMD loss reference total({loss_edmd_T.item():.2f})|current_best_loss({current_best_total_loss})')
                    print(f': eDMD loss reference total(xxx)|current_best_loss({current_best_total_loss})')
                    A += (A_tar - A) * rl_rate
                    predict_traj_dict_dnn = get_predict_traj_dict(test_traj_dict, A, 'dnn', model)
                    stepwise_rmse_dnn = get_stepwise_rmse(test_traj_dict, predict_traj_dict_dnn)
                    # print(f': eDMD rmse({stepwise_rmse_edmd[-1]:.4f})|DNN rmse({stepwise_rmse_dnn[-1]:.4f})|best rmse({current_best_rmse})')
                    print(f': eDMD rmse(xxx)|DNN rmse({stepwise_rmse_dnn[-1]:.4f})|best rmse({current_best_rmse})')

                    print('------------------')
                    loss_info[epoch, :] = np.array([loss_total_T.item(), loss_total_T_tar.item(), stepwise_rmse_dnn[-1]])
                    if stepwise_rmse_dnn[-1] < current_best_rmse:
                        current_best_rmse = stepwise_rmse_dnn[-1]
                        model_save = model
                        A_save = A
        if epoch == num_epochs - 1:
            break
        else:
            x_predict = model(inputs)
            y_predict = model(labels)
            loss, loss_T = my_loss(inputs, x_predict, labels, y_predict, A)
            loss_T.backward()
            optimizer.step()
            optimizer.zero_grad()
            update_flag = 0
    epoch += 1


# pd.DataFrame(loss_info).to_csv('./training_results/case6_final/loss_info.csv')
# pd.DataFrame(A_save).to_csv('./training_results/case6_final/A_dnn.csv')
# torch.save(model_save.state_dict(), './training_results/case6_final/model.pt')


# plot results
# steps = np.arange(0, 1000)
# testa = np.ones((1000,2), dtype=np.float64)
# testa[:, 0] *= 81.01
# testa[:, 1] *= 0.8339
# fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6.5))
# fig.subplots_adjust(wspace=0.35)
# ax0.grid()
# ax0.set_yscale('log')
# ax0.set_xlim(-5, 1000)
# line0, = ax0.plot(steps, loss_info[:, 1], color='r', linewidth=1.5, label='DNN training loss')
# line1, = ax0.plot(steps, testa[:,0], color='b', linewidth=1.5, label='eDMD loss')
# line2, = ax0.plot(steps, loss_info[:, 2], color='r', linewidth=1.5, linestyle=':', label='DNN RMSE(300 steps)')
# line3, = ax0.plot(steps, testa[:,1], color='b', linewidth=2.0, linestyle=':', label='eDMD RMSE(300 steps)')
# ax0.legend(handles=[line0, line1, line2, line3], loc='upper right', bbox_to_anchor=(1, 0.8))
# ax0.set_xlabel('training epochs', fontweight='bold')
# ax0.set_ylabel('RMSE & training loss', fontweight='bold')
# ax0.grid(True, which='both')
# ax0.set_title('Training process', fontweight='bold')
# textvar0 = ax0.text(1010, 0.44, '0.44')
# textvar1 = ax0.text(1010, 81.01, '81.01')
# textvar2 = ax0.text(1010, 0.08, '0.08')
# textvar3 = ax0.text(1010, 0.83, '0.83')
#
# steps_pred = np.arange(0, 300)
# line4, = ax1.plot(steps_pred, stepwise_rmse_dnn, color='r', linewidth=1.5, label='DNN prediction')
# line5, = ax1.plot(steps_pred, stepwise_rmse_edmd, color='b', linewidth=1.5, label='eDMD prediction')
# line6, = ax1.plot(steps_pred, stepwise_rmse_linear, color='k', linewidth=1.5, label='first-order approx')
# ax1.legend(handles=[line4, line5, line6], loc='upper right', bbox_to_anchor=(0.4, 0.9))
# ax1.set_xlabel('prediction steps', fontweight='bold')
# ax1.set_ylabel('RMSE(euler angles)', fontweight='bold')
# ax1.set_xlim(0, 300)
# ax1.set_ylim(0, 7.5)
# ax1.grid()
# ax1.set_title('Trajectory prediction with control', fontweight='bold')
# textvar4 = ax1.text(303, 0.08, '0.08')
# textvar5 = ax1.text(303, 0.83, '0.83')
# textvar6 = ax1.text(303, 7.2, '7.21')