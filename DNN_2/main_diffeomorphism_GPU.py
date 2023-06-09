import numpy as np
import sympy as sp
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import itertools
import airsim_para as para
import time
import random
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')

torch.manual_seed(923)
np.random.seed(923)
random.seed(923)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class DiffeomorphismDNN(nn.Module):
    def __init__(self, Acl, ub, lb, dim_in=6, dim_out=6, layer_width=100, constraint_soften=1e5,
                 num_epoch=10, batch_size=10000, learning_rate=1e-3):
        super(DiffeomorphismDNN, self).__init__()  # initialize the parent class before initializing the current class
        self.linear_0 = nn.Linear(dim_in, layer_width)
        self.linear_1 = nn.Linear(layer_width, layer_width)
        self.linear_2 = nn.Linear(layer_width, layer_width)
        self.linear_3 = nn.Linear(layer_width, layer_width)
        self.linear_4 = nn.Linear(layer_width, dim_out)
        self.activ = nn.Tanh()
        self.Acl = Acl
        self.constraint_soften = constraint_soften
        self.ub = ub
        self.lb = lb
        self.Aaug, self.Baug = None, None
        self.lambd_principle, self.w = self.construct_principle_eigpairs()
        self.lambd, self.Mi = self.construct_linear_eigpairs()
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.info = torch.zeros(7, dtype=torch.double)

    def forward(self, x):
        out_0 = self.activ(self.linear_0(x))
        out_1 = self.activ(self.linear_1(out_0))
        out_2 = self.activ(self.linear_2(out_1))
        out_3 = self.activ(self.linear_3(out_2))
        out_4 = self.activ(self.linear_4(out_3))
        return out_4

    def training_loss(self, x, dx):
        # here, dx should be xdot_nom
        if x.dim() == 1:
            x = x.view(x.dim(), x.__len__())
            dx = dx.view(dx.dim(), dx.__len__())
        batch_size = x.shape[0]
        hx = self.forward(x)
        d_hx = torch.zeros_like(hx)
        for i in range(batch_size):
            jaco_i = torch.autograd.functional.jacobian(self.forward, x[i], create_graph=True)
            d_hx[i] = jaco_i @ dx[i]
        diffeo_linear_error_matrix = d_hx.T - self.Acl @ hx.T - self.Acl @ x.T + dx.T
        loss_diffeo_linear = torch.mean(diffeo_linear_error_matrix ** 2)
        x0 = torch.zeros(x.shape[1], dtype=torch.double, device=device)
        jaco_0 = torch.autograd.functional.jacobian(self.forward, x0, create_graph=True)
        loss_zero_jaco = self.constraint_soften * torch.mean(jaco_0 ** 2)
        loss = loss_diffeo_linear + loss_zero_jaco
        return loss_diffeo_linear, loss_zero_jaco, loss

    def training_loss_test(self, x, dx):
        # here, dx should be xdot_nom
        if x.dim() == 1:
            x = x.view(x.dim(), x.__len__())
            dx = dx.view(dx.dim(), dx.__len__())
        batch_size = x.shape[0]
        hx = self.forward(x)
        d_hx = torch.zeros_like(hx)
        for i in range(batch_size):
            jaco_i = torch.autograd.functional.jacobian(self.forward, x[i], create_graph=False)
            d_hx[i] = jaco_i @ dx[i]
        diffeo_linear_error_matrix = d_hx.T - self.Acl @ hx.T - self.Acl @ x.T + dx.T
        loss_diffeo_linear = torch.mean(diffeo_linear_error_matrix ** 2)
        x0 = torch.zeros(x.shape[1], dtype=torch.double, device=device)
        jaco_0 = torch.autograd.functional.jacobian(self.forward, x0, create_graph=False)
        loss_zero_jaco = self.constraint_soften * torch.mean(jaco_0 ** 2)
        loss = loss_diffeo_linear + loss_zero_jaco
        return loss_diffeo_linear, loss_zero_jaco, loss

    def construct_principle_eigpairs(self):
        res = torch.linalg.eig(self.Acl)
        lambd, v = res.eigenvalues.real, res.eigenvectors.real
        res_conj = torch.linalg.eig(self.Acl.T)
        w = res_conj.eigenvectors.real
        w_scaling = torch.diag(v.T @ w)
        w /= w_scaling  # w = [w1, ..., wn], each wi is a column vector
        return lambd, w

    def construct_linear_eigpairs(self):
        # 1st power: all combinations; 2nd power: upto 2 combinations
        num_principle_eig = self.lambd_principle.shape[0]
        Mi = torch.zeros((1, num_principle_eig), dtype=torch.double, device=device)
        select_array = np.arange(0, num_principle_eig, 1)
        for select_num in range(1, select_array.size + 1):
            for element in itertools.combinations(select_array, select_num):
                mi = torch.zeros((1, select_array.size), dtype=torch.double, device=device)
                for power_idx in element:
                    mi[0, power_idx] = 1
                Mi = torch.vstack((Mi, mi))
        for select_num in range(1, 3):
            for element in itertools.combinations(select_array, select_num):
                if select_num == 1:
                    mi = torch.zeros((1, select_array.size), dtype=torch.double, device=device)
                    for power_idx in element:
                        mi[0, power_idx] = 2
                else:
                    mi = torch.zeros((3, select_array.size), dtype=torch.double, device=device)
                    mi[0, element[0]], mi[0, element[1]] = 1, 2
                    mi[1, element[0]], mi[1, element[1]] = 2, 1
                    mi[2, element[0]], mi[2, element[1]] = 2, 2
                Mi = torch.vstack((Mi, mi))
        Mi = Mi[1:]
        lambd = Mi @ self.lambd_principle
        return lambd, Mi

    def generate_phi_x(self, xin):
        if xin.dim() == 1:
            xin = xin.view(xin.dim(), xin.__len__())
        with torch.no_grad():
            hxin = self.forward(xin)
        g_scaling = 1 / (self.ub - self.lb)
        g_c_xin = g_scaling * (xin + hxin)
        phi_xin = torch.zeros((xin.shape[0], self.Mi.shape[0]), dtype=torch.double, device=device)
        for i in range(self.Mi.shape[0]):
            mi = self.Mi[i]
            phi_xin_i = torch.prod(torch.pow(g_c_xin @ self.w, mi), dim=1)
            phi_xin[:, i] = phi_xin_i
        return phi_xin

    def keedmd_update(self, x, dx, u, Symbolic):
        if x.dim() == 1:
            x = x.view(x.dim(), x.__len__())
            dx = dx.view(dx.dim(), dx.__len__())
            u = u.view(u.dim(), u.__len__())
        phi_x = self.generate_phi_x(x)
        d_phi_x = torch.zeros_like(phi_x)
        Symbolic.update_jaco_phi_x()
        dphi_x_python = sp.lambdify(Symbolic.symbols, Symbolic.jaco)
        print('-> Start calculating jaco...')
        last_count = 0
        t0 = time.time_ns() * 1e-9
        for i in range(d_phi_x.shape[0]):
            xc = x[i].to('cpu').numpy()
            dphi_xi = dphi_x_python(xc[0], xc[1], xc[2], xc[3], xc[4], xc[5])
            d_phi_x[i] = torch.from_numpy(dphi_xi).to(device) @ dx[i]
            if (i - last_count) / d_phi_x.shape[0] >= 0.01:
                tc = time.time_ns() * 1e-9
                print(f'{int(100 * i / d_phi_x.shape[0])}% --- (time = {tc - t0}s)')
                last_count = i
        print('-> Jaco complete!')
        x, dx, phi_x, d_phi_x, u = x.T, dx.T, phi_x.T, d_phi_x.T, u.T
        # derive A_dx
        Xnext = dx - torch.from_numpy(para.Bnom).to(device) @ u
        Xc = torch.vstack((x, phi_x))
        A_dx = Xnext @ Xc.T @ torch.inverse(Xc @ Xc.T)
        # derive B_phi
        Lambd = torch.diag(self.lambd)
        Xnext = d_phi_x - Lambd @ phi_x
        Xc = u + torch.from_numpy(para.K).to(device) @ x
        B_phi = Xnext @ Xc.T @ torch.inverse(Xc @ Xc.T)
        A_dphi = torch.hstack((B_phi @ torch.from_numpy(para.K).to(device), Lambd))
        self.Aaug = torch.vstack((A_dx, A_dphi))
        self.Baug = torch.vstack((torch.from_numpy(para.Bnom).to(device), B_phi))


class SymbolicFunc:
    def __init__(self):
        self.weight_list = None
        self.bias_list = None
        self.ub = None
        self.lb = None
        self.lambd_principle, self.w = None, None
        self.lambd, self.Mi = None, None
        self.symbols = None
        self.jaco = None

    def syn_model_parameters(self, model):
        parameters = model.parameters()
        parameters_list = [param.detach().to('cpu').numpy() for param in parameters]
        weight_list = [param for param in parameters_list if len(param.shape) > 1]
        bias_list = [param for param in parameters_list if len(param.shape) == 1]
        for i in range(len(bias_list)):
            bias_list[i] = bias_list[i].reshape((bias_list[i].size, 1))
        self.weight_list = weight_list
        self.bias_list = bias_list
        self.ub, self.lb = model.ub.to('cpu').numpy(), model.lb.to('cpu').numpy()
        self.lambd_principle, self.w = model.lambd_principle.to('cpu').numpy(), model.w.to('cpu').numpy()
        self.lambd, self.Mi = model.lambd.to('cpu').numpy(), model.Mi.to('cpu').numpy()

    def forward_single(self, xc):
        # xc should be symbolics matrix 6x1
        for weight, bias in zip(self.weight_list, self.bias_list):
            xc = weight @ xc + bias
        return xc

    def update_jaco_phi_x(self):
        symbols = sp.symbols('x0, x1, x2, x3, x4, x5')
        x = sp.Matrix(symbols)
        hx = self.forward_single(x)
        g_scaling = 1 / (self.ub - self.lb)
        g_c_xin = np.diag(g_scaling) @ (x + hx)
        phi_x = sp.zeros(self.Mi.shape[0], 1)
        for i in range(self.Mi.shape[0]):
            mi = self.Mi[i]
            phi_xi = sp.prod(sp.Matrix([element ** power for element, power in zip(g_c_xin, mi)]))
            phi_x[i, 0] = phi_xi
        self.symbols = symbols
        self.jaco = phi_x.jacobian(x)


class KoopmanDataset(Dataset):

    def __init__(self, x, y):
        self.X_train = x
        self.Y_train = y
        self.n_samples = x.shape[0]

    def __getitem__(self, index):
        return self.X_train[index], self.Y_train[index]

    def __len__(self):
        return self.n_samples


class MyDataloader:
    def __init__(self):
        self.x_train = torch.from_numpy(np.genfromtxt('./training_data/lqr_new/x_train.csv',
                                                      delimiter=',')).to(device)
        self.xdot_train = torch.from_numpy(np.genfromtxt('./training_data/lqr_new/xdot_train.csv',
                                                         delimiter=',')).to(device)
        self.xdot_nom_train = torch.from_numpy(np.genfromtxt('./training_data/lqr_new/xdot_nom_train.csv',
                                                             delimiter=',')).to(device)
        self.u_train = torch.from_numpy(np.genfromtxt('./training_data/lqr_new/u_train.csv',
                                                      delimiter=',')).to(device)
        self.unom_train = torch.from_numpy(np.genfromtxt('./training_data/lqr_new/unom_train.csv',
                                                         delimiter=',')).to(device)
        AB_edmd = torch.from_numpy(np.genfromtxt('./AB_edmd.csv', delimiter=',')).to(device)
        self.Aedmd = AB_edmd[:, :106]
        self.Bedmd = AB_edmd[:, -3:]
        self.rbf_points = torch.from_numpy(np.genfromtxt('./rbf_points.csv', delimiter=',').T).to(device)


def test_prediction_precision(model, dataloader, method='DNN', compen=0):
    example_traj = torch.from_numpy(np.genfromtxt('./training_data/test_trajectory/test_traj_0.csv',
                                                  delimiter=',')).to(device)
    prediction_rmse = torch.zeros((1, example_traj.shape[0]), dtype=torch.double, device=device)
    for i in range(compen, compen + 100):
        test_traj = torch.from_numpy(np.genfromtxt('./training_data/test_trajectory/test_traj_' + str(i) + '.csv',
                                                   delimiter=',')).to(device)
        test_x = test_traj[:, :6]
        test_u = test_traj[:, -3:]
        pred_traj = torch.zeros((example_traj.shape[0], 6), dtype=torch.double, device=device)
        pred_traj[0] = test_x[0]
        for j in range(1, example_traj.shape[0]):
            xc = pred_traj[j-1]
            xc = torch.reshape(xc, (xc.__len__(), 1))
            uc = test_u[j-1]
            uc = torch.reshape(uc, (uc.__len__(), 1))
            if method == 'DNN':
                with torch.no_grad():
                    phi_xc = model.generate_phi_x(xc.T)
                xc_aug = torch.vstack((xc, phi_xc.T))
                xc_aug_next = xc_aug + (model.Aaug @ xc_aug + model.Baug @ uc) * para.ts
                xnext = xc_aug_next[:6]
            elif method == 'edmd':
                x_aug = edmd_aug_single(xc, dataloader.rbf_points)
                x_aug_next = dataloader.Aedmd @ x_aug + dataloader.Bedmd @ uc
                xnext = x_aug_next[:6]
            else:  # method == 'first_order'
                xnext = xc + (torch.from_numpy(para.Anom).to(device) @ xc +
                              torch.from_numpy(para.Bnom).to(device) @ uc) * para.ts
            pred_traj[j] = xnext.flatten()
        rmse = torch.norm(test_x - pred_traj, dim=1)
        prediction_rmse = torch.vstack((prediction_rmse, rmse))
    prediction_rmse = prediction_rmse[1:]
    return torch.mean(prediction_rmse, 0)


def check_prediction(model, dataloader, method='DNN', select=0):
    test_traj = torch.from_numpy(np.genfromtxt('./training_data/test_trajectory/test_traj_' + str(select) + '.csv',
                                               delimiter=',')).to(device)
    test_x = test_traj[:, :6]
    test_u = test_traj[:, -3:]
    pred_traj = torch.zeros((test_traj.shape[0], 6), dtype=torch.double, device=device)
    pred_traj[0] = test_x[0]
    for j in range(1, test_traj.shape[0]):
        xc = pred_traj[j - 1]
        xc = torch.reshape(xc, (xc.__len__(), 1))
        uc = test_u[j - 1]
        uc = torch.reshape(uc, (uc.__len__(), 1))
        if method == 'DNN':
            with torch.no_grad():
                phi_xc = model.generate_phi_x(xc.T)
            xc_aug = torch.vstack((xc, phi_xc.T))
            xc_aug_next = xc_aug + (model.Aaug @ xc_aug + model.Baug @ uc) * para.ts
            xnext = xc_aug_next[:6]
        elif method == 'edmd':
            x_aug = edmd_aug_single(xc, dataloader.rbf_points)
            x_aug_next = dataloader.Aedmd @ x_aug + dataloader.Bedmd @ uc
            xnext = x_aug_next[:6]
        else:  # method == 'first_order'
            xnext = xc + (torch.from_numpy(para.Anom).to(device) @ xc +
                          torch.from_numpy(para.Bnom).to(device) @ uc) * para.ts
        pred_traj[j] = xnext.flatten()
    return pred_traj


def plot_traj(traj_ref, traj_pred):
    fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 6)
    time_steps = para.ts * np.arange(0, traj_ref.shape[0], 1)
    ax0.plot(time_steps, traj_ref[:, 0], label='phi ref')
    ax0.plot(time_steps, traj_pred[:, 0], label='phi pred')
    ax0.grid()
    ax0.legend()
    ax1.plot(time_steps, traj_ref[:, 1], label='theta ref')
    ax1.plot(time_steps, traj_pred[:, 1], label='theta pred')
    ax1.grid()
    ax1.legend()
    ax2.plot(time_steps, traj_ref[:, 2], label='psi ref')
    ax2.plot(time_steps, traj_pred[:, 2], label='psi pred')
    ax2.grid()
    ax2.legend()

    ax3.plot(time_steps, traj_ref[:, 3], label='Bwx ref')
    ax3.plot(time_steps, traj_pred[:, 3], label='Bwx pred')
    ax3.grid()
    ax3.legend()
    ax4.plot(time_steps, traj_ref[:, 4], label='Bwy ref')
    ax4.plot(time_steps, traj_pred[:, 4], label='Bwy pred')
    ax4.grid()
    ax4.legend()
    ax5.plot(time_steps, traj_ref[:, 5], label='Bwz ref')
    ax5.plot(time_steps, traj_pred[:, 5], label='Bwz pred')
    ax5.grid()
    ax5.legend()

def edmd_aug_single(xin, rbf_points):
    # xin: dim X 1
    two_norm = torch.norm(rbf_points - xin, dim=0)
    sudo_state = two_norm ** 2 * torch.log(two_norm)
    sudo_state = torch.reshape(sudo_state, (sudo_state.__len__(), 1))
    xaug = torch.vstack((xin, sudo_state))
    return xaug


def init_training(num_epoch, batch_size, learning_rate):
    MyData = MyDataloader()
    Acl = torch.from_numpy(para.Anom - para.Bnom @ para.K).to(device)
    x_ub, _ = torch.max(MyData.x_train, dim=0)
    x_lb, _ = torch.min(MyData.x_train, dim=0)
    model = DiffeomorphismDNN(Acl, x_ub, x_lb, num_epoch=num_epoch, batch_size=batch_size,
                              learning_rate=learning_rate).double().to(device)
    Symbolic = SymbolicFunc()
    return MyData, model, Symbolic


# step 1: initialize training model
MyData, model, Symbolic = init_training(num_epoch=100, batch_size=5000, learning_rate=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate, amsgrad=True)
train_dataset = KoopmanDataset(MyData.x_train, MyData.xdot_nom_train)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=model.batch_size, shuffle=True, num_workers=0,
                              worker_init_fn=np.random.seed(923))
best_loss = torch.inf
best_model_dict = None
t0 = time.time_ns() * 1e-9
tc = time.time_ns() * 1e-9
t_train = 3600 * 2 + 60 * 0
# step 2: start training
for epoch in range(model.num_epoch):
    # check prediction precision
    rmse_a, rmse_b = -1, -1
    if epoch % 2 == 0 and epoch > 0:
        Symbolic.syn_model_parameters(model)
        with torch.no_grad():
            model.keedmd_update(MyData.x_train, MyData.xdot_train, MyData.u_train, Symbolic)
            rmse_a = test_prediction_precision(model, MyData, compen=0)[-1].item()
            rmse_b = test_prediction_precision(model, MyData, compen=100)[-1].item()
            print(f'rmse_a: {rmse_a}')
            print(f'rmse_b: {rmse_b}')
    # break if time out
    if tc - t0 >= t_train:
        print('time out')
        break
    for i, (inputs, labels) in enumerate(train_dataloader):
        tc = time.time_ns() * 1e-9
        # Derive loss and log info
        loss_diffeo_linear, loss_zero_jaco, loss = model.training_loss(inputs, labels)
        if loss.item() < best_loss:
            best_model_dict = model.state_dict()
            best_loss = loss.item()
        loss_info = torch.tensor([loss_diffeo_linear.item(), loss_zero_jaco.item(), loss.item(), rmse_a, rmse_b, epoch,
                                  i], dtype=torch.double)
        model.info = torch.vstack((model.info, loss_info))
        # back stepping
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # print
        print(f'--------- Current epoch-batch: {epoch}-{i} --------')
        print(f'loss_diffeo: {loss_diffeo_linear.item():.6f}')
        print(f'loss_zero_jaco: {loss_zero_jaco.item():.6f}')
        print(f'loss: {loss.item():.6f}')
        print(f'training time: {int(tc - t0)}')
