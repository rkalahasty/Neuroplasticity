import numpy as np
import torch
from torch.nn import functional as F  # https://twitter.com/francoisfleuret/status/1247576431762210816
from torch import nn


# device = torch.device('cpu')# device = torch.device('cuda')

# %%
##############################################################################
# continuous time recurrent neural network
# Tau * dah/dt = -ah + Wahh @ f(ah) + Wahx @ x + bah
#
# ah[t] = ah[t-1] + (dt/Tau) * (-ah[t-1] + Wahh @ h[t−1] + 􏰨Wahx @ x[t] +  bah)􏰩
# h[t] = f(ah[t]) + bhneverlearn[t], if t > 0
# y[t] = Wyh @ h[t] + by  output

# parameters to be learned: Wahh, Wahx, Wyh, bah, by, ah0(optional). In this implementation h[0] = f(ah[0]) with no noise added to h[0] except potentially through ah[0]
# constants that are not learned: dt, Tau, bhneverlearn
# Equation 1 from Miller & Fumarola 2012 "Mathematical Equivalence of Two Common Forms of Firing Rate Models of Neural Networks"
class CTRNN(nn.Module):  # class CTRNN inherits from class torch.nn.Module
    def __init__(self, dim_input, dim_recurrent, dim_output, Wahx=None, Wahh=None, Wyh=None, bah=None, by=None,
                 nonlinearity='retanh', ah0=None, LEARN_ah0=False):
        super().__init__()  # super allows you to call methods of the superclass in your subclass
        # dim_recurrent, dim_input = Wahx.shape# dim_recurrent x dim_input tensor
        # dim_output = Wyh.shape[0]# dim_output x dim_recurrent tensor
        self.fc_x2ah = nn.Linear(dim_input, dim_recurrent)  # Wahx @ x + bah
        self.fc_h2ah = nn.Linear(dim_recurrent, dim_recurrent, bias=False)  # Wahh @ h
        self.fc_h2y = nn.Linear(dim_recurrent, dim_output)  # y = Wyh @ h + by
        self.numparameters = dim_recurrent ** 2 + dim_recurrent * dim_input + dim_recurrent + dim_output * dim_recurrent + dim_output  # number of learned parameters in model
        # ------------------------------
        # initialize the biases bah and by
        if bah is not None:
            self.fc_x2ah.bias = torch.nn.Parameter(
                torch.squeeze(bah))  # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L48-L52
        if by is not None:
            self.fc_h2y.bias = torch.nn.Parameter(
                torch.squeeze(by))  # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L48-L52
        # ------------------------------
        # initialize input(Wahx), recurrent(Wahh), output(Wyh) weights
        if Wahx is not None:
            self.fc_x2ah.weight = torch.nn.Parameter(Wahx)  # Wahx @ x + bah
        if Wahh is not None:
            self.fc_h2ah.weight = torch.nn.Parameter(Wahh)  # Wahh @ h
        if Wyh is not None:
            self.fc_h2y.weight = torch.nn.Parameter(Wyh)  # y = Wyh @ h + by
        # ------------------------------
        # set the nonlinearity for h
        # pytorch seems to have difficulty saving the model architecture when using lambda functions
        # https://discuss.pytorch.org/t/beginner-should-relu-sigmoid-be-called-in-the-init-method/18689/3
        # self.nonlinearity = lambda x: f(x, nonlinearity)
        self.nonlinearity = nonlinearity
        # ------------------------------
        # set the initial state ah0
        if ah0 is None:
            self.ah0 = torch.nn.Parameter(torch.zeros(dim_recurrent), requires_grad=False)  # (dim_recurrent,) tensor
        else:
            self.ah0 = torch.nn.Parameter(ah0, requires_grad=False)  # (dim_recurrent,) tensor
        if LEARN_ah0:
            # self.ah0 = self.ah0.requires_grad=True# learn initial value for h, https://discuss.pytorch.org/t/learn-initial-hidden-state-h0-for-rnn/10013/6  https://discuss.pytorch.org/t/solved-train-initial-hidden-state-of-rnns/2589/8
            self.ah0 = torch.nn.Parameter(self.ah0, requires_grad=True)  # (dim_recurrent,) tensor
            self.numparameters = self.numparameters + dim_recurrent  # number of learned parameters in model
        # ------------------------------
        # self.LEARN_ah0 = LEARN_ah0
        # if LEARN_ah0:
        #    self.ah0 = torch.nn.Parameter(torch.zeros(dim_recurrent), requires_grad=True)# learn initial value for h, https://discuss.pytorch.org/t/learn-initial-hidden-state-h0-for-rnn/10013/6  https://discuss.pytorch.org/t/solved-train-initial-hidden-state-of-rnns/2589/8
        #    self.numparameters = self.numparameters + dim_recurrent# number of learned parameters in model

    # output y for all numT timesteps
    def forward(self, input, dt, Tau,
                bhneverlearn):  # nn.Linear expects inputs of size (numtrials, *, dim_input) where * is optional and could be numT
        # numtrials, numT, dim_input = input.size()# METHOD 1
        numtrials, numT, dim_input = input.shape  # METHOD 2
        # dim_recurrent = self.fc_h2y.weight.size(1)# y = Wyh @ h + by, METHOD 1
        # dim_recurrent = self.fc_h2y.weight.shape[1]# y = Wyh @ h + by, METHOD 2
        ah = self.ah0.repeat(numtrials,
                             1)  # (numtrials, dim_recurrent) tensor, all trials should have the same initial value for h, not different values for each trial
        # if self.LEARN_ah0:
        #    ah = self.ah0.repeat(numtrials, 1)# (numtrials, dim_recurrent) tensor, all trials should have the same initial value for h, not different values for each trial
        # else:
        #    ah = input.new_zeros(numtrials, dim_recurrent)# tensor.new_zeros(size) returns a tensor of size size filled with 0. By default, the returned tensor has the same torch.dtype and torch.device as this tensor.
        # h = self.nonlinearity(ah)# h0
        h = computef(ah, self.nonlinearity)  # h0, this implementation doesn't add noise to h0
        hstore = []  # (numtrials, numT, dim_recurrent)
        for t in range(numT):
            ah = ah + (dt / Tau) * (-ah + self.fc_h2ah(h) + self.fc_x2ah(
                input[:, t]))  # ah[t] = ah[t-1] + (dt/Tau) * (-ah[t-1] + Wahh @ h[t−1] + 􏰨Wahx @ x[t] +  bah)
            # h = self.nonlinearity(ah)  +  bhneverlearn[:,t,:]# bhneverlearn has shape (numtrials, numT, dim_recurrent)
            h = computef(ah, self.nonlinearity) + bhneverlearn[:, t,
                                                  :]  # bhneverlearn has shape (numtrials, numT, dim_recurrent)
            hstore.append(h)  # hstore += [h]
        hstore = torch.stack(hstore,
                             dim=1)  # (numtrials, numT, dim_recurrent), each appended h is stored in hstore[:,i,:], nn.Linear expects inputs of size (numtrials, *, dim_recurrent) where * means any number of additional dimensions
        return self.fc_h2y(hstore), hstore, None  # (numtrials, numT, dim_output) tensor, y = Wyh @ h + by


class CTRNN_HL(nn.Module):  # class CTRNN inherits from class torch.nn.Module
    def __init__(self, dim_input, dim_recurrent, dim_output, Wahx=None, Wahh0=None, Wyh=None, bah=None, by=None,
                 nonlinearity='retanh', ah0=None, LEARN_ah0=False, A = None, B = None, C = None, D = None, lrs = None):
        super().__init__()  # super allows you to call methods of the superclass in your subclass
        # dim_recurrent, dim_input = Wahx.shape# dim_recurrent x dim_input tensor
        # dim_output = Wyh.shape[0]# dim_output x dim_recurrent tensor
        self.fc_x2ah = nn.Linear(dim_input, dim_recurrent)  # Wahx @ x + bah
        self.fc_h2y = nn.Linear(dim_recurrent, dim_output)  # y = Wyh @ h + by
        self.numparameters = dim_recurrent ** 2 + dim_recurrent * dim_input + dim_recurrent + dim_output * dim_recurrent + dim_output  # number of learned parameters in model
        self.Wahh0 = nn.Parameter(Wahh0)
        self.mdW0 = torch.zeros(Wahh0.shape).float()
        self.A = nn.Parameter(A.float())
        self.B = nn.Parameter(B.float())
        self.C = nn.Parameter(C.float())
        self.D = nn.Parameter(D.float())
        self.lrs = nn.Parameter(lrs.float())

        # ------------------------------
        # initialize the biases bah and by
        if bah is not None:
            self.fc_x2ah.bias = torch.nn.Parameter(
                torch.squeeze(bah))  # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L48-L52
        if by is not None:
            self.fc_h2y.bias = torch.nn.Parameter(
                torch.squeeze(by))  # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L48-L52
        # ------------------------------
        # initialize input(Wahx), recurrent(Wahh), output(Wyh) weights
        if Wahx is not None:
            self.fc_x2ah.weight = torch.nn.Parameter(Wahx)  # Wahx @ x + bah
        if Wyh is not None:
            self.fc_h2y.weight = torch.nn.Parameter(Wyh)  # y = Wyh @ h + by
        # ------------------------------
        # set the nonlinearity for h
        # pytorch seems to have difficulty saving the model architecture when using lambda functions
        # https://discuss.pytorch.org/t/beginner-should-relu-sigmoid-be-called-in-the-init-method/18689/3
        # self.nonlinearity = lambda x: f(x, nonlinearity)
        self.nonlinearity = nonlinearity
        # ------------------------------
        # set the initial state ah0
        if ah0 is None:
            self.ah0 = torch.nn.Parameter(torch.zeros(dim_recurrent), requires_grad=False)  # (dim_recurrent,) tensor
        else:
            self.ah0 = torch.nn.Parameter(ah0, requires_grad=False)  # (dim_recurrent,) tensor
        if LEARN_ah0:
            # self.ah0 = self.ah0.requires_grad=True# learn initial value for h, https://discuss.pytorch.org/t/learn-initial-hidden-state-h0-for-rnn/10013/6  https://discuss.pytorch.org/t/solved-train-initial-hidden-state-of-rnns/2589/8
            self.ah0 = torch.nn.Parameter(self.ah0, requires_grad=True)  # (dim_recurrent,) tensor
            self.numparameters = self.numparameters + dim_recurrent  # number of learned parameters in model
        # ------------------------------
        # self.LEARN_ah0 = LEARN_ah0
        # if LEARN_ah0:
        #    self.ah0 = torch.nn.Parameter(torch.zeros(dim_recurrent), requires_grad=True)# learn initial value for h, https://discuss.pytorch.org/t/learn-initial-hidden-state-h0-for-rnn/10013/6  https://discuss.pytorch.org/t/solved-train-initial-hidden-state-of-rnns/2589/8
        #    self.numparameters = self.numparameters + dim_recurrent# number of learned parameters in model

    # output y for all numT timesteps
    def forward(self, input, dt, Tau, bhneverlearn):  # nn.Linear expects inputs of size (numtrials, *, dim_input) where * is optional and could be numT
        # numtrials, numT, dim_input = input.size()# METHOD 1
        numtrials, numT, dim_input = input.shape  # METHOD 2
        # dim_recurrent = self.fc_h2y.weight.size(1)# y = Wyh @ h + by, METHOD 1
        # dim_recurrent = self.fc_h2y.weight.shape[1]# y = Wyh @ h + by, METHOD 2
        ah = self.ah0.repeat(numtrials, 1)  # (numtrials, dim_recurrent) tensor, all trials should have the same initial value for h, not different values for each trial
        Wahh = torch.tile(self.Wahh0, (numtrials, 1, 1))
        mdW = torch.tile(self.mdW0, (numtrials, 1, 1))
        # mdW = self.mdW0.repeat(numtrials, 1)

        h = computef(ah, self.nonlinearity)  # h0, this implementation doesn't add noise to h0
        hstore = []  # (numtrials, numT, dim_recurrent)
        mdWstore = []
        for t in range(numT):
            ah = ah + (dt / Tau) * (-ah + torch.einsum('ijk,ik->ij', Wahh + mdW, h) + self.fc_x2ah(input[:, t]))  # ah[t] = ah[t-1] + (dt/Tau) * (-ah[t-1] + Wahh @ h[t−1] + 􏰨Wahx @ x[t] +  bah)

            h = computef(ah, self.nonlinearity) + bhneverlearn[:, t, :]  # bhneverlearn has shape (numtrials, numT, dim_recurrent)
            hstore.append(h)  # hstore += [h]

            zeta = torch.std(Wahh * h[:,:,None], dim=1)
            phi = mdW / t if t != 0 else mdW

            mdW = mdW + self.lrs[None,:,:] * (self.A[None,:,:] * zeta[:,:,None] + self.B[None,:,:] * phi + self.C[None,:,:] * (h * hstore[t - 1])[:,:,None] + self.D[None,:,:])
            mdWstore.append(mdW)
        hstore = torch.stack(hstore,  dim=1)  # (numtrials, numT, dim_recurrent), each appended h is stored in hstore[:,i,:], nn.Linear expects inputs of size (numtrials, *, dim_recurrent) where * means any number of additional dimensions
        return self.fc_h2y(hstore), hstore, Wahh + mdW # (numtrials, numT, dim_output) tensor, y = Wyh @ h + by

class CTRNN_HL_Analysis(nn.Module):  # class CTRNN inherits from class torch.nn.Module
    def __init__(self, dim_input, dim_recurrent, dim_output, Wahx=None, Wahh0=None, Wyh=None, bah=None, by=None,
                 nonlinearity='retanh', ah0=None, LEARN_ah0=False, A = None, B = None, C = None, D = None, lrs = None):
        super().__init__()  # super allows you to call methods of the superclass in your subclass
        # dim_recurrent, dim_input = Wahx.shape# dim_recurrent x dim_input tensor
        # dim_output = Wyh.shape[0]# dim_output x dim_recurrent tensor
        self.fc_x2ah = nn.Linear(dim_input, dim_recurrent)  # Wahx @ x + bah
        self.fc_h2y = nn.Linear(dim_recurrent, dim_output)  # y = Wyh @ h + by
        self.numparameters = dim_recurrent ** 2 + dim_recurrent * dim_input + dim_recurrent + dim_output * dim_recurrent + dim_output  # number of learned parameters in model
        self.Wahh0 = nn.Parameter(Wahh0)
        self.mdW0 = torch.zeros(Wahh0.shape).float()
        self.A = nn.Parameter(A.float())
        self.B = nn.Parameter(B.float())
        self.C = nn.Parameter(C.float())
        self.D = nn.Parameter(D.float())
        self.lrs = nn.Parameter(lrs.float())

        # ------------------------------
        # initialize the biases bah and by
        if bah is not None:
            self.fc_x2ah.bias = torch.nn.Parameter(
                torch.squeeze(bah))  # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L48-L52
        if by is not None:
            self.fc_h2y.bias = torch.nn.Parameter(
                torch.squeeze(by))  # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L48-L52
        # ------------------------------
        # initialize input(Wahx), recurrent(Wahh), output(Wyh) weights
        if Wahx is not None:
            self.fc_x2ah.weight = torch.nn.Parameter(Wahx)  # Wahx @ x + bah
        if Wyh is not None:
            self.fc_h2y.weight = torch.nn.Parameter(Wyh)  # y = Wyh @ h + by
        # ------------------------------
        # set the nonlinearity for h
        # pytorch seems to have difficulty saving the model architecture when using lambda functions
        # https://discuss.pytorch.org/t/beginner-should-relu-sigmoid-be-called-in-the-init-method/18689/3
        # self.nonlinearity = lambda x: f(x, nonlinearity)
        self.nonlinearity = nonlinearity
        # ------------------------------
        # set the initial state ah0
        if ah0 is None:
            self.ah0 = torch.nn.Parameter(torch.zeros(dim_recurrent), requires_grad=False)  # (dim_recurrent,) tensor
        else:
            self.ah0 = torch.nn.Parameter(ah0, requires_grad=False)  # (dim_recurrent,) tensor
        if LEARN_ah0:
            # self.ah0 = self.ah0.requires_grad=True# learn initial value for h, https://discuss.pytorch.org/t/learn-initial-hidden-state-h0-for-rnn/10013/6  https://discuss.pytorch.org/t/solved-train-initial-hidden-state-of-rnns/2589/8
            self.ah0 = torch.nn.Parameter(self.ah0, requires_grad=True)  # (dim_recurrent,) tensor
            self.numparameters = self.numparameters + dim_recurrent  # number of learned parameters in model
        # ------------------------------
        # self.LEARN_ah0 = LEARN_ah0
        # if LEARN_ah0:
        #    self.ah0 = torch.nn.Parameter(torch.zeros(dim_recurrent), requires_grad=True)# learn initial value for h, https://discuss.pytorch.org/t/learn-initial-hidden-state-h0-for-rnn/10013/6  https://discuss.pytorch.org/t/solved-train-initial-hidden-state-of-rnns/2589/8
        #    self.numparameters = self.numparameters + dim_recurrent# number of learned parameters in model

    # output y for all numT timesteps
    def forward(self, input, dt, Tau, bhneverlearn):  # nn.Linear expects inputs of size (numtrials, *, dim_input) where * is optional and could be numT
        # numtrials, numT, dim_input = input.size()# METHOD 1
        numtrials, numT, dim_input = input.shape  # METHOD 2
        # dim_recurrent = self.fc_h2y.weight.size(1)# y = Wyh @ h + by, METHOD 1
        # dim_recurrent = self.fc_h2y.weight.shape[1]# y = Wyh @ h + by, METHOD 2
        ah = self.ah0.repeat(numtrials, 1)  # (numtrials, dim_recurrent) tensor, all trials should have the same initial value for h, not different values for each trial
        Wahh = torch.tile(self.Wahh0, (numtrials, 1, 1))
        mdW = torch.tile(self.mdW0, (numtrials, 1, 1))
        # mdW = self.mdW0.repeat(numtrials, 1)

        h = computef(ah, self.nonlinearity)  # h0, this implementation doesn't add noise to h0
        hstore = []  # (numtrials, numT, dim_recurrent)
        mdWstore = []
        for t in range(numT):
            ah = ah + (dt / Tau) * (-ah + torch.einsum('ijk,ik->ij', Wahh + mdW, h) + self.fc_x2ah(input[:, t]))  # ah[t] = ah[t-1] + (dt/Tau) * (-ah[t-1] + Wahh @ h[t−1] + 􏰨Wahx @ x[t] +  bah)

            h = computef(ah, self.nonlinearity) + bhneverlearn[:, t, :]  # bhneverlearn has shape (numtrials, numT, dim_recurrent)
            hstore.append(h)  # hstore += [h]

            zeta = torch.std(Wahh * h[:,:,None], dim=1)
            phi = mdW / t if t != 0 else mdW

            mdW = mdW + self.lrs[None,:,:] * (self.A[None,:,:] * zeta[:,:,None] + self.B[None,:,:] * phi + self.C[None,:,:] * (h * hstore[t - 1])[:,:,None] + self.D[None,:,:])
            mdWstore.append(mdW)
        hstore = torch.stack(hstore,  dim=1)  # (numtrials, numT, dim_recurrent), each appended h is stored in hstore[:,i,:], nn.Linear expects inputs of size (numtrials, *, dim_recurrent) where * means any number of additional dimensions
        return self.fc_h2y(hstore), hstore, Wahh + mdW, mdWstore  # (numtrials, numT, dim_output) tensor, y = Wyh @ h + by



'''    
# A note on broadcasting:
# multiplying a (N,) array by a (M,N) matrix with * will broadcast element-wise
torch.manual_seed(123)# set random seed for reproducible results  
numtrials = 2  
Tau = torch.randn(5); Tau[-1] = 10
ah = torch.randn(numtrials,5)
A = ah + 1/Tau * (-ah)
A_check = -700*torch.ones(numtrials,5)
for i in range(numtrials):
    A_check[i,:] = ah[i,:] + 1/Tau * (-ah[i,:])# * performs elementwise multiplication
print(f"Do A and A_check have the same shape and are element-wise equal within a tolerance? {A.shape == A_check.shape and np.allclose(A, A_check)}")
'''


# -----------------------------------------------------------------------------
#                      compute specified nonlinearity
# -----------------------------------------------------------------------------
def computef(IN, string, *args):  # ags[0] is the slope for string='tanhwithslope'
    if string == 'linear':
        F = IN
        return F
    elif string == 'logistic':
        F = 1 / (1 + torch.exp(-IN))
        return F
    elif string == 'smoothReLU':  # smoothReLU or softplus
        F = torch.log(1 + torch.exp(IN))  # always greater than zero
        return F
    elif string == 'ReLU':  # rectified linear units
        # F = torch.maximum(IN,torch.tensor(0))
        F = torch.clamp(IN, min=0)
        return F
    elif string == 'swish':  # swish or SiLU (sigmoid linear unit)
        # Hendrycks and Gimpel 2016 "Gaussian Error Linear Units (GELUs)"
        # Elfwing et al. 2017 "Sigmoid-weighted linear units for neural network function approximation in reinforcement learning"
        # Ramachandran et al. 2017 "Searching for activation functions"
        sigmoid = 1 / (1 + torch.exp(-IN))
        F = torch.mul(IN, sigmoid)  # x*sigmoid(x), torch.mul performs elementwise multiplication
        return F
    elif string == 'mish':  # Misra 2019 "Mish: A Self Regularized Non-Monotonic Neural Activation Function
        F = torch.mul(IN, torch.tanh(torch.log(1 + torch.exp(IN))))  # torch.mul performs elementwise multiplication
        return F
    elif string == 'GELU':  # Hendrycks and Gimpel 2016 "Gaussian Error Linear Units (GELUs)"
        F = 0.5 * torch.mul(IN, (1 + torch.tanh(torch.sqrt(torch.tensor(2 / np.pi)) * (
                    IN + 0.044715 * IN ** 3))))  # fast approximating used in original paper
        # F = x.*normcdf(x,0,1);% x.*normcdf(x,0,1)  =  x*0.5.*(1 + erf(x/sqrt(2)))
        # figure; hold on; x = linspace(-5,5,100); plot(x,x.*normcdf(x,0,1),'k-'); plot(x,0.5*x.*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x.^3))),'r--')
        return F
    elif string == 'ELU':  # exponential linear units, Clevert et al. 2015 "FAST AND ACCURATE DEEP NETWORK LEARNING BY EXPONENTIAL LINEAR UNITS (ELUS)"
        alpha = 1
        inegativeIN = (IN < 0)
        F = IN.clone()
        F[inegativeIN] = alpha * (torch.exp(IN[inegativeIN]) - 1)
        return F
    elif string == 'tanh':
        F = torch.tanh(IN)
        return F
    elif string == 'tanhwithslope':
        a = args[0]
        F = torch.tanh(a * IN)  # F(x)=tanh(a*x), dFdx=a-a*(tanh(a*x).^2), d2dFdx=-2*a^2*tanh(a*x)*(1-tanh(a*x).^2)
        return F
    elif string == 'tanhlecun':  # LeCun 1998 "Efficient Backprop"
        F = 1.7159 * torch.tanh(
            2 / 3 * IN)  # F(x)=a*tanh(b*x), dFdx=a*b-a*b*(tanh(b*x).^2), d2dFdx=-2*a*b^2*tanh(b*x)*(1-tanh(b*x).^2)
        return F
    elif string == 'lineartanh':
        # F = torch.minimum(torch.maximum(IN,torch.tensor(-1)),torch.tensor(1))# -1(x<-1), x(-1<=x<=1), 1(x>1)
        F = torch.clamp(IN, min=-1, max=1)
        return F
    elif string == 'retanh':  # rectified tanh
        F = torch.maximum(torch.tanh(IN), torch.tensor(0))
        return F
    elif string == 'binarymeanzero':  # binary units with output values -1 and +1
        # F = (IN>=0) - (IN<0)# matlab code
        F = 1 * (IN >= 0) - 1 * (IN < 0)  # multiplying by 1 converts True to 1 and False to 0
        return F
    else:
        print('Unknown transfer function type')


##############################################################################
# %% test forward pass of continous time recurrent neural network
if __name__ == "__main__":

    import matplotlib.pyplot as plt

    torch.manual_seed(123)  # set random seed for reproducible results
    dir = r'C:\Users\17033\BiologicalSF\HL\saving'  # use r'path with spaces' if there are spaces in the path name

    dim_input = 2
    dim_recurrent = 1
    dim_output = 1
    numT = 351
    numtrials = 1
    nonlinearity = 'logistic'
    # parameters to be learned: Wahh, Wahx, Wyh, bah, by, ah0(optional)
    Wahx = torch.randn(dim_recurrent, dim_input) / np.sqrt(dim_input)
    Wahh = 1.5 * torch.randn(dim_recurrent, dim_recurrent) / np.sqrt(dim_recurrent)
    A = .5 * torch.randn(dim_recurrent, dim_recurrent) / np.sqrt(dim_recurrent)
    B = .5 * torch.randn(dim_recurrent, dim_recurrent) / np.sqrt(dim_recurrent)
    C = .5 * torch.randn(dim_recurrent, dim_recurrent) / np.sqrt(dim_recurrent)
    D = .5 * torch.randn(dim_recurrent, dim_recurrent) / np.sqrt(dim_recurrent)
    lrs = .01 * torch.randn(dim_recurrent, dim_recurrent) / np.sqrt(dim_recurrent)
    Wyh = torch.randn(dim_output, dim_recurrent)
    bah = torch.randn(dim_recurrent)
    by = torch.randn(dim_output)
    ah0 = torch.randn(dim_recurrent)
    # constants that are not learned: dt, Tau, bhneverlearn
    dt = 1
    Tau = 10 * torch.ones(dim_recurrent)
    bhneverlearn = torch.randn(numtrials, numT, dim_recurrent)  # (numtrials, numT, dim_recurrent) tensor

    # IN, TARGETOUT, itimeRNN = generateINandTARGETOUT(dim_input=2, dim_output=1, numT=numT, numtrials=numtrials)
    IN = torch.randn(numtrials, numT, dim_input)
    model = CTRNN_HL(dim_input, dim_recurrent, dim_output, Wahx=Wahx, Wahh=Wahh, Wyh=Wyh, bah=bah, by=by,
                 nonlinearity=nonlinearity, ah0=ah0, LEARN_ah0=False, A=A, B=B, C=C, D=D, lrs=lrs)
    output_pytorch, h_pytorch = model(IN, dt, Tau, bhneverlearn)

    # ---------------check number of learned parameters---------------
    numparameters = sum(p.numel() for p in model.parameters() if
                        p.requires_grad)  # model.parameters include those defined in __init__ even if they are not used in forward pass
    # assert np.allclose(model.numparameters, numparameters), "Number of learned parameters don't match!"

    fig, ax = plt.subplots()  # activity of RNN units over time
    fontsize = 14
    itrial = 1
    handle1 = ax.plot(np.arange(1, numT + 1), h_pytorch.detach().numpy()[itrial - 1, :, :], linewidth=1)
    ax.set_xlabel('Timestep', fontsize=fontsize)
    ax.set_ylabel(f'Activity of {dim_recurrent} units', fontsize=fontsize)
    ax.set_title(f'CTRNN with Hybrid Synaptic Plasticity using {nonlinearity} nonlinearity', fontsize=fontsize)
    plt.show()
    # METHOD 1: compare to forward pass written manually with torch.mm
    # ah[t] = ah[t-1] + (dt/Tau) * (-ah[t-1] + Wahh @ h[t−1] + 􏰨Wahx @ x[t] +  bah)􏰩
    # h[t] = f(ah[t]) + bhneverlearn[t]
    # y[t] = Wyh @ h[t] + by  output
    ah = -700 * torch.ones(numtrials, numT, dim_recurrent)
    h = -700 * torch.ones(numtrials, numT, dim_recurrent)
    output = -700 * torch.ones(numtrials, numT, dim_output)
    for itrial in range(0,
                        numtrials):  # note that we can eliminate this for-loop over trials by using matrix multiplication
        ahold = ah0[:,
                None]  # N x 1 tensor, add an extra dimension to ahold because adding two tensors of size (dim_recurrent,) and (dim_recurrent,1) will result in a tensor of size (dim_recurrent, dim_recurrent)
        hold = computef(ah0, nonlinearity)[:,
               None]  # N x 1 tensor, add an extra dimension so we can perform matrix multiplication with torch.mm
        for t in range(0, numT):
            ah[itrial, [t], :] = (ahold + (dt / Tau[:, None]) * (
                        -ahold + torch.mm(Wahh, hold) + torch.mm(Wahx, IN[itrial, t, :][:, None]) + bah[:,
                                                                                                    None])).transpose(0,
                                                                                                                      1)  # 1 x N tensor, add an extra dimension to bah because adding two tensors of size (dim_recurrent,) and (dim_recurrent,1) will result in a tensor of size (dim_recurrent, dim_recurrent)
            h[itrial, t, :] = computef(ah[itrial, t, :], nonlinearity) + bhneverlearn[itrial, t, :]  # (N,) tensor
            output[itrial, [t], :] = torch.transpose(torch.mm(Wyh, h[itrial, t, :][:, None]) + by[:, None], 0,
                                                     1)  # 1 x N tensor, add extra dimension to bah because adding two tensors of size (dim_recurrent,) and (dim_recurrent,1) will result in a tensor of size (dim_recurrent, dim_recurrent)

            ahold = ah[itrial, t, :][:,
                    None]  # N x 1 tensor, add an extra dimension to ahold because adding two tensors of size (dim_recurrent,) and (dim_recurrent,1) will result in a tensor of size (dim_recurrent, dim_recurrent)
            hold = h[itrial, t, :][:,
                   None]  # N x 1 tensor, add an extra dimension so we can perform matrix multiplication with torch.mm

    h = h.numpy();
    output = output.numpy()  # convert to numpy
    h_pytorch = h_pytorch.detach().numpy();
    output_pytorch = output_pytorch.detach().numpy()  # convert to numpy
    print(
        f"METHOD 1: Do h_pytorch and h have the same shape and are element-wise equal within a tolerance? {h_pytorch.shape == h.shape and np.allclose(h_pytorch, h)}")
    print(
        f"METHOD 1:Do output_pytorch and output have the same shape and are element-wise equal within a tolerance? {output_pytorch.shape == output.shape and np.allclose(output_pytorch, output)}")
    # import sys; sys.exit()# stop script at current line

    # METHOD 2: compare to forward pass written manually with torch.matmul
    ah = -700 * torch.ones(numtrials, numT, dim_recurrent)
    h = -700 * torch.ones(numtrials, numT, dim_recurrent)
    output = -700 * torch.ones(numtrials, numT, dim_output)
    for itrial in range(0,
                        numtrials):  # note that we can eliminate this for-loop over trials by using matrix multiplication
        ahold = ah0  # (N,) tensor
        hold = computef(ah0, nonlinearity)  # (N,) tensor
        for t in range(0, numT):
            ah[itrial, t, :] = ahold + (dt / Tau) * (
                        -ahold + torch.matmul(Wahh, hold) + torch.matmul(Wahx, IN[itrial, t, :]) + bah)  # (N,) tensor
            h[itrial, t, :] = computef(ah[itrial, t, :], nonlinearity) + bhneverlearn[itrial, t, :]  # (N,) tensor
            output[itrial, t, :] = torch.matmul(Wyh, h[itrial, t, :]) + by  # (N,) tensor

            ahold = ah[itrial, t, :]  # (N,) tensor
            hold = h[itrial, t, :]  # (N,) tensor

    h = h.numpy();
    output = output.numpy()  # convert to numpy
    print(
        f"METHOD 2: Do h_pytorch and h have the same shape and are element-wise equal within a tolerance? {h_pytorch.shape == h.shape and np.allclose(h_pytorch, h)}")
    print(
        f"METHOD 2: Do output_pytorch and output have the same shape and are element-wise equal within a tolerance? {output_pytorch.shape == output.shape and np.allclose(output_pytorch, output)}")
    # import sys; sys.exit()# stop script at current line
