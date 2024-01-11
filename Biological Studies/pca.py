'''
N     # N neurons
g     # variance of the initial weights

# Network activity of untrained and trained network
t     # time vector
R0    # Rates of untrained network  (N x time)
R     # Rates of trained network  (N x time)
RS    # [R0, R]

# Target activitiy and corresponding times
target_t            # time vector for targets
target_activity     # targets

J0    # Initial random weight matrix (NxN)
J     # Trained weight matrix (NxN)
'''
import time
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

n_comp = 100


import resultutils
from resultutils import saveTuningData, getConnectivityFigures
import numpy as np  # https://stackoverflow.com/questions/11788950/importing-numpy-into-functions
import torch
import matplotlib.pyplot as plt
from matplotlib import cm

from HLmodel import CTRNN, CTRNN_HL # from file import function
from generateINandTARGETOUT import generateINandTARGETOUT  # from file import function
torch.autograd.set_detect_anomaly(True)

randseed = 123
np.random.seed(randseed);
torch.manual_seed(randseed)  # set random seed for reproducible results

##############################################################################

def AdapGenerateINandTARGETOUT(dim_input=2, dim_output=4, numT=200, numtrials=100):
    IN = np.zeros((numtrials, numT, dim_input))
    TARGETOUT = np.zeros((numtrials, numT, dim_output))
    itimeRNN = np.ones((numtrials, numT, dim_output))

    delay1 = np.random.randint(low=0, high=50,
                               size=numtrials)  # on each trial the RNN is supposed to remember a different constant
    orientation1 = np.radians(np.random.randint(low=340, high=345, size=numtrials))
    cosOrientation1 = np.cos(
        orientation1)  # on each trial the RNN is supposed to remember a different constant    cosOrientation1 = np.random.randint(low = 0, high = 360, size = numtrials)# on each trial the RNN is supposed to remember a different constant
    sinOrientation1 = np.sin(orientation1)  # on each trial the RNN is supposed to remember a different constant

    delay2 = np.random.randint(low=0, high=50,
                               size=numtrials)  # on each trial the RNN is supposed to remember a different constant
    orientation2 = np.radians(np.random.randint(low=0, high=360,
                                                size=numtrials))  # on each trial the RNN is supposed to remember a different constant
    cosOrientation2 = np.cos(
        orientation2)  # on each trial the RNN is supposed to remember a different constant    cosOrientation1 = np.random.randint(low = 0, high = 360, size = numtrials)# on each trial the RNN is supposed to remember a different constant
    sinOrientation2 = np.sin(orientation2)  # on each trial the RNN is supposed to remember a different constant

    for itrial in range(0, numtrials):  # 0, 1, 2, ... numtrials-1
        start = delay1[itrial]
        IN[itrial, start:start + 10, 0] = cosOrientation1[
            itrial]  # the constant is presented as an input to the RNN for the first 10 timesteps of the trial
        IN[itrial, start:start + 10, 1] = sinOrientation1[
            itrial]  # the constant is presented as an input to the RNN for the first 10 timesteps of the trial

        start += delay2[itrial] + 10
        IN[itrial, start:start + 10, 0] = cosOrientation2[
            itrial]  # the constant is presented as an input to the RNN for the first 10 timesteps of the trial
        IN[itrial, start:start + 10, 1] = sinOrientation2[
            itrial]  # the constant is presented as an input to the RNN for the first 10 timesteps of the trial

        TARGETOUT[itrial, start + 10:, 0] = cosOrientation1[itrial]  # the RNN outputs
        TARGETOUT[itrial, start + 10:, 1] = sinOrientation1[itrial]  # the RNN outputs
        TARGETOUT[itrial, start + 10:, 2] = cosOrientation2[itrial]  # the RNN outputs
        TARGETOUT[itrial, start + 10:, 3] = sinOrientation2[itrial]  # the RNN outputs

        itimeRNN[itrial, :start + 10, :] = 0
    # convert to pytorch tensors
    dtype = torch.float32
    # IN = torch.from_numpy(IN, dtype=dtype); TARGETOUT = torch.from_numpy(TARGETOUT, dtype=dtype);
    IN = torch.tensor(IN, dtype=dtype);
    TARGETOUT = torch.tensor(TARGETOUT, dtype=dtype);
    itimeRNN = torch.tensor(itimeRNN, dtype=dtype);
    return IN, TARGETOUT, itimeRNN


##############################################################################
# %% initialize network
dim_input = 2  # number of inputs, the first input dimension is for the values to be remembered, the second input dimension is for the go-cue
dim_recurrent = 100  # number of recurrent units
dim_output = 4  # number of outputs, the first output dimension is for the value inputted first, the second output dimension is for the value inputted second
numT = 150  # number of timesteps in a trial
numtrials = 1  # number of trials used for each parameter update, i.e. the number of trials generated by the function generateINandTARGETOUT for each minibatch
nonlinearity = 'retanh'
numparameterupdates = 10000  # number of parameter updates
pset_saveparameters = np.unique(np.concatenate((np.arange(0, 6), np.array([50, 100, 150, 200]), np.round(
    np.linspace(0, numparameterupdates, num=20, endpoint=True))))).astype(
    int)  # save parameters when parameter update p is a member of pset_saveparameters, save as int so we can use elements to load model parameters: for example, model_parameterupdate211.pth versus model_parameterupdate211.0.pth
dt = 1
Tau = 10 * torch.ones(dim_recurrent)
NOISEAMPLITUDE = 0.1  # 0.1, standard deviation of firing rate noise, bhneverlearn = NOISEAMPLITUDE*torch.randn(numtrials, numT, dim_recurrent)
bhneverlearn = NOISEAMPLITUDE * torch.randn(numtrials, numT, dim_recurrent)  # (numtrials, numT, dim_recurrent) tensor
ah0 = torch.zeros(dim_recurrent)
bah = torch.zeros(dim_recurrent)
by = torch.zeros(dim_output)
Wahx = torch.randn(dim_recurrent, dim_input) / np.sqrt(dim_input)
Wahh = 1.5 * torch.randn(dim_recurrent, dim_recurrent) / np.sqrt(dim_recurrent)
A = .5 * torch.randn(dim_recurrent, dim_recurrent) / np.sqrt(dim_recurrent)
B = .5 * torch.randn(dim_recurrent, dim_recurrent) / np.sqrt(dim_recurrent)
C = .5 * torch.randn(dim_recurrent, dim_recurrent) / np.sqrt(dim_recurrent)
D = .5 * torch.randn(dim_recurrent, dim_recurrent) / np.sqrt(dim_recurrent)
lrs = .01 * torch.randn(dim_recurrent, dim_recurrent) / np.sqrt(dim_recurrent)

initname = '_initWahhsussillo'
Wyh = torch.zeros(dim_output, dim_recurrent)

###########################################################################################

colors = np.array([[0.1254902 , 0.29019608, 0.52941176],
       [0.80784314, 0.36078431, 0.        ],
       [0.30588235, 0.60392157, 0.02352941],
       [0.64313725, 0.        , 0.        ]])
figsize = (8, 6)
lbls = ["Before training", "After training"]

###########################################################################################
untrainedCTRNN = CTRNN(dim_input, dim_recurrent, dim_output, Wahx=Wahx, Wahh=Wahh, Wyh=Wyh, bah=bah, by=by,
                 nonlinearity=nonlinearity, ah0=ah0, LEARN_ah0=False)
ctrnn_path = r"C:\Users\17033\BiologicalSF\HL\saving\CTRNN_retanh_dimrecurrent100_10704parameters_10000parameterupdates_initWahhsussillo_NOISEAMPLITUDE0.1_rng123\FinalCTRNN.pth"
trainedCTRNN = torch.load(ctrnn_path)

untrainedHL_CTRNN = CTRNN_HL(dim_input, dim_recurrent, dim_output, Wahx=Wahx.clone(), Wahh0=Wahh.clone(), Wyh=Wyh.clone(), bah=bah.clone(), by=by.clone(),
                 nonlinearity=nonlinearity, ah0=ah0.clone(), LEARN_ah0=False, A=A, B=B, C=C, D=D, lrs=lrs)
hl_ctrnn_path = r"C:\Users\17033\BiologicalSF\HL\saving\HL+CTRNN_retanh_dimrecurrent100_60704parameters_10000parameterupdates_initWahhsussillo_NOISEAMPLITUDE0.1_rng123\FinalHL+CTRNN.pth"
trainedHL_CTRNN = torch.load(hl_ctrnn_path)

###########################################################################################

IN, TARGETOUT, itimeRNN = AdapGenerateINandTARGETOUT(dim_input, dim_output, numT, numtrials)
bhneverlearn = NOISEAMPLITUDE * torch.randn(numtrials, numT, dim_recurrent)  # (numtrials, numT, dim_recurrent) tensor

###########################################################################################

ut_aCTRNN, CTRNN_r0, Wahhend = untrainedCTRNN(IN, dt, Tau, bhneverlearn)
aCTRNN, CTRNN_r, Wahhend = trainedCTRNN(IN, dt, Tau, bhneverlearn)
ut_aHL_CTRNN, HL_CTRNN_r0, Wahhend = untrainedHL_CTRNN(IN, dt, Tau, bhneverlearn)
aHL_CTRNN, HL_CTRNN_r, Wahhend = trainedHL_CTRNN(IN, dt, Tau, bhneverlearn)

CTRNN_r0 = np.swapaxes(CTRNN_r0.detach().numpy()[0], 1, 0)
CTRNN_r = np.swapaxes(CTRNN_r.detach().numpy()[0], 1, 0)
HL_CTRNN_r0 = np.swapaxes(HL_CTRNN_r0.detach().numpy()[0], 1, 0)
HL_CTRNN_r = np.swapaxes(HL_CTRNN_r.detach().numpy()[0], 1, 0)

print(HL_CTRNN_r0.shape)
print(HL_CTRNN_r.shape)

def getPCA(Rs, model_name):
    pcas = []
    R_projs = []
    cevrs = []
    time0 = time.time()
    for R_i in Rs:
        pca_i = PCA(n_comp)
        pca_i.fit(R_i.T)
        pcas.append(pca_i)

        # Projections
        R_i_proj = pca_i.transform(R_i.T)
        R_projs.append(R_i_proj)

        # Cumulative explained variance ratio
        cum_evr = pca_i.explained_variance_ratio_.cumsum()
        cevrs.append(cum_evr)

    print("Computing PCA took %.1f sec." % (time.time() - time0))

    # Plot activity in state space (projected separately on each datasets' PCs)

    # 2D

    fig = plt.figure(figsize=figsize)
    axes = fig.subplots(1, 2)

    for i, R_i_proj in enumerate(R_projs):
        ax = axes[i]

        ax.plot(R_i_proj[:, 0], R_i_proj[:, 1], c=colors[i])
        # Init and final
        ax.plot(R_i_proj[0, 0], R_i_proj[0, 1], 'o', ms=7, c=colors[i], alpha=0.8, label="Init")
        ax.plot(R_i_proj[-1, 0], R_i_proj[-1, 1], '*', ms=10, c=colors[i], alpha=0.8, label="Final")

        ax.axhline(0, c='0.5', zorder=-1)
        if i == 0:
            ax.legend(loc=4)
            ax.set_ylabel("PC2")
        ax.set_xlabel("PC1")
        ax.set_title(lbls[i])
    fig.suptitle(f"Projection of 2D PCA of trained and untrained {model_name} separately", y=1.05)
    plt.savefig(f"{model_name}_2D_PCA.png")
    # 3D
    fig = plt.figure(figsize=figsize)
    for i, R_i_proj in enumerate(R_projs):
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        ax.plot(R_i_proj[:, 0], R_i_proj[:, 1], R_i_proj[:, 2], c=colors[i], label=lbls[i], alpha=0.8)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')

    fig.suptitle(f"Projection of 3D PCA of trained and untrained {model_name} separately", y=1.05)
    plt.savefig(f"{model_name}_3D_PCA.png")


getPCA([CTRNN_r0, CTRNN_r], "CTRNN")
getPCA([HL_CTRNN_r0, HL_CTRNN_r], "HL_CTRNN")

###########################################################################################
#                                   Adaptation PCA                                        #
###########################################################################################

untrainedAdapCTRNN = CTRNN(dim_input, dim_recurrent, dim_output, Wahx=Wahx, Wahh=Wahh, Wyh=Wyh, bah=bah, by=by,
                 nonlinearity=nonlinearity, ah0=ah0, LEARN_ah0=False)
adap_ctrnn_path = r"C:\Users\17033\BiologicalSF\HL\saving\CTRNN_Adaptation_retanh_dimrecurrent100_10704parameters_10000parameterupdates_initWahhsussillo_NOISEAMPLITUDE0.1_rng123\FinalCTRNN_Adaptation.pth"
AdapTrainedCTRNN = torch.load(ctrnn_path)

untrainedAdapHL_CTRNN = CTRNN_HL(dim_input, dim_recurrent, dim_output, Wahx=Wahx.clone(), Wahh0=Wahh.clone(), Wyh=Wyh.clone(), bah=bah.clone(), by=by.clone(),
                 nonlinearity=nonlinearity, ah0=ah0.clone(), LEARN_ah0=False, A=A, B=B, C=C, D=D, lrs=lrs)
adap_hl_ctrnn_path = r"C:\Users\17033\BiologicalSF\HL\saving\HL+CTRNN_Adaptation_retanh_dimrecurrent100_60704parameters_10000parameterupdates_initWahhsussillo_NOISEAMPLITUDE0.1_rng123\FinalHL+CTRNN_Adaptation.pth"
AdapTrainedHL_CTRNN = torch.load(adap_hl_ctrnn_path)

###########################################################################################

IN, TARGETOUT, itimeRNN = AdapGenerateINandTARGETOUT(dim_input, dim_output, numT, numtrials)
bhneverlearn = NOISEAMPLITUDE * torch.randn(numtrials, numT, dim_recurrent)  # (numtrials, numT, dim_recurrent) tensor

###########################################################################################

ut_aCTRNN, CTRNN_r0, Wahhend = untrainedAdapCTRNN(IN, dt, Tau, bhneverlearn)
aCTRNN, CTRNN_r, Wahhend = AdapTrainedCTRNN(IN, dt, Tau, bhneverlearn)
ut_aHL_CTRNN, HL_CTRNN_r0, Wahhend = untrainedAdapHL_CTRNN(IN, dt, Tau, bhneverlearn)
# aHL_CTRNN, HL_CTRNN_r, Wahhend = AdapTrainedHL_CTRNN(IN, dt, Tau, bhneverlearn)
aHL_CTRNN, HL_CTRNN_r, Wahhend = trainedHL_CTRNN(IN, dt, Tau, bhneverlearn)

CTRNN_r0 = np.swapaxes(CTRNN_r0.detach().numpy()[0], 1, 0)
CTRNN_r = np.swapaxes(CTRNN_r.detach().numpy()[0], 1, 0)
HL_CTRNN_r0 = np.swapaxes(HL_CTRNN_r0.detach().numpy()[0], 1, 0)
HL_CTRNN_r = np.swapaxes(HL_CTRNN_r.detach().numpy()[0], 1, 0)

getPCA([CTRNN_r0, CTRNN_r], "Adap_CTRNN")
getPCA([HL_CTRNN_r0, HL_CTRNN_r], "Adap_HL_CTRNN")
