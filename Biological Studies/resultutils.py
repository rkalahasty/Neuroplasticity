import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from sklearn.preprocessing import normalize
import torch
from matplotlib import cm
from tqdm import tqdm
import os
import bct
from generateINandTARGETOUT import generateINandTARGETOUT
def normalize_2d(matrix):
    # Only this is changed to use 2-norm put 2 instead of 1
    norm = np.linalg.norm(matrix, 1)
    # normalized matrix
    matrix = matrix/norm
    return matrix

def smallWorldIndex(Wahh):
    #################################################################################
    #                                                                               #
    #                              Small world index                                #
    #                                                                               #
    # Takes in the trained recurrent weight matrix                                  #
    # Converts it to a binary adjacency matrix using avg weight as threshold        #
    # Path lengths are equivalent to the weight of the path                         #
    # Small world index = (ACCtrained / APLtrained) / (ACCrand / APLrand)           #
    #################################################################################


    # Wahh = normalize_2d(Wahh)
    #
    # APL = np.mean(Wahh)
    # badj = torch.randn(Wahh.shape)
    # for i in range(Wahh.shape[0]):
    #     for j in range(Wahh.shape[0]):
    #         badj[i][j] = 0 if Wahh[i][j] < APL else 1
    # print(badj)
    # CCVsum = 0
    # for i in range(Wahh.shape[0]):
    #     connectedNodes = {i for i, c in enumerate(badj[i]) if c == 1}
    #     degree = len(connectedNodes)
    #     Nv = 0
    #     for neighbor in connectedNodes:
    #         neighborConnections = {i for i, nde in enumerate(badj[neighbor]) if nde == 1}
    #         Nv += len(connectedNodes.union(neighborConnections))
    #     Nv = Nv/2 #links between neighbors are double counted
    #     CCVsum += (2 * Nv) / (degree * degree - 1)
    # avgAcc = CCVsum/Wahh.shape[0]
    #
    # untrainedWahh = 1.5 * torch.rand(Wahh.shape) / np.sqrt(Wahh.shape[0])
    # untrainedWahh = untrainedWahh.numpy()
    # untrainedWahh = normalize_2d(untrainedWahh)
    # print(untrainedWahh)
    # uAPL = np.mean(untrainedWahh)
    # ubadj = torch.rand(Wahh.shape)
    # for i in range(Wahh.shape[0]):
    #     for j in range(Wahh.shape[0]):
    #         ubadj[i][j] = 0 if untrainedWahh[i][j] < uAPL else 1
    #
    # uCCVsum = 0
    # for i in range(Wahh.shape[0]):
    #     connectedNodes = {i for i, c in enumerate(ubadj[i]) if c == 1}
    #     degree = len(connectedNodes)
    #     Nv = 0
    #     for neighbor in connectedNodes:
    #         neighborConnections = {i for i, nde in enumerate(ubadj[neighbor]) if nde == 1}
    #         Nv += len(connectedNodes.union(neighborConnections))
    #     Nv = Nv/2 #links between neighbors are double counted
    #     uCCVsum += (2 * Nv) / (degree * degree - 1)
    # uAvgAcc = uCCVsum/Wahh.shape[0]
    #
    # return (avgAcc/APL) / (uAvgAcc/uAPL)
    def binarize(Wahh):
        APL = np.mean(  Wahh)
        badj = np.random.randn(Wahh.shape[0], Wahh.shape[1])
        for i in range(Wahh.shape[0]):
            for j in range(Wahh.shape[0]):
                badj[i][j] = 0 if Wahh[i][j] < APL else 1
        return badj
    trianedAvgClustering = np.mean(bct.clustering_coef_bd(normalize(Wahh)))
    trainedShortestPaths = bct.distance_bin(binarize(normalize(Wahh)))
    untrainedWahh = 1.5 * torch.rand(Wahh.shape) / np.sqrt(Wahh.shape[0])
    untrainedWahh = untrainedWahh.numpy()
    untrianedAvgClustering = np.mean(bct.clustering_coef_bd(normalize(untrainedWahh)))
    untrainedShortestPaths = bct.distance_bin(binarize(normalize(untrainedWahh)))

    return (trianedAvgClustering/np.mean(trainedShortestPaths)) / (untrianedAvgClustering/np.mean(untrainedShortestPaths))

def saveTuningData(model_path, model_name, fig_dir, dim_input, dim_recurrent, dim_output, numT, numtrials, dt, Tau, bhneverlearn, NOISEAMPLITUDE):
    model = torch.load(model_path)

    def TuningCurves(dim_input=2, dim_output=4, numT=50, numtrials=100, constant1=1):
        IN = np.zeros((numtrials, numT, dim_input))
        itimeRNN = np.ones((numtrials, numT, dim_output))

        delay1 = np.random.randint(low=0, high=50, size=numtrials)  # on each trial the RNN is supposed to remember a different constant
        # orientation1 = np.full(shape = (numtrials), fill_value = np.radians(constant1))
        orientation1 = np.radians(np.random.randint(low=0, high=360,    size=numtrials))  # on each trial the RNN is supposed to remember a different constant
        cosOrientation1 = np.cos(  orientation1)  # on each trial the RNN is supposed to remember a different constant    cosOrientation1 = np.random.randint(low = 0, high = 360, size = numtrials)# on each trial the RNN is supposed to remember a different constant
        sinOrientation1 = np.sin(orientation1)  # on each trial the RNN is supposed to remember a different constant

        delay2 = np.random.randint(low=0, high=50, size=numtrials)  # on each trial the RNN is supposed to remember a different constant
        orientation2 = np.full(shape=(numtrials), fill_value=np.radians(constant1))
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

            # convert to pytorch tensors
            dtype = torch.float32
            # IN = torch.from_numpy(IN, dtype=dtype); TARGETOUT = torch.from_numpy(TARGETOUT, dtype=dtype);
            IN = torch.tensor(IN, dtype=dtype)
            itimeRNN = torch.tensor(itimeRNN, dtype=dtype)
            return IN, itimeRNN

    tuningData = []
    times = []
    for i in tqdm(range(360)):
        IN, itimeRNN = TuningCurves(dim_input, dim_output, numT, numtrials, constant1=i)
        output, h, Wahh = model(IN, dt, Tau, bhneverlearn)
        tuningH = np.average(h.detach().numpy(), axis=0)
        tuningData.append(tuningH[numT - 1, :])
        plt.show()
        times.append(i)

    arr = np.vstack(tuningData)

    plt.close()

    print(arr.shape)

    plt.plot(arr[:, 0])

    fig, ax = plt.subplots(10, 10)

    col = 0
    row = 0

    import matplotlib.ticker as ticker

    for i in tqdm(range(dim_recurrent)):
        ax[row][col].plot(times, arr[:, i])
        ax[row][col].axes.xaxis.set_major_locator(ticker.NullLocator())
        ax[row][col].axes.yaxis.set_major_locator(ticker.NullLocator())
        if (col + 1) % 10 == 0: col = 0
        if (i + 1) % 10 == 0: row += 1
        else: col += 1
        print(row, col)

    fig.suptitle(f'{model_name} Constant Shown (0 to 360)')
    os.chdir(fig_dir)
    plt.savefig(f"{model_name} memoryTuningCurves.pdf")

    def TuningCurves(dim_input=2, dim_output=4, numT=50, numtrials=100, constant1=1, constant2=1):
        IN = np.zeros((numtrials, numT, dim_input))
        itimeRNN = np.ones((numtrials, numT, dim_output))

        delay1 = np.random.randint(low=0, high=50,
                                   size=numtrials)  # on each trial the RNN is supposed to remember a different constant
        orientation1 = np.radians(np.full(shape=(numtrials), fill_value=constant1))
        cosOrientation1 = np.cos(
            orientation1)  # on each trial the RNN is supposed to remember a different constant    cosOrientation1 = np.random.randint(low = 0, high = 360, size = numtrials)# on each trial the RNN is supposed to remember a different constant
        sinOrientation1 = np.sin(orientation1)  # on each trial the RNN is supposed to remember a different constant

        delay2 = np.random.randint(low=0, high=50,
                                   size=numtrials)  # on each trial the RNN is supposed to remember a different constant
        orientation2 = np.radians(np.full(shape=(numtrials), fill_value=constant2))
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

            # convert to pytorch tensors
            dtype = torch.float32
            # IN = torch.from_numpy(IN, dtype=dtype); TARGETOUT = torch.from_numpy(TARGETOUT, dtype=dtype);
            IN = torch.tensor(IN, dtype=dtype)
            itimeRNN = torch.tensor(itimeRNN, dtype=dtype)
            return IN, itimeRNN

    tuningData = np.zeros((361, 361, 100))
    numtrials = 10
    bhneverlearn = NOISEAMPLITUDE * torch.randn(numtrials, numT,
                                                dim_recurrent)  # (numtrials, numT, dim_recurrent) tensor

    for i in tqdm(range(361)):
        for j in range(361):
            IN, itimeRNN = TuningCurves(dim_input, dim_output, numT, numtrials, constant1=i, constant2=j)
            output, h, Wahh = model(IN, dt, Tau, bhneverlearn)
            tuningH = np.average(h.detach().numpy(), axis=0)
            tuningData[i][j] = tuningH[dim_recurrent - 1, :]

    np.save(f"{model_name}tuningData.npy", tuningData)

    # print(tuningData[:, :, 0])

    plt.close()

    fig, ax = plt.subplots(10, 10)

    col = 0
    row = 0

    import matplotlib.ticker as ticker

    for i in tqdm(range(100)):
        ax[row][col].imshow((tuningData[:, :, i]))
        ax[row][col].axes.xaxis.set_major_locator(ticker.NullLocator())
        ax[row][col].axes.yaxis.set_major_locator(ticker.NullLocator())

        if (col + 1) % 10 == 0:
            col = 0
        if (i + 1) % 10 == 0:
            row += 1
        else:
            col += 1

    fig.suptitle(f'{model_name} Orientation Shown (0 to 360)')
    plt.savefig(f"{model_name} memoryTuningCurves.pdf")

def getConnectivityFigures(model_path, tuning_path, model_name, fig_dir, dim_input, dim_recurrent, dim_output, numT, numtrials, dt, Tau, bhneverlearn, NOISEAMPLITUDE):
    dir = fig_dir
    os.chdir(dir)  # print(f'current working direction is {os.getcwd()}')

    model = torch.load(model_path)
    # Wahh = model.Wahh0.detach().numpy()
    Wahh = model.fc_h2ah.weight.detach().numpy()

    def TuningCurvesO1(dim_input=2, dim_output=4, numT=50, numtrials=100, constant1 = 1):
        IN = np.zeros((numtrials, numT, dim_input))
        itimeRNN = np.ones((numtrials, numT, dim_output))

        delay1 = np.random.randint(low = 0, high = 50, size = numtrials)# on each trial the RNN is supposed to remember a different constant
        orientation1 = np.full(shape = (numtrials), fill_value = np.radians(constant1))
        # orientation1 = np.radians(np.random.randint(low = 0, high = 360, size = numtrials))# on each trial the RNN is supposed to remember a different constant
        cosOrientation1 = np.cos(orientation1)  # on each trial the RNN is supposed to remember a different constant    cosOrientation1 = np.random.randint(low = 0, high = 360, size = numtrials)# on each trial the RNN is supposed to remember a different constant
        sinOrientation1 = np.sin(orientation1)  # on each trial the RNN is supposed to remember a different constant

        delay2 = np.random.randint(low = 0, high = 50, size = numtrials)# on each trial the RNN is supposed to remember a different constant
        orientation2 = np.radians(np.random.randint(low = 0, high = 360, size = numtrials))# on each trial the RNN is supposed to remember a different constant
        cosOrientation2 = np.cos(orientation2)  # on each trial the RNN is supposed to remember a different constant    cosOrientation1 = np.random.randint(low = 0, high = 360, size = numtrials)# on each trial the RNN is supposed to remember a different constant
        sinOrientation2 = np.sin(orientation2)  # on each trial the RNN is supposed to remember a different constant

        for itrial in range(0, numtrials):  # 0, 1, 2, ... numtrials-1
            start = delay1[itrial]
            IN[itrial, start:start + 10, 0] = cosOrientation1[itrial]  # the constant is presented as an input to the RNN for the first 10 timesteps of the trial
            IN[itrial, start:start + 10, 1] = sinOrientation1[itrial]  # the constant is presented as an input to the RNN for the first 10 timesteps of the trial
            start += delay2[itrial] + 10
            IN[itrial, start:start + 10, 0] = cosOrientation2[itrial]  # the constant is presented as an input to the RNN for the first 10 timesteps of the trial
            IN[itrial, start:start + 10, 1] = sinOrientation2[itrial]  # the constant is presented as an input to the RNN for the first 10 timesteps of the trial

            # convert to pytorch tensors
            dtype = torch.float32
            # IN = torch.from_numpy(IN, dtype=dtype); TARGETOUT = torch.from_numpy(TARGETOUT, dtype=dtype);
            IN = torch.tensor(IN, dtype=dtype)
            itimeRNN = torch.tensor(itimeRNN, dtype=dtype)
            return IN, itimeRNN


    tuningData = []
    times = []

    print("FINDING PREFERENCES OF O1")
    for i in tqdm(range(360)):
        IN, itimeRNN = TuningCurvesO1(dim_input, dim_output, numT, numtrials, constant1 = i)
        output, h, Wahhend = model(IN, dt, Tau, bhneverlearn)
        tuningH = np.average(h.detach().numpy(), axis = 0)
        tuningData.append(tuningH[numT - 1, :])
        times.append(i)

    arr = np.vstack(tuningData)
    # plt.plot(arr[:, 0])
    # plt.show()

    prefO1 = []
    for i in range(100):
       prefO1.append(np.argmax(arr[:, i]))

    def TuningCurvesO2(dim_input=2, dim_output=4, numT=50, numtrials=100, constant1 = 1):
        IN = np.zeros((numtrials, numT, dim_input))
        itimeRNN = np.ones((numtrials, numT, dim_output))

        delay1 = np.random.randint(low = 0, high = 50, size = numtrials)# on each trial the RNN is supposed to remember a different constant
        # orientation1 = np.full(shape = (numtrials), fill_value = np.radians(constant1))
        orientation1 = np.radians(np.random.randint(low = 0, high = 360, size = numtrials))# on each trial the RNN is supposed to remember a different constant
        cosOrientation1 = np.cos(orientation1)  # on each trial the RNN is supposed to remember a different constant    cosOrientation1 = np.random.randint(low = 0, high = 360, size = numtrials)# on each trial the RNN is supposed to remember a different constant
        sinOrientation1 = np.sin(orientation1)  # on each trial the RNN is supposed to remember a different constant

        delay2 = np.random.randint(low = 0, high = 50, size = numtrials)# on each trial the RNN is supposed to remember a different constant
        orientation2 = np.full(shape = (numtrials), fill_value = np.radians(constant1))
        cosOrientation2 = np.cos(orientation2)  # on each trial the RNN is supposed to remember a different constant    cosOrientation1 = np.random.randint(low = 0, high = 360, size = numtrials)# on each trial the RNN is supposed to remember a different constant
        sinOrientation2 = np.sin(orientation2)  # on each trial the RNN is supposed to remember a different constant

        for itrial in range(0, numtrials):  # 0, 1, 2, ... numtrials-1
            start = delay1[itrial]
            IN[itrial, start:start + 10, 0] = cosOrientation1[itrial]  # the constant is presented as an input to the RNN for the first 10 timesteps of the trial
            IN[itrial, start:start + 10, 1] = sinOrientation1[itrial]  # the constant is presented as an input to the RNN for the first 10 timesteps of the trial
            start += delay2[itrial] + 10
            IN[itrial, start:start + 10, 0] = cosOrientation2[itrial]  # the constant is presented as an input to the RNN for the first 10 timesteps of the trial
            IN[itrial, start:start + 10, 1] = sinOrientation2[itrial]  # the constant is presented as an input to the RNN for the first 10 timesteps of the trial

            # convert to pytorch tensors
            dtype = torch.float32
            # IN = torch.from_numpy(IN, dtype=dtype); TARGETOUT = torch.from_numpy(TARGETOUT, dtype=dtype);
            IN = torch.tensor(IN, dtype=dtype)
            itimeRNN = torch.tensor(itimeRNN, dtype=dtype)
            return IN, itimeRNN


    tuningData = []
    times = []
    print("FINDING PREFERENCES OF O2")
    for i in tqdm(range(360)):
        IN, itimeRNN = TuningCurvesO2(dim_input, dim_output, numT, numtrials, constant1 = i)
        output, h, Wahhend = model(IN, dt, Tau, bhneverlearn)
        tuningH = np.average(h.detach().numpy(), axis = 0)
        tuningData.append(tuningH[numT - 1, :])
        times.append(i)

    arr = np.vstack(tuningData)
    plt.close()
    plt.plot(arr[:, 0])
    plt.show()
    # IN, OUT, itimeRNN = generateINandTARGETOUT(dim_input, dim_output, numT, numtrials)
    # output, h, Wahh = model(IN, dt, Tau, bhneverlearn)
    # Wahh = Wahh.detach().numpy()
    # Wahh = np.mean(Wahh, axis=0)
    prefO2 = []
    for i in range(100):
       prefO2.append(np.argmax(arr[:, i]))

    print(prefO1)
    print(prefO2)

    tuningData = np.load(tuning_path)

    X = []
    Y = []

    group1 = [[], [], [], []]
    group2 = [[], [], [], []]
    group3 = [[], [], [], []]
    group4 = [[], [], [], []]

    for i in range(100):
        p1 = np.mean(np.std(tuningData[:, :, i], axis=0))
        p2 = np.mean(np.std(tuningData[:, :, i], axis=1))
        if p1 == 0 and p2 == 0:
            group3[0].append(p1)
            group3[1].append(p2)
            group3[2].append(i)
        elif p1 > .03 and p2 < .02:
            group1[0].append(p1)
            group1[1].append(p2)
            group1[2].append(i)
            group1[3].append(prefO1[i])
        elif p1 < .02 and p2 > 0.03:
            group2[0].append(p1)
            group2[1].append(p2)
            group2[2].append(i)
            group1[3].append(prefO2[i])
        else:
            group4[0].append(p1)
            group4[1].append(p2)
            group4[2].append(i)


    plt.close()

    plt.scatter(group1[0], group1[1], marker='o', c='blue')
    plt.scatter(group2[0], group2[1], marker='o', c='red')
    plt.scatter(group3[0], group3[1], marker='o', c='black')
    plt.scatter(group4[0], group4[1], marker='o', c='green')

    plt.legend(["Angle1", "Angle2", "Other", "Remove"])

    plt.suptitle(f'{model_name} Preference Chart')
    plt.savefig(f"{model_name} Preference Graph.pdf")
    plt.close()
    plt.imshow(Wahh)
    plt.suptitle(f'{model_name} Unsorted Weights of Model')
    plt.savefig(f"{model_name} WeightsUnsorted.pdf")

    C = []
    PAdA = []

    for i in range(10):
        for j in range(15):
            i1 = group1[2][i]
            i2 = group1[2][j]

            angle = prefO1[int(i1)] - prefO2[int(i2)]
            if angle > 180: angle -= 360
            elif angle < -180: angle += 360

            # PAdA.append(angle)
            # C.append(Wahh[int(i1)][int(i2)])

            for q in range(angle-4, angle+4):
                PAdA.append(q)
                C.append(Wahh[int(i1)][int(i2)])

    print(len(C))

    plt.scatter(PAdA, C, s=3)

    plt.suptitle(f'{model_name} Preferences')
    plt.savefig(f"{model_name} Preferences.pdf")

    Slice = []
    Std = []
    Means = []
    for i in range(min(PAdA), max(PAdA)):
        if i in PAdA:
            indexes = np.where(np.array(PAdA) == i)[0]
            slice = [C[k] for k in indexes]
            stddev = np.std(slice)
            mean = np.mean(slice)
            Slice.append(i)
            Std.append(stddev)
            Means.append(mean)

    Slice = np.asarray(Slice)
    Std = np.asarray(Std)
    Means = np.asarray(Means)

    print(Std)

    plt.close()

    plt.plot(Slice, Means, c='black', linewidth=2.0)
    plt.fill_between(Slice, Means + Std, Means - Std, facecolor='gray')


    plt.suptitle(f'{model_name} Sliced Preferences')
    plt.savefig(f"{model_name} Sliced Preferences.pdf")

    from operator import itemgetter


    sg1 = sorted([[k, prefO1[k]] for k in group1[2]], key=itemgetter(1))
    sg2 = sorted([[k, prefO2[k]] for k in group2[2]], key=itemgetter(1))

    print(sg1)
    print(sg1[:][0])

    order = [k[0] for k in sg1] + [k[0] for k in sg2]  + group4[2] + group3[2]

    Wahh = Wahh[:, order]
    Wahh = Wahh[order, :]

    # def NormalizeData(data):
    #     return (data - np.min(data)) / (np.max(data) - np.min(data))

    fig, ax = plt.subplots()
    data = Wahh

    from matplotlib.colors import TwoSlopeNorm

    divnorm= TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)# divnorm=colors.TwoSlopeNorm(vmin=-1., vcenter=0., vmax=1), use divnorm to set the midpoint of the blue-white-red colormap to 0
    im = ax.imshow(data, cmap='bwr', norm=divnorm, origin='upper')# use divnorm to set the midpoint of the blue-white-red colormap to 0
    # create an axes on the right side of ax. The width of cax will be 5% of ax and the padding between cax and ax will be fixed at 0.05 inch. https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, ticks=[np.min(data)+abs(np.min(data)/100), 0, np.max(data)-abs(np.max(data)/100)])
    cbar.ax.set_yticklabels([f'{np.min(data):.2g}', 0, f'{np.max(data):.2g}'])# cbar.ax.set_yticklabels(['< -1', '0', '> 1'])

    plt.suptitle(f'{model_name} Connectivity')
    plt.savefig(f"{model_name} Post Connectivity.pdf")
