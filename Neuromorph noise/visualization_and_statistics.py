import pickle
import numpy as np
from matplotlib import pyplot as plt


def weight_noise():
    """
    Visualization for the simulations in Fig. 5b
    Returns:

    """
    worker_results = pickle.load(open('Logfiles/Weight_noise_Final.p', 'rb'))
    results_all, results_w, results_wd, results_wtc, results_wdtc = np.zeros((4, 11)), [], [], [], []
    fig, axs = plt.subplots(1, 1, gridspec_kw={'right': 0.9}, layout='tight')
    font_axis = 16
    for i, result in enumerate(worker_results):
        if result['Para'] == 'W':
            results_w.append(result['Results'][0]['Mini_loss'])
        elif result['Para'] == 'WD':
            results_wd.append(result['Results'][0]['Mini_loss'])
        elif result['Para'] == 'WTC':
            results_wtc.append(result['Results'][0]['Mini_loss'])
        elif result['Para'] == 'WDTC':
            results_wdtc.append(result['Results'][0]['Mini_loss'])
    results_all[0, :] = np.array(results_w)
    results_all[1, :] = np.array(results_wd)
    results_all[2, :] = np.array(results_wtc)
    results_all[3, :] = np.array(results_wdtc)
    im = axs.imshow(results_all.T, vmin=0.5, vmax=1.8, origin='lower', aspect=0.7)
    axs.set_yticks(np.arange(0, 12, 2), labels=[str(np.round(x, decimals=1)) for x in np.arange(0, 1.2, 0.2)],
                   fontsize=font_axis)
    axs.set_xticks(np.arange(4), labels=['W', 'WD', r'$W\tau_c$', r'$WD\tau_c$'], fontsize=font_axis)
    axs.set_ylabel('Weights noise STD (mV)', fontsize=font_axis)
    axs.set_xlabel('Mutable parameters combinations', fontsize=font_axis)
    cbar_ax = fig.add_axes([0.64, 0.2, 0.03, 0.6])
    clb = fig.colorbar(im, cax=cbar_ax)
    clb.ax.tick_params(labelsize=font_axis)
    clb.set_label(label='Loss', fontsize=font_axis)
    plt.show()


def gaussian_noise():
    """
    Visualization for the simulations in Fig. 5a
    Returns:

    """
    result_all = pickle.load(open('Logfiles/Gaussian_Final.p', 'rb'))

    worker_results_wdtc = result_all['WDTC']
    worker_results_dtc = result_all['DTC']
    worker_results_wtc = result_all['WTC']
    worker_results_wd = result_all['WD']

    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    worker_results = [worker_results_wd, worker_results_wtc, worker_results_dtc, worker_results_wdtc]
    titles = ['$WD$', r'$W\tau_{c}$', r'$D\tau_{c}$', r'$WD\tau_{c}$']
    font_axis = 16
    font_title = 20
    for i, ax in enumerate(axs.flatten()):
        loss_map = []
        gaussian_std = []
        noise_chance = []
        xlim, ylim = np.arange(6), np.arange(6)
        for j in range(36):
            loss_map.append(worker_results[i][j]['Results'][0]['Mini_loss'])
            if j % 6 == 0:
                gaussian_std.append(worker_results[i][j]['Gaussian STD'])
            if j < 6:
                noise_chance.append(worker_results[i][j]['Noise chance'])
        loss_map, gaussian_std, noise_chance = np.array(loss_map).reshape(6, 6), np.array(gaussian_std)\
            , np.array(noise_chance)
        im = ax.imshow(loss_map, origin='lower', vmax=0.4, vmin=0)
        ylbls = [str(y) for y in gaussian_std]
        xlbls = [str(x) for x in noise_chance]
        if i == 0:
            ax.set_yticks(ylim, labels=ylbls, fontsize=font_axis)
            ax.set_xticks([])
            ax.set_ylabel('Spike jitter STD', fontsize=font_axis)
            ax.set_title(titles[i], fontsize=font_title)
        elif i == 1:
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title(titles[i], fontsize=font_title)
        elif i == 2:
            ax.set_yticks(ylim, labels=ylbls, fontsize=font_axis)
            ax.set_xticks(ylim, labels=xlbls, fontsize=font_axis)
            ax.set_xlabel('Spike insertion probability', fontsize=font_axis)
            ax.set_ylabel('Spike jitter STD', fontsize=font_axis)
            ax.set_title(titles[i], fontsize=font_title)
        else:
            ax.set_yticks([])
            ax.set_xticks(ylim, labels=xlbls, fontsize=font_axis)
            ax.set_xlabel('Spike insertion probability', fontsize=font_axis)
            ax.set_title(titles[i], fontsize=font_title)
    cbar_ax = fig.add_axes([0.897, 0.25, 0.03, 0.6])
    # noinspection PyUnboundLocalVariable
    clb = fig.colorbar(im, cax=cbar_ax)
    clb.ax.tick_params(labelsize=font_axis)
    clb.set_label(label='Loss', fontsize=font_axis)
    plt.show()