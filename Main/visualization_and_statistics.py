"""This is the visualization module that visualizes the various experiment"""

"""Define imports"""
import pickle
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def visualize_exp_one():
    """
    Visualizes Fig.2a&b
    Returns:

    """
    na, divider, no_sum_val = 1000, 5, 10

    def get_avg_count(dic):
        sum_dic, sum_arr = {}, []
        count_dic, count_arr = {}, []

        for i, key in enumerate(dic.keys()):
            avg_arr = np.empty((dic[key].shape[0],))
            c_arr = np.empty((dic[key].shape[0],))
            for j in range(dic[key].shape[0]):
                value_index = list(np.where(dic[key][j] != na)[0])
                if value_index:
                    avg_o = np.sum(dic[key][j][value_index]) + (5-len(value_index))*no_sum_val
                    c_arr[j] = len(value_index)
                    avg_arr[j] = avg_o / divider
                else:
                    c_arr[j] = np.nan
                    avg_arr[j] = np.nan
            sum_dic[key] = avg_arr
            sum_arr.append(avg_arr)
            count_dic[key] = c_arr
            count_arr.append(c_arr)
        sum_arr = np.array(sum_arr).T
        count_arr = np.array(count_arr).T
        return sum_arr, count_arr

    file = open("Logfiles/All_simple_FINAL_RESULT_EXP_1.p", 'rb')
    all_data = pickle.load(file)
    file.close()
    data_mut = {}
    for k1, v1 in all_data.items():
        data_para = {}
        for k2, v2 in v1.items():
            data_in = {}
            for k3, v3 in v2.items():
                data_gate = []
                for ig in v3[0]:
                    data_seed = []
                    for ise in ig['Seed']:
                        data_seed.append(ise)
                    data_gate.append(data_seed)
                data_gate = np.array(data_gate)
                data_in[k3] = data_gate
            data_para[k2] = data_in
        data_mut[k1] = data_para

    w11 = data_mut['W11']['W']
    wtc11 = data_mut['W11']['WTC']
    dtc11 = data_mut['W11']['DTC']
    wd11 = data_mut['W11']['WD']
    wdtc11 = data_mut['W11']['WDTC']
    w22 = data_mut['W22']['W']
    wtc22 = data_mut['W22']['WTC']
    dtc22 = data_mut['W22']['DTC']
    wd22 = data_mut['W22']['WD']
    wdtc22 = data_mut['W22']['WDTC']

    sum_w11, count_w11 = get_avg_count(w11)
    sum_wd11, count_wd11 = get_avg_count(wd11)
    sum_wtc11, count_wtc11 = get_avg_count(wtc11)
    sum_dtc11, count_dtc11 = get_avg_count(dtc11)
    sum_wdtc11, count_wdtc11 = get_avg_count(wdtc11)

    sum_w22, count_w22 = get_avg_count(w22)
    sum_wd22, count_wd22 = get_avg_count(wd22)
    sum_wtc22, count_wtc22 = get_avg_count(wtc22)
    sum_dtc22, count_dtc22 = get_avg_count(dtc22)
    sum_wdtc22, count_wdtc22 = get_avg_count(wdtc22)

    list_sum_11 = [sum_w11, sum_wtc11, sum_dtc11, sum_wd11, sum_wdtc11]
    list_count_11 = [count_w11, count_wtc11, count_dtc11, count_wd11, count_wdtc11]
    list_sum_22 = [sum_w22, sum_wtc22, sum_dtc22, sum_wd22, sum_wdtc22]
    list_count_22 = [count_w22, count_wtc22, count_dtc22, count_wd22, count_wdtc22]
    list_titles = ['$W$', r'$W\tau_{c}$', r'$D\tau_{c}$', '$WD$', r'$WD\tau_{c}$']
    ylbls = ['XOR', 'XNOR', 'OR', 'NOR', 'AND', 'NAND']

    font_axis = 28
    font_title = 32
    fig, axa = plt.subplots(2, 5, constrained_layout=True, figsize=(10, 5), dpi=50)
    ax_top = axa[0, :]
    xlbls = ['001\n011\n\n0\n1', '001\n011\n\n1\n2', '011\n111\n\n1\n2', '011\n111\n\n2\n3',
                 '011\n111\n\n3\n4']

    for h, ax in enumerate(ax_top):
        xlim = np.arange(5)
        ylim = np.arange(6)
        cmap = matplotlib.colormaps['viridis']
        cmap.set_bad(color='tan')
        im = ax.imshow(list_sum_22[h], origin='lower', cmap=cmap, vmin=0, vmax=7, aspect='auto')
        # subfig.suptitle('Weights clipping range (-2, 2)', fontsize=20)
        ax.set_xticks([])
        ax.set_title(list_titles[h], fontsize=font_title)
        if h == 0:
            ax.set_ylabel('Problem type', fontsize=font_axis)
            ax.set_yticks(ylim, labels=ylbls, fontsize=font_axis)
        else:
            ax.set_yticks([])

    ax_mid = axa[1, :]

    for h, ax in enumerate(ax_mid):
        xlim = np.arange(5)
        ylim = np.arange(6)
        cmap = matplotlib.colormaps['viridis']
        cmap.set_bad(color='tan')
        im = ax.imshow(list_sum_11[h], origin='lower', cmap=cmap, vmin=0, vmax=7, aspect='auto')
        # subfig.suptitle('Weights clipping range (-2, 2)', fontsize=20)
        ax.set_title(list_titles[h], fontsize=font_title)
        ax.set_xticks(xlim, labels=xlbls, fontsize=font_axis)
        if h == 0:
            ax.set_ylabel('Problem type', fontsize=font_axis)
            ax.set_yticks(ylim, labels=ylbls, fontsize=font_axis)
        else:
            ax.set_yticks([])
    axins = inset_axes(ax_mid[4], width="15%", height="200%", loc='lower left', bbox_to_anchor=(1.15, 0.25, 1, 1),
                       bbox_transform=ax_mid[4].transAxes, borderpad=0,)
    clb = fig.colorbar(im, cax=axins)
    clb.ax.tick_params(labelsize=font_axis)
    clb.set_label(label='Number of generations', fontsize=font_axis)
    plt.show()


def visualize_exp_two():
    """
    Visualizes Fig.6a&b
    Returns:

    """
    na, divider, no_sum_val = 1000, 5, 200

    def get_avg_count(dic):
        sum_dic, sum_arr = {}, []
        count_dic, count_arr = {}, []

        for i, key in enumerate(dic.keys()):
            avg_arr = np.empty((dic[key].shape[0],))
            c_arr = np.empty((dic[key].shape[0],))
            for j in range(dic[key].shape[0]):
                value_index = list(np.where(dic[key][j] != na)[0])
                if value_index:
                    avg_o = np.sum(dic[key][j][value_index]) + (5-len(value_index))*no_sum_val
                    c_arr[j] = len(value_index)
                    avg_arr[j] = avg_o / divider
                else:
                    c_arr[j] = np.nan
                    avg_arr[j] = np.nan
            sum_dic[key] = avg_arr
            sum_arr.append(avg_arr)
            count_dic[key] = c_arr
            count_arr.append(c_arr)
        sum_arr = np.array(sum_arr).T
        count_arr = np.array(count_arr).T
        return sum_arr, count_arr

    file = open("Logfiles/All_spike_FINAL_RESULT_EXP_2.p", 'rb')
    all_data = pickle.load(file)
    file.close()

    data_mut = {}
    for k1, v1 in all_data.items():
        data_para = {}
        for k2, v2 in v1.items():
            data_in = {}
            for k3, v3 in v2.items():
                data_gate = []
                for ig in v3[0]:
                    data_seed = []
                    for ise in ig['Seed']:
                        data_seed.append(ise)
                    data_gate.append(data_seed)
                data_gate = np.array(data_gate)
                data_in[k3] = data_gate
            data_para[k2] = data_in
        data_mut[k1] = data_para

    wtc = data_mut['W11']['WTC']
    dtc = data_mut['W11']['DTC']
    wd = data_mut['W11']['WD']
    wdtc = data_mut['W11']['WDTC']
    wbdtc = data_mut['W11']['WBDTC']

    sum_wd, count_wd = get_avg_count(wd)
    sum_wtc, count_wtc = get_avg_count(wtc)
    sum_dtc, count_dtc = get_avg_count(dtc)
    sum_wdtc, count_wdtc = get_avg_count(wdtc)
    sum_wbdtc, count_wbdtc = get_avg_count(wbdtc)

    list_sum = [sum_wtc, sum_dtc, sum_wd, sum_wdtc, sum_wbdtc]
    list_count = [count_wtc, count_dtc, count_wd, count_wdtc, count_wbdtc]
    list_titles = ['$WT_{c}$', '$DT_{c}$', '$WD$', '$WDT_{c}$', '$WBDT_{c}$']
    ylbls = ['XOR', 'XNOR', 'OR', 'NOR', 'AND', 'NAND']

    font_axis = 26
    font_title = 28
    fig, axa = plt.subplots(2, 5, constrained_layout=True, figsize=(10, 5), dpi=50)
    ax_top = axa[0, :]

    xlbls = ['001\n011\n\n000\n001', '001\n011\n\n001\n011', '011\n111\n\n001\n011', '011\n111\n\n011\n111']
    for h, ax in enumerate(ax_top):
        ylim = np.arange(6)
        cmap = matplotlib.colormaps['viridis_r']
        cmap.set_bad(color='gray')
        im = ax.imshow(list_count[h], origin='lower', cmap=cmap, vmin=0, aspect='auto')
        # subfig.suptitle('Count map', fontsize=20)
        ax.set_xticks([])
        ax.set_title(list_titles[h], fontsize=font_title)
        if h == 0:
            ax.set_ylabel('Problem type', fontsize=font_axis)
            ax.set_yticks(ylim, labels=ylbls, fontsize=font_axis)
        else:
            ax.set_yticks([])

    axins1 = inset_axes(ax_top[4], width="15%", height="100%", loc='lower left', bbox_to_anchor=(1.15, 0, 1, 1),
                       bbox_transform=ax_top[4].transAxes, borderpad=0,)
    clb = fig.colorbar(im, cax=axins1)
    clb.ax.tick_params(labelsize=font_axis)
    clb.set_label(label='Solution count', fontsize=font_axis)

    ax_mid = axa[1, :]

    for h, ax in enumerate(ax_mid):
        xlim = np.arange(4)
        ylim = np.arange(6)
        cmap = matplotlib.colormaps['viridis']
        cmap.set_bad(color='gray')
        im = ax.imshow(list_sum[h], origin='lower', cmap=cmap, aspect='auto')
        # subfig.suptitle('Average map', fontsize=20)
        ax.set_title(list_titles[h], fontsize=font_title)
        ax.set_xticks(xlim, labels=xlbls, fontsize=font_axis)
        if h == 0:
            ax.set_ylabel('Problem type', fontsize=font_axis)
            ax.set_yticks(ylim, labels=ylbls, fontsize=font_axis)
        else:
            ax.set_yticks([])
    axins2 = inset_axes(ax_mid[4], width="15%", height="100%", loc='lower left', bbox_to_anchor=(1.15, 0, 1, 1),
                       bbox_transform=ax_mid[4].transAxes, borderpad=0,)
    clb = fig.colorbar(im, cax=axins2)
    clb.ax.tick_params(labelsize=font_axis)
    clb.set_label(label='Avg. number of generations', fontsize=font_axis)
    plt.show()


def visualize_exp_three():
    """
    Visualizes Fig.3a
    Returns:

    """
    font_legend = 8
    font_axis = 10
    font_title = 12

    dist_all = open('Logfiles/Quality_sol_FINAL_RESULT_EXP_3.p', 'rb')
    dist_all = pickle.load(dist_all)
    list_enc = ('13-01', '13-13', '37-13', '37-37', '37-715')

    dist_final_w_avg, dist_final_w_std = {}, {}
    dist_final_tc_avg, dist_final_tc_std = {}, {}
    for kh, vh in dist_all['W11'].items():  # Loop over parameter flag
        dist_all_w_avg, dist_all_w_std = [], []
        dist_all_tc_avg, dist_all_tc_std = [], []
        for kj in list_enc:  # Loop over encoding
            if kj in vh.keys():
                data_h_w_avg = vh[kj]['Avg'][2][0]
                data_h_tc_avg = vh[kj]['Avg'][2][1]
                data_h_w_std = vh[kj]['STD'][2][0]
                data_h_tc_std = vh[kj]['STD'][2][1]
                dist_all_w_avg.append(data_h_w_avg)
                dist_all_tc_avg.append(data_h_tc_avg)
                dist_all_w_std.append(data_h_w_std)
                dist_all_tc_std.append(data_h_tc_std)
            else:
                dist_all_w_avg.append(np.nan)
                dist_all_tc_avg.append(np.nan)
                dist_all_w_std.append(np.nan)
                dist_all_tc_std.append(np.nan)
        dist_final_w_avg[kh] = np.array(dist_all_w_avg)
        dist_final_tc_avg[kh] = np.array(dist_all_tc_avg)
        dist_final_w_std[kh] = np.array(dist_all_w_std)
        dist_final_tc_std[kh] = np.array(dist_all_tc_std)

    fig, ax = plt.subplots(2, 1)
    ax_ei = ax[0]
    ax_ls = ax[1]

    xlbls = ['001\n011\n\n0\n1', '001\n011\n\n1\n2', '011\n111\n\n1\n2', '011\n111\n\n2\n3', '011\n111\n\n3\n4']
    results_ei_avg = {'W': dist_final_w_avg['W'], 'WD': dist_final_w_avg['WD'], 'W$T_C$': dist_final_w_avg['WTC'],
                  'WD$T_C$': dist_final_w_avg['WDTC']}
    results_ei_std = {'W': dist_final_w_std['W'], 'WD': dist_final_w_std['WD'], 'W$T_C$': dist_final_w_std['WTC'],
                      'WD$T_C$': dist_final_w_std['WDTC']}

    x = np.arange(len(xlbls)) * 1.5  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0

    for ij, (parameters, data) in enumerate(results_ei_avg.items()):
        offset = width * multiplier
        bars = ax_ei.bar(x + offset, data, width, label=parameters, yerr=results_ei_std[parameters])
        multiplier += 1

    ax_ei.set_ylabel('EI ration (au)', fontsize=font_axis)
    ax_ei.set_title('Excitatory inhibitory connections ratio', fontsize=font_title)
    ax_ei.set_xticks([])
    ax_ei.legend(loc='upper left', ncols=4, fontsize=font_legend)
    ax_ei.tick_params(axis='both', which='major', labelsize=font_axis)
    ax_ei.set_ylim(0, 8)
    ####################################################################################################################
    """LS ratio"""

    results_ls_avg = {'$T_c$D': dist_final_tc_avg['DTC'], '$T_c$W': dist_final_tc_avg['WTC'],
                      '$T_c$WD': dist_final_tc_avg['WDTC']}
    results_ls_std = {'$T_c$D': dist_final_tc_std['DTC'], '$T_c$W': dist_final_tc_std['WTC'],
                      '$T_c$WD': dist_final_tc_std['WDTC']}
    multiplier = 0
    for parameters, data in results_ls_avg.items():
        offset = width * multiplier
        bars = ax_ls.bar(x + offset, data, width, label=parameters, yerr=results_ls_std[parameters])
        # ax_ei.bar_label(bars, padding=5, fontsize=font_axis)
        multiplier += 1

    ax_ls.set_ylabel('LS ration (au)', fontsize=font_axis)
    ax_ls.set_title('Long short time constants ratio', fontsize=font_title)
    ax_ls.set_xticks(x + width * 1.5, xlbls, fontsize=font_axis)
    ax_ls.legend(loc='upper left', ncols=4, fontsize=font_legend)
    ax_ls.tick_params(axis='both', which='major', labelsize=font_axis)
    ax_ls.set_ylim(0, 2)

    plt.show()


def visualize_exp_four():
    """
    Visualizes Fig.3d
    Returns:

    """
    font_legend = 8
    font_axis = 10
    font_title = 12

    dist_w_all_vtc = open('Logfiles/Dist_WTC_FINAL_RESULT_EXP_4.p', 'rb')
    dist_w_all_vtc = pickle.load(dist_w_all_vtc)

    dist_final = {}
    dict_flag = {'DTC': 2, 'WD': 0}
    for kh, vh in dist_w_all_vtc['W11'].items():  # Loop over parameter flag
        dist_all = {}
        for kj, vj in vh.items():  # Loop over encoding
            data_h = vj['Dist'][2]
            data_add = []
            for ik in range(len(data_h)):
                data_add += list(data_h[ik][dict_flag[kh]])
            dist_all[kj] = np.array(data_add)
        dist_final[kh] = dist_all

    dist_w_01_data = dist_final['WD']['37-01']
    dist_w_12_data = dist_final['WD']['37-13']
    dist_w_23_data = dist_final['WD']['37-37']
    dist_w_34_data = dist_final['WD']['37-715']

    fig, axs = plt.subplots(2, 4)
    ax_dist_w_01, ax_dist_w_12, ax_dist_w_23, ax_dist_w_34 = axs[0, :]
    dist_w_01 = ax_dist_w_01.hist(dist_w_01_data, density=True, color='green')
    dist_w_01 = sum(dist_w_01[0][5:]) / sum(dist_w_01[0][:5])
    labels = ['EI ratio = ' + str(np.round(dist_w_01, decimals=2))]
    ax_dist_w_01.legend(labels=labels, fontsize=font_legend, loc='upper left')

    dist_w_12 = ax_dist_w_12.hist(dist_w_12_data, density=True, color='green')
    dist_w_12 = sum(dist_w_12[0][5:]) / sum(dist_w_12[0][:5])
    labels = ['EI ratio = ' + str(np.round(dist_w_12, decimals=2))]
    ax_dist_w_12.legend(labels=labels, fontsize=font_legend, loc='upper left')

    dist_w_23 = ax_dist_w_23.hist(dist_w_23_data, density=True, color='green')
    dist_w_23 = sum(dist_w_23[0][5:]) / sum(dist_w_23[0][:5])
    labels = ['EI ratio = ' + str(np.round(dist_w_23, decimals=2))]
    ax_dist_w_23.legend(labels=labels, fontsize=font_legend, loc='upper left')

    dist_w_34 = ax_dist_w_34.hist(dist_w_34_data, density=True, color='green')
    dist_w_34 = sum(dist_w_34[0][5:]) / sum(dist_w_34[0][:5])
    labels = ['EI ratio = ' + str(np.round(dist_w_34, decimals=2))]
    ax_dist_w_34.legend(labels=labels, fontsize=font_legend, loc='upper left')

    ax_dist_w_01.set_ylim([0, 2.7])
    ax_dist_w_01.set_title('Output code 0-1', fontsize=font_title)
    ax_dist_w_12.set_ylim([0, 2.7])
    ax_dist_w_12.set_title('Output code 1-2', fontsize=font_title)
    ax_dist_w_12.set_yticks([])
    ax_dist_w_23.set_ylim([0, 2.7])
    ax_dist_w_23.set_title('Output code 2-3', fontsize=font_title)
    ax_dist_w_23.set_yticks([])
    ax_dist_w_34.set_ylim([0, 2.7])
    ax_dist_w_34.set_title('Output code 3-4', fontsize=font_title)
    ax_dist_w_34.set_yticks([])
    # xlbls, ylbls = [str(x) for x in xlim], [str(y) for y in ylim]
    # ax.set_xticks(xlim, labels=xlbls, fontsize=30)
    # ax.set_yticks(ylim, labels=ylbls, fontsize=30)
    ax_dist_w_01.set_xlabel('Weights (mv)', fontsize=font_axis)
    ax_dist_w_01.set_ylabel('Prob. density (au)', fontsize=font_axis)
    ax_dist_w_12.set_xlabel('Weights (mv)', fontsize=font_axis)
    ax_dist_w_23.set_xlabel('Weights (mv)', fontsize=font_axis)
    ax_dist_w_34.set_xlabel('Weights (mv)', fontsize=font_axis)
    ax_dist_w_01.tick_params(axis='both', which='major', labelsize=font_axis)
    ax_dist_w_12.tick_params(axis='both', which='major', labelsize=font_axis)
    ax_dist_w_23.tick_params(axis='both', which='major', labelsize=font_axis)
    ax_dist_w_34.tick_params(axis='both', which='major', labelsize=font_axis)

    ax_dist_tc_01, ax_dist_tc_12, ax_dist_tc_23, ax_dist_tc_34 = axs[1, :]
    dist_tc_01_data = dist_final['DTC']['37-01']
    dist_tc_12_data = dist_final['DTC']['37-13']
    dist_tc_23_data = dist_final['DTC']['37-37']
    dist_tc_34_data = dist_final['DTC']['37-715']

    dist_tc_01 = ax_dist_tc_01.hist(dist_tc_01_data, density=True, color='green')
    dist_tc_01 = sum(dist_tc_01[0][5:]) / sum(dist_tc_01[0][:5])
    labels = ['EI ratio = ' + str(np.round(dist_tc_01, decimals=2))]
    ax_dist_tc_01.legend(labels=labels, fontsize=font_legend, loc='upper right')

    dist_tc_12 = ax_dist_tc_12.hist(dist_tc_12_data, density=True, color='green')
    dist_tc_12 = sum(dist_tc_12[0][5:]) / sum(dist_tc_12[0][:5])
    labels = ['EI ratio = ' + str(np.round(dist_tc_12, decimals=2))]
    ax_dist_tc_12.legend(labels=labels, fontsize=font_legend, loc='upper right')

    dist_tc_23 = ax_dist_tc_23.hist(dist_tc_23_data, density=True, color='green')
    dist_tc_23 = sum(dist_tc_23[0][5:]) / sum(dist_tc_23[0][:5])
    labels = ['EI ratio = ' + str(np.round(dist_tc_23, decimals=2))]
    ax_dist_tc_23.legend(labels=labels, fontsize=font_legend, loc='upper right')

    dist_tc_34 = ax_dist_tc_34.hist(dist_tc_34_data, density=True, color='green')
    dist_tc_34 = sum(dist_tc_34[0][5:]) / sum(dist_tc_34[0][:5])
    labels = ['EI ratio = ' + str(np.round(dist_tc_34, decimals=2))]
    ax_dist_tc_34.legend(labels=labels, fontsize=font_legend, loc='upper right')

    ax_dist_tc_01.set_ylim([0, 0.9])
    ax_dist_tc_01.set_title('Output code 0-1', fontsize=font_title)
    ax_dist_tc_12.set_ylim([0, 0.9])
    ax_dist_tc_12.set_title('Output code 1-2', fontsize=font_title)
    ax_dist_tc_12.set_yticks([])
    ax_dist_tc_23.set_ylim([0, 0.9])
    ax_dist_tc_23.set_title('Output code 2-3', fontsize=font_title)
    ax_dist_tc_23.set_yticks([])
    ax_dist_tc_34.set_ylim([0, 0.9])
    ax_dist_tc_34.set_title('Output code 3-4', fontsize=font_title)
    ax_dist_tc_34.set_yticks([])
    # xlbls, ylbls = [str(x) for x in xlim], [str(y) for y in ylim]
    # ax.set_xticks(xlim, labels=xlbls, fontsize=30)
    # ax.set_yticks(ylim, labels=ylbls, fontsize=30)
    ax_dist_tc_01.set_xlabel('Time constants (ms)', fontsize=font_axis)
    ax_dist_tc_01.set_ylabel('Prob. density (au)', fontsize=font_axis)
    ax_dist_tc_12.set_xlabel('Time constants (ms)', fontsize=font_axis)
    ax_dist_tc_23.set_xlabel('Time constants (ms)', fontsize=font_axis)
    ax_dist_tc_34.set_xlabel('Time constants (ms)', fontsize=font_axis)
    ax_dist_tc_01.tick_params(axis='both', which='major', labelsize=font_axis)
    ax_dist_tc_12.tick_params(axis='both', which='major', labelsize=font_axis)
    ax_dist_tc_23.tick_params(axis='both', which='major', labelsize=font_axis)
    ax_dist_tc_34.tick_params(axis='both', which='major', labelsize=font_axis)
    plt.show()


def visualize_exp_five():
    """
    Visualizes Fig.3b
    Returns:

    """
    font_axis = 10
    font_title = 12

    dist_w_all_v = open('Logfiles/Clip_effect_FINAL_RESULT_EXP_5.p', 'rb')
    dist_w_all_v = pickle.load(dist_w_all_v)
    dist_w_all_final = {}
    for kh, vh in dist_w_all_v.items():
        data_in_all = vh['WD']['13-13']['Dist'][2]  # Then seed, parameter,
        list_w = []
        for i in range(len(data_in_all)):
            list_w += list(data_in_all[i][0])
        list_w = np.array(list_w)
        dist_w_all_final[kh] = list_w

    dist_w_08v_data = dist_w_all_final['W08']
    dist_w_09v_data = dist_w_all_final['W09']
    dist_w_12v_data = dist_w_all_final['W12']
    dist_w_13v_data = dist_w_all_final['W13']

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].hist(dist_w_08v_data, density=True, color='green')
    axs[0, 1].hist(dist_w_09v_data, density=True, color='green')
    axs[1, 0].hist(dist_w_12v_data, density=True, color='green')
    axs[1, 1].hist(dist_w_13v_data, density=True, color='green')

    axs[0, 0].set_ylim([0, 3.0])
    axs[0, 0].set_xlim([-1.4, 1.4])
    axs[0, 0].set_title('(-0.8, 0.8) mv', fontsize=font_title)
    axs[0, 0].set_xticks([])
    axs[0, 1].set_ylim([0, 3.0])
    axs[0, 1].set_xlim([-1.4, 1.4])
    axs[0, 1].set_title('(-0.9, 0.9) mv', fontsize=font_title)
    axs[0, 1].set_yticks([])
    axs[0, 1].set_xticks([])
    axs[1, 0].set_ylim([0, 3.0])
    axs[1, 0].set_xlim([-1.4, 1.4])
    axs[1, 0].set_title('(-1.2, 1.2) mv', fontsize=font_title)
    axs[1, 1].set_ylim([0, 3.0])
    axs[1, 1].set_xlim([-1.4, 1.4])
    axs[1, 1].set_title('(-1.3, 1.3) mv', fontsize=font_title)
    axs[1, 1].set_yticks([])
    # xlbls, ylbls = [str(x) for x in xlim], [str(y) for y in ylim]
    # ax.set_xticks(xlim, labels=xlbls, fontsize=30)
    # ax.set_yticks(ylim, labels=ylbls, fontsize=30)
    axs[1, 0].set_xlabel('Weights (mv)', fontsize=font_axis)
    axs[1, 1].set_xlabel('Weights (mv)', fontsize=font_axis)
    axs[0, 0].set_ylabel('Prob. density (au)', fontsize=font_axis)
    axs[1, 0].set_ylabel('Prob. density (au)', fontsize=font_axis)
    axs[0, 0].tick_params(axis='both', which='major', labelsize=font_axis)
    axs[0, 1].tick_params(axis='both', which='major', labelsize=font_axis)
    axs[1, 0].tick_params(axis='both', which='major', labelsize=font_axis)
    axs[1, 1].tick_params(axis='both', which='major', labelsize=font_axis)
    plt.show()


def visualize_exp_six():
    """
    Visualizes Fig.4a
    Returns:

    """
    choice = 1
    data_all = open('Logfiles/Clip_WTC_effect_FINAL_RESULT_EXP_6.p', 'rb')
    data_all = pickle.load(data_all)
    dic_avg = {'15-13': np.empty((7, 4)), '19-13': np.empty((7, 4)), '117-13': np.empty((7, 4))}
    dic_count = {'15-13': np.empty((7, 4)), '19-13': np.empty((7, 4)), '117-13': np.empty((7, 4))}
    dic_std = {'15-13': np.empty((7, 4)), '19-13': np.empty((7, 4)), '117-13': np.empty((7, 4))}
    for i, (k1, v1) in enumerate(data_all.items()):  # Clip range w
        for k2, v2 in v1['WD'].items():  # Object encoding
            val_tc_avg, val_tc_std, val_tc_count = [], [], []
            for k3 in range(len(v2)):  # TC
                val_avg = []
                val_count = 0
                val = v2[k3][0]['Seed']
                for k4 in range(len(val)):
                    if val[k4] > 990:
                        continue
                    else:
                        val_avg.append(val[k4])
                        val_count += 1
                if val_avg:
                    val_avg = np.array(val_avg)
                    val_std = np.std(val_avg)
                    val_avg = np.average(val_avg)
                else:
                    val_avg = np.nan
                    val_std = np.nan
                    val_count = np.nan
                val_tc_avg.append(val_avg)
                val_tc_std.append(val_std)
                val_tc_count.append(val_count)
            val_tc_avg = np.array(val_tc_avg)
            val_tc_std = np.array(val_tc_std)
            val_tc_count = np.array(val_tc_count)
            dic_avg[k2][:, i] = val_tc_avg
            dic_count[k2][:, i] = val_tc_count
            dic_std[k2][:, i] = val_tc_std

    data_in_plot = {1: (dic_avg, 40), 2: (dic_std, 15), 3: (dic_count, 5)}
    data_in_fig = data_in_plot[choice]

    fig = plt.figure(constrained_layout=True, figsize=(80, 40), dpi=50)
    gs_outer = gridspec.GridSpec(2, 3, wspace=0.1, hspace=0.1)
    gs_outer.update(left=0.1, right=0.9, top=0.965, bottom=0.05, wspace=0.0001, hspace=0.5)
    font_axis = 24
    font_sub_title = 26
    grid_top = gridspec.GridSpecFromSubplotSpec(1, 3, gs_outer[:, :])
    ax_top = [fig.add_subplot(grid_top[0, 0]), fig.add_subplot(grid_top[0, 1]), fig.add_subplot(grid_top[0, 2])]

    title_list = ['101', '1001', '10001']
    title_list = ['Input two encoding = ' + x for x in title_list]

    for i, ax in enumerate(ax_top):
        range_tc = np.arange(1, 8, 1)
        xlbls = [u"\u00B1"'0.6', u"\u00B1"'0.7', u"\u00B1"'0.8', u"\u00B1"'0.9']
        ylbls = [str(x) for x in range_tc]
        xlim = np.arange(4)
        ylim = np.arange(7)
        if i == 0:
            ax.set_xlabel('Weight clipping range (mv)', fontsize=font_axis)
            ax.set_ylabel('Time constant (ms)', fontsize=font_axis)
            ax.set_xticks(xlim, labels=xlbls, fontsize=font_axis)
            ax.set_yticks(ylim, labels=ylbls, fontsize=font_axis)
        else:
            ax.set_xlabel('Weight clipping range (mv)', fontsize=font_axis)
            ax.set_xticks(xlim, labels=xlbls, fontsize=font_axis)
            ax.set_yticks([])
        cmap = matplotlib.colormaps['viridis']
        cmap.set_bad(color='gray')
        image = np.where(data_in_fig[0][list(data_in_fig[0].keys())[i]] < 90,
                         data_in_fig[0][list(data_in_fig[0].keys())[i]], np.nan)
        im = ax.imshow(image, vmin=0, vmax=data_in_fig[1], origin='lower', aspect=0.5, cmap=cmap)
        ax.set_title(title_list[i], fontsize=font_sub_title)

    axins = inset_axes(ax_top[2], width="15%", height="100%", loc='lower left', bbox_to_anchor=(1.15, 0, 1, 1),
                       bbox_transform=ax_top[2].transAxes, borderpad=0, )
    clb = fig.colorbar(im, cax=axins)
    clb.ax.tick_params(labelsize=font_axis)
    clb.set_label(label='Average number of generations', fontsize=font_axis)
    plt.show()


def visualize_exp_seven():
    """
    Visualizes Fig.4b
    Returns:

    """
    font_legend = 14
    font_axis = 24

    dtc_effect_data = open('Logfiles/Delay_TC_effect_FINAL_RESULT_EXP_7.p', 'rb')
    dtc_effect_data = pickle.load(dtc_effect_data)
    keys_l = ('1', '2', '3', '4', '5', '6')
    results_oi_avg = {'1': [], '2': [], '3': [], '4': [], '5': [], '6': []}
    results_oi_std = {'1': [], '2': [], '3': [], '4': [], '5': [], '6': []}
    for i1, k1 in enumerate(dtc_effect_data.keys()):
        data_in_avg = dtc_effect_data[k1]['WD']['15-13']['Avg']
        data_in_std = dtc_effect_data[k1]['WD']['15-13']['STD']
        for i, (tc_avg, tc_std) in enumerate(zip(data_in_avg.values(), data_in_std.values())):
            results_oi_avg[keys_l[i]].append(tc_avg)
            results_oi_std[keys_l[i]].append(tc_std)
    for i1, k1 in enumerate(results_oi_avg.keys()):
        results_oi_avg[k1] = np.round(np.array(results_oi_avg[k1]), decimals=3)
        results_oi_std[k1] = np.round(np.array(results_oi_std[k1]), decimals=3)

    xlbls = [u"\u00B1"'0.7', u"\u00B1"'0.8', u"\u00B1"'0.9']
    fig, ax_bot = plt.subplots(1, 1)
    x = np.arange(len(xlbls)) * 1.5  # the label locations
    width = 0.22  # the width of the bars
    multiplier = 0

    labels_i = [['4 ms', '2.5 ms', '1.5 ms'], ['4.5 ms', '3 ms', '2 ms'], ['5 ms', '4 ms', '2.5 ms'],
                ['8 ms', '5 ms', '4 ms'], ['10 ms', '6 ms', ' 6ms'], ['12 ms', '8 ms', '8 ms']]
    cmap = matplotlib.colormaps['viridis']
    for i, (parameters, data) in enumerate(results_oi_avg.items()):
        offset = width * multiplier
        data = np.round(data, decimals=3)
        bars = ax_bot.bar(x + offset, data, width, label=parameters, color=['royalblue', 'darkslateblue'
            , 'yellowgreen'], yerr=results_oi_std[parameters])
        ax_bot.bar_label(bars, padding=5, fontsize=font_legend, labels=labels_i[i])
        multiplier += 1

    ax_bot.set_ylabel('Average change in delays (ms)', fontsize=font_axis)
    # ax_bot.set_title('Distribution ratios', fontsize=font_title)
    ax_bot.set_xticks(x + width * 2.5, xlbls, fontsize=font_axis)
    ax_bot.set_xlabel('Weight clipping range (mv)', fontsize=font_axis)
    # ax_bot.legend(loc='upper left', ncols=11, fontsize=font_legend)
    ax_bot.tick_params(axis='both', which='major', labelsize=font_axis)
    shift_y = (1.4, 1.7)
    ax_bot.set_ylim(shift_y[0], shift_y[1])
    ax_bot.set_yticks(np.arange(shift_y[0], shift_y[1], 0.2), [str(np.round(x, decimals=1))
                                                               for x in np.arange(shift_y[0], shift_y[1], 0.2)],
                      fontsize=font_axis)

    plt.show()


def visualize_exp_eight():
    """
    Visualizes Fig.6c
    Returns:

    """
    font_axis = 12

    fig, ax_bot = plt.subplots(1, 6)
    delay_shift_data = open('Logfiles/Delay_shift_FINAL_RESULT_EXP_8.p', 'rb')
    delay_shift_data = pickle.load(delay_shift_data)
    list_keys = list(delay_shift_data['W11']['WDTC'].keys())
    dist_data = []
    for kl in list_keys:
        list_data = []
        for i in range(5):
            list_data += list(delay_shift_data['W11']['WDTC'][kl]['Dist'][5][i][3])
        dist_data.append(list_data)

    list_title = ['Out 1: .|...............................\n Out 2: ......|..........................',
                  'Out 1: ......|..........................\n Out 2: ...........|.....................',
                  'Out 1: ...........|.....................\n Out 2: ................|................',
                  'Out 1: ................|................\n Out 2: .....................|...........',
                  'Out 1: .....................|...........\n Out 2: ..........................|......',
                  'Out 1: ..........................|......\n Out 2: ...............................|.']
    for i, ax in enumerate(ax_bot):
        ax.hist(dist_data[i], density=True)
        ax.set_ylim(0, 0.2)
        ax.set_xticks(np.arange(-10, 15, 5), labels=[str(x) for x in np.arange(-10, 15, 5)], fontsize=font_axis)
        ax.set_xlabel('Change in delays (ms)', fontsize=font_axis)
        ax.set_title(list_title[i], fontsize=12)
        if i == 0:
            ax.set_ylabel('Probability density (au)', fontsize=font_axis)
            ax.set_yticks(np.arange(0, 0.25, 0.05), labels=[str(np.round(y, decimals=2)) for y in
                                                            np.arange(0, 0.25, 0.05)], fontsize=font_axis)
        else:
            ax.set_yticks([])

    fig.subplots_adjust(wspace=0.2)
    fig.savefig('Panel2_mainmap.png')
    plt.show()



