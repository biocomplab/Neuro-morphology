"""This is the main module where simulations are performed"""

"""Define imports"""
import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import torch

import pickle
import torch.multiprocessing as mp
from functools import partial

from input_problems import FUNCTIONS
from settings import (get_exp_one, get_exp_two, get_exp_three, get_exp_four, get_exp_five, get_exp_six, get_exp_seven,
                      get_exp_eight, EXP_FLAGS)
from visualization_and_statistics import (visualize_exp_one, visualize_exp_two, visualize_exp_three, visualize_exp_four,
                                          visualize_exp_five, visualize_exp_six, visualize_exp_seven,
                                          visualize_exp_eight)

from evolutionary_algorithms import ea_map
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.autograd.set_detect_anomaly(True)

VISUALIZE, EXP_SINGLE = False, False  # Visualize directly without running simulations, EXP_single: Whether to run a
# single defined experiment
MULTIPROCESSING, CORE_COUNT, CUDA_COUNT = True, mp.cpu_count(), torch.cuda.device_count()  # Use multiprocessing,
# number of CPU cores, number of CUDA cores
CROSS_WORKER_COM, SEED_BIAS = True, 4  # If GPU share elites, single seed bias
PROBLEM_SPECS = {'Problem_type': FUNCTIONS['Delay_ms'], 'Out_spikes': False, 'Encoding_length': 5, 'Problem_pad': 0,
                 'Digits': 2, 'Decay': False, 'Change_objects': True, 'Objects': ('01', '05', '21'), 'd_s': 1,
                 'Logic_type': 'XOR', 'Out_simple': (1, 2), 'Input_pad': '0000', 'Delay_shift': (False, 3),
                 'Ordered_unique': True}  # Input problem used, Out_spikes: if target is spike train,
# Encoding_length: the size of the decimal to binary encoding, Problem_pad: the padding of the decimal to binary
# encoding, Digits: the size of the decimal chunk, Decay: if decay is applied to the output, Change_objects: whether to
# change the problem type during testing, Objects: the intput decimal encoding, d_s a decorative parameter for the
# delay sign, Logic_type, the problem being investigated, Out_simple: the output encoding, Input_pad, the size of
# the pad to the input, Delay_shift: whether to shift the output and the count of the shifts
MAPE = {'Flag': False, 'Size': 20, 'Scale': (10, 10), 'Decimals': 2, 'Keep_e': True, 'Grid_score': False,
        'Double_sort': False, 'Gene_time': (1800000, 202, 5), 'Choice_corr': ('W', 'Dist_ini', 'B', 'Dist_ini'),
        'Alt_dim': (True, 'Spike Count', 'Ratio EX')}  # Not used except for the Gene_time: which holds the maximum
# and target number of generations
LOSS_FN, LOSS_CLIP = 'MSE', {'Flag': True, 'Range': (0, 6)}  # Loss function and  the loss clip range
SHAPE_INPUT, SHAPE_OUT = 2, 1  # Shape of input and output
BATCH_SIZE = 4  # Batch size
EPOCH = 8000  # Number of epochs
POPULATION_SIZE = 160000  # Must be divisible by NUM_ELITES
NUM_ELITES = 2000  # Number of elites


DEVICE, DTYPE = torch.device("cuda" if torch.cuda.is_available() else "cpu"), torch.float16  # Device used and data type
NET_ARCH = {'Max_sizes': (SHAPE_INPUT, 4, SHAPE_OUT), 'Rand_num': False, 'Rand_pos': 'none',
            'Range_x': (-3.5, 3.5, 0.5), 'Type_con': 'uniform'}  # For geometric networks, not used except for
# Max_sizes, which specify the size of the network
NET_MUT_PROP = {'Prop_con_type': 'Const', 'Prop_con': 1, 'Prop_w': 0.2}  # In the case of evolving the connections
MUT_RATE = {'Flag_w': True, 'Value_w': 0.25, 'Range_w': (-1, 1), 'Flag_b': False, 'Value_b': 0.1,
            'Range_b': (-0.4, 0), 'Flag_t': False, 'Value_t': 3, 'Range_t': (0.1, 10), 'Flag_d': True,
            'Value_d': 2, 'Range_d': (-4, 4)}  # Flags, mutations rates and clipping ranges for all the parameters
NET_PARA = {'Range_w': MUT_RATE['Range_w'], 'Grad_w': False, 'Range_b': MUT_RATE['Range_b'], 'Grad_b': False,
            'Range_d': MUT_RATE['Range_d'], 'Grad_d': False, 'Range_t': MUT_RATE['Range_t'], 'Grad_t': False,
            'Synapse_t': True, 'Synapse_b': True}  # Adds the possibility to differentiate the parameters
OTHER_PARA = {'Const_tc': True, 'Const_tc_val': 4, 'Const_delay': False, 'Const_d_val': 0, 'Fix_first': False,
              'Zero_pad_d': False, 'Zero_pad_val': None, 'V_thresh': 1.1, 'Const_b': True, 'Const_b_val': 0,
              'Const_w': False, 'Const_w_val': 1, 'Surr_slope': 2, 'Surr_shift': 0, 'Device': DEVICE, 'Dtype': DTYPE,
              'Collective_size': POPULATION_SIZE, 'Loss_fn': LOSS_FN, 'Out_decay': PROBLEM_SPECS['Decay'],
              'Spike_ap': (False, 4), 'Default_delay': 15, 'Roll_delay': True, 'Seed_bias': SEED_BIAS,
              'd_s': PROBLEM_SPECS['d_s']}  # Const_tc: if the time constants are held fixed, Const_tc_val: value for
# the fixed time constants and output time constant, Fix_first: fix the first intput to the delay layer, Zero_pad_d:
# apply zero padding to the delays, zero_pad_val: the size of the zero pad, V_threshold: the threshold voltage value,
# Const_b: fix the bias/ after potential, Const_b_val: the value of the constant bias/after potential,
# Const_w: fix the weights, Const_w_val: value of the constant weights, Surr_slope: the slope of the surrogate
# activation, Surr_shift: the bias of the surrogate activation, Spike_ap: the spike after potential flag and
# its time constant value, Default_delay: the delay added to the spiking plots,
# Roll_delay: which method is used to add delays
NOW_TIME = datetime.now()
NOW_TIME = NOW_TIME.strftime("%d_%m_%H%M%S")
DIR_PATH = [x if x != '\\' else '/' for x in os.getcwd()]
DIR_PATH = ''.join([str(x) for x in DIR_PATH])
LOG_LOCATION = DIR_PATH + '/Logfiles/' + NOW_TIME[:6] + '2024' + '/'
LOG_LOCATION_FIXED = DIR_PATH + '/Logfiles/'
os.makedirs(LOG_LOCATION, exist_ok=True)
os.makedirs(LOG_LOCATION_FIXED, exist_ok=True)
SAVE_FILE_NAME = LOG_LOCATION + NOW_TIME


def save_excel(name):
    """
    Initialize the log file
    Args:
        name(str): the destination of the log file

    Returns:

    """
    training_main = PROBLEM_SPECS | OTHER_PARA | MAPE
    training_para = {'Loss fn': LOSS_FN, 'Loss clip flag': LOSS_CLIP['Flag'], 'Loss clip range': LOSS_CLIP['Range'],
                     'Shape input': SHAPE_INPUT, 'Shape_out': SHAPE_OUT, 'Batch size': BATCH_SIZE, 'Epoch': EPOCH,
                     'Population size': POPULATION_SIZE, 'Num Elites': NUM_ELITES}

    pd_writer_statistics = pd.ExcelWriter(name + '_statistics.xlsx', engine='openpyxl')
    global_data_frame = pd.DataFrame(
        data=list(training_main.keys()) + [''] + list(training_para.keys()) + [''] + list(NET_ARCH.keys()) +
        [''] + list(NET_PARA.keys()) + [''] + list(NET_MUT_PROP.keys())
        + [''] + list(MUT_RATE.keys()) + [''] + list(OTHER_PARA.keys()))
    global_data_frame.to_excel(pd_writer_statistics, sheet_name='Global_variables', startcol=0,
                               index=False)
    global_data_frame = pd.DataFrame(
        data=list(training_main.values()) + [''] + list(training_para.values()) + [''] +
        list(NET_ARCH.values()) + [''] + list(NET_PARA.values()) + [''] + list(NET_MUT_PROP.values())
        + [''] + list(MUT_RATE.values()) + [''] + list(OTHER_PARA.values()))
    global_data_frame.to_excel(pd_writer_statistics, sheet_name='Global_variables', startcol=1,
                               index=False)
    pd_writer_statistics.close()


def get_para_dist(para_in=None, seed_count=1, condition_d2=False, tc_in=np.array([5])):
    """
    Calculate the distributions, ratios and related
    Args:
        para_in(dict): the results of the simulations from varying the logic problem and seed.
        seed_count(int): number of seeds used
        condition_d2(bool): flag for experiment 7
        tc_in(numpy array): the array of fixed time constants used

    Returns:
        dict_out_all(dict): holds all the parameters of the simulation
        dict_out_avg(dict): holds the average ratio for the parameters W and TC
        dict_out_std(dict): holds the std of the ratios of the parameters W and TC
    """
    dict_out_avg, dict_out_std, dict_out_all = {}, {}, {}
    for itc, vtc in enumerate(tc_in):
        count = len(para_in[itc])
        dist_all, ratios_all_w, ratios_all_tc = [], [], []
        for sed in range(seed_count):
            ############################################################################################################
            layers = ('L1', 'L2')
            dict_para_w, dict_para_b, dict_para_tc, dict_para_d = {}, {}, {}, {}
            ############################################################################################################
            for ct in range(count):
                ########################################################################################################
                solution_condition = (1000 in np.array(para_in[itc][ct]['Seed']))
                if solution_condition:
                    continue
                ########################################################################################################
                for cn in range(CUDA_COUNT):
                    dict_para = para_in[itc][ct]['Data'][sed][cn]
                    cma_weights, cma_tc, cma_bias, cma_delays = dict_para['Weights'], dict_para['TC'], dict_para['Bias'], \
                        dict_para['Delays']
                    for h in range(2):
                        if layers[h] not in dict_para_w.keys():
                            dict_para_w[layers[h]] = cma_weights[h]
                            dict_para_b[layers[h]] = cma_bias[h]
                            dict_para_tc[layers[h]] = cma_tc[h]
                            dict_para_d[layers[h]] = cma_delays[h]
                        else:
                            dict_para_w[layers[h]] = np.concatenate((dict_para_w[layers[h]], cma_weights[h]), axis=0)
                            dict_para_b[layers[h]] = np.concatenate((dict_para_b[layers[h]], cma_bias[h]), axis=0)
                            dict_para_tc[layers[h]] = np.concatenate((dict_para_tc[layers[h]], cma_tc[h]), axis=0)
                            dict_para_d[layers[h]] = np.concatenate((dict_para_d[layers[h]], cma_delays[h]), axis=0)
            list_data = [np.concatenate((dict_para_w['L1'].flatten(), dict_para_w['L2'].flatten())),
                         np.concatenate((dict_para_b['L1'].flatten(), dict_para_b['L2'].flatten())),
                         np.concatenate((dict_para_tc['L1'].flatten(), dict_para_tc['L2'].flatten())),
                         np.concatenate((dict_para_d['L1'].flatten(), dict_para_d['L2'].flatten()))]
            if not condition_d2:
                dist_results, ratios = [], []
                for ih in range(len(list_data)):
                    dist = np.histogram(list_data[ih], density=True)
                    dist_results.append(list_data[ih])
                    sum_1 = sum(dist[0][5:])
                    sum_2 = sum(dist[0][:5])
                    if sum_2 > 1e-6:
                        ratio = sum_1/sum_2
                    else:
                        ratio = np.nan
                    ratios.append(ratio)
                print(ratios)
                dist_all.append(dist_results)
                ratios_all_w.append(ratios[0])
                ratios_all_tc.append(ratios[2])
            else:
                data_filt = list_data[3]
                data_filt = data_filt[(data_filt >= -3) & (data_filt <= 3)]
                avg_res = np.average(np.abs(data_filt))
                dist_all.append(avg_res)
        if not condition_d2:
            dict_out_all[vtc] = dist_all
            dict_out_avg[vtc] = [np.average(np.array(ratios_all_w)), np.average(np.array(ratios_all_tc))]
            dict_out_std[vtc] = [np.std(np.array(ratios_all_w)), np.std(np.array(ratios_all_tc))]
        else:
            dist_all = np.array(dist_all)
            dict_out_all[vtc] = dist_all
            dict_out_avg[vtc] = np.average(dist_all)
            dict_out_std[vtc] = np.std(dist_all)
            print('TC: ', vtc, '  Avg: ', np.average(dist_all), '  STD: ', np.std(dist_all))
    return dict_out_all, dict_out_avg, dict_out_std


if __name__ == '__main__':
    sys._excepthook = sys.excepthook
    def exception_hook(exctype, value, traceback):
        print(exctype, value, traceback)
        sys._excepthook(exctype, value, traceback)
        sys.exit(1)
    sys.excepthook = exception_hook

    if MULTIPROCESSING:
        mp.set_start_method('spawn')
    net_info = (OTHER_PARA, NET_ARCH, NET_PARA, NET_MUT_PROP, MUT_RATE, MAPE)
    core_number = list(range(CUDA_COUNT))
    core_count = CUDA_COUNT

    if not VISUALIZE:
        sum_flags = 0
        for i, value in enumerate(EXP_FLAGS.values()):
            sum_flags += value*1
        EXP_TAG = ''
        if sum_flags == 1:
            if EXP_FLAGS['Exp_one']:
                problem_settings = get_exp_one()
                para_flags_all, in_out_encoding = problem_settings['Flags'], problem_settings['In_out_encod']
                hold_res = {}
                SAVE_FILE_NAME_FIXED = LOG_LOCATION_FIXED + 'All_simple'
                EXP_TAG = 'EXP_1'
            elif EXP_FLAGS['Exp_two']:
                problem_settings = get_exp_two()
                para_flags_all, in_out_encoding = problem_settings['Flags'], problem_settings['In_out_encod']
                hold_res = {}
                SAVE_FILE_NAME_FIXED = LOG_LOCATION_FIXED + 'All_spike'
                EXP_TAG = 'EXP_2'
            elif EXP_FLAGS['Exp_three']:
                problem_settings = get_exp_three()
                para_flags_all, in_out_encoding = problem_settings['Flags'], problem_settings['In_out_encod']
                hold_res = {}
                SAVE_FILE_NAME_FIXED = LOG_LOCATION_FIXED + 'Quality_sol'
                EXP_TAG = 'EXP_3'
            elif EXP_FLAGS['Exp_four']:
                problem_settings = get_exp_four()
                para_flags_all, in_out_encoding = problem_settings['Flags'], problem_settings['In_out_encod']
                hold_res = {}
                SAVE_FILE_NAME_FIXED = LOG_LOCATION_FIXED + 'Dist_WTC'
                EXP_TAG = 'EXP_4'
            elif EXP_FLAGS['Exp_five']:
                problem_settings = get_exp_five()
                para_flags_all, in_out_encoding = problem_settings['Flags'], problem_settings['In_out_encod']
                hold_res = {'W08': 0, 'W09': 0, 'W12': 0, 'W13': 0}
                SAVE_FILE_NAME_FIXED = LOG_LOCATION_FIXED + 'Clip_effect'
                EXP_TAG = 'EXP_5'
            elif EXP_FLAGS['Exp_six']:
                problem_settings = get_exp_six()
                para_flags_all, in_out_encoding = problem_settings['Flags'], problem_settings['In_out_encod']
                hold_res = {}
                hold_res_tc = {}
                SAVE_FILE_NAME_FIXED = LOG_LOCATION_FIXED + 'Clip_WTC_effect'
                EXP_TAG = 'EXP_6'
            elif EXP_FLAGS['Exp_seven']:
                problem_settings = get_exp_seven()
                para_flags_all, in_out_encoding = problem_settings['Flags'], problem_settings['In_out_encod']
                hold_res = {}
                SAVE_FILE_NAME_FIXED = LOG_LOCATION_FIXED + 'Delay_TC_effect'
                condition_tc = {'W07': np.array([4, 4.5, 5, 8, 10, 12]), 'W08': np.array([2.5, 3, 4, 5, 6, 8]),
                                'W09': np.array([1.5, 2, 2.5, 4, 6, 8])}
                EXP_TAG = 'EXP_7'
            elif EXP_FLAGS['Exp_eight']:
                problem_settings = get_exp_eight()
                para_flags_all, in_out_encoding = problem_settings['Flags'], problem_settings['In_out_encod']
                hold_res = {}
                SAVE_FILE_NAME_FIXED = LOG_LOCATION_FIXED + 'Delay_shift'
                PROBLEM_SPECS['d_s'] = -1
                OTHER_PARA['d_s'] = PROBLEM_SPECS['d_s']
                PROBLEM_SPECS['Input_pad'] = '0000000000'
                PROBLEM_SPECS['Delay_shift'] = (True, 3)
                MUT_RATE['Range_d'], NET_PARA['Range_d'] = (-10, 10), (-10, 10)

                EXP_TAG = 'EXP_8'
            else:
                print('Choose an EXP')
                SAVE_FILE_NAME_FIXED = ''
                problem_settings = None
                in_out_encoding = None
                para_flags_all = None
                condition_tc = None
                hold_res = None
                save_name = 'None'

            single_result = problem_settings['Single_r']
            w_mut_all, w_mut_tag = problem_settings['W_mut'], problem_settings['W_mut_tag']
            logic_g, seed_values = problem_settings['Probs'], problem_settings['Seed']
            rate_d, rate_tc = problem_settings['DTC_mut']
            for key, value in PROBLEM_SPECS.items():
                if key not in ('Problem_type', 'd_s', 'Input_pad', 'Delay_shift', 'Stability_counter', 'Gaussian',
                               'Robust', 'Ordered_unique'):
                    PROBLEM_SPECS[key] = problem_settings['Problem_specs'][key]
            ############################################################################################################
            MAPE['Gene_time'] = (1800000, problem_settings['Num_gen'][0], problem_settings['Num_gen'][1])
            gene_fixed = (problem_settings['Num_gen'][2], MAPE['Gene_time'][2])
            POPULATION_SIZE, NUM_ELITES = (problem_settings['Train_para']['Pop_size'],
                                           problem_settings['Train_para']['Num_elites'])
            OTHER_PARA['Collective_size'], OTHER_PARA['Out_decay'] = POPULATION_SIZE, PROBLEM_SPECS['Decay']
            OTHER_PARA['d_s'] = PROBLEM_SPECS['d_s']
            NET_ARCH['Max_sizes'] = (SHAPE_INPUT, problem_settings['Train_para']['Hid_size'], SHAPE_OUT)
            const_tc_all = problem_settings['Const_tc']
            ############################################################################################################
            for i, w_mut in enumerate(w_mut_all):
                result_flags = {}
                for j, para_flags in enumerate(para_flags_all):
                    ####################################################################################################
                    MUT_RATE['Range_w'] = w_mut
                    MUT_RATE['Flag_w'], MUT_RATE['Flag_b'], MUT_RATE['Flag_t'], MUT_RATE['Flag_d'] = (
                        para_flags[0], para_flags[1], para_flags[2], para_flags[3])
                    MUT_RATE['Value_t'], MUT_RATE['Value_d'] = rate_tc, rate_d
                    OTHER_PARA['Const_w'], OTHER_PARA['Const_b'], OTHER_PARA['Const_tc'], OTHER_PARA['Const_delay'] = (
                        not para_flags[0], not para_flags[1], not para_flags[2], not para_flags[3])
                    OTHER_PARA['Spike_ap'] = (para_flags[1], OTHER_PARA['Spike_ap'][1])
                    ####################################################################################################
                    if EXP_FLAGS['Exp_three']:
                        condition_w = (para_flags[4] == 'W')
                        condition_wtc = (para_flags[4] == 'WTC')
                    ####################################################################################################
                    result_io = {}
                    for k, io_encod in enumerate(in_out_encoding):
                        ################################################################################################
                        if EXP_FLAGS['Exp_eight']:
                            PROBLEM_SPECS['Delay_shift'] = (True, io_encod[2])
                        ################################################################################################
                        if EXP_FLAGS['Exp_three']:
                            if (io_encod[4] in ('13-13', '37-37', '37-715')) and condition_w:
                                continue
                            if (io_encod[4] == '13-13') and condition_wtc:
                                continue
                        ################################################################################################
                        result_tc = []
                        tc_counter = []
                        for c_tc in const_tc_all:
                            ############################################################################################
                            if EXP_FLAGS['Exp_seven']:
                                if c_tc not in condition_tc[w_mut_tag[i]]:
                                    print('Continued for ', 'Tag:  ', w_mut_tag[i], ' and TC: ', c_tc)
                                    continue
                            ############################################################################################
                            tc_counter.append(c_tc)
                            result_gate = []
                            for m, l_gate in enumerate(logic_g):
                                ########################################################################################
                                PROBLEM_SPECS['Objects'] = (io_encod[0], io_encod[1], '21')
                                PROBLEM_SPECS['Out_simple'] = (io_encod[2], io_encod[3])
                                PROBLEM_SPECS['Logic_type'] = l_gate
                                OTHER_PARA['Const_tc_val'] = c_tc
                                ########################################################################################
                                if EXP_FLAGS['Exp_three']:
                                    if (io_encod[4] == '37-715') and condition_wtc and (l_gate in ('XOR', 'OR', 'NAND')):
                                        continue
                                    if (l_gate in ('NOR', 'NAND')) and condition_w:
                                        continue
                                ########################################################################################
                                if ((EXP_FLAGS['Exp_five']) and (w_mut_tag[i] == 'W08') and
                                    (l_gate == 'NAND' or l_gate == 'NOR')) or ((EXP_FLAGS['Exp_five']) and
                                                                               (w_mut_tag[i] == 'W09') and
                                                                               (l_gate == 'NAND' or l_gate == 'NOR')):
                                    continue
                                ########################################################################################
                                result_seed, result_all = [], []
                                for n, seed in enumerate(seed_values):
                                    OTHER_PARA['Seed_bias'] = seed[0]
                                    net_info = (OTHER_PARA, NET_ARCH, NET_PARA, NET_MUT_PROP, MUT_RATE, MAPE)
                                    with (mp.Manager() as manager):
                                        # Need modify for multiple GPUs
                                        queue_all, queue_gene = [], []
                                        for _ in range(CUDA_COUNT):
                                            queue_all.append(manager.Queue(maxsize=CUDA_COUNT - 1))
                                            queue_gene.append(manager.Queue(maxsize=CUDA_COUNT - 1))
                                        queue_all, queue_gene = tuple(queue_all), tuple(queue_gene)
                                        worker_pool = mp.Pool(core_count)
                                        worker_result = worker_pool.map(partial(ea_map,
                                                                                population_size=POPULATION_SIZE,
                                                                                mut_rate=MUT_RATE,
                                                                                batch_size=BATCH_SIZE,
                                                                                num_elites=NUM_ELITES,
                                                                                device=DEVICE,
                                                                                dtype=DTYPE,
                                                                                net_info=net_info, mape=MAPE,
                                                                                loss_clip=LOSS_CLIP,
                                                                                problem_specs=PROBLEM_SPECS,
                                                                                queue_com=queue_all,
                                                                                queue_gen=queue_gene,
                                                                                cross_com=CROSS_WORKER_COM,
                                                                                gene_fixed=gene_fixed,
                                                                                exp_flags=EXP_FLAGS),
                                                                        core_number)
                                        worker_pool.close()
                                        worker_pool.join()
                                        result_seed.append(worker_result[0]['Sol_gene'])
                                        result_all.append(worker_result)
                                        save_name = (SAVE_FILE_NAME + '_' + l_gate + '_' + w_mut_tag[i] + '_' +
                                                     para_flags[4] + '_' + io_encod[4] + '_' + seed[1] + '_' +
                                                     str(c_tc) + '_' + str(worker_result[0]['Sol_gene']))
                                        print(save_name)
                                        if single_result:
                                            pickle.dump(worker_result, open(save_name + ".p", "wb"))
                                            save_excel(save_name)
                                result_gate.append({'Seed': result_seed, 'Data': result_all})
                            result_tc.append(result_gate)
                        if (EXP_FLAGS['Exp_three'] or EXP_FLAGS['Exp_four'] or EXP_FLAGS['Exp_five'] or
                                EXP_FLAGS['Exp_seven'] or EXP_FLAGS['Exp_eight']):
                            if result_tc:
                                condition_d = False
                                if EXP_FLAGS['Exp_seven']:
                                    condition_d = True
                                result_dist, result_avg, result_std = get_para_dist(para_in=result_tc,
                                                                                    seed_count=len(seed_values),
                                                                                    condition_d2=condition_d,
                                                                                    tc_in=np.array(tc_counter))
                                result_io[io_encod[4]] = {'Dist': result_dist, 'Avg': result_avg, 'STD': result_std}
                        else:
                            result_io[io_encod[4]] = result_tc
                    result_flags[para_flags[4]] = result_io
                ########################################################################################################
                save_name = SAVE_FILE_NAME
                hold_res[w_mut_tag[i]] = result_flags
            # print(hold_res)
            pickle.dump(hold_res, open(SAVE_FILE_NAME_FIXED + '_' + 'FINAL_RESULT_' + EXP_TAG + ".p", "wb"))
        elif EXP_SINGLE:
            with (mp.Manager() as manager):
                # Need modify for multiple GPUs
                queue_all, queue_gene = [], []
                for _ in range(CUDA_COUNT):
                    queue_all.append(manager.Queue(maxsize=CUDA_COUNT - 1))
                    queue_gene.append(manager.Queue(maxsize=CUDA_COUNT - 1))
                queue_all, queue_gene = tuple(queue_all), tuple(queue_gene)
                worker_pool = mp.Pool(core_count)
                worker_result = worker_pool.map(partial(ea_map,
                                                        population_size=POPULATION_SIZE,
                                                        mut_rate=MUT_RATE,
                                                        batch_size=BATCH_SIZE,
                                                        num_elites=NUM_ELITES,
                                                        device=DEVICE,
                                                        dtype=DTYPE,
                                                        net_info=net_info,
                                                        mape=MAPE, loss_clip=LOSS_CLIP,
                                                        problem_specs=PROBLEM_SPECS,
                                                        queue_com=queue_all,
                                                        queue_gen=queue_gene,
                                                        cross_com=CROSS_WORKER_COM,
                                                        exp_flags=EXP_FLAGS),
                                                core_number)
                worker_pool.close()
                worker_pool.join()
                pickle.dump(worker_result, open(SAVE_FILE_NAME + ".p", "wb"))
                save_excel(SAVE_FILE_NAME)
        else:
            print('Please choose 1 experiment by setting 1 flag in  EXP_FLAGS')
        if not VISUALIZE:
            if EXP_FLAGS['Exp_one']:
                visualize_exp_one()
            elif EXP_FLAGS['Exp_two']:
                visualize_exp_two()
            elif EXP_FLAGS['Exp_three']:
                visualize_exp_three()
            elif EXP_FLAGS['Exp_four']:
                visualize_exp_four()
            elif EXP_FLAGS['Exp_five']:
                visualize_exp_five()
            elif EXP_FLAGS['Exp_six']:
                visualize_exp_six()
            elif EXP_FLAGS['Exp_seven']:
                visualize_exp_seven()
            elif EXP_FLAGS['Exp_eight']:
                visualize_exp_eight()
    else:
        if EXP_FLAGS['Exp_one']:
            visualize_exp_one()
        elif EXP_FLAGS['Exp_two']:
            visualize_exp_two()
        elif EXP_FLAGS['Exp_three']:
            visualize_exp_three()
        elif EXP_FLAGS['Exp_four']:
            visualize_exp_four()
        elif EXP_FLAGS['Exp_five']:
            visualize_exp_five()
        elif EXP_FLAGS['Exp_six']:
            visualize_exp_six()
        elif EXP_FLAGS['Exp_seven']:
            visualize_exp_seven()
        elif EXP_FLAGS['Exp_eight']:
            visualize_exp_eight()






