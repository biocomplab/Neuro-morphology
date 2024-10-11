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
from evolutionary_algorithms import ea_map
from visualization_and_statistics import weight_noise, gaussian_noise
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.autograd.set_detect_anomaly(True)

EXP_CHOICE = {'Weight_noise': False, 'Gaussian': True}  # Choice of experiment, Weight_noise is Fig.5b, Gaussian is Fig. 5a
VISUALIZE = True  # Visualize directly
if EXP_CHOICE['Weight_noise']:
    out_spikes_a = False
    decay_a = False
    ordered_a = True
    epoch_a = 1
    pop_a = 1800
    elites_a = 100
    gene_a = (18000, 1200, 1000)
elif EXP_CHOICE['Gaussian']:
    out_spikes_a = True
    decay_a = True
    ordered_a = False
    epoch_a = 32
    pop_a = 180000
    elites_a = 3000
    gene_a = (18000, 105, 100)
elif EXP_CHOICE['Weight_noise'] and EXP_CHOICE['Gaussian']:
    'Choose one experiment'
    assert False
else:
    print('Please choose an experiment')
    assert False

GAUSSIAN_STD_RANGE = (0.1, 0.2, 0.4, 0.6, 0.8, 1)  # For the gaussian noise experiment
NOISE_CHANCE_RANGE = (0.01, 0.02, 0.04, 0.06, 0.08, 0.1)  # For the gaussian noise experiment
NOISE_STD_RANGE = np.arange(0, 1.1, 0.1)  # For the para noise experiment

MULTIPROCESSING, CORE_COUNT, CUDA_COUNT = True, mp.cpu_count(), torch.cuda.device_count()  # Use multiprocessing,
# number of CPU cores, number of CUDA cores
CROSS_WORKER_COM, SEED_BIAS = True, 4  # If GPU share elites, single seed bias
PROBLEM_SPECS = {'Problem_type': FUNCTIONS['Delay_ms'], 'Out_spikes': out_spikes_a, 'Encoding_length': 10, 'Problem_pad': 0,
                 'Digits': 4, 'Decay': decay_a, 'Change_objects': True, 'Objects': ('0120', '0248', '0007'),
                 'Ordered_unique': ordered_a, 'Stability_counter': 20,
                 'Gaussian': (EXP_CHOICE['Gaussian'], 0.2, True, 0.01), 'Robust': (100, 0, {'W': True, 'TC': False,
                                                                                            'D': False})}
# Input problem used, Out_spikes: if target is spike train,
# Encoding_length: the size of the decimal to binary encoding, Problem_pad: the padding of the decimal to binary
# encoding, Digits: the size of the decimal chunk, Decay: if decay is applied to the output, Change_objects: whether to
# change the problem type during testing, Objects: the intput decimal encoding, d_s a decorative parameter for the
# delay sign, Logic_type, the problem being investigated, Out_simple: the output encoding, Input_pad, the size of
# the pad to the input, Delay_shift: whether to shift the output and the count of the shifts
# Gaussian: flag, STD, Noise flag, Noise prop per time step
MAPE = {'Flag': False, 'Size': 20, 'Scale': (10, 10), 'Decimals': 2, 'Keep_e': True, 'Grid_score': False,
        'Double_sort': False, 'Gene_time': gene_a, 'Choice_corr': ('W', 'Dist_ini', 'B', 'Dist_ini'),
        'Alt_dim': (True, 'Spike Count', 'Ratio EX')}  # Not used except for the Gene_time: which holds the maximum
# and target number of generations
LOSS_FN, LOSS_CLIP = 'MSE', {'Flag': True, 'Range': (0, 50)}  # Loss function and  the loss clip range
SHAPE_INPUT, SHAPE_OUT = 2, 1  # Shape of input and output
BATCH_SIZE = 4  # Batch size
EPOCH = epoch_a  # Number of epochs
POPULATION_SIZE = pop_a  # Must be divisible by NUM_ELITES
NUM_ELITES = elites_a   # Number of elites

DEVICE, DTYPE = torch.device("cuda" if torch.cuda.is_available() else "cpu"), torch.float16  # Device used and data type
NET_ARCH = {'Max_sizes': (SHAPE_INPUT, 4, SHAPE_OUT), 'Rand_num': False, 'Rand_pos': 'none',
            'Range_x': (-3.5, 3.5, 0.5), 'Type_con': 'uniform'}  # For geometric networks, not used except for
# Max_sizes, which specify the size of the network
NET_MUT_PROP = {'Prop_con_type': 'Const', 'Prop_con': 1, 'Prop_w': 0.2}  # In the case of evolving the connections
MUT_RATE = {'Flag_w': True, 'Value_w': 0.25, 'Range_w': (-1, 1), 'Flag_b': False, 'Value_b': 2,
            'Range_b': (-10, 0), 'Flag_t': False, 'Value_t': 3, 'Range_t': (0.1, 10), 'Flag_d': True,
            'Value_d': 2, 'Range_d': (-5, 5)}  # Flags, mutations rates and clipping ranges for all the parameters
NET_PARA = {'Range_w': MUT_RATE['Range_w'], 'Grad_w': False, 'Range_b': MUT_RATE['Range_b'], 'Grad_b': False,
            'Range_d': MUT_RATE['Range_d'], 'Grad_d': False, 'Range_t': MUT_RATE['Range_t'], 'Grad_t': False,
            'Synapse_t': True, 'Synapse_b': True}  # Adds the possibility to differentiate the parameters
OTHER_PARA = {'Const_tc': True, 'Const_tc_val': 6, 'Const_delay': False, 'Const_d_val': 0, 'Fix_first': False,
              'Zero_pad_d': False, 'Zero_pad_val': None, 'V_thresh': 1.1, 'Const_b': True, 'Const_b_val': 0,
              'Const_w': False, 'Const_w_val': 1, 'Surr_slope': 2, 'Surr_shift': 0, 'Device': DEVICE, 'Dtype': DTYPE,
              'Collective_size': POPULATION_SIZE, 'Loss_fn': LOSS_FN, 'Out_decay': PROBLEM_SPECS['Decay'],
              'Spike_ap': (False, 2), 'Default_delay': 15, 'Roll_delay': True, 'Seed_bias': SEED_BIAS}
# Const_tc: if the time constants are held fixed, Const_tc_val: value for
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
LOG_LOCATION = DIR_PATH + '/Logfiles/'
os.makedirs(LOG_LOCATION, exist_ok=True)


""" Initialize log file """
TRAINING_MAIN = PROBLEM_SPECS
TRAINING_PARA = {'Loss fn': LOSS_FN, 'Loss clip flag': LOSS_CLIP['Flag'], 'Loss clip range': LOSS_CLIP['Range'],
                 'Shape input': SHAPE_INPUT, 'Shape_out': SHAPE_OUT, 'Batch size': BATCH_SIZE, 'Epoch': EPOCH,
                 'Population size': POPULATION_SIZE, 'Num Elites': NUM_ELITES}


def save_excel(name=''):
    """
    Initialize the log file
    Args:
        name(str): the destination of the log file

    Returns:

    """
    pd_writer_statistics = pd.ExcelWriter(name + '_statistics.xlsx',
                                          engine='openpyxl')
    global_data_frame = pd.DataFrame(
        data=list(TRAINING_MAIN.keys()) + [''] + list(TRAINING_PARA.keys()) + [''] + list(NET_ARCH.keys()) +
        [''] + list(NET_PARA.keys()) + [''] + list(NET_MUT_PROP.keys())
        + [''] + list(MUT_RATE.keys()) + [''] + list(OTHER_PARA.keys()))
    global_data_frame.to_excel(pd_writer_statistics, sheet_name='Global_variables', startcol=0,
                               index=False)
    global_data_frame = pd.DataFrame(
        data=list(TRAINING_MAIN.values()) + [''] + list(TRAINING_PARA.values()) + [''] +
        list(NET_ARCH.values()) + [''] + list(NET_PARA.values()) + [''] + list(NET_MUT_PROP.values())
        + [''] + list(MUT_RATE.values()) + [''] + list(OTHER_PARA.values()))
    global_data_frame.to_excel(pd_writer_statistics, sheet_name='Global_variables', startcol=1,
                               index=False)
    pd_writer_statistics.close()


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
        if PROBLEM_SPECS['Gaussian'][0]:
            worker_result_mut = {}
            list_mut = ['WD', 'WTC', 'DTC', 'WDTC']
            for para_m in list_mut:
                if para_m == 'WD':
                    MUT_RATE['Flag_w'] = True
                    MUT_RATE['Flag_t'] = False
                    MUT_RATE['Flag_d'] = True
                    OTHER_PARA['Const_w'] = not MUT_RATE['Flag_w']
                    OTHER_PARA['Const_tc'] = not MUT_RATE['Flag_t']
                    OTHER_PARA['Const_delay'] = not MUT_RATE['Flag_d']
                elif para_m == 'WTC':
                    MUT_RATE['Flag_w'] = True
                    MUT_RATE['Flag_t'] = True
                    MUT_RATE['Flag_d'] = False
                    OTHER_PARA['Const_w'] = not MUT_RATE['Flag_w']
                    OTHER_PARA['Const_tc'] = not MUT_RATE['Flag_t']
                    OTHER_PARA['Const_delay'] = not MUT_RATE['Flag_d']
                elif para_m == 'DTC':
                    MUT_RATE['Flag_w'] = False
                    MUT_RATE['Flag_t'] = True
                    MUT_RATE['Flag_d'] = True
                    OTHER_PARA['Const_w'] = not MUT_RATE['Flag_w']
                    OTHER_PARA['Const_tc'] = not MUT_RATE['Flag_t']
                    OTHER_PARA['Const_delay'] = not MUT_RATE['Flag_d']
                elif para_m == 'WDTC':
                    MUT_RATE['Flag_w'] = True
                    MUT_RATE['Flag_t'] = True
                    MUT_RATE['Flag_d'] = True
                    OTHER_PARA['Const_w'] = not MUT_RATE['Flag_w']
                    OTHER_PARA['Const_tc'] = not MUT_RATE['Flag_t']
                    OTHER_PARA['Const_delay'] = not MUT_RATE['Flag_d']
                worker_result_all = []
                for i in range(len(GAUSSIAN_STD_RANGE)):
                    for j in range(len(NOISE_CHANCE_RANGE)):
                        ps = PROBLEM_SPECS['Gaussian']
                        PROBLEM_SPECS['Gaussian'] = (ps[0], GAUSSIAN_STD_RANGE[i], ps[2], NOISE_CHANCE_RANGE[j])
                        net_info = (OTHER_PARA, NET_ARCH, NET_PARA, NET_MUT_PROP, MUT_RATE, MAPE)
                        with mp.Manager() as manager:
                            queue_all = []
                            for _ in range(CUDA_COUNT):
                                queue_all.append(manager.Queue(maxsize=CUDA_COUNT-1))
                            queue_all = tuple(queue_all)
                            worker_pool = mp.Pool(core_count)
                            worker_result = worker_pool.map(partial(ea_map, population_size=POPULATION_SIZE,
                                                                    mut_rate=MUT_RATE, batch_size=BATCH_SIZE,
                                                                    num_elites=NUM_ELITES, device=DEVICE,
                                                                    dtype=DTYPE,
                                                                    net_info=net_info,
                                                                    mape=MAPE, loss_clip=LOSS_CLIP,
                                                                    problem_specs=PROBLEM_SPECS,
                                                                    queue_com=queue_all,
                                                                    cross_com=CROSS_WORKER_COM, epoch=EPOCH),
                                                            core_number)
                            worker_pool.close()
                            worker_pool.join()
                            worker_result_all.append({'Parameters': para_m, 'Gaussian STD': GAUSSIAN_STD_RANGE[i],
                                                      'Noise chance': NOISE_CHANCE_RANGE[j], 'Results': worker_result})
                worker_result_mut[para_m] = worker_result_all
            SAVE_FILE_NAME = LOG_LOCATION + 'Gaussian_Final'
            pickle.dump(worker_result_mut, open(SAVE_FILE_NAME+".p", "wb"))
            save_excel()
            gaussian_noise()
        else:
            list_mut = ['W', 'WTC', 'WD', 'WDTC']
            worker_result_all = []
            for para_m in list_mut:
                if para_m == 'W':
                    MUT_RATE['Flag_t'] = False
                    MUT_RATE['Flag_d'] = False
                    OTHER_PARA['Const_tc'] = True
                    OTHER_PARA['Const_delay'] = True
                elif para_m == 'WTC':
                    MUT_RATE['Flag_t'] = True
                    MUT_RATE['Flag_d'] = False
                    OTHER_PARA['Const_tc'] = False
                    OTHER_PARA['Const_delay'] = True
                elif para_m == 'WD':
                    MUT_RATE['Flag_t'] = False
                    MUT_RATE['Flag_d'] = True
                    OTHER_PARA['Const_tc'] = True
                    OTHER_PARA['Const_delay'] = False
                elif para_m == 'WDTC':
                    MUT_RATE['Flag_t'] = True
                    MUT_RATE['Flag_d'] = True
                    OTHER_PARA['Const_tc'] = False
                    OTHER_PARA['Const_delay'] = False
                net_info = (OTHER_PARA, NET_ARCH, NET_PARA, NET_MUT_PROP, MUT_RATE, MAPE)
                for std in NOISE_STD_RANGE:
                    PROBLEM_SPECS['Robust'] = (PROBLEM_SPECS['Robust'][0],
                                               std, {'W': PROBLEM_SPECS['Robust'][2]['W'], 'TC':
                        PROBLEM_SPECS['Robust'][2]['TC'], 'D': PROBLEM_SPECS['Robust'][2]['D']})
                    with mp.Manager() as manager:
                        # Need modify for multiple GPUs
                        queue_all = []
                        for _ in range(CUDA_COUNT):
                            queue_all.append(manager.Queue(maxsize=CUDA_COUNT - 1))
                        queue_all = tuple(queue_all)
                        worker_pool = mp.Pool(core_count)
                        worker_result = worker_pool.map(partial(ea_map, population_size=POPULATION_SIZE,
                                                                mut_rate=MUT_RATE, batch_size=BATCH_SIZE,
                                                                num_elites=NUM_ELITES, device=DEVICE,
                                                                dtype=DTYPE,
                                                                net_info=net_info,
                                                                mape=MAPE, loss_clip=LOSS_CLIP,
                                                                problem_specs=PROBLEM_SPECS,
                                                                queue_com=queue_all,
                                                                cross_com=CROSS_WORKER_COM, epoch=EPOCH),
                                                        core_number)
                        worker_pool.close()
                        worker_pool.join()
                        worker_result_all.append({'Para': para_m, 'STD': std, 'Results': worker_result})
            SAVE_FILE_NAME = LOG_LOCATION + 'Weight_noise_Final'
            pickle.dump(worker_result_all, open(SAVE_FILE_NAME + ".p", "wb"))
            save_excel()
            weight_noise()
    else:
        if EXP_CHOICE['Weight_noise']:
            weight_noise()
        elif EXP_CHOICE['Gaussian']:
            gaussian_noise()





