"""This is simulations settings module for the user to try and change"""

"""Define imports"""
import numpy as np

# Choice of experiment by setting one flag True and other False
EXP_FLAGS = {'Exp_one': False, 'Exp_two': True, 'Exp_three': False, 'Exp_four': False, 'Exp_five': False,
             'Exp_six': False, 'Exp_seven': False, 'Exp_eight': False}


# See the first experiment for the definition of the parameters
def get_exp_one():
    """
    Defines the settings for Fig.2 A&B
    Returns:

    """
    # Seeds
    seed_values = [(0, 'T1'), (4, 'T2'), (8, 'T3'), (12, 'T4'), (16, 'T5')]
    # Input and output encodings  (input 1, input 2, output 1, output 2, tag)
    in_out_encoding = [('01', '03', 0, 1, '13-01'), ('01', '03', 1, 2, '13-13'), ('03', '07', 1, 2, '37-13'),
                       ('03', '07', 2, 3, '37-37'), ('03', '07', 3, 4, '37-715')]
    # Which parameters are mutable (W, B, TC, D, tag)
    para_flags = [(True, False, False, False, 'W'), (True, False, True, False, 'WTC'), (False, False, True, True, 'DTC'),
                  (True, False, False, True, 'WD'), (True, False, True, True, 'WDTC')]
    # Which problems are used in the simulations
    logic_g = ['XOR', 'XNOR', 'OR', 'NOR', 'AND', 'NAND']
    # The weight clipping range and its tag
    w_mt, w_mut_tag = [(-1, 1), (-2, 2)], ['W11', 'W22']
    # Number of allowed generation (maximum, condition, fixed condition or not)
    num_generation = (13, 10, False)
    # Constant time constant used for output and when time constant is fixed
    # Single result is a flag for whether the simulation saves all solutions
    const_tc, single_result = np.array([5]), False
    # Mutation rate of delays and time constants (delays, time constants)
    dtc_mut = (1, 2)
    # Population size per GPU, number of elites per GPU and hidden size
    training_para = {'Pop_size': 460000, 'Num_elites': 4000, 'Hid_size': 4}
    # Outspikes: whether the output is a spike train or not, Encoding_length: is the size of the decimal to
    # binary conversion, Problem_pad is the size of the padding for each string chunk, Digits: the size of the converted
    # digits from decimal to binary, Decay: whether time constant decay is applied to the output or not
    # Change_objects: is for testing different problems, Objects: input encodings, Logic_type: the type of problem
    # being solved, Out_simple: the output encoding
    problem_specs = {'Out_spikes': False, 'Encoding_length': 5, 'Problem_pad': 0, 'Digits': 2, 'Decay': False,
                     'Change_objects': True, 'Objects': ('01', '03', '21'), 'Logic_type': 'XOR', 'Out_simple': (0, 1)}
    exp_one = {'Seed': seed_values, 'In_out_encod': in_out_encoding, 'Flags': para_flags, 'Probs': logic_g,
               'W_mut': w_mt, 'W_mut_tag': w_mut_tag, 'Num_gen': num_generation, 'Train_para': training_para,
               'Problem_specs': problem_specs, 'DTC_mut': dtc_mut, 'Const_tc': const_tc, 'Single_r': single_result}
    return exp_one


def get_exp_two():
    """
    Defines the settings for Fig.6 A&B
    Returns:

    """
    seed_values = [(0, 'T1'), (4, 'T2'), (8, 'T3'), (12, 'T4'), (16, 'T5')]
    in_out_encoding = [('01', '03', '00', '01', '13-01'), ('01', '03', '01', '03', '13-13'),
                       ('03', '07', '01', '03', '37-13'), ('03', '07', '03', '07', '37-37')]
    para_flags = [(True, False, True, False, 'WTC'), (False, False, True, True, 'DTC'), (True, False, False, True, 'WD'),
                  (True, False, True, True, 'WDTC'), (True, True, True, True, 'WBDTC')]
    logic_g = ['XOR', 'XNOR', 'OR', 'NOR', 'AND', 'NAND']
    w_mt, w_mut_tag = [(-1, 1)], ['W11']
    num_generation = (202, 5, False)
    const_tc, single_result = np.array([5]), False
    dtc_mut = (1, 2)
    training_para = {'Pop_size': 270000, 'Num_elites': 3000, 'Hid_size': 6}
    problem_specs = {'Out_spikes': True, 'Encoding_length': 5, 'Problem_pad': 0, 'Digits': 2, 'Decay': True,
                     'Change_objects': True, 'Objects': ('01', '03', '21'),
                     'Logic_type': 'XOR', 'Out_simple': ('00', '01')}
    exp_two = {'Seed': seed_values, 'In_out_encod': in_out_encoding, 'Flags': para_flags, 'Probs': logic_g,
               'W_mut': w_mt, 'W_mut_tag': w_mut_tag, 'Num_gen': num_generation, 'Train_para': training_para,
               'Problem_specs': problem_specs, 'DTC_mut': dtc_mut, 'Const_tc': const_tc, 'Single_r': single_result}
    return exp_two


def get_exp_three():
    """
    Defines the settings for Fig.3 A
    Returns:

    """
    seed_values = [(0, 'T1'), (4, 'T2'), (8, 'T3'), (12, 'T4'), (16, 'T5')]
    in_out_encoding = [('01', '03', 0, 1, '13-01'), ('01', '03', 1, 2, '13-13'), ('03', '07', 1, 2, '37-13'),
                       ('03', '07', 2, 3, '37-37'), ('03', '07', 3, 4, '37-715')]
    para_flags = [(True, False, False, False, 'W'), (True, False, True, False, 'WTC'),
                  (False, False, True, True, 'DTC'),
                  (True, False, False, True, 'WD'), (True, False, True, True, 'WDTC')]
    logic_g = ['XOR', 'XNOR', 'OR', 'NOR', 'AND', 'NAND']
    w_mt, w_mut_tag = [(-1, 1)], ['W11']
    num_generation = (13, 10, False)
    const_tc, single_result = np.array([2]), False
    dtc_mut = (3, 4)
    training_para = {'Pop_size': 280000, 'Num_elites': 2400, 'Hid_size': 6}
    problem_specs = {'Out_spikes': False, 'Encoding_length': 5, 'Problem_pad': 0, 'Digits': 2, 'Decay': False,
                     'Change_objects': True, 'Objects': ('01', '03', '21'), 'Logic_type': 'XOR', 'Out_simple': (0, 1)}
    exp_three = {'Seed': seed_values, 'In_out_encod': in_out_encoding, 'Flags': para_flags, 'Probs': logic_g,
               'W_mut': w_mt, 'W_mut_tag': w_mut_tag, 'Num_gen': num_generation, 'Train_para': training_para,
                 'Problem_specs': problem_specs, 'DTC_mut': dtc_mut, 'Const_tc': const_tc, 'Single_r': single_result}
    return exp_three


def get_exp_four():
    """
    Defines the settings for Fig.3 D
    Returns:

    """
    seed_values = [(0, 'T1'), (4, 'T2'), (8, 'T3'), (12, 'T4'), (16, 'T5')]
    in_out_encoding = [('03', '07', 0, 1, '37-01'), ('03', '07', 1, 2, '37-13'), ('03', '07', 2, 3, '37-37'),
                       ('03', '07', 3, 4, '37-715')]
    para_flags = [(False, False, True, True, 'DTC'), (True, False, False, True, 'WD')]
    logic_g = ['XOR', 'XNOR', 'OR', 'NOR', 'AND', 'NAND']
    w_mt, w_mut_tag = [(-1, 1)], ['W11']
    num_generation = (13, 10, False)
    dtc_mut = (3, 4)
    const_tc, single_result = np.array([2]), False
    training_para = {'Pop_size': 256000, 'Num_elites': 4000, 'Hid_size': 6}
    problem_specs = {'Out_spikes': False, 'Encoding_length': 5, 'Problem_pad': 0, 'Digits': 2, 'Decay': False,
                     'Change_objects': True, 'Objects': ('01', '03', '21'), 'Logic_type': 'XOR', 'Out_simple': (0, 1)}
    exp_four = {'Seed': seed_values, 'In_out_encod': in_out_encoding, 'Flags': para_flags, 'Probs': logic_g,
               'W_mut': w_mt, 'W_mut_tag': w_mut_tag, 'Num_gen': num_generation, 'Train_para': training_para,
                'Problem_specs': problem_specs, 'DTC_mut': dtc_mut, 'Const_tc': const_tc, 'Single_r': single_result}
    return exp_four


def get_exp_five():
    """
    Defines the settings for Fig.3 B
    Returns:

    """
    seed_values = [(0, 'T1'), (4, 'T2'), (8, 'T3'), (12, 'T4'), (16, 'T5')]
    in_out_encoding = [('01', '03', 1, 2, '13-13')]
    para_flags = [(True, False, False, True, 'WD')]
    logic_g = ['XOR', 'XNOR', 'OR', 'NOR', 'AND', 'NAND']
    w_mt, w_mut_tag = [(-0.9, 0.9), (-0.8, 0.8), (-1.2, 1.2), (-1.3, 1.3)], ['W09', 'W08', 'W12', 'W13']
    num_generation = (15, 12,  False)
    dtc_mut = (3, 4)
    const_tc, single_result = np.array([2]), False
    training_para = {'Pop_size': 380000, 'Num_elites': 4000, 'Hid_size': 4}
    problem_specs = {'Out_spikes': False, 'Encoding_length': 5, 'Problem_pad': 0, 'Digits': 2, 'Decay': False,
                     'Change_objects': True, 'Objects': ('01', '03', '21'), 'Logic_type': 'XOR', 'Out_simple': (0, 1)}
    exp_five = {'Seed': seed_values, 'In_out_encod': in_out_encoding, 'Flags': para_flags, 'Probs': logic_g,
               'W_mut': w_mt, 'W_mut_tag': w_mut_tag, 'Num_gen': num_generation, 'Train_para': training_para,
                'Problem_specs': problem_specs, 'DTC_mut': dtc_mut, 'Const_tc': const_tc, 'Single_r': single_result}
    return exp_five


def get_exp_six():
    """
    Defines the settings for Fig.4 A
    Returns:

    """
    seed_values = [(0, 'T1'), (4, 'T2'), (8, 'T3'), (12, 'T4'), (16, 'T5')]
    in_out_encoding = [('01', '05', 1, 2, '15-13'), ('01', '09', 1, 2, '19-13'), ('01', '17', 1, 2, '117-13')]
    para_flags = [(True, False, False, True, 'WD')]
    logic_g = ['XOR']
    w_mt, w_mut_tag = [(-0.6, 0.6), (-0.7, 0.7), (-0.8, 0.8), (-0.9, 0.9)], ['W06', 'W07', 'W08', 'W09']
    num_generation = (43, 40, False)
    dtc_mut = (2, 3)  # Delays, then TC
    const_tc, single_result = np.arange(1, 8, 1), False
    training_para = {'Pop_size': 160000, 'Num_elites': 2000, 'Hid_size': 4}
    problem_specs = {'Out_spikes': False, 'Encoding_length': 5, 'Problem_pad': 0, 'Digits': 2, 'Decay': False,
                     'Change_objects': True, 'Objects': ('01', '05', '07'), 'Logic_type': 'XOR', 'Out_simple': (0, 1)}
    exp_six = {'Seed': seed_values, 'In_out_encod': in_out_encoding, 'Flags': para_flags, 'Probs': logic_g,
               'W_mut': w_mt, 'W_mut_tag': w_mut_tag, 'Num_gen': num_generation, 'Train_para': training_para,
               'Problem_specs': problem_specs, 'DTC_mut': dtc_mut, 'Const_tc': const_tc, 'Single_r': single_result}
    return exp_six


def get_exp_seven():
    """
    Defines the settings for Fig.4 B
    Returns:

    """
    seed_values = [(0, 'T1'), (4, 'T2'), (8, 'T3'), (12, 'T4'), (16, 'T5')]
    in_out_encoding = [('01', '05', 1, 2, '15-13')]
    para_flags = [(True, False, False, True, 'WD')]
    logic_g = ['XOR', 'XNOR', 'OR', 'NOR', 'AND', 'NAND']
    w_mt, w_mut_tag = [(-0.7, 0.7), (-0.8, 0.8), (-0.9, 0.9)], ['W07', 'W08', 'W09']
    num_generation = (23, 20, True)
    dtc_mut = (2, 3)  # Delays, then TC
    const_tc, single_result = np.arange(1.5, 12.5, 0.5), False
    training_para = {'Pop_size': 160000, 'Num_elites': 2000, 'Hid_size': 4}
    problem_specs = {'Out_spikes': False, 'Encoding_length': 5, 'Problem_pad': 0, 'Digits': 2, 'Decay': False,
                     'Change_objects': True, 'Objects': ('01', '05', '07'), 'Logic_type': 'XOR', 'Out_simple': (0, 1)}
    exp_seven = {'Seed': seed_values, 'In_out_encod': in_out_encoding, 'Flags': para_flags, 'Probs': logic_g,
               'W_mut': w_mt, 'W_mut_tag': w_mut_tag, 'Num_gen': num_generation, 'Train_para': training_para,
               'Problem_specs': problem_specs, 'DTC_mut': dtc_mut, 'Const_tc': const_tc, 'Single_r': single_result}
    return exp_seven


def get_exp_eight():
    """
    Defines the settings for Fig.6 C
    Returns:

    """
    seed_values = [(0, 'T1'), (4, 'T2'), (8, 'T3'), (12, 'T4'), (16, 'T5')]
    in_out_encoding = [('01', '03', 5, 7, '13-57'), ('01', '03', 7, 9, '13-79'), ('01', '03', 9, 11, '13-911'),
                       ('01', '03', 11, 13, '13-1113'), ('01', '03', 13, 15, '13-1315'),
                       ('01', '03', 15, 17, '15-1517')]
    para_flags = [(True, False, True, True, 'WDTC')]
    logic_g = ['XOR', 'XNOR', 'OR', 'NOR', 'AND', 'NAND']
    w_mt, w_mut_tag = [(-1, 1)], ['W11']
    num_generation = (43, 20, False)
    dtc_mut = (4, 4)  # Delays, then TC
    const_tc, single_result = np.array([5]), False
    training_para = {'Pop_size': 128000, 'Num_elites': 2000, 'Hid_size': 4}
    problem_specs = {'Out_spikes': True, 'Encoding_length': 5, 'Problem_pad': 0, 'Digits': 2, 'Decay': True,
                     'Change_objects': True, 'Objects': ('01', '05', '07'), 'Logic_type': 'XOR', 'Out_simple': (0, 1)}
    exp_eight = {'Seed': seed_values, 'In_out_encod': in_out_encoding, 'Flags': para_flags, 'Probs': logic_g,
               'W_mut': w_mt, 'W_mut_tag': w_mut_tag, 'Num_gen': num_generation, 'Train_para': training_para,
               'Problem_specs': problem_specs, 'DTC_mut': dtc_mut, 'Const_tc': const_tc, 'Single_r': single_result}
    return exp_eight

