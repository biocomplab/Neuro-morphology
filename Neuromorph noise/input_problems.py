"""This is the input problems module, which specifies two main functions: 1) the delay_ms function which provides the
input output combination of all the logic problems, 2) the encode_inputs_bursts_3, which specify the encoding of the
input/output"""

"""Define imports"""
import numpy as np
from scipy.stats import norm

ENCODING = 'Burst'  # For now, 'Burst' or time to first spike 'TFS' for future implementations
OUT1 = '0'  # An object holding the encoding of the zero state/ logic zero
OUT2 = '0'  # An object holding the encoding of the one state/ logic one


# noinspection PyIncorrectDocstring
def delay_ms(batch_size=4, ordered_unique=False, target_spikes=False, encoding_length=10, pad=0, digits=4,
             decay=False, tau=5, os=('0032', '0048', '0005'), change_prop=(False, 'XOR'), gaussian=(False, 2, 0.1, 10),
             seeder=None):
    """
    The main input function specifying all input/output combination of the logic problems
    Args:
        batch_size(int): the size of the input batch
        ordered_unique(bool): whether to use the batch size, or the default batch of 4 (all input/output combinations)
        target_spikes(bool): specifies whether the target is spiking timing or number of spikes
        encoding_length(int): the size of the binary encoding
        pad(int): the pad of the binary encoding
        digits(int): the parsing size of the binary code
        decay(bool): flag to apply decay to the output spikes
        tau(int): if decay is applied, specifies the decay time constant
        os(tuple): the binary encoding of the input
        change_prop(tuple): specified whether the problem is changeable with specifying the new problem
        gaussian(tuple): a tuple holding various parameters regarding the gaussian noise study
        seeder(numpy.random.randomstate): the seeder for the random functions
    Returns:
        all_inputs(numpy.array): the input spike trains
        all_targets(numpy.array): the target spikes or spike trains
        spikes_target_g(numpy.array): the required number of spikes in the output
        type_inputs(int): the number of unique inputs to the problem
    """
    def app_decay(spikes_in):
        """
        Applies a decay spike trains
        Args:
            spikes_in(numpy.array): the input spike trains

        Returns:
            v-out(numpy.array): the voltage/state after the application of the decay
        """
        v_out = np.zeros_like(spikes_in).astype(np.float32)
        alpha = np.exp(-1 / tau)
        for t in range(spikes_in.shape[1]):
            v_out[:, t] = (alpha * v_out[:, t - 1]) + spikes_in[:, t]
        return v_out
    raw_input = np.array([['00000000'+os[0]+'00000000', '00000000'+os[0]+'00000000'],
                          ['00000000'+os[1]+'00000000', '00000000'+os[1]+'00000000'],
                          ['00000000'+os[1]+'00000000', '00000000'+os[0]+'00000000'],
                          ['00000000'+os[0]+'00000000', '00000000'+os[1]+'00000000']
                          ])
    spikes_input = encode_inputs_bursts_3(raw_input, encoding_length=encoding_length, pad=pad, digits=digits,
                                          gaussian=gaussian, seeder=seeder)
    if target_spikes:
        global OUT1
        global OUT2
        o1 = '00000000003200000000'
        o2 = '00000000025600000000'
        OUT1, OUT2 = o1, o2
        raw_target = np.array([[o1, o1, o2, o2]])
        spikes_target = encode_inputs_bursts_3(raw_target, encoding_length=encoding_length, pad=pad, digits=digits,
                                               seeder=seeder).squeeze()

        dic_target = {'XOR': np.array([[o1, o1, o2, o2]]), 'XNOR': np.array([[o2, o2, o1, o1]]),
                      'OR': np.array([[o1, o2, o2, o2]]), 'NOR': np.array([[o2, o1, o1, o1]]),
                      'AND': np.array([[o1, o2, o1, o1]]), 'NAND': np.array([[o2, o1, o2, o2]])}
        for y, key in enumerate(dic_target.keys()):
            dic_target[key] = encode_inputs_bursts_3(dic_target[key], encoding_length=encoding_length, pad=pad,
                                                     digits=digits, seeder=seeder).squeeze()
        if decay:
            spikes_target = app_decay(spikes_target)
            for y, key in enumerate(dic_target.keys()):
                dic_target[key] = app_decay(dic_target[key])
    else:
        o1 = 1
        o2 = 2
        dic_target = {'XOR': np.array([o1, o1, o2, o2], dtype=np.float32), 'XNOR': np.array([o2, o2, o1, o1],
                                                                                            dtype=np.float32),
                      'OR': np.array([o1, o2, o2, o2], dtype=np.float32), 'NOR': np.array([o2, o1, o1, o1],
                                                                                          dtype=np.float32),
                      'AND': np.array([o1, o2, o1, o1], dtype=np.float32), 'NAND': np.array([o2, o1, o2, o2],
                                                                                            dtype=np.float32)}
        spikes_target = np.array([o1, o1, o2, o2], dtype=np.float32)
    spikes_target_g = 0
    if gaussian[0]:
        o1 = 1
        o2 = 2
        spikes_target_g = np.array([o1, o1, o2, o2], dtype=np.float32)

    all_inputs = []
    all_targets = []
    type_inputs = spikes_target.shape[0]
    if ordered_unique:
        for choice in range(type_inputs):
            all_inputs.append(spikes_input[choice])
            if change_prop[0]:
                all_targets.append(dic_target[change_prop[1]][choice])
            else:
                all_targets.append(spikes_target[choice])
        yield np.array(all_inputs), np.array(all_targets), spikes_target_g, type_inputs
    else:
        repetitions = batch_size/type_inputs
        for rep in range(batch_size):
            if gaussian[0]:
                choice = int(np.floor(rep / repetitions))
            else:
                choice = seeder.choice(spikes_target.shape[0])
            all_inputs.append(encode_inputs_bursts_3(np.array([raw_input[choice]]), encoding_length=encoding_length,
                                                     pad=pad, digits=digits, gaussian=gaussian, seeder=seeder)[0])
            if change_prop[0]:
                all_targets.append(dic_target[change_prop[1]][choice])
            else:
                if gaussian[0]:
                    all_targets.append(spikes_target_g[choice])
                else:
                    all_targets.append(spikes_target[choice])
        if not gaussian[0]:
            yield np.array(all_inputs), np.array(all_targets), spikes_target_g, type_inputs
        else:
            all_targets_pdf = []
            for ti in range(type_inputs):
                pdf = np.zeros_like(spikes_target[ti]).astype(np.float64)
                locations = np.where(np.array(spikes_target[ti]) >= 0.5)
                for loc in locations[0]:
                    pdf += norm.pdf(np.arange(spikes_target.shape[1]), loc, gaussian[1])
                pdf = pdf / pdf.sum()
                all_targets_pdf.append(pdf)
            yield np.array(all_inputs), np.array(all_targets), np.array(all_targets_pdf), type_inputs


def encode_inputs_bursts_3(arr_in, pad=0, encoding_length=5, digits=2, gaussian=(False, 2, 0.1, 10), seeder=None):
    """
    Encodes the inputs/outputs using the burst code specified below
    Args:
        arr_in(numpy.array): the un-encoded input/output array
        pad(int): the pad of the binary encoding
        encoding_length(int): the size of the binary encoding
        digits(int): the parsing size of the binary code
        gaussian(tuple): a tuple holding various parameters regarding the gaussian noise study
        seeder(numpy.random.randomstate): the seeder for the random functions
    Returns:
        arr_out(numpy.array): the encoded inputs/outputs
    """
    shape = arr_in.shape

    def get_spikes(word_in='012'):
        """
        Encodes strings into a binary code
        Args:
            word_in(str): the string to be converted

        Returns:
            word_out(list): the encoded string
        """
        word_out = []
        for i in range(int(len(word_in)/digits)):
            if word_in[i*digits:i*digits+digits] == 'RN':
                word_out += list(seeder.choice(('0', '1'), size=encoding_length, p=(0.5, 0.5)))
                continue
            if i != (int(len(word_in)/digits)-1):
                word_out += (list('{1:0{0}b}'.format(encoding_length, int(word_in[i*digits:i*digits+digits]))) +
                             ['0']*pad)
            else:
                word_out += list('{1:0{0}b}'.format(encoding_length, int(word_in[i*digits:])))
        if gaussian[0]:
            chunk = [int(x) for x in word_out[encoding_length*2:encoding_length*3]]
            summed = sum(chunk)
            if summed > 0:
                pdf = np.zeros_like(np.array(chunk)).astype(np.float64)
                locations = np.where(np.array(chunk) >= 0.5)

                for loc in locations[0]:
                    pdf += norm.pdf(np.arange(encoding_length), loc, gaussian[1])
                pdf = pdf / pdf.sum()
                pos_choice = seeder.choice(np.arange(encoding_length), size=int(summed), p=pdf, replace=False)
                word_new = np.zeros_like(np.array(chunk)).astype(np.float64)
                word_new[pos_choice] = 1
                word_new = [str(int(x)) for x in word_new]
                word_out[encoding_length*2:encoding_length*3] = word_new
        if gaussian[0] and gaussian[2]:
            noise = seeder.choice([0, 1], size=encoding_length*3, p=(1-gaussian[3], gaussian[3]))
            word_out_int = np.array([int(x) for x in word_out])
            word_out_int[encoding_length: encoding_length*4] = (
                np.clip(word_out_int[encoding_length: encoding_length*4] + noise, a_max=1, a_min=0))
            word_out = [str(x) for x in word_out_int]
        return word_out

    arr_out = np.array([[get_spikes(arr_in[i, j]) for j in range(shape[1])] for i in range(shape[0])], dtype=np.int32)
    return arr_out


FUNCTIONS = {'Delay_ms': delay_ms}  # Helps in calling the input from the main module
