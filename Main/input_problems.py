"""This is the input problems module, which specifies two main functions: 1) the delay_ms function which provides the
input output combination of all the logic problems, 2) the encode_inputs_bursts_3, which specify the encoding of the
input/output"""

"""Define imports"""
import numpy as np

ENCODING = 'Burst'  # For now, 'Burst' or time to first spike 'TFS' for future implementations
OUT1 = '0'  # An object holding the encoding of the zero state/ logic zero
OUT2 = '0'  # An object holding the encoding of the one state/ logic one


def delay_ms(batch_size=1000, ordered_unique=False, target_spikes=False, encoding_length=5, pad=0, digits=2,
             decay=False, tau=5, os=('01', '03', '05'), change_prop=(False, 'XOR'), logic_type='XOR',
             out_simple=(1, 2), input_pad='0000', delay_shift=(False, 3)):
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
        logic_type(str): specifies the logic problem
        out_simple(tuple): specifies the encoding of the output (logic 0, logic 1)
        input_pad(str): the zero padding to the input
        delay_shift(tuple): holds parameters for the delay shift experiment (Exp 8)

    Returns:
        all_inputs(numpy.array): the input spike trains
        all_targets(numpy.array): the target spikes or spike trains
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
    raw_input = np.array([[input_pad + os[0] + input_pad, input_pad + os[0] + input_pad],
                          [input_pad + os[1] + input_pad, input_pad + os[1] + input_pad],
                          [input_pad + os[1] + input_pad, input_pad + os[0] + input_pad],
                          [input_pad + os[0] + input_pad, input_pad + os[1] + input_pad]
                          ])
    spikes_input = encode_inputs_bursts_3(raw_input, encoding_length=encoding_length, pad=pad, digits=digits)
    if target_spikes:
        global OUT1
        global OUT2
        if delay_shift[0]:
            o1 = list('0000000000000000000000')
            o2 = list('0000000000000000000000')
            o1[delay_shift[1]] = '1'
            o2[delay_shift[1]+2] = '1'
            o1 = ''.join(o1)
            o2 = ''.join(o2)
        else:
            o1 = '00' + out_simple[0] + '000000'
            o2 = '0000' + out_simple[1] + '0000'
        OUT1, OUT2 = o1, o2
        dic_target = {'XOR': np.array([[o1, o1, o2, o2]]), 'XNOR': np.array([[o2, o2, o1, o1]]),
                      'OR': np.array([[o1, o2, o2, o2]]), 'NOR': np.array([[o2, o1, o1, o1]]),
                      'AND': np.array([[o1, o2, o1, o1]]), 'NAND': np.array([[o2, o1, o2, o2]])}
        raw_target = dic_target[logic_type]
        spikes_target = encode_inputs_bursts_3(raw_target, encoding_length=encoding_length, pad=pad, digits=digits).squeeze()

        for y, key in enumerate(dic_target.keys()):
            dic_target[key] = encode_inputs_bursts_3(dic_target[key], encoding_length=encoding_length, pad=pad,
                                                     digits=digits).squeeze()
        if decay:
            spikes_target = app_decay(spikes_target)
            for y, key in enumerate(dic_target.keys()):
                dic_target[key] = app_decay(dic_target[key])
    else:
        o1 = out_simple[0]
        o2 = out_simple[1]
        dic_target = {'XOR': np.array([o1, o1, o2, o2], dtype=np.float32), 'XNOR': np.array([o2, o2, o1, o1],
                                                                                            dtype=np.float32),
                      'OR': np.array([o1, o2, o2, o2], dtype=np.float32), 'NOR': np.array([o2, o1, o1, o1],
                                                                                          dtype=np.float32),
                      'AND': np.array([o1, o2, o1, o1], dtype=np.float32), 'NAND': np.array([o2, o1, o2, o2],
                                                                                            dtype=np.float32)}
        spikes_target = dic_target[logic_type]
    all_inputs = []
    all_targets = []
    if ordered_unique:
        for choice in range(spikes_target.shape[0]):
            all_inputs.append(spikes_input[choice])
            if change_prop[0]:
                all_targets.append(dic_target[change_prop[1]][choice])
            else:
                all_targets.append(spikes_target[choice])
        yield np.array(all_inputs), np.array(all_targets)
    else:
        for _ in range(batch_size):
            choice = np.random.choice(spikes_target.shape[0])
            all_inputs.append(spikes_input[choice])
            if change_prop[0]:
                all_targets.append(dic_target[change_prop[1]][choice])
            else:
                all_targets.append(spikes_target[choice])
        yield np.array(all_inputs), np.array(all_targets)


def encode_inputs_bursts_3(arr_in, pad=3, encoding_length=3, digits=1):
    """
    Encodes the inputs/outputs using the burst code specified below
    Args:
        arr_in(numpy.array): the un-encoded input/output array
        pad(int): the pad of the binary encoding
        encoding_length(int): the size of the binary encoding
        digits(int): the parsing size of the binary code

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
            if i != (int(len(word_in)/digits)-1):
                word_out += (list('{1:0{0}b}'.format(encoding_length, int(word_in[i*digits:i*digits+digits]))) +
                             ['0']*pad)
            else:
                word_out += list('{1:0{0}b}'.format(encoding_length, int(word_in[i*digits:])))
        return word_out

    arr_out = np.array([[get_spikes(arr_in[i, j]) for j in range(shape[1])] for i in range(shape[0])], dtype=np.int32)
    return arr_out


FUNCTIONS = {'Delay_ms': delay_ms}  # Helps in calling the input from the main module
