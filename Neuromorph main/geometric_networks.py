"""This is the networks module defining four classes 1) The SurrGradSpike, for passing surrogate gradients during
backprop, 2) DelayUpdate, for passing surrogate gradients through the differentiable delay layer, 3) DelayLayer,
which always applied delays to be differentiated, 4) GeoNetwork, which defines the main network"""

"""Define imports"""
import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import copy


class SurrGradSpike(torch.autograd.Function):
    """ The spike surrogate class
    Applies the surrogate gradient through the backward pass
    Attributes
    ---------
    slope: int
        slope of the sigmoid function
    shift: int
        shift of the sigmoid function
    Methods
    ------
    forward(ctx, input):
        applies the forward pass
    backward(ctx, grad_out):
        applies the backward pass
    """
    slope = 1
    shift = 0

    @staticmethod
    def forward(ctx, input):
        """
        Applies the forward pass
        Args:
            ctx: Holder for the Saved the input tensors
            input(torch.Tensor): input tensors

        Returns:
            out(torch.Tensor): output after application of heaviside function
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out
    @staticmethod
    def backward(ctx, grad_out):
        """
        Applies the backward pass
        Args:
            ctx: Saved input tensors
            grad_out(torch.Tensor): Calculated gradients

        Returns:
            grad(torch.Tensor): the gradients after the application of the differentiated sigmoid
        """
        input_tens, = ctx.saved_tensors
        sig = torch.sigmoid(SurrGradSpike.slope*(input_tens-SurrGradSpike.shift))
        grad = grad_out*sig*(1-sig)
        return grad


class DelayUpdate(torch.autograd.Function):
    """The delay surrogate class
        Applies the surrogate gradient through the backward pass
        Attributes
        ---------
        delay_min: int
            minimum allowed delay
        delay_max: int
            maximum allowed delay
        Methods
        ------
        forward(ctx, delays):
            applies the forward pass
        backward(ctx, grad_out):
            applies the backward pass
        """
    delay_min = -10
    delay_max = 10
    @staticmethod
    def forward(ctx, delays):
        """
        Applies the forward pass
        Args:
            ctx: Holder for the Saved the input tensors
            delays(torch.Tensor): input tensors

        Returns:
            delay_forward(torch.Tensor): output after application of truncation
        """
        delays_forward = torch.round(torch.clamp(delays, min=DelayUpdate.delay_min, max=DelayUpdate.delay_max))
        return delays_forward

    @staticmethod
    def backward(ctx, grad_output):
        """
        Applies the backward pass
        Args:
            ctx: Saved input tensors
            grad_output(torch.Tensor): Calculated gradients

        Returns:
            grad(torch.Tensor): the gradients are passed directly
        """
        delays_in = grad_output
        return delays_in


class DelayLayer(nn.Module):
    """The delay layer class
    This class defines an array of differentiable delays of size (NUMBER_INPUTS, NUMBER_CLASSES)
    to applied between any SNN layers
    Attributes
    ---------
    Methods
    ------
    _init_delay_vector:
        initialize the delay vector
    _init_optimizer:
        initialize the delay optimizer when backprop
    forward:
        apply forward pass through the delay layer
    """
    def __init__(self, train_delays=True, constant_delays=False, constant_value=0, lr_delay=1e-3, delay_min=-10,
                 delay_max=10, true_sizes=(2, 2, 1), fix_first_input=False, zero_pad=False, zero_pad_v=None,
                 round_d=4, device=None, dtype=None, collective_size=1, seeder=None, mut_rate=None):
        """
        Initialize the delay layer
        Args:
            train_delays(bool): A flag that defines whether the delay array is differentiable or not
            constant_delays(bool): the initialized delays all have a constant value
            constant_value(int): the initialized delays constant value
            lr_delay(float): lr for backprop of delays
            delay_min(int): minimum value of delays
            delay_max(int): maximum value of delays
            true_sizes(tuple): network size
            fix_first_input(boolean): mask the first input spike train to a neuron from training
            zero_pad(bool): flag for zero padding the delay tensors
            zero_pad_v((int): zero pad size
            round_d(int): number of decimal places when rounding the delay values
            device(torch.device): the device to do computation on, CPU or GPU#
            dtype(dtype): the data type of the array/tensor objects
            collective_size(int): the size of the population
            seeder(numpy.random.randomstate): the local seeder for random operations
            mut_rate(float): mutation rate of the delay values
        """
        super().__init__()
        self.delay_max = delay_max
        self.delay_min = delay_min
        self.trainable_delays = train_delays
        self.true_sizes = true_sizes
        self.count_layer = len(true_sizes)
        self.constant_delays = constant_delays
        self.constant_value = constant_value
        self.lr_delay = lr_delay
        self.fix_first_input = fix_first_input
        self.round_d = round_d
        self.device = device
        self.dtype = dtype
        self.zero_pad, self.zero_pad_v = zero_pad, zero_pad_v
        self.collective_size = collective_size
        self.local_seed = seeder
        self.mut_rate = mut_rate

        self.delays_out, self.delays_out_ini = self._init_delay_vector()
        self.optimizer_delay = self._init_optimizer()
        DelayUpdate.delay_min = delay_min
        DelayUpdate.delay_max = delay_max
        self.delay_fn = DelayUpdate.apply

    def _init_delay_vector(self):
        """ Initialize the delay layer

        Returns:
            mat_delays(torch.Tensor, (COLLECTIVE_SIZE, NUMBER_INPUTS, NUMBER_CLASSES)): the delays array/tensor that will be updated
            mat_delays_ini(torch.Tensor, (COLLECTIVE_SIZE, NUMBER_INPUTS, NUMBER_CLASSES)): the initial delay values that are fixed
        """
        mat_delays, mat_delays_ini, fix_first = [], [], 0
        for i in range(self.count_layer-1):
            if self.fix_first_input:
                fix_first = 1
            else:
                fix_first = 0
            if self.constant_delays:
                delays = torch.nn.parameter.Parameter(torch.FloatTensor(
                    self.constant_value * np.ones((self.collective_size, self.true_sizes[i]-fix_first,
                                                   self.true_sizes[i+1]), dtype=int)).to(self.device),
                                                      requires_grad=self.trainable_delays)
            else:
                delays_numpy = self.local_seed.randint(self.delay_min, self.delay_max,
                                                       size=(self.collective_size, self.true_sizes[i]-fix_first,
                                                             self.true_sizes[i+1]), dtype=int)
                delays = torch.nn.parameter.Parameter(torch.FloatTensor(delays_numpy).to(self.device),
                                                      requires_grad=self.trainable_delays)
            mat_delays.append(delays.detach())
            mat_delays_ini.append(delays.detach())
        return mat_delays, mat_delays_ini

    def _init_optimizer(self):
        """Initialize the optimizer

        Returns:
            optimizer_delay(torch.optim): the optimizer used to update delays when backprop
        """
        optimizer_delay = torch.optim.SGD(self.delays_out, lr=self.lr_delay)
        return optimizer_delay

    def forward(self, spikes_in, choice_layer=0):
        """forward pass through the delay layer

        Parameters:
            spikes_in(torch.Tensor, (BATCH_SIZE, COLLECTIVE_SIZE, NUMBER_INPUTS, NUMBER_CLASSES, EFFECTIVE_DURATION)): the input spike trains to be shifted
            choice_layer(int): the chosen delay layer

        Returns:
            output_train(torch.Tensor, (BATCH_SIZE, COLLECTIVE_SIZE, NUMBER_INPUTS, NUMBER_CLASSES, EFFECTIVE_DURATION)): the shifted(delay applied) spike trains
        """
        if (len(spikes_in.shape) == 4) and (self.collective_size == 1):
            spikes_in = spikes_in[None, :, :, :, :]
        if self.zero_pad:
            if self.zero_pad_v is None:
                spikes_zeros = torch.zeros_like(spikes_in)
            else:
                spikes_zeros = torch.zeros((spikes_in[0], spikes_in[1], spikes_in[2], spikes_in[3], self.zero_pad_v))
            spikes_in = torch.concatenate((spikes_zeros, spikes_in, spikes_zeros), dim=3)
        input_train = spikes_in[:, :, :, :, :, None]
        if self.fix_first_input:
            input_first = input_train[:, :, 0:1, :, :, :]
            input_train = input_train[:, :, 1:, :, :, :]
        else:
            input_first = None
        dlys = self.delay_fn(self.delays_out[choice_layer])
        collective_size, batch_size, inputs, classes, duration, _ = input_train.size()
        # initialize M to identity transform and resize
        translate_mat = np.array([[1., 0., 0.], [0., 1., 0.]])
        translate_mat = torch.FloatTensor(np.resize(translate_mat, (collective_size, batch_size, inputs, classes,
                                                                    2, 3))).to(self.device)
        # translate with delays
        translate_mat[:, :, :, :, 0, 2] = 2 / (duration - 1) * dlys[:, None, :, :]
        # create normalized 1D grid and resize
        x_t = np.linspace(-1, 1, duration)
        y_t = np.zeros((1, duration))  # 1D: all y points are zeros
        ones = np.ones(np.prod(x_t.shape))
        grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])   # an array of points (x, y, 1) shape (3, :)
        grid = torch.FloatTensor(np.resize(grid, (collective_size, batch_size, inputs, classes, 3,
                                                  duration))).to(self.device)
        # transform the sampling grid i.e. batch multiply
        translate_grid = torch.matmul(translate_mat, grid)
        # reshape to (num_batch, height, width, 2)
        translate_grid = torch.transpose(translate_grid, 4, 5)
        x_points = translate_grid[:, :, :, :, :, 0]
        corr_center = ((x_points + 1.) * (duration - 1)) * 0.5
        # grab 4 nearest corner points for each (x_t, y_t)
        corr_left = torch.floor(torch.round(corr_center, decimals=self.round_d)).type(torch.int32)
        corr_right = corr_left + 1
        # Calculate weights
        weight_right = (corr_right - corr_center)
        weight_left = (corr_center - corr_left)

        # Padding for values that are evaluated outside the input range
        pad_right = torch.maximum(torch.amax(corr_right) + 1 - duration, torch.abs(torch.amin(corr_left))+1)
        zeros_right = torch.zeros(size=(collective_size, batch_size, inputs, classes, pad_right, 1)).to(self.device)
        ################################################################################################################
        input_train = torch.cat((input_train, zeros_right), dim=4)  # care no zeros left
        ################################################################################################################
        # Get the new values after the transformation
        value_left = input_train[np.arange(self.collective_size)[:, None, None, None, None],
                                 np.arange(batch_size)[None, :, None, None, None],
                                 np.arange(inputs)[None, None, :, None, None],
                                 np.arange(classes)[None, None, None, :, None], corr_left][:, :, :, :, :, 0]
        value_right = input_train[np.arange(self.collective_size)[:, None, None, None, None],
                                  np.arange(batch_size)[None, :, None, None, None],
                                  np.arange(inputs)[None, None, :, None, None],
                                  np.arange(classes)[None, None, None, :, None], corr_right][:, :, :, :, :, 0]
        # compute output
        output_train = weight_right*value_left + weight_left*value_right
        if self.fix_first_input:
            output_train = torch.concatenate((input_first[:, :, :, :, :, 0], output_train), dim=1)
        return output_train


# noinspection PyProtectedMember
class GeoNetwork(nn.Module):
    """The class define the main network architecture and properties
    Attributes
    ---------
    Methods
    ------
    __get_loc_x__:
        get coordinates of each node in the network
    __init_bias__:
        initialize the network bias
    __init_weights__:
        initialize the network weights
    __init_tc__:
        initialize the network tc
    __init_con_mat__:
        initialize the network connections
    set_weights:
        set the network weights from an outside array
    set_tc:
        set the network time constants from an outside array
    set_bias:
        set the network bias from an outside array
    set_delays:
        set the network delays from an outside array
    set_conn:
        set the networks conn probabilities from an outside array
    set_col_size:
        set the networks population size
    connection_count:
        count the number of connections in a network
    distance_square:
        calculate the distance between two nodes
    connection_cost:
        calculate the cost of connections in a network
    get_statistics:
        calculate some statistics of the network, like the total distance between nodes.
    forward:
        apply a forward pass through the network
    copy:
        copy the parameters of the network
    delete:
        delete the parameters of the network to save memory
    """
    def __init__(self, seed=0, record=False, other_para=None, net_arch=None, net_para=None, net_mut_prop=None,
                 mut_rate=None, exp_flags=None):
        """
        Initialize the class
        Args:
            self.other_para(Dict): holding various network configurations
            self.net_arch(tuple): architecture of the network
            self.net_para(Dict): holding various network configurations related to the net architecture
            self.net_mut_prop(float): mutation probability of the network architecture
            self.record(boolean): flag to record spikes or not during the forward pass
            self.seed(int): the seed of the network of random operations
            self.local_seed(numpy.random.Randomstate): the (seeder) of the network for various operations
            self.prop_con(float): the probability of connection between two nodes
            self.prop_con_type(boolean): the evolvability of the connections, constant or evolvable
            self.prop_w(float): the rate of connections evolvability
            self.v_thresh(float): the value of the neuron threshold
            self.device(torch.device): the device for computations, CPU or GPU#
            self.dtype(dtype): the data type of the array/tensors
            self.collective_size(int): the population size of the network
            self.loss_fn(torch loss function): the loss function used in computation
            self.mut_rate(dict): a dictionary holding various parameters related to the network mutarion rates
            self.out_decay(boolean): dictates whether a decay is applied to the spikes or not
            self.roll_delay(boolean): dictates which method is used to update delays
            self.ap_flag(boolean): the spike after potential flag
            self.ap_tc(float): the spike after potential time constant
            self.rand_nodes(boolean): a flag dictating whether the initial node number are random
            self.rand_pos(str): a condition for the initial node position distribution
            self.max_sizes(tuple): the maximum size of the network for all layer
            self.d_s(integer): a decorative parameter for the interpretation of the delay shit (left or right)
            self.true_sizes(tuple): the true/final architecture of the network
            self.hidden_sizes(int): the hidden layers architecture
            self.count_layer(int): the number of layers in the network, including input and output
            self.count_hidden(int): the number of hidden layers
            self.range_x(numpy array): the x position range of the nodes
            self.range_y(numpy array): the y position range of the nodes
            self.range_w(tuples): the allowed range of weights
            self.grad_w(boolean): flag for weights grads retention
            self.const_w(boolean): flag for whether the weights are constant or not
            self.const_w_val(float): constant value of the weights
            self.range_b(tuple): the allowed range of the bias
            self.grad_b(boolean): flag for bias grads retention
            self.synapse_b(boolean): whether the bias/after potential is synaptic or somatic
            self.const_b(boolean): flag for whether the bias/after potential is constant or not
            self.const_b_val(float): the constant value of the bias/after potential
            self.range_d(tuple): the allowed values of the delays
            self.grad_d(boolean): a flag for delays grads retention
            self.range_t(tuple): the allowed values of the time constants
            self.grad_t(boolean): flag for time constants grad retention
            self.synapse_t(boolean): flag for whether the time constants are synaptic or somatic
            self.const_tc(boolean): a flag for whether the time constants are fixed or not
            self.const_tc_val(float): the constant time constant value
            self.loc_y(tuple): the y locations of all nodes in the network
            self.loc_x(tuple): the x locations of all nodes in the network
            self.locations(tuple): the x and y locations of all nodes in the network
            self.spike_fn(object): the surrogate spike object
            SurrGradSpike.slope(int): the slope of the activation function of the surrogate spike class
            SurrGradSpike.shift(int): the shift of the activation function of the surrogate spike class
            self.delay_layer(object): the delay layer object
            self.mat_weights(torch.Tensor, ((COLLECTIVE_SIZE, NUMBER_INPUTS, NUMBER_OUTPUTS))): the weights of the network
            self.mat_weights_ini(torch.Tensor, ((COLLECTIVE_SIZE, NUMBER_INPUTS, NUMBER_OUTPUTS))): the initial weights of the network (unchanged)
            self.mat_tc(torch.Tensor, ((COLLECTIVE_SIZE, NUMBER_INPUTS, NUMBER_OUTPUTS))): the time constants of the network
            self.mat_tc_ini(torch.Tensor, ((COLLECTIVE_SIZE, NUMBER_INPUTS, NUMBER_OUTPUTS))): the initial time constants of the network (unchanged)
            self.mat_con(torch.Tensor, ((COLLECTIVE_SIZE, NUMBER_INPUTS, NUMBER_OUTPUTS))): the connection probability of the network
            self.bias(torch.Tensor, ((COLLECTIVE_SIZE, NUMBER_INPUTS, NUMBER_OUTPUTS))): the bias/after potential of the network
            self.bias_ini(torch.Tensor, ((COLLECTIVE_SIZE, NUMBER_INPUTS, NUMBER_OUTPUTS))): the initial bias/after potential of the network
        """
        super().__init__()

        self.other_para = other_para
        self.net_arch = net_arch
        self.net_para = net_para
        self.net_mut_prop = net_mut_prop
        self.record = record
        self.seed, self.exp_flags = seed, exp_flags
        self.local_seed = np.random.RandomState(seed=2 * seed + 11)
        self.prop_con = net_mut_prop['Prop_con']
        self.prop_con_type = net_mut_prop['Prop_con_type']
        self.prop_w = net_mut_prop['Prop_w']
        self.v_thresh = other_para['V_thresh']
        self.device, self.dtype = other_para['Device'], other_para['Dtype']
        self.collective_size = other_para['Collective_size']
        self.loss_fn = other_para['Loss_fn']
        self.mut_rate, self.out_decay, self.roll_delay = mut_rate, other_para['Out_decay'], other_para['Roll_delay']
        self.ap_flag, self.ap_tc = other_para['Spike_ap'][0], other_para['Spike_ap'][1]
        self.rand_nodes, self.rand_pos, self.max_sizes = (net_arch['Rand_num'], net_arch['Rand_pos'],
                                                          net_arch['Max_sizes'])
        self.d_s = other_para['d_s']
        if self.rand_nodes:
            self.true_sizes = self.local_seed.randint(low=1, high=self.max_sizes, size=len(self.max_sizes))
        else:
            self.true_sizes = np.array(self.max_sizes)
        self.hidden_sizes = self.true_sizes[1:-1]
        self.count_layer, self.count_hidden = len(self.max_sizes), len(self.hidden_sizes)
        self.range_x = net_arch['Range_x']
        self.range_y = np.arange(self.count_layer)

        self.range_w, self.grad_w = net_para['Range_w'], net_para['Grad_w']
        self.const_w, self.const_w_val = other_para['Const_w'], other_para['Const_w_val']
        self.range_b, self.grad_b, self.synapse_b = net_para['Range_b'], net_para['Grad_b'], net_para['Synapse_b']
        self.const_b, self.const_b_val = other_para['Const_b'], other_para['Const_b_val']
        self.range_d, self.grad_d = net_para['Range_d'], net_para['Grad_d']
        self.range_t, self.grad_t, self.synapse_t = net_para['Range_t'], net_para['Grad_t'], net_para['Synapse_t']
        self.const_tc, self.const_tc_val = other_para['Const_tc'], other_para['Const_tc_val']

        # Need collective update #######################################################################################
        self.loc_y = tuple(np.array([i] * j) for i, j in zip(self.range_y, self.true_sizes))
        self.loc_x = self.__get_loc_x__()
        self.locations = tuple(tuple((x, y) for x, y in zip(row1, row2)) for row1, row2 in zip(self.loc_x, self.loc_y))
        ################################################################################################################

        self.spike_fn = SurrGradSpike.apply
        SurrGradSpike.slope = other_para['Surr_slope']
        SurrGradSpike.shift = other_para['Surr_shift']
        self.delay_layer = DelayLayer(constant_delays=other_para['Const_delay'],
                                      constant_value=other_para['Const_d_val'], delay_min=self.range_d[0],
                                      true_sizes=self.true_sizes, delay_max=self.range_d[1], train_delays=self.grad_d,
                                      device=self.device, dtype=self.dtype, fix_first_input=other_para['Fix_first'],
                                      zero_pad=other_para['Zero_pad_d'], zero_pad_v=other_para['Zero_pad_val'],
                                      collective_size=self.collective_size, seeder=self.local_seed, mut_rate=mut_rate)

        self.mat_weights, self.mat_weights_ini = self.__init_weights__()
        self.mat_tc, self.mat_tc_ini = self.__init_tc__()
        self.mat_con = self.__init_con_mat__()
        self.bias, self.bias_ini = self.__init_bias__()

    def __get_loc_x__(self):
        """
        Get the x coordinates of the network nodes
        Returns:
            loc_x(tuple): x locations of the whole network
        """
        range_x = self.range_x
        possible_loc = np.arange(range_x[0], range_x[1] + range_x[2], range_x[2])
        possible_loc_shape = possible_loc.shape[0]
        if self.rand_pos == 'uniform':
            loc_x = tuple(np.sort((self.local_seed.choice(possible_loc, size=sx, replace=False)))
                               for sx in self.true_sizes)
        elif self.rand_pos == 'inv_gauss':
            ga = sp.signal.gaussian(M=possible_loc_shape, std=possible_loc_shape / 4, sym=True)
            ga = np.amax(ga) - ga
            ga = ga / np.sum(ga)
            loc_x = tuple(
                np.sort((self.local_seed.choice(possible_loc, size=sx, replace=False, p=ga))) for sx in
                self.true_sizes)
        else:
            loc_x = tuple(np.linspace(range_x[0], range_x[1] + range_x[2], sx) for sx in self.true_sizes)

        return loc_x

    def __init_bias__(self):
        """
        Initialize the bias/after potential of the network
        Returns:
            arr_bias(torch.Tensor, ((COLLECTIVE_SIZE, NUMBER_INPUTS, NUMBER_OUTPUTS))): the bias/after potential of the network
            arr_bias_ini(torch.Tensor, ((COLLECTIVE_SIZE, NUMBER_INPUTS, NUMBER_OUTPUTS))): the initial bias/after potential of the network
        """
        arr_bias, arr_bias_ini = [], []
        if self.synapse_b:
            if self.const_b:
                for i in range(self.count_hidden + 1):
                    bias = self.const_b_val * np.ones((self.collective_size, self.true_sizes[i], self.true_sizes[i+1]))
                    bias = torch.nn.parameter.Parameter(torch.from_numpy(bias).to(self.device).type(self.dtype),
                                                        requires_grad=self.grad_b)
                    arr_bias.append(bias.detach())
                    arr_bias_ini.append(bias.detach())
            else:
                for i in range(self.count_hidden + 1):
                    tmp_bias_zeros = np.zeros((self.true_sizes[i]*self.true_sizes[i+1],))
                    tmp_bias_avg = np.zeros((self.true_sizes[i]*self.true_sizes[i+1],))
                    tmp_bias_cm = np.identity(self.true_sizes[i]*self.true_sizes[i+1])
                    arr_bb = (tmp_bias_avg + self.local_seed.multivariate_normal(tmp_bias_zeros, tmp_bias_cm,
                                                                                 size=self.collective_size))\
                        .reshape((self.collective_size, self.true_sizes[i], self.true_sizes[i+1]))
                    arr_bb = nn.Parameter(torch.from_numpy(np.clip(arr_bb, a_min=self.mut_rate['Range_b'][0],
                                                                   a_max=self.mut_rate['Range_b'][1])).to(self.device).
                                          type(self.dtype), requires_grad=self.grad_b)
                    arr_bias.append(arr_bb.detach())
                    arr_bias_ini.append(arr_bb.detach())
            return arr_bias, arr_bias_ini
        else:
            if self.const_b:
                for i in range(self.count_hidden + 1):
                    bias = self.const_b_val * np.ones((self.collective_size, self.true_sizes[i+1]))
                    bias = torch.nn.parameter.Parameter(torch.from_numpy(bias).to(self.device).type(self.dtype),
                                                        requires_grad=self.grad_b)
                    arr_bias.append(bias.detach())
                    arr_bias_ini.append(bias.detach())
            else:
                for i in range(self.count_hidden + 1):
                    tmp_bias_zeros = np.zeros((self.true_sizes[i+1],))
                    tmp_bias_avg = np.zeros((self.true_sizes[i+1],))
                    tmp_bias_cm = np.identity(self.true_sizes[i+1])
                    arr_bb = (tmp_bias_avg + self.local_seed.multivariate_normal(tmp_bias_zeros, tmp_bias_cm,
                                                                                 size=self.collective_size))\
                        .reshape((self.collective_size, self.true_sizes[i+1]))
                    arr_bb = nn.Parameter(torch.from_numpy(np.clip(arr_bb, a_min=self.mut_rate['Range_b'][0],
                                                                   a_max=self.mut_rate['Range_b'][1])).to(self.device).
                                          type(self.dtype), requires_grad=self.grad_b)
                    arr_bias.append(arr_bb.detach())
                    arr_bias_ini.append(arr_bb.detach())
            return arr_bias, arr_bias_ini

    def __init_weights__(self):
        """
        Initialize the weights of the network
        Returns:
            weights(torch.Tensor, ((COLLECTIVE_SIZE, NUMBER_INPUTS, NUMBER_OUTPUTS))): the weights of the network
            weights_ini(torch.Tensor, ((COLLECTIVE_SIZE, NUMBER_INPUTS, NUMBER_OUTPUTS))): the initial weights of the network
        """
        weights, weights_ini = [], []
        if self.const_w:
            for i in range(self.count_layer - 1):
                arr_weights = nn.Parameter(self.const_w_val*torch.ones(size=(self.collective_size, self.true_sizes[i],
                                                                             self.true_sizes[i+1]),
                                                                       requires_grad=self.grad_w).
                                           to(self.device).type(self.dtype))
                weights.append(arr_weights.detach())
                weights_ini.append(arr_weights.detach())
        else:
            if self.exp_flags['Exp_one'] or self.exp_flags['Exp_two']:
                for i in range(self.count_layer - 1):
                    tmp_weights_zeros = np.zeros((self.true_sizes[i]*self.true_sizes[i+1],))
                    tmp_weights_avg = np.zeros((self.true_sizes[i]*self.true_sizes[i+1],))
                    tmp_weights_cm = np.identity(self.true_sizes[i]*self.true_sizes[i+1])
                    arr_weights = (tmp_weights_avg + self.local_seed.multivariate_normal(tmp_weights_zeros,
                                                                                         tmp_weights_cm,
                                                                                         size=self.collective_size))\
                        .reshape((self.collective_size, self.true_sizes[i], self.true_sizes[i+1]))
                    arr_weights = nn.Parameter(torch.from_numpy(np.clip(arr_weights, a_min=self.mut_rate['Range_w'][0],
                                                                        a_max=self.mut_rate['Range_w'][1])).
                                               to(self.device).type(self.dtype), requires_grad=self.grad_w)
                    weights.append(arr_weights.detach())
                    weights_ini.append(arr_weights.detach())
            else:
                for i in range(self.count_layer - 1):
                    arr_weights = nn.Parameter(torch.empty(size=(self.collective_size, self.true_sizes[i],
                                                                 self.true_sizes[i + 1]), requires_grad=self.grad_w).
                                               to(self.device).type(self.dtype))
                    torch.manual_seed(self.seed)
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(arr_weights[0])
                    boundary = 1 / np.sqrt(fan_in)
                    nn.init.uniform_(arr_weights, -boundary, boundary)
                    weights.append(arr_weights.detach())
                    weights_ini.append(arr_weights.detach())
        return weights, weights_ini

    def __init_tc__(self):
        """
        Initialize the time constants of the network
        Returns:
            arr_tc(torch.Tensor, ((COLLECTIVE_SIZE, NUMBER_INPUTS, NUMBER_OUTPUTS))): the time constants of the network
            init_tc(torch.Tensor, ((COLLECTIVE_SIZE, NUMBER_INPUTS, NUMBER_OUTPUTS))): the initial time constants of the network
        """
        arr_tc, init_tc = [], []
        if self.const_tc:
            if self.synapse_t:
                for i in range(self.count_layer - 1):
                    arr_tc_temp = self.const_tc_val * np.ones((self.collective_size, self.true_sizes[i],
                                                               self.true_sizes[i + 1]))
                    arr_tc_temp = torch.nn.parameter.Parameter(torch.from_numpy(arr_tc_temp).
                                                               to(self.device).type(self.dtype),
                                                               requires_grad=self.grad_t)
                    arr_tc.append(arr_tc_temp.detach())
                    init_tc.append(arr_tc_temp.detach())
            else:
                for i in range(self.count_layer - 1):
                    arr_tc_temp = self.const_tc_val * np.ones((self.collective_size, self.true_sizes[i + 1]))
                    arr_tc_temp = torch.nn.parameter.Parameter(torch.from_numpy(arr_tc_temp).
                                                               to(self.device).type(self.dtype),
                                                               requires_grad=self.grad_t)
                    arr_tc.append(arr_tc_temp.detach())
                    init_tc.append(arr_tc_temp.detach())
        else:
            if self.synapse_t:
                if self.exp_flags['Exp_one'] or self.exp_flags['Exp_two']:
                    for i in range(self.count_layer - 1):
                        tmp_tc_zeros = np.zeros((self.true_sizes[i] * self.true_sizes[i+1],))
                        tmp_tc_avg = np.zeros((self.true_sizes[i] * self.true_sizes[i+1],))
                        tmp_tc_cm = np.identity(self.true_sizes[i] * self.true_sizes[i+1])
                        arr_tcc = (tmp_tc_avg + self.local_seed.multivariate_normal(tmp_tc_zeros, tmp_tc_cm,
                                                                                    size=self.collective_size)) \
                            .reshape((self.collective_size, self.true_sizes[i], self.true_sizes[i + 1]))
                        arr_tcc = nn.Parameter(torch.from_numpy(np.clip(arr_tcc, a_min=self.mut_rate['Range_t'][0],
                                                                        a_max=self.mut_rate['Range_t'][1])).
                                               to(self.device).type(self.dtype), requires_grad=self.grad_t)
                        arr_tc.append(arr_tcc.detach())
                        init_tc.append(arr_tcc.detach())
                else:
                    for i in range(self.count_layer - 1):
                        arr_tc_temp = self.local_seed.uniform(low=self.range_t[0], high=self.range_t[-1],
                                                              size=(self.collective_size, self.true_sizes[i],
                                                                    self.true_sizes[i + 1]))
                        arr_tc_temp = torch.nn.parameter.Parameter(torch.from_numpy(arr_tc_temp).
                                                                   to(self.device).type(self.dtype),
                                                                   requires_grad=self.grad_t)
                        arr_tc.append(arr_tc_temp.detach())
                        init_tc.append(arr_tc_temp.detach())
            else:
                if self.exp_flags['Exp_one'] or self.exp_flags['Exp_two']:
                    for i in range(self.count_layer - 1):
                        tmp_tc_zeros = np.zeros((self.true_sizes[i+1],))
                        tmp_tc_avg = np.zeros((self.true_sizes[i+1],))
                        tmp_tc_cm = np.identity(self.true_sizes[i+1])
                        arr_tcc = (tmp_tc_avg + self.local_seed.multivariate_normal(tmp_tc_zeros, tmp_tc_cm,
                                                                                    size=self.collective_size)) \
                            .reshape((self.collective_size, self.true_sizes[i + 1]))
                        arr_tcc = nn.Parameter(torch.from_numpy(np.clip(arr_tcc, a_min=self.mut_rate['Range_t'][0],
                                                                        a_max=self.mut_rate['Range_t'][1]))
                                               .to(self.device).type(self.dtype), requires_grad=self.grad_t)
                        arr_tc.append(arr_tcc.detach())
                        init_tc.append(arr_tcc.detach())
                else:
                    for i in range(self.count_layer - 1):
                        arr_tc_temp = self.local_seed.uniform(low=self.range_t[0], high=self.range_t[-1],
                                                              size=(self.collective_size, self.true_sizes[i + 1]))
                        arr_tc_temp = torch.nn.parameter.Parameter(torch.from_numpy(arr_tc_temp).
                                                                   to(self.device).type(self.dtype),
                                                                   requires_grad=self.grad_t)
                        arr_tc.append(arr_tc_temp.detach())
                        init_tc.append(arr_tc_temp.detach())
        return arr_tc, init_tc

    def __init_con_mat__(self):
        """
        Initialize the connection probabilities of the network
        Returns:
            mat_con(torch.Tensor, ((COLLECTIVE_SIZE, NUMBER_INPUTS, NUMBER_OUTPUTS))): the connection probabilities of the network
        """
        mat_con = []
        if self.prop_con_type == 'Const':
            for i in range(self.count_layer - 1):
                mat_con.append(torch.from_numpy(self.local_seed.choice((0, 1), size=(self.collective_size,
                                                                                     self.true_sizes[i],
                                                                                     self.true_sizes[i + 1]),
                                                                       p=[1 - self.prop_con, self.prop_con])).
                               to(self.device).type(self.dtype))
        else:
            for i in range(self.count_layer - 1):
                con_para = self.local_seed.uniform(low=0.0, high=1.0, size=(self.collective_size, self.true_sizes[i],
                                                                            self.true_sizes[i + 1]))
                mat_con.append(torch.nn.parameter.Parameter(torch.from_numpy(con_para).to(self.device).type(self.dtype),
                                                            requires_grad=True))
        return mat_con

    def set_weights(self, new_weights):
        """
        Set weights to the network
        Args:
            new_weights: a list of layer weights to be assigned

        Returns:

        """
        for i in range(self.count_hidden+1):
            if type(new_weights[i]) == torch.Tensor:
                self.mat_weights[i] = new_weights[i].clone()
            else:
                self.mat_weights[i] = torch.nn.parameter.Parameter(torch.from_numpy(new_weights[i]).to(self.device)
                                                                   .type(self.dtype), requires_grad=self.grad_w)

    def set_tc(self, new_tc):
        """
        Set new time constants to the network
        Args:
            new_tc: a list of layer time constants to be assigned

        Returns:

        """
        for i in range(self.count_hidden+1):
            if type(new_tc[i]) == torch.Tensor:
                self.mat_tc[i] = new_tc[i].clone()
            else:
                self.mat_tc[i] = torch.nn.parameter.Parameter(torch.from_numpy(new_tc[i]).to(self.device).
                                                              type(self.dtype), requires_grad=self.grad_t)

    def set_bias(self, new_bias):
        """
        Set new bias/after potentials the network
        Args:
            new_bias: a list of layer bias/after potentials to be assigned

        Returns:

        """
        for i in range(self.count_hidden+1):
            if type(new_bias[i]) == torch.Tensor:
                self.bias[i] = new_bias[i].clone()
            else:
                self.bias[i] = torch.nn.parameter.Parameter(torch.from_numpy(new_bias[i]).to(self.device).type(self.dtype),
                                                            requires_grad=self.grad_w)

    def set_delays(self, new_delays):
        """
        Set new delays to the network
        Args:
            new_delays: a list of layer delays to be assigned to the network

        Returns:

        """
        for i in range(self.count_hidden+1):
            if type(new_delays[i]) == torch.Tensor:
                self.delay_layer.delays_out[i] = new_delays[i].clone()
            else:
                self.delay_layer.delays_out[i] = torch.nn.parameter.Parameter(torch.from_numpy(new_delays[i]).
                                                                              to(self.device).type(self.dtype),
                                                                              requires_grad=self.grad_w)

    def set_conn(self, new_conn):
        """
        Set new connection probabilities to the network
        Args:
            new_conn: a list layer connection probabilities to be assigned to the network

        Returns:

        """
        for i in range(self.count_hidden+1):
            if type(new_conn[i]) == torch.Tensor:
                self.mat_con[i] = new_conn[i].clone()
            else:
                self.mat_con[i] = torch.nn.parameter.Parameter(torch.from_numpy(new_conn[i]).to(self.device).
                                                               type(self.dtype),requires_grad=self.grad_w)

    def set_col_size(self, col_size):
        """
        Assigns a new population size
        Args:
            col_size: new population size

        Returns:

        """
        self.collective_size = col_size
        self.delay_layer.collective_size = col_size

    def connection_count(self):
        """
        Counts the connections in the network
        Returns:
            con_count: number of connections in the network
        """
        con_count = 0
        for i in range(self.count_hidden+1):
            con_count += np.sum(self.mat_con[i])
        if con_count == 0:
            con_count = 1
        return con_count

    @staticmethod
    def distance_square(point1, point2):
        """
        Computes the distance between two nodes
        Args:
            point1: node 1
            point2: node 2

        Returns:

        """
        return (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2

    def connection_cost(self):
        """
        Calculate tje connection cost of the network
        Returns:
            cost: the connection cost of the network
        """
        cost = 0
        for i in np.arange(0, self.count_hidden+1):
            mask, idx = self.mat_con[i].flatten(), 0
            for point1 in self.locations[i]:
                for point2 in self.locations[i+1]:
                    cost += self.distance_square(point1, point2) * mask[idx]
                    idx += 1
        return cost

    def get_statistics(self):
        """
        Calculate the connection/distance statistics of the network
        Returns:
            stats_all: the connection/distance statistics of the network
        """
        def get_stat(para_value=None, para_ini=None):
            dist_mean, dist_ini, sol_mean, sol_median = [], [], [], []
            for i in range(self.count_hidden+1):
                arr_p, arr_p_ini = para_value[i], para_ini[i]
                arr_shape_p = arr_p.shape
                mean_p_cur, mean_p_ini = torch.mean(arr_p, dim=0), torch.mean(arr_p_ini, dim=0)
                dist_p_cur, dist_p_ini = torch.mean((arr_p - mean_p_cur[None, ...])**2,
                                                    dim=tuple(np.arange(len(arr_shape_p))[1:])), \
                    torch.mean((arr_p - mean_p_ini[None, ...])**2, dim=tuple(np.arange(len(arr_shape_p))[1:]))
                dist_mean.append(dist_p_cur)
                dist_ini.append(dist_p_ini)
                sol_mean_new_p = torch.mean(arr_p, dim=tuple(np.arange(len(arr_shape_p))[1:]))
                sol_mean.append(sol_mean_new_p)
                arr_median_p = torch.median(arr_p.reshape(-1, np.prod(arr_shape_p[1:])), dim=1)
                sol_median.append(arr_median_p[0])
            return dist_mean, dist_ini, sol_mean, sol_median
        dist_mean_w, dist_ini_w, sol_mean_w, sol_median_w = get_stat(self.mat_weights, self.mat_weights_ini)
        dist_mean_tc, dist_ini_tc, sol_mean_tc, sol_median_tc = get_stat(self.mat_tc, self.mat_tc_ini)
        dist_mean_b, dist_ini_b, sol_mean_b, sol_median_b = get_stat(self.bias, self.bias_ini)
        dist_mean_d, dist_ini_d, sol_mean_d, sol_median_d = get_stat(self.delay_layer.delays_out,
                                                                     self.delay_layer.delays_out_ini)
        stats_dic_w = {'Dist_mean': dist_mean_w, 'Dist_ini': dist_ini_w, 'Sol_mean': sol_mean_w,
                       'Sol_median': sol_median_w}
        stats_dic_tc = {'Dist_mean': dist_mean_tc, 'Dist_ini': dist_ini_tc, 'Sol_mean': sol_mean_tc,
                        'Sol_median': sol_median_tc}
        stats_dic_b = {'Dist_mean': dist_mean_b, 'Dist_ini': dist_ini_b, 'Sol_mean': sol_mean_b,
                       'Sol_median': sol_median_b}
        stats_dic_d = {'Dist_mean': dist_mean_d, 'Dist_ini': dist_ini_d, 'Sol_mean': sol_mean_d,
                       'Sol_median': sol_median_d}
        stats_all = {'W': stats_dic_w, 'TC': stats_dic_tc, 'D': stats_dic_d, 'B': stats_dic_b}
        return stats_all

    # noinspection PyStatementEffect
    def forward(self, states_in=0, test=False):
        """
        Forward pass through the main network
        Args:
            states_in: the network input
            test(boolean): test various aspects to the network

        Returns:
            spikes_out: the spikes of each neuron in the network
            states_out: the spikes of the final layer of the network
            v_out: the voltage of the final layer of the network
        """
        spikes_all = []
        states_out = torch.FloatTensor(states_in).to(self.device).type(self.dtype)
        size_batch, _, duration = states_out.size()
        states_out = states_out.unsqueeze(0).repeat(self.collective_size, 1, 1, 1)  # _, batch, inputs, duration
        for i in range(self.count_hidden + 1):
            states_out = states_out.unsqueeze(3).repeat(1, 1, 1, self.true_sizes[i+1], 1)
            ############################################################################################################
            if self.roll_delay:
                size_col, size_batch, size_in, size_out, len_time = states_out.shape
                times_array = (torch.FloatTensor(np.arange(len_time)[None, None, None, None, :])
                               .repeat(size_col, size_batch, size_in, size_out, 1)
                               .to(self.device))
                times_array = (times_array + torch.round(self.d_s * self.delay_layer.delays_out[i])
                [:, None, :, :, None]).type(torch.int)
                zeros_pad = (torch.zeros(size_col, size_batch, size_in, size_out, self.delay_layer.delay_max)
                             .to(self.device).type(self.dtype))
                states_out_old = torch.concatenate((states_out, zeros_pad), dim=-1)
                states_out = states_out_old[np.arange(size_col)[:, None, None, None, None],
                np.arange(size_batch)[None, :, None, None, None], np.arange(size_in)[None, None, :, None, None],
                np.arange(size_out)[None, None, None, :, None], times_array]
            else:
                states_out = self.delay_layer(states_out, choice_layer=i).type(self.dtype)  # Problem with visualization
            ############################################################################################################
            if self.prop_con_type == 'Const':
                masked_weights = torch.mul(self.mat_weights[i], self.mat_con[i])
            else:
                masked_weights = torch.mul(self.mat_weights[i], torch.sigmoid(self.mat_con[i]))
            if not self.synapse_b:
                if self.ap_flag:
                    weighted_in_bias = torch.einsum("labdc,lbd->labdc", (states_out, masked_weights))
                else:
                    weighted_in_bias = torch.einsum("labdc,lbd->labdc", (states_out, masked_weights)) + \
                        self.bias[i][:, None, None, :, None]
            else:
                if self.ap_flag:
                    weighted_in_bias = torch.einsum("labdc,lbd->labdc", (states_out, masked_weights))
                else:
                    weighted_in_bias = torch.einsum("labdc,lbd->labdc", (states_out, masked_weights)) + \
                                       self.bias[i][:, None, :, :, None]
            v_soma = torch.zeros((self.collective_size, size_batch, self.true_sizes[i], self.true_sizes[i+1],
                                  duration)).to(self.device).type(self.dtype)
            spike_temp = torch.zeros((self.collective_size, size_batch,
                                      self.true_sizes[i + 1])).to(self.device).type(self.dtype)
            spike_ap = torch.zeros((self.collective_size, size_batch, self.true_sizes[i], self.true_sizes[i+1],
                                    duration)).to(self.device).type(self.dtype)
            spikes_out = []
            ############################################################################################################
            alpha = torch.exp(-1 / self.mat_tc[i])  # Need to add the ability for different synapses
            alpha_sa = torch.exp(-1 / torch.FloatTensor([self.ap_tc]).to(self.device).type(self.dtype))
            if not self.synapse_t:
                alpha = alpha.unsqueeze(dim=1).repeat(1, self.true_sizes[i], 1)
            ############################################################################################################
            for t in range(duration):
                if self.ap_flag:
                    v_soma[:, :, :, :, t] = alpha[:, None, :, :]*(1*v_soma[:, :, :, :, t-1]) \
                                            + weighted_in_bias[:, :, :, :, t] + spike_ap[:, :, :, :, t-1]
                    v_sum_temp = torch.sum(v_soma[:, :, :, :, t], dim=2)
                    spike_temp = self.spike_fn(v_sum_temp-self.v_thresh)
                    spike_ap[:, :, :, :, t] = alpha_sa*spike_ap[:, :, :, :, t-1] + \
                        self.bias[i][:, None, :, :]*spike_temp[:, :, None, :]
                else:
                    v_soma[:, :, :, :, t] = alpha[:, None, :, :] * (1 * v_soma[:, :, :, :, t - 1]) * \
                                            (1 - spike_temp[:, :, None, :]) + weighted_in_bias[:, :, :, :, t]
                    v_sum_temp = torch.sum(v_soma[:, :, :, :, t], dim=2)
                    spike_temp = self.spike_fn(v_sum_temp - self.v_thresh)
                spikes_out.append(spike_temp)
            states_out = torch.stack(spikes_out, dim=3).to(self.device)
            if self.record:
                spikes_all.append(states_out)
        if (self.loss_fn == 'MSE') and (not self.out_decay):
            return states_out, spikes_all, None
        elif (self.loss_fn != 'MSE') or self.out_decay:
            v_out = torch.zeros((self.collective_size, size_batch, self.true_sizes[-1],
                                 duration)).to(self.device).type(self.dtype)
            alpha_out = torch.exp(-1 / torch.FloatTensor([self.const_tc_val]).to(self.device).type(self.dtype))
            for t in range(duration):
                v_out[:, :, :, t] = alpha_out*v_out[:, :, :, t-1] + states_out[:, :, :, t]
            return v_out, spikes_all, states_out

    def copy(self, solution):
        """
        Copy another network
        Args:
            solution: the input network

        Returns:

        """
        self.true_sizes = solution.true_sizes.copy()
        self.seed = solution.seed
        self.local_seed = solution.local_seed
        self.loc_x = solution.loc_x
        self.loc_y = solution.loc_y
        self.locations = solution.locations
        self.spike_fn = solution.spike_fn
        self.collective_size = solution.collective_size

        self.mat_weights = copy.deepcopy(solution.mat_weights)
        self.mat_tc = copy.deepcopy(solution.mat_tc)
        self.mat_con = copy.deepcopy(solution.mat_con)
        self.bias = copy.deepcopy(solution.bias)
        self.delay_layer.delays_out = copy.deepcopy(solution.delay_layer.delays_out)

    def delete(self):
        """
        Delete the variables of the network
        Returns:

        """
        del self.mat_weights
        del self.mat_tc
        del self.mat_con
        del self.bias
        del self.delay_layer.delays_out
        del self.mat_weights_ini
        del self.mat_tc_ini
        del self.bias_ini
        del self.delay_layer.delays_out_ini
        del self.delay_layer





