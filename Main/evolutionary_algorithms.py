"""This is the networks module defining three functions: 1) the ea_map function which the main handler of applying the
evolutionary algorithm and communicates with the other three functions, 2) mutate, which takes care of applying a ]
mutation to each parameter set, 3)get_loss, which evaluates the loss of each solution in the population"""

"""Define imports"""
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import torch.multiprocessing as mp

from geometric_networks import GeoNetwork

# An assurance to the fixed seed
GLOBAL_SEED = np.random.RandomState(seed=12345)


def ea_map(core_number, iteration=0, population_size=10, num_elites=100, mut_rate=None, batch_size=1000,
           device=None, dtype=None, net_info=None, mape=None, loss_clip=None, problem_specs=None, queue_com=None,
           cross_com=True, queue_gen=None, gene_fixed=(False, 20), exp_flags=None):
    """
    The main function that handles the evolutionary algorithm
    Args:
        core_number(int): the number of the GPU core used in the computation
        iteration(int): a number to bias the seed
        population_size(int): the number of solutions in the population
        num_elites(int): the number of allowed elites in the population
        mut_rate(Dict): a dictionary holding various parameters related to evolving the population
        batch_size(int): the batch size
        device(torch.Device): the device used for computation CPU or CUDA#
        dtype(torch.Dtype): the data type of the array/Tensors
        net_info(Dict): a dictionary holding various parameters related to training
        mape(Dict): a dictionary holding various parameters related to training
        loss_clip(tuple): the clipping range of the loss
        problem_specs(Dict): a dictionary holding various parameters related to the problem specifications
        queue_com(multiprocessor object): the queue to hold the data from each GPU
        cross_com(bool): a flag for communication between GPUs
        queue_gen(multiprocessor object): the queue to hold the generation number from each GPU
        gene_fixed(Dict): a dictionary holding parameters related to end of training/evolution
        exp_flags(dict): the experiment flags

    Returns:
        A dictionary holding the solutions 1) weights, 2) time constants, 3) biases/after potentials, 4) delays
        5) minimum loss of the solution, 6) the number of generations of each solution
    """
    if not cross_com:
        device_number = core_number % 4
    else:
        device_number = core_number
    cuda_count = len(queue_com)
    device = torch.device("cuda:"+str(device_number))
    net_info[0]['Device'] = device
    gene_elapsed, repetition_count, seed_bias = 0, int(population_size/num_elites), net_info[0]['Seed_bias']
    instance = GeoNetwork(seed=core_number+iteration*mp.cpu_count()+seed_bias, other_para=net_info[0],
                          net_arch=net_info[1], net_para=net_info[2], net_mut_prop=net_info[3],
                          record=True, mut_rate=mut_rate, exp_flags=exp_flags)
    log_softmax_fn = nn.LogSoftmax(dim=2)
    if instance.loss_fn == 'MSE':
        loss_fn = nn.MSELoss(reduction='none')
    else:
        loss_fn = nn.NLLLoss(reduction='none')
    done_flag, condition_terminate = 0, False
    loss_min, sol_gene = deque(maxlen=2), {'Val': 1000, 'Flag': True}
    loss_min.append(5)
    loss_min.append(5)

    while True:
        if (gene_elapsed >= mape['Gene_time'][1]) and (not gene_fixed[0]):
            condition_terminate = True
        if not condition_terminate:
            loss_d, spikes_d, _ = get_loss(instance=instance, batch_size=batch_size, device=device, dtype=dtype,
                                           log_softmax_fn=log_softmax_fn, loss_fn=loss_fn, loss_clip=loss_clip,
                                           problem_specs=problem_specs, final_tc=net_info[0]['Const_tc_val'])
            loss_min_temp = np.amin(loss_d)
            if loss_min_temp < loss_min[1]:
                loss_min.append(loss_min_temp)
            loss_index = np.argsort(loss_d)[:num_elites]
            if loss_min[1] < loss_min[0]:
                loss_min.append(loss_min[1])
                print('Worker_' + str(core_number) + ' new Alpha:', loss_min[1])
                print('\n')
            if (np.any((loss_d >= -2e-6) & (loss_d <= 2e-6))) and sol_gene['Flag']:
                sol_gene['Val'] = gene_elapsed
                sol_gene['Flag'] = False
                mape['Gene_time'] = (mape['Gene_time'][0], sol_gene['Val'] + 5, gene_elapsed+2)
        if gene_fixed[0]:
            if_condition = (gene_elapsed >= gene_fixed[1])
        else:
            if_condition = ((np.any((loss_d >= -2e-6) & (loss_d <= 2e-6)) and (gene_elapsed >= mape['Gene_time'][2])
                             and (not sol_gene['Flag'])) or ((gene_elapsed >= mape['Gene_time'][1]) and
                                                             (not sol_gene['Flag'])) or condition_terminate)
        if if_condition:
            if done_flag == 1:
                out_w, out_tc, out_d, out_b = [], [], [], []
                loss_d_arg = np.flatnonzero(loss_d == np.min(loss_d))
                for i in range(instance.count_hidden+1):
                    out_w.append(instance.mat_weights[i][loss_d_arg].cpu().numpy())
                    out_tc.append(instance.mat_tc[i][loss_d_arg].cpu().numpy())
                    out_b.append(instance.bias[i][loss_d_arg].cpu().numpy())
                    out_d.append(instance.delay_layer.delays_out[i][loss_d_arg].cpu().numpy())
                instance.delete()
                del instance
                return {'Weights': out_w, 'TC': out_tc, 'Bias': out_b, 'Delays': out_d,
                        'Mini_loss': loss_min[1], 'Sol_gene': sol_gene['Val']}
            elif done_flag == 0:
                done_flag = 1
        elites_weights, elites_tcc, elites_bb, elites_dd = [], [], [], []
        for i in range(instance.count_hidden+1):
            elites_w, elites_tc, elites_bias, elites_delays = instance.mat_weights[i][loss_index], \
                instance.mat_tc[i][loss_index], instance.bias[i][loss_index], \
                instance.delay_layer.delays_out[i][loss_index]
            ####################################################################################################
            if cuda_count > 1:
                if cross_com:
                    if core_number == 0:
                        for _ in range(3):
                            queue_com[0].put((elites_w[:int(num_elites/4)].cpu().numpy(),
                                              elites_tc[:int(num_elites/4)].cpu().numpy(),
                                              elites_bias[:int(num_elites/4)].cpu().numpy(),
                                              elites_delays[:int(num_elites/4)].cpu().numpy()))
                            queue_gen[0].put(sol_gene['Val'])
                        elites_cross_1 = queue_com[1].get()
                        elites_cross_2 = queue_com[2].get()
                        elites_cross_3 = queue_com[3].get()

                        gene_cross_1 = queue_gen[1].get()
                        gene_cross_2 = queue_gen[2].get()
                        gene_cross_3 = queue_gen[3].get()
                    elif core_number == 1:
                        for _ in range(3):
                            queue_com[1].put((elites_w[:int(num_elites/4)].cpu().numpy(),
                                              elites_tc[:int(num_elites/4)].cpu().numpy(),
                                              elites_bias[:int(num_elites/4)].cpu().numpy(),
                                              elites_delays[:int(num_elites/4)].cpu().numpy()))
                            queue_gen[1].put(sol_gene['Val'])
                        elites_cross_1 = queue_com[0].get()
                        elites_cross_2 = queue_com[2].get()
                        elites_cross_3 = queue_com[3].get()

                        gene_cross_1 = queue_gen[0].get()
                        gene_cross_2 = queue_gen[2].get()
                        gene_cross_3 = queue_gen[3].get()
                    elif core_number == 2:
                        for _ in range(3):
                            queue_com[2].put((elites_w[:int(num_elites/4)].cpu().numpy(),
                                              elites_tc[:int(num_elites/4)].cpu().numpy(),
                                              elites_bias[:int(num_elites/4)].cpu().numpy(),
                                              elites_delays[:int(num_elites/4)].cpu().numpy()))
                            queue_gen[2].put(sol_gene['Val'])
                        elites_cross_1 = queue_com[0].get()
                        elites_cross_2 = queue_com[1].get()
                        elites_cross_3 = queue_com[3].get()

                        gene_cross_1 = queue_gen[0].get()
                        gene_cross_2 = queue_gen[1].get()
                        gene_cross_3 = queue_gen[3].get()
                    elif core_number == 3:
                        for _ in range(3):
                            queue_com[3].put((elites_w[:int(num_elites/4)].cpu().numpy(),
                                              elites_tc[:int(num_elites/4)].cpu().numpy(),
                                              elites_bias[:int(num_elites/4)].cpu().numpy(),
                                              elites_delays[:int(num_elites/4)].cpu().numpy()))
                            queue_gen[3].put(sol_gene['Val'])
                        elites_cross_1 = queue_com[0].get()
                        elites_cross_2 = queue_com[1].get()
                        elites_cross_3 = queue_com[2].get()

                        gene_cross_1 = queue_gen[0].get()
                        gene_cross_2 = queue_gen[1].get()
                        gene_cross_3 = queue_gen[2].get()
                    while True:
                        if (queue_com[0].empty and queue_com[1].empty and queue_com[2].empty and queue_com[3].empty and
                                queue_gen[0].empty and queue_gen[1].empty and queue_gen[2].empty and queue_gen[3].empty):
                            break
                    sol_gene['Val'] = min(gene_cross_1, gene_cross_2, gene_cross_3)
                    if (sol_gene['Val'] < 990) and sol_gene['Flag']:
                        sol_gene['Flag'] = False
                        mape['Gene_time'] = (mape['Gene_time'][0], sol_gene['Val'] + 5, sol_gene['Val'] + 2)
                    elites_cross_all = [elites_cross_1, elites_cross_2, elites_cross_3]
                    for k in range(len(elites_cross_all)):
                        (elites_w[int(num_elites*(k+1)/4): int(num_elites*(k+2)/4)],
                         elites_tc[int(num_elites*(k+1)/4): int(num_elites*(k+2)/4)],
                         elites_bias[int(num_elites*(k+1)/4): int(num_elites*(k+2)/4)],
                         elites_delays[int(num_elites*(k+1)/4): int(num_elites*(k+2)/4)]) = (
                            torch.FloatTensor(elites_cross_all[k][0]).to(device),
                            torch.FloatTensor(elites_cross_all[k][1]).to(device),
                            torch.FloatTensor(elites_cross_all[k][2]).to(device),
                            torch.FloatTensor(elites_cross_all[k][3]).to(device))
            ############################################################################################################

            repetition_count = int((instance.collective_size / elites_w.shape[0])) + 1

            elites_w = elites_w.repeat(repetition_count, 1, 1)[:instance.collective_size]
            elites_tc = elites_tc.repeat(repetition_count, *np.ones((len(elites_tc.shape, )) - 1,
                                                                    dtype=np.int32))[:instance.collective_size]
            elites_bias = elites_bias.repeat(repetition_count, 1, 1)[:instance.collective_size]
            elites_delays = elites_delays.repeat(repetition_count, 1, 1)[:instance.collective_size]
            elites_weights.append(elites_w)
            elites_tcc.append(elites_tc)
            elites_bb.append(elites_bias)
            elites_dd.append(elites_delays)

        para_temp = {'W': elites_weights, 'TC': elites_tcc, 'B': elites_bb, 'D': elites_dd}
        temp_weights, temp_tc, temp_bias, temp_delays = mutate(mut_rate=mut_rate, para_temp=para_temp,
                                                               num_elites=num_elites, device=device, dtype=dtype,
                                                               instance=instance)
        if mut_rate['Flag_w']:
            instance.set_weights(temp_weights)
        if mut_rate['Flag_t']:
            instance.set_tc(temp_tc)
        if mut_rate['Flag_b']:
            instance.set_bias(temp_bias)
        if mut_rate['Flag_d']:
            instance.set_delays(temp_delays)

        gene_elapsed += 1
        if core_number == 0:
            print('Generations elapsed:  ', gene_elapsed, '\n\n')
            pass


def mutate(mut_rate=None, para_temp=None, device=None, dtype=None, num_elites=200, instance=None):
    """
    Mutate the parameters of the model
    Args:
        mut_rate(Dict): a dictionary holding various parameters regarding the mutation operation
        para_temp(torch.Tensor): the passed down parameters for mutation
        device(torch.Device): the device used for computation, CPU or GPU#
        dtype(dtype): the date type of the arrays/Tensors
        num_elites(int): the number of elites used for generation of new population
        instance(object): an object of GeoNetwork

    Returns:
        the mutated, weights, time constants, bias/after potentials and delays
    """
    def random_mut(parameters=None, hid_count=1, mut_rate_v=0.1, para_range=(-1, 1), seeder=None):
        """
        Apply the mutation operation
        Args:
            parameters(torch.Tensor): the parameter to mutate
            hid_count(int): the number of hidden
            mut_rate_v(float): the mutation rate of the parameter
            para_range(tuple): the clipping range of the parameter
            seeder(numpy.random.Randomstate): the seeder for generation of random values

        Returns:
            the mutated parameter
        """
        para_mut = []
        for i in range(hid_count+1):
            para_shape = parameters[i][num_elites:].size()
            para_add = torch.from_numpy(seeder.normal(0, 1, size=para_shape)).to(device).type(dtype)
            parameters[i][num_elites:] = parameters[i][num_elites:] + (mut_rate_v * para_add)
            value_add = parameters[i].clone()
            para_mut.append(torch.clip(value_add, min=para_range[0], max=para_range[1]).float())
        return para_mut

    if mut_rate['Flag_w']:
        temp_weights = random_mut(parameters=para_temp['W'], hid_count=instance.count_hidden,
                                  mut_rate_v=mut_rate['Value_w'], para_range=mut_rate['Range_w'],
                                  seeder=instance.local_seed)
    else:
        temp_weights = para_temp['W']
    if mut_rate['Flag_t']:
        temp_tc = random_mut(parameters=para_temp['TC'], hid_count=instance.count_hidden,
                             mut_rate_v=mut_rate['Value_t'], para_range=mut_rate['Range_t'], seeder=instance.local_seed)
    else:
        temp_tc = para_temp['TC']
    if mut_rate['Flag_b']:
        temp_bias = random_mut(parameters=para_temp['B'], hid_count=instance.count_hidden,
                               mut_rate_v=mut_rate['Value_b'], para_range=mut_rate['Range_b'],
                               seeder=instance.local_seed)
    else:
        temp_bias = para_temp['B']
    if mut_rate['Flag_d']:
        temp_delays = random_mut(parameters=para_temp['D'], hid_count=instance.count_hidden,
                                 mut_rate_v=mut_rate['Value_d'], para_range=mut_rate['Range_d'],
                                 seeder=instance.local_seed)
    else:
        temp_delays = para_temp['D']
    return temp_weights, temp_tc, temp_bias, temp_delays


def get_loss(instance=None, batch_size=100, device=None, dtype=None, log_softmax_fn=None, loss_fn=None, loss_clip=None,
             problem_specs=None, final_tc=5, change_prop=(False, 'XOR')):
    """
    Acquire the loss of the entire population
    Args:
        instance(object): an object of GeoNetwork
        batch_size(int): the size of the batch used
        device(torch.device): the device used for computations CPU or GPU#
        dtype(dtype): the data type of the arrays/tensors
        log_softmax_fn(torch object): applies the log softmax function
        loss_fn(torch object): the loss function used for computation ex: MSE or CE
        loss_clip(tuple): the clipping range of the loss
        problem_specs(Dict): a dictionary holding various parameters regarding the problem specification
        final_tc(int): the time constant used in the final layer when decay is applied
        change_prop(tuple): it specifies the problem type and whether it changes or not

    Returns:
        loss_d(float): the loss of the whole population
        spikes(torch.Tensor): the generated spikes by the whole population
        x_local(numpy.ndarray): the used input during computation
    """
    x_local, y_local = next(problem_specs['Problem_type'](batch_size=batch_size,
                                                          ordered_unique=problem_specs['Ordered_unique'],
                                                          target_spikes=problem_specs['Out_spikes'],
                                                          encoding_length=problem_specs['Encoding_length'],
                                                          digits=problem_specs['Digits'],
                                                          pad=problem_specs['Problem_pad'], tau=final_tc,
                                                          decay=problem_specs['Decay'], os=problem_specs['Objects'],
                                                          change_prop=change_prop,
                                                          logic_type=problem_specs['Logic_type'],
                                                          out_simple=problem_specs['Out_simple'],
                                                          input_pad=problem_specs['Input_pad'],
                                                          delay_shift=problem_specs['Delay_shift']))

    net_outputs, spikes, _ = instance(x_local)
    if instance.loss_fn == 'MSE':
        if problem_specs['Out_spikes']:
            net_outputs = net_outputs.squeeze(dim=2)
        else:
            net_outputs = torch.sum(net_outputs.squeeze(dim=2), dim=2)
    else:
        net_outputs = torch.sum(net_outputs, dim=3)
    if problem_specs['Out_spikes']:
        net_targets = torch.from_numpy(y_local).repeat(instance.collective_size, 1, 1).to(device).type(dtype)
    else:
        net_targets = torch.from_numpy(y_local).unsqueeze(dim=0).repeat(instance.collective_size, 1).\
            to(device).type(dtype)
    if instance.loss_fn == 'MSE':
        if problem_specs['Out_spikes']:
            loss = (net_outputs - net_targets)**2
            loss = torch.mean(torch.sum(loss, dim=2), dim=1)
        else:
            loss = torch.mean(loss_fn(net_outputs.float(), net_targets.float()), dim=1)
    else:
        net_outputs_prop = log_softmax_fn(net_outputs).swapaxes(0, 2).swapaxes(0, 1)
        net_targets = net_targets.swapaxes(0, 1)
        loss = torch.mean(loss_fn(net_outputs_prop.float(), net_targets.type(torch.LongTensor).to(device)), dim=0)
    loss_d = loss.cpu().detach().numpy()
    if loss_clip['Flag']:
        loss_d = np.clip(loss_d, a_min=loss_clip['Range'][0], a_max=loss_clip['Range'][1])
    return loss_d, spikes, x_local



