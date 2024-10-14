"""This is the networks module defining three functions: 1) the ea_map function which the main handler of applying the
evolutionary algorithm and communicates with the other three functions, 2) mutate, which takes care of applying a ]
mutation to each parameter set, 3)get_loss, which evaluates the loss of each solution in the population"""

"""Define imports"""
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import time
import torch.multiprocessing as mp

from geometric_networks import GeoNetwork

GLOBAL_SEED = np.random.RandomState(seed=12345)


def ea_map(core_number, iteration=0, population_size=10, num_elites=100, mut_rate=None, batch_size=1000,
           device=None, dtype=None, net_info=None, mape=None, loss_clip=None, problem_specs=None, queue_com=None,
           cross_com=True, epoch=64):
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
        epoch(int): the number of training cycles required to average the loss
    Returns:
        A dictionary holding the solutions 1) weights, 2) time constants, 3) biases/after potentials, 4) delays
        5) minimum loss of the solution, 6) the number of generations of each solution
    """
    stability_counter = 0
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
                          record=True, mut_rate=mut_rate)
    instance_temp = GeoNetwork(seed=core_number, other_para=net_info[0], net_arch=net_info[1],
                               net_para=net_info[2], net_mut_prop=net_info[3], record=True, mut_rate=mut_rate)
    log_softmax_fn = nn.LogSoftmax(dim=2)
    if instance.loss_fn == 'MSE':
        loss_fn = nn.MSELoss(reduction='none')
    else:
        loss_fn = nn.NLLLoss(reduction='none')
    done_flag = 0
    loss_min = deque(maxlen=2)
    loss_min.append(5)
    loss_min.append(5)

    while True:
        loss_d = get_loss(instance=instance, batch_size=batch_size, device=device, dtype=dtype,
                          log_softmax_fn=log_softmax_fn, loss_fn=loss_fn, loss_clip=loss_clip,
                          problem_specs=problem_specs, final_tc=net_info[0]['Const_tc_val'],
                          cn=core_number, epoch=epoch)
        loss_min_temp = np.amin(loss_d)
        if np.any(loss_d <= 1e-8):
            stability_counter += 1
        else:
            stability_counter = 0
        if loss_min_temp < loss_min[1]:
            loss_min.append(loss_min_temp)
        loss_index = np.argsort(loss_d)[:num_elites]
        if loss_min[1] < loss_min[0]:
            loss_min.append(loss_min[1])
            print('Worker_' + str(core_number) + ' new Alpha:', loss_min[1])
            print('\n')
        if ((np.any((loss_d >= -1e-8) & (loss_d <= 1e-8)) and (stability_counter == problem_specs['Stability_counter']))
                or (gene_elapsed >= mape['Gene_time'][2])):
            if (done_flag == 1) and (gene_elapsed >= mape['Gene_time'][2]):
                out_w, out_tc, out_d, out_b = [], [], [], []
                loss_d_arg = np.flatnonzero(loss_d == np.min(loss_d))
                for i in range(instance_temp.count_hidden+1):
                    out_w.append(instance_temp.mat_weights[i][loss_d_arg].cpu().numpy())
                    out_tc.append(instance_temp.mat_tc[i][loss_d_arg].cpu().numpy())
                    out_b.append(instance_temp.bias[i][loss_d_arg].cpu().numpy())
                    out_d.append(instance_temp.delay_layer.delays_out[i][loss_d_arg].cpu().numpy())
                instance.delete()
                del instance
                instance_temp.delete()
                del instance_temp
                return {'Weights': out_w, 'TC': out_tc, 'Bias': out_b, 'Delays': out_d,
                        'Mini_loss': loss_min[1]}
            elif done_flag == 0:
                instance_temp.copy(instance)
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
                        time.sleep(0.01)
                        for _ in range(3):
                            queue_com[0].put((elites_w[:int(num_elites / 4)].cpu().numpy(),
                                              elites_tc[:int(num_elites / 4)].cpu().numpy(),
                                              elites_bias[:int(num_elites / 4)].cpu().numpy(),
                                              elites_delays[:int(num_elites / 4)].cpu().numpy()))
                        time.sleep(0.01)
                        elites_cross_1 = queue_com[1].get()
                        elites_cross_2 = queue_com[2].get()
                        elites_cross_3 = queue_com[3].get()
                    elif core_number == 1:
                        time.sleep(0.01)
                        for _ in range(3):
                            queue_com[1].put((elites_w[:int(num_elites / 4)].cpu().numpy(),
                                              elites_tc[:int(num_elites / 4)].cpu().numpy(),
                                              elites_bias[:int(num_elites / 4)].cpu().numpy(),
                                              elites_delays[:int(num_elites / 4)].cpu().numpy()))
                        time.sleep(0.01)
                        elites_cross_1 = queue_com[0].get()
                        elites_cross_2 = queue_com[2].get()
                        elites_cross_3 = queue_com[3].get()
                    elif core_number == 2:
                        time.sleep(0.01)
                        for _ in range(3):
                            queue_com[2].put((elites_w[:int(num_elites / 4)].cpu().numpy(),
                                              elites_tc[:int(num_elites / 4)].cpu().numpy(),
                                              elites_bias[:int(num_elites / 4)].cpu().numpy(),
                                              elites_delays[:int(num_elites / 4)].cpu().numpy()))
                        time.sleep(0.01)
                        elites_cross_1 = queue_com[0].get()
                        elites_cross_2 = queue_com[1].get()
                        elites_cross_3 = queue_com[3].get()
                    elif core_number == 3:
                        time.sleep(0.01)
                        for _ in range(3):
                            queue_com[3].put((elites_w[:int(num_elites / 4)].cpu().numpy(),
                                              elites_tc[:int(num_elites / 4)].cpu().numpy(),
                                              elites_bias[:int(num_elites / 4)].cpu().numpy(),
                                              elites_delays[:int(num_elites / 4)].cpu().numpy()))
                        time.sleep(0.01)
                        elites_cross_1 = queue_com[0].get()
                        elites_cross_2 = queue_com[1].get()
                        elites_cross_3 = queue_com[2].get()

                    elites_cross_all = [elites_cross_1, elites_cross_2, elites_cross_3]
                    for k in range(len(elites_cross_all)):
                        (elites_w[int(num_elites * (k + 1) / 4): int(num_elites * (k + 2) / 4)],
                         elites_tc[int(num_elites * (k + 1) / 4): int(num_elites * (k + 2) / 4)],
                         elites_bias[int(num_elites * (k + 1) / 4): int(num_elites * (k + 2) / 4)],
                         elites_delays[int(num_elites * (k + 1) / 4): int(num_elites * (k + 2) / 4)]) = (
                            torch.FloatTensor(elites_cross_all[k][0]).to(device),
                            torch.FloatTensor(elites_cross_all[k][1]).to(device),
                            torch.FloatTensor(elites_cross_all[k][2]).to(device),
                            torch.FloatTensor(elites_cross_all[k][3]).to(device))
            ####################################################################################################
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
            print('Stability counter: ', stability_counter)
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
             problem_specs=None, final_tc=5, cn=0, change_prop=(False, 'XOR'), epoch=4):
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
        epoch(int): the number of training cycles required to average the loss

    Returns:
        loss_d(float): the loss of the whole population
    """
    loss_mse, loss, sump, pdf_local, net_out, xl, best_idx, spikes_out = 0, 0, 0, 0, 0, 0, [], 0
    col_size = instance.mat_weights[0].shape[0]
    weights_ini, tc_ini, bias_ini, delays_ini, conn_ini = [], [], [], [], []
    repeat_r = problem_specs['Robust'][0]
    repeat_std = problem_specs['Robust'][1]
    flag_w, flag_tc, flag_d = (problem_specs['Robust'][2]['W'], problem_specs['Robust'][2]['TC']
                               , problem_specs['Robust'][2]['D'])
    seeder = instance.local_seed
    for ep in range(epoch):
        x_local, y_local, tmp_pdf_local, type_inputs = next(problem_specs['Problem_type'](batch_size=batch_size,
                                                                                          ordered_unique=problem_specs[
                                                                                              'Ordered_unique'],
                                                                                          target_spikes=problem_specs[
                                                                                              'Out_spikes'],
                                                                                          encoding_length=problem_specs[
                                                                                              'Encoding_length'],
                                                                                          digits=problem_specs[
                                                                                              'Digits'],
                                                                                          pad=problem_specs[
                                                                                              'Problem_pad'],
                                                                                          tau=final_tc,
                                                                                          decay=problem_specs['Decay'],
                                                                                          os=problem_specs['Objects'],
                                                                                          change_prop=change_prop,
                                                                                          gaussian=problem_specs[
                                                                                              'Gaussian'],
                                                                                          seeder=seeder))
        pdf_local, xl = tmp_pdf_local, x_local
        ################################################################################################################
        """Robust"""
        if not problem_specs['Gaussian'][0]:
            weights_new, tc_new, bias_new, delays_new, conn_new = [], [], [], [], []
            for i in range(instance.count_hidden + 1):
                weights_ini.append(instance.mat_weights[i].cpu().numpy())
                conn_ini.append(instance.mat_con[i])
                weights_old = (instance.mat_weights[i].repeat(repeat_r, *([1] * len(instance.mat_weights[i].shape[1:])))
                               .cpu().numpy())
                conn_old = instance.mat_con[i].repeat(repeat_r, *([1] * len(instance.mat_con[i].shape[1:]))).cpu().numpy()
                conn_new.append(conn_old)
                if flag_w:
                    weights_white_noise = seeder.normal(0, 1, size=(repeat_r, *weights_old.shape[1:]))*repeat_std
                    weights_white_noise = np.repeat(weights_white_noise, col_size, axis=0)
                    weights_new_temp = weights_old + weights_white_noise
                    weights_new.append(weights_new_temp)
                else:
                    weights_new.append(weights_old)

                tc_ini.append(instance.mat_tc[i].cpu().numpy())
                tc_old = instance.mat_tc[i].repeat(repeat_r, *([1] * len(instance.mat_tc[i].shape[1:]))).cpu().numpy()
                if flag_tc:
                    tc_white_noise = seeder.normal(0, 1, size=(repeat_r, *tc_old.shape[1:]))*repeat_std
                    tc_white_noise = np.repeat(tc_white_noise, col_size, axis=0)
                    tc_new_temp = tc_old + tc_white_noise
                    tc_new.append(np.clip(tc_new_temp, a_min=0.1, a_max=None))
                else:
                    tc_new.append(tc_old)

                bias_ini.append(instance.bias[i].cpu().numpy())
                bias_old = instance.bias[i].repeat(repeat_r, *([1] * len(instance.bias[i].shape[1:]))).cpu().numpy()
                bias_new.append(bias_old)

                delays_ini.append(instance.delay_layer.delays_out[i].cpu().numpy())
                delays_old = (instance.delay_layer.delays_out[i].repeat(repeat_r, *([1] * len(instance.delay_layer.
                                                                                             delays_out[i].shape[1:]))).
                              cpu().numpy())
                if flag_d:
                    delays_white_noise = seeder.normal(0, 1, size=(repeat_r, *delays_old.shape[1:]))*repeat_std
                    delays_white_noise = np.repeat(delays_white_noise, col_size, axis=0)
                    delays_new_temp = delays_old + delays_white_noise
                    delays_new.append(delays_new_temp)
                else:
                    delays_new.append(delays_old)

            instance.set_weights(weights_new)
            instance.set_conn(conn_new)
            instance.set_tc(tc_new)
            instance.set_bias(bias_new)
            instance.set_delays(delays_new)
            instance.set_col_size(instance.collective_size*repeat_r)
        ################################################################################################################
        net_outputs, spikes, net_outputs_g = instance(x_local)
        spikes_out = spikes
        size_col, size_in, out_shape, size_t = net_outputs.shape
        net_outputs_s = 0
        net_outputs_n = 0
        if instance.loss_fn == 'MSE':
            if out_shape == 1:
                if problem_specs['Out_spikes'] and (not problem_specs['Gaussian'][0]):
                    net_outputs = net_outputs.squeeze(dim=2)
                elif problem_specs['Gaussian'][0]:
                    net_outputs_s = net_outputs_g.squeeze(dim=2)
                    net_outputs_n = torch.sum(net_outputs_g.squeeze(dim=2), dim=2)
                else:
                    net_outputs = torch.sum(net_outputs.squeeze(dim=2), dim=2)
        else:
            net_outputs = torch.sum(net_outputs, dim=3)
        if not problem_specs['Gaussian'][0]:
            if problem_specs['Out_spikes']:
                if out_shape == 1:
                    net_targets = torch.from_numpy(y_local).repeat(instance.collective_size, 1, 1).to(device).type(
                        dtype)
                else:
                    net_targets = torch.from_numpy(y_local).repeat(instance.collective_size, 1, 1, 1).to(device).type(
                        dtype)
            else:
                net_targets = torch.from_numpy(y_local).unsqueeze(dim=0).repeat(instance.collective_size, 1). \
                    to(device).type(dtype)
            if instance.loss_fn == 'MSE':
                if problem_specs['Out_spikes']:
                    loss = (net_outputs - net_targets) ** 2
                    if out_shape == 1:
                        loss = torch.mean(torch.sum(loss, dim=2), dim=1)
                    else:
                        loss = torch.mean(torch.sum(loss, dim=(2, 3)), dim=1)
                else:
                    loss = torch.mean(loss_fn(net_outputs.float(), net_targets.float()), dim=1)
            else:
                net_outputs_prop = log_softmax_fn(net_outputs).swapaxes(0, 2).swapaxes(0, 1)
                net_targets = net_targets.swapaxes(0, 1)
                loss = torch.mean(loss_fn(net_outputs_prop.float(), net_targets.type(torch.LongTensor).to(device)),
                                  dim=0)
        else:
            pdf_local = torch.from_numpy(pdf_local).to(device).type(dtype)
            net_targets = torch.from_numpy(y_local).unsqueeze(dim=0).repeat(instance.collective_size, 1). \
                to(device).type(dtype)
            tmp_loss_mse = torch.mean(loss_fn(net_outputs_n.float(), net_targets.float()), dim=1)
            net_out = net_outputs_n

            size_c, size_b, size_t = net_outputs_s.shape
            net_outputs_s = net_outputs_s.view(size_c, type_inputs, -1, size_t)
            temp_sump = torch.sum(net_outputs_s, dim=2)

            if ep == 0:
                loss_mse = tmp_loss_mse[None, :]
                sump = temp_sump
            else:
                loss_mse = torch.concatenate((loss_mse, tmp_loss_mse[None, :]), dim=0)
                sump += temp_sump
    if problem_specs['Gaussian'][0]:
        # out_prop = sump / (torch.sum(sump, dim=2).unsqueeze(dim=2) + 1e-6)
        # kl_div = torch.sum(torch.log((pdf_local[None, :, :] / (out_prop + 1e-5)) + 1e-6) * pdf_local[None, :, :], dim=2)
        # loss_kl = torch.sum(kl_div, dim=1)
        # loss = 0.01*loss_kl + torch.mean(loss_mse, dim=0)
        loss = torch.mean(loss_mse, dim=0)
        print('Mini MSE Loss: ', torch.amin(torch.mean(loss_mse, dim=0)), torch.argmin(torch.mean(loss_mse, dim=0)))
        # print('Mini MSE Loss: ', torch.amin(torch.mean(loss_mse, dim=0)), 'Mini KL Loss: ', torch.amin(loss_kl))
    loss_d = loss.cpu().detach().numpy()

    if not problem_specs['Gaussian'][0]:
        loss_d = np.average(loss_d.reshape(repeat_r, -1), axis=0)
        instance.set_col_size(col_size)
        instance.set_weights(weights_ini)
        instance.set_conn(conn_ini)
        instance.set_tc(tc_ini)
        instance.set_bias(bias_ini)
        instance.set_delays(delays_ini)

    if loss_clip['Flag']:
        loss_d = np.clip(loss_d, a_min=loss_clip['Range'][0], a_max=loss_clip['Range'][1])
    return loss_d

