import os
import time
from collections import defaultdict
from typing import List

import Optimizer
import ray
import torch
import torch.nn.functional as F
import numpy as np
import wandb
from NetworkES import Network, _compute_job_priority

from instance_reader import Task, Machine, Job, CapacityManager2, perf_measures

wandb.init(project='es_strikes_back')
ray.init(ignore_reinit_error=True)


class Storage:
    alpha = None
    sigma = None
    adam = None

def solve_with_es(task_list: List[Task], machine_list: List[Machine], job_list: List[Job], setup, cm: CapacityManager2, obj_dv=None, instance_name='general'):
    config = dict()
    config['sigma'] = 0.04
    config['alpha'] = 0.05
    config['pop_size'] = 300
    machine_days = defaultdict(lambda: defaultdict(lambda: 0))

    pop_size = config['pop_size']

    iter_nb = 300

    # prepocess data
    bijection_job_id = dict()
    bijection_machine_id = dict()
    bijection_task_id = dict()
    number_jobs = 0
    number_machines = 0
    number_tasks = 0
    all_jobs = []
    for job in job_list:
        if len(job.tasks) > 0:
            all_jobs.append(job)
            bijection_job_id[job.id] = number_jobs
            number_jobs += 1
            for task in job.tasks:
                if task.machine.id not in bijection_machine_id:
                    bijection_machine_id[task.machine.id] = number_machines
                    number_machines += 1
                bijection_task_id[task.id] = number_tasks
                number_tasks += 1

    pm = PenaltyManager(setup['objective'])
    job_weights = torch.full((number_jobs, ), pm.default_weight, dtype=torch.double)
    deadlines = torch.zeros((number_jobs, ), dtype=torch.double)
    for job in job_list:
        if len(job.tasks) > 0:
            index_job = bijection_job_id[job.id]
            deadlines[index_job] = job.deadline
            if job.project in pm.project_weights:
                job_weights[index_job] = pm.project(job.project)

    torch.manual_seed(0)
    np.random.seed(0)

    pm_ray = ray.put(pm)
    deadlines_ray = ray.put(deadlines)
    cm_ray = ray.put(cm)
    bijection_job_id_ray = ray.put(bijection_job_id)
    bijection_task_id_ray = ray.put(bijection_task_id)
    all_jobs_ray = ray.put(all_jobs)
    job_weights_ray = ray.put(job_weights)
    job_list_ray = ray.put(job_list)

    print("----------Start evolutionary strategy----------")
    print(f"number of jobs to allocates {number_jobs}, total of {number_tasks} tasks to perform on {number_machines} machines")
    nb_features = 6
    best_solution = [float('inf'), None]
    start = time.time()
    i = 0
    with torch.no_grad():
        master_net = Network(nb_features, 1)
        master_net.train()
        if os.path.isfile('pretrained.pt'):
            master_net.load_state_dict(torch.load('pretrained.pt'))
            master_net.eval()
        if Storage.sigma is None:
            Storage.sigma = config['sigma']
            Storage.alpha = config['alpha']
            Storage.adam = Optimizer.Adam(master_net, Storage.alpha)
        adam = Storage.adam
        while ('timeLimit' in setup and start + setup['timeLimit'] > time.time()) or ('timeLimit' not in setup and i < iter_nb):
            alpha = Storage.alpha
            sigma = Storage.sigma
            master_weights = master_net.get_weights()
            noises = np.random.randn(pop_size, len(master_weights))
            noises[len(noises) // 2:] = -1 * noises[:len(noises) // 2]
            networks = [Network(nb_features, 1) for _ in range(pop_size)]
            for n in range(pop_size):
                networks[n].load_state_dict(master_net.state_dict())
                network_weights = master_weights + sigma * noises[n]
                networks[n].set_weights(network_weights)
            output_iter = ray.get([one_pop_iter.remote(pm_ray, deadlines_ray, cm_ray, bijection_job_id_ray, bijection_task_id_ray, number_jobs, number_tasks, all_jobs_ray, job_weights_ray, job_list_ray, networks[k]) for k in range(pop_size)])
            output_iter = np.array(output_iter, dtype=object)
            results = np.array(output_iter[:, 0], dtype=float)
            solutions = np.array(output_iter[:, 1], dtype=object)
            best_pop = np.argmin(results)
            if results[best_pop] < best_solution[0]:
                best_solution[0] = results[best_pop]
                best_solution[1] = solutions[best_pop]
            mean_result = np.mean(results)
            print(f"iteration {i} best solution found this iter {results[best_pop] + obj_dv}, with a mean of {mean_result + obj_dv}, best solution so far {best_solution[0] + obj_dv}")
            advantage = (results - mean_result) / np.std(results)
            weight_update = np.dot(noises.T, advantage)
            update_ratio = adam.update(weight_update)
            wandb.log({f'{instance_name}/best_iter_solution': results[best_pop] + obj_dv,
                       f'{instance_name}/mean_iter_solution': mean_result + obj_dv,
                       f'{instance_name}/best_solution': best_solution[0] + obj_dv,
                       'general/sigma': sigma,
                       'general/alpha': alpha,
                       'general/update_ratio': update_ratio,
                       })
            #momentum_update = momentum_friction * momentum_update + weight_update
            #master_net.set_weights(master_weights - ((alpha / (pop_size * sigma)) * momentum_update))
            master_net.eval()
            i += 1
    solver_result = np.array(best_solution[1], dtype=int)
    for job in job_list:
        for task in job.tasks:
            task_id = bijection_task_id[task.id]
            task.result_processing_day = int(solver_result[task_id])
            machine_days[task.result_processing_day][task.machine.id] += task.length

    for job in job_list:
        if len(job.tasks) > 0:
            job.solver_delay = max(0, job.tasks[-1].result_processing_day - job.deadline)
            job.solver_has_delay = 1 if job.solver_delay > 0 else 0
        else:
            job.solver_has_delay = 0
            job.solver_delay = 0
    p = perf_measures(job_list)
    obj = sum([pm.project(job.project) * pm.pen_per_day * job.solver_delay for job in job_list]) \
          + sum([pm.project(job.project) * pm.pen_per_job * job.solver_has_delay for job in job_list]) + obj_dv
    p['obj'] = obj
    p['tasks'] = [t.to_json() for t in task_list]
    torch.save(master_net.state_dict(), 'pretrained.pt')
    wandb.save('pretrained.pt')
    return dict(machine_days=machine_days, **p)



@ray.remote
def one_pop_iter(pm, deadlines, cm, bijection_job_id, bijection_task_id, number_jobs, number_tasks, all_jobs, job_weights, job_list, network):
    with torch.no_grad():
        machine_days = defaultdict(lambda: defaultdict(lambda: 0))

        # END of preprocess
        task_day_allocation = torch.zeros(number_tasks, dtype=torch.int)
        current_job_op = torch.zeros(number_jobs, dtype=torch.int)
        min_starts = torch.zeros(number_jobs, dtype=torch.int)
        nb_day_left = torch.zeros(number_jobs, dtype=torch.double)
        nb_day_outside_left = torch.zeros(number_jobs, dtype=torch.double)
        nb_coupling_day_left = torch.zeros(number_jobs, dtype=torch.double)
        all_task_length = torch.zeros(number_jobs, dtype=torch.double)
        for job in job_list:
            if len(job.tasks) > 0:
                job_nb = bijection_job_id[job.id]
                first_task = job.tasks[0]
                i = bijection_job_id[job.id]
                min_starts[i] = max(first_task.earliest_start, first_task.free_days_before)
                all_task_length[i] = first_task.length
                for task_id, task in enumerate(job.tasks):
                    nb_day_left[job_nb] += 1
                    nb_day_left[job_nb] += task.free_days_before
                    nb_day_outside_left[job_nb] += task.free_days_before
                    if task_id + 1 < len(job.tasks):
                        if job.tasks[task_id + 1].directly_after_last:
                            nb_coupling_day_left[job_nb] += 1
                        elif task.directly_after_last and not job.tasks[task_id + 1].directly_after_last:
                            nb_coupling_day_left[job_nb] += 1
                    elif task.directly_after_last:
                        nb_coupling_day_left[job_nb] += 1

        number_finished_job = 0
        day = 0
        #all_jobs_index = np.arange(number_jobs)
        # sorted_job_list = sorted(job_list, key=lambda j: j.deadline - len(j.tasks) - sum([t.free_days_before for t in j.tasks]))
        # order_jobs = [bijection_job_id[job.id] for job in sorted_job_list if len(job.tasks) > 0]
        not_finished_job = torch.ones((number_jobs, ), dtype=torch.bool)
        while number_finished_job < number_jobs:
            to_allocate_this_step = torch.logical_and(not_finished_job, torch.le(min_starts, day))
            job_to_allocate_this_step = torch.nonzero(to_allocate_this_step).view(-1)
            if job_to_allocate_this_step.shape[0] > 0:
                if len(job_to_allocate_this_step) == 1:
                    # just for performance, avoid a softmax on 2000+ job for only one action
                    order_jobs = job_to_allocate_this_step
                else:
                    priority_jobs = _compute_job_priority((deadlines - day)[job_to_allocate_this_step], nb_day_left[job_to_allocate_this_step], job_weights[job_to_allocate_this_step], nb_day_outside_left[job_to_allocate_this_step], nb_coupling_day_left[job_to_allocate_this_step], all_task_length[job_to_allocate_this_step],network)
                    probabilities = F.softmax(priority_jobs, dim=0)
                    #print(probabilities)
                    order_jobs = np.random.choice(job_to_allocate_this_step, size=len(job_to_allocate_this_step), replace=False, p=probabilities.numpy())
                for i in order_jobs:
                    job_object = all_jobs[i]
                    # check if compatible
                    # not finished
                    current_task_nb = current_job_op[i]
                    if to_allocate_this_step[i]:
                        current_task = job_object.tasks[current_task_nb]
                        task_length = current_task.length
                        needed_machine = current_task.machine.id
                        # machine capacity present
                        if machine_days[day][needed_machine] + task_length <= current_task.machine.capacity(day, cm):
                            can_allocate = True
                            k = 1
                            machine_day = day
                            all_tasks_to_allocate = {machine_day: current_task}
                            while can_allocate and current_task_nb + k < len(job_object.tasks) and job_object.tasks[
                                current_task_nb + k].directly_after_last:
                                machine_day = current_task.machine.next_timestep(
                                    machine_day + job_object.tasks[current_task_nb + k].free_days_before, cm)
                                if machine_days[machine_day][needed_machine] + job_object.tasks[
                                    current_task_nb + k].length > current_task.machine.capacity(machine_day, cm):
                                    can_allocate = False
                                all_tasks_to_allocate[machine_day] = job_object.tasks[current_task_nb + k]
                                k += 1
                            if can_allocate:
                                for day_allocation in all_tasks_to_allocate:
                                    current_task_allocation = all_tasks_to_allocate[day_allocation]
                                    task_bijection = bijection_task_id[current_task_allocation.id]
                                    task_day_allocation[task_bijection] = day_allocation
                                    machine_days[day_allocation][current_task.machine.id] += current_task_allocation.length
                                    nb_day_left[i] -= current_task_allocation.free_days_before
                                    nb_day_outside_left[i] -= current_task_allocation.free_days_before
                                current_job_op[i] += len(all_tasks_to_allocate)
                                if len(all_tasks_to_allocate) > 1:
                                    nb_coupling_day_left[i] -= len(all_tasks_to_allocate)
                                nb_day_left[i] -= len(all_tasks_to_allocate)
                                if current_job_op[i] == len(job_object.tasks):
                                    number_finished_job += 1
                                    not_finished_job[i] = False
                                else:
                                    last_task = job_object.tasks[current_job_op[i] - 1]
                                    next_task = job_object.tasks[current_job_op[i]]

                                    task_bijection = bijection_task_id[last_task.id]
                                    min_starts[i] = max(next_task.earliest_start,
                                                        task_day_allocation[task_bijection] + next_task.free_days_before + 1)
                                    all_task_length[i] = next_task.length
            day += 1

        delay_jobs = np.zeros(len(job_list), dtype=int)
        for i, job in enumerate(job_list):
            if len(job.tasks) > 0:
                delay_jobs[i] = max(0, task_day_allocation[bijection_task_id[job.tasks[-1].id]] - job.deadline)
        obj = sum([pm.project(job.project) * pm.pen_per_day * delay_jobs[i] for i, job in enumerate(job_list)]) + sum([pm.project(job.project) * pm.pen_per_job * (delay_jobs[i] > 0) for i, job in enumerate(job_list)])
        return obj, task_day_allocation
