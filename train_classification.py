from collections import defaultdict
from math import floor
from typing import List

import transformers

import torch
import torch.nn as nn
import numpy as np

import collections
import multiprocessing as mp
from collections import defaultdict
from math import floor

from ortools.sat.python import cp_model
from torch.utils.data import TensorDataset, DataLoader

from NetworkES import Network
from instance_reader import *

from instance_reader import Task, Machine, Job, CapacityManager2, perf_measures, PenaltyManager


def hours_to_minutes(hours):
    return int(floor(hours * 60))


def solve_with_cp_get_result(
        job_list: List[Job],
        setup,
        cm: CapacityManager2,
):
    job_list = sorted(job_list, key=lambda j: j.deadline)

    FAKE_ID = "12345678987654321"

    MAX_DELAY = 20

    model = cp_model.CpModel()

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple("task_type", "start end interval height")
    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple(
        "assigned_task_type", "start job index duration"
    )
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)
    machine_bijection = dict()

    # compute meta-data
    oneTimePenalty = setup["objective"]["oneTimePenalty"]
    penaltyPerDay = setup["objective"]["penaltyPerDay"]
    default_weight = setup["objective"]["jobWeight"]
    warm_start = "warmStart" in setup and setup["warmStart"]
    max_task_job = 0
    max_default_weight = default_weight

    all_starts = []
    all_ends = []
    all_final_tasks = []

    bijection_job_id = dict()
    number_jobs = 0
    for job in job_list:
        if len(job.tasks) > 0:
            bijection_job_id[job.id] = number_jobs
            number_jobs += 1
            max_task_job = max(len(job.tasks), max_task_job)

    project_weight = {
        a["project"]: a["jobWeight"] for a in setup["objective"]["projects"]
    }
    job_weights = [default_weight for _ in range(number_jobs)]
    for job in job_list:
        if len(job.tasks) > 0:
            index_job = bijection_job_id[job.id]
            if job.project in project_weight:
                job_weights[index_job] = project_weight[job.project]
                max_default_weight = max(max_default_weight, job_weights[index_job])

    max_end_date = 0

    first_tasks = []

    machine_max_end_date = collections.defaultdict(int)
    for job in job_list:
        for num_task, task in enumerate(job.tasks):
            duration = 1
            start_var = model.NewIntVar(
                task.earliest_start,
                task.job.tasks[-1].result_processing_day + 10 - task.min_days_after,
                task.original_id,
            )
            end_var = model.NewIntVar(
                task.earliest_start,
                task.job.tasks[-1].result_processing_day + 10 - task.min_days_after,
                task.original_id,
            )
            interval_var = model.NewIntervalVar(
                start_var, duration, end_var, f"interval_{task.original_id}"
            )
            # model.Add(start_var == task.result_processing_day)
            if warm_start:
                model.AddHint(start_var, task.result_processing_day)
            taskType = task_type(
                start=start_var, end=end_var, interval=interval_var, height=task.length
            )
            all_tasks[job.id, task.id] = taskType
            machine_to_intervals[task.machine.id].append(taskType)
            machine_bijection[task.machine.id] = task.machine
            all_starts.append(start_var)
            all_ends.append(end_var)
            max_end_date = max(max_end_date, task.result_processing_day + MAX_DELAY + 1)
            machine_max_end_date[task.machine.id] = max(
                machine_max_end_date[task.machine.id],
                task.result_processing_day + MAX_DELAY + 1,
            )
            if num_task == len(job.tasks) - 1:
                all_final_tasks.append(start_var)

    max_machine_capacity = collections.defaultdict(int)
    waiting_days_machine = collections.defaultdict(list)
    for machine in machine_to_intervals:
        machine_object = machine_bijection[machine]
        for day in range(machine_max_end_date[machine]):
            waiting_days_machine[machine].append(
                machine_object.next_timestep(day, cm) - (day + 1)
            )
            max_machine_capacity[machine] = max(
                max_machine_capacity[machine], machine_object.capacity(day, cm)
            )

    for machine in machine_to_intervals:
        machine_task_type = machine_to_intervals[machine]
        intervals = []
        demands = []
        for task_type in machine_task_type:
            intervals.append(task_type.interval)
            demands.append(hours_to_minutes(task_type.height))
        machine_object = machine_bijection[machine]
        for day in range(machine_max_end_date[machine]):
            to_remove = abs(
                machine_object.capacity(day, cm) - max_machine_capacity[machine]
            )
            if to_remove > 0:
                interval_var = model.NewIntervalVar(
                    day, 1, day + 1, f"{FAKE_ID}_{day}_{machine}"
                )
                intervals.append(interval_var)
                demands.append(hours_to_minutes(to_remove))
        model.AddCumulative(
            intervals, demands, hours_to_minutes(max_machine_capacity[machine])
        )

    for job in job_list:
        for i in range(1, len(job.tasks)):
            previous = job.tasks[i - 1]
            task = job.tasks[i]
            waiting_days = model.NewIntVar(
                0, max(waiting_days_machine[task.machine.id]), "end"
            )
            model.AddElement(
                all_tasks[job.id, previous.id].start,
                waiting_days_machine[task.machine.id],
                waiting_days,
            )
            if task.directly_after_last:
                model.Add(
                    all_tasks[job.id, task.id].start
                    == all_tasks[job.id, previous.id].end + waiting_days
                )
            else:
                waiting_time = model.NewIntVar(
                    0,
                    max(
                        max(waiting_days_machine[task.machine.id]),
                        task.free_days_before,
                    ),
                    "end",
                )
                model.AddMaxEquality(
                    waiting_time,
                    [model.NewConstant(task.free_days_before), waiting_days],
                )
                model.Add(
                    all_tasks[job.id, task.id].start
                    >= all_tasks[job.id, previous.id].end + waiting_time
                )

    job_late = [model.NewBoolVar("job_late") for _ in range(number_jobs)]
    weighted_one_time_penality = [
        model.NewIntVar(0, oneTimePenalty * max_default_weight, "weighted_delay")
        for _ in range(number_jobs)
    ]
    sum_weighted_one_time_penality = model.NewIntVar(
        0,
        oneTimePenalty * max_default_weight * number_jobs,
        "sum_weighted_one_time_penality",
    )
    model.Add(sum(weighted_one_time_penality) == sum_weighted_one_time_penality)
    for job in job_list:
        if len(job.tasks) > 0:
            index_job = bijection_job_id[job.id]
            last_task = job.tasks[-1]
            last_task_job = all_tasks[job.id, last_task.id]
            model.Add(last_task_job.end <= job.deadline).OnlyEnforceIf(
                job_late[index_job].Not()
            )
            model.Add(last_task_job.end > job.deadline).OnlyEnforceIf(
                job_late[index_job]
            )
            model.Add(
                weighted_one_time_penality[index_job]
                == (job_late[index_job] * job_weights[index_job] * oneTimePenalty)
            )

    deviation_job = [
        model.NewIntVar(
            -(max_task_job * MAX_DELAY), max_task_job * MAX_DELAY, "devation"
        )
        for _ in range(number_jobs)
    ]
    delay_job = [
        model.NewIntVar(0, max_task_job * MAX_DELAY, "devation")
        for _ in range(number_jobs)
    ]
    weighted_delay = [
        model.NewIntVar(
            0, max_task_job * MAX_DELAY * max_default_weight, "weighted_delay"
        )
        for _ in range(number_jobs)
    ]
    sum_weighted_delay = model.NewIntVar(
        0,
        max_task_job * MAX_DELAY * max_default_weight * number_jobs,
        "sum_weighted_delay",
    )
    model.Add(sum(weighted_delay) == sum_weighted_delay)
    for job in job_list:
        if len(job.tasks) > 0:
            index_job = bijection_job_id[job.id]
            last_task = job.tasks[-1]
            last_task_job = all_tasks[job.id, last_task.id]
            model.Add(deviation_job[index_job] == (last_task_job.end - job.deadline))
            model.AddMaxEquality(delay_job[index_job], [0, deviation_job[index_job]])
            model.Add(
                weighted_delay[index_job]
                == (delay_job[index_job] * job_weights[index_job] * penaltyPerDay)
            )

    objective = model.NewIntVar(
        0,
        (oneTimePenalty * max_default_weight * number_jobs)
        + (max_task_job * MAX_DELAY * max_default_weight * number_jobs),
        "objective",
    )
    model.Add(sum_weighted_delay + sum_weighted_one_time_penality == objective)
    model.AddDecisionStrategy(
        all_final_tasks, cp_model.CHOOSE_FIRST, cp_model.SELECT_MIN_VALUE
    )
    model.AddDecisionStrategy(
        all_starts, cp_model.CHOOSE_FIRST, cp_model.SELECT_MIN_VALUE
    )
    model.Minimize(objective)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = mp.cpu_count()
    solver.parameters.max_time_in_seconds = setup["timeLimit"]
    print("Start solving")
    status = solver.Solve(model)

    machine_days = defaultdict(lambda: defaultdict(lambda: 0))
    day_job = defaultdict(lambda: list())
    task_day = defaultdict(lambda: int)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"We found something {solver.ObjectiveValue()}!")
        print(f"Weighted sum delay {solver.Value(sum_weighted_delay)}")
        print(
            f"Weighted one time penalty {solver.Value(sum_weighted_one_time_penality)}"
        )
        for job in job_list:
            for task in job.tasks:
                cp_task = all_tasks[job.id, task.id]
                task.solver_result_processing_day = solver.Value(cp_task.start)
                day_job[task.solver_result_processing_day].append((job.id, task.id))
                task_day[task.id] = task.solver_result_processing_day
                machine_days[task.solver_result_processing_day][
                    task.machine.id
                ] += task.length

        for job in job_list:
            if len(job.tasks) > 0:
                job.solver_delay = max(
                    0, job.tasks[-1].solver_result_processing_day - job.deadline
                )
                job.solver_has_delay = 1 if job.solver_delay > 0 else 0
            else:
                job.solver_delay = 0
                job.solver_has_delay = 0

    elif status == cp_model.INFEASIBLE:
        print("Infeasible model")
    elif status == cp_model.UNKNOWN:
        print("Unknown")
    elif status == cp_model.MODEL_INVALID:
        print("Invalid model!")
    else:
        print(f"Status is {status}")

    return day_job, task_day


def compute_input_tensor(
        nb_day_deadline,
        nb_day_left,
        job_weights,
        days_outside,
        coupling_days,
        task_length,
):
    with torch.no_grad():
        nb_jobs = len(nb_day_deadline)
        # normalize
        nb_day_deadline = (nb_day_deadline - torch.min(nb_day_deadline)) / max(
            torch.max(nb_day_deadline) - torch.min(nb_day_deadline), 1
        )
        nb_day_left = (nb_day_left - torch.min(nb_day_left)) / max(
            torch.max(nb_day_left) - torch.min(nb_day_left), 1
        )
        days_outside = (days_outside - torch.min(days_outside)) / max(
            torch.max(days_outside) - torch.min(days_outside), 1
        )
        coupling_days = (coupling_days - torch.min(coupling_days)) / max(
            torch.max(coupling_days) - torch.min(coupling_days), 1
        )
        job_weights = (job_weights - torch.min(job_weights)) / max(
            torch.max(job_weights) - torch.min(job_weights), 1
        )
        task_length = (task_length - torch.min(task_length)) / max(
            torch.max(task_length) - torch.min(task_length), 1
        )
        # view
        nb_day_deadline = nb_day_deadline.view(nb_jobs, 1)
        nb_day_left = nb_day_left.view(nb_jobs, 1)
        job_weights = job_weights.view(nb_jobs, 1)
        coupling_days = coupling_days.view(nb_jobs, 1)
        days_outside = days_outside.view(nb_jobs, 1)
        task_length = task_length.view(nb_jobs, 1)
        input_tensor = torch.hstack(
            (
                nb_day_deadline,
                nb_day_left,
                job_weights,
                days_outside,
                coupling_days,
                task_length,
            )
        )
        return input_tensor


def generate_data(
        task_list: List[Task],
        machine_list: List[Machine],
        job_list: List[Job],
        setup,
        cm: CapacityManager2,
        obj_dv=0,
        instance_name="general",
):
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

    pm = PenaltyManager(setup["objective"])
    job_weights = torch.full((number_jobs,), pm.default_weight, dtype=torch.double)
    deadlines = torch.zeros((number_jobs,), dtype=torch.double)
    for job in job_list:
        if len(job.tasks) > 0:
            index_job = bijection_job_id[job.id]
            deadlines[index_job] = job.deadline
            if job.project in pm.project_weights:
                job_weights[index_job] = pm.project(job.project)

    torch.manual_seed(0)
    np.random.seed(0)

    print("----------Start generating data----------")
    print(
        f"number of jobs to allocates {number_jobs}, total of {number_tasks} tasks to perform on {number_machines} machines"
    )

    days_allocation, task_day = solve_with_cp_get_result(job_list, setup, cm)
    result = one_pop_iter(
        pm,
        deadlines,
        cm,
        bijection_job_id,
        bijection_task_id,
        number_jobs,
        number_tasks,
        all_jobs,
        job_weights,
        job_list,
        days_allocation,
        task_day
    )
    return None


def one_pop_iter(
        pm,
        deadlines,
        cm,
        bijection_job_id,
        bijection_task_id,
        number_jobs,
        number_tasks,
        all_jobs,
        job_weights,
        job_list,
        days_allocation,
        task_day
):
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
                min_starts[i] = max(
                    first_task.earliest_start, first_task.free_days_before
                )
                all_task_length[i] = first_task.length
                for task_id, task in enumerate(job.tasks):
                    nb_day_left[job_nb] += 1
                    nb_day_left[job_nb] += task.free_days_before
                    nb_day_outside_left[job_nb] += task.free_days_before
                    if task_id + 1 < len(job.tasks):
                        if job.tasks[task_id + 1].directly_after_last:
                            nb_coupling_day_left[job_nb] += 1
                        elif (
                                task.directly_after_last
                                and not job.tasks[task_id + 1].directly_after_last
                        ):
                            nb_coupling_day_left[job_nb] += 1
                    elif task.directly_after_last:
                        nb_coupling_day_left[job_nb] += 1

        number_finished_job = 0
        day = 0
        not_finished_job = torch.ones((number_jobs,), dtype=torch.bool)
        all_job_representation = []
        all_labels = []
        all_loss_weights = []
        while day <= max(days_allocation.keys()):
            print(f"day {day}, {number_finished_job} jobs finished total of {number_jobs}")
            to_allocate_this_step = torch.logical_and(
                not_finished_job, torch.le(min_starts, day)
            )
            job_to_allocate_this_step = torch.nonzero(to_allocate_this_step).view(-1)
            machine_to_allocate = defaultdict(lambda: list())
            task_to_machine_to_allocate = defaultdict(lambda: list())
            for job_nb in job_to_allocate_this_step:
                job = all_jobs[job_nb]
                task = job.tasks[current_job_op[job_nb]]
                if task_day[task.id] >= max(day - 7, 0):
                    machine_to_allocate[task.machine.id].append(job_nb)
                    task_to_machine_to_allocate[task.machine.id].append(task.id)
            for machine_id, machine_job_list in machine_to_allocate.items():
                job_to_allocate_this_step_machine = torch.tensor(machine_job_list, dtype=torch.long)
                job_to_allocate_this_step_machine_copy = job_to_allocate_this_step_machine.clone()
                if job_to_allocate_this_step_machine.shape[0] > 0:
                    job_representation = compute_input_tensor(
                        (deadlines - day)[job_to_allocate_this_step_machine],
                        nb_day_left[job_to_allocate_this_step_machine],
                        job_weights[job_to_allocate_this_step_machine],
                        nb_day_outside_left[job_to_allocate_this_step_machine],
                        nb_coupling_day_left[job_to_allocate_this_step_machine],
                        all_task_length[job_to_allocate_this_step_machine],
                    )
                    # sort per job_representation[:, 0], select the first 100
                    indexes_sort_deadline = torch.argsort(job_representation[:, 0])[:100]
                    job_to_allocate_this_step_machine = job_to_allocate_this_step_machine[indexes_sort_deadline]
                    job_representation = job_representation[indexes_sort_deadline, :]

                    all_job_representation.append(job_representation)
                    order_jobs = days_allocation[day]
                    get_biject_to_allocate = [bijection_job_id[x[0]] for x in order_jobs if
                                              bijection_job_id[x[0]] in job_to_allocate_this_step_machine]
                    #print(f"get_biject_to_allocate {len(get_biject_to_allocate)}")
                    get_biject_to_allocate.sort()
                    y_to_pred = torch.zeros((job_to_allocate_this_step_machine.shape[0],), dtype=torch.bool)
                    for idx, job_candidate in enumerate(job_to_allocate_this_step_machine):
                        if job_candidate.item() in get_biject_to_allocate:
                            y_to_pred[idx] = True
                    assert y_to_pred.sum() == len(get_biject_to_allocate), f'{y_to_pred.sum()} not the same {len(get_biject_to_allocate)}'
                    #for i, job in enumerate(get_biject_to_allocate):
                    #    if
                    #y_to_pred[job_to_allocate_this_step_machine == torch.tensor(get_biject_to_allocate)] = True
                    all_labels.append(y_to_pred)

                    this_day_machine_loss_weight = []
                    for idx_task, job_day_machine in enumerate(task_to_machine_to_allocate[machine_id]):
                        if idx_task in indexes_sort_deadline:
                            this_day_machine_loss_weight.append((task_day[job_day_machine] - day) ** 2)
                    this_day_machine_loss_weight = torch.tensor(this_day_machine_loss_weight, dtype=torch.double)
                    all_loss_weights.append(this_day_machine_loss_weight)
                    assert y_to_pred.shape[0] == this_day_machine_loss_weight.shape[0], f'{y_to_pred.shape[0]} not the same {this_day_machine_loss_weight.shape[0]}'
                    #print('-' * 10)
                    #print(f'shape of labels {y_to_pred.shape}')
                    #print(f'shape of representation {job_representation.shape}')
                    #print(f'-' * 10)
                    get_biject_to_allocate_without_filtering = [bijection_job_id[x[0]] for x in order_jobs if
                                              bijection_job_id[x[0]] in job_to_allocate_this_step_machine_copy]
                    for i in get_biject_to_allocate_without_filtering:
                        job_object = all_jobs[i]
                        # check if compatible
                        # not finished
                        current_task_nb = current_job_op[i]
                        if to_allocate_this_step[i]:
                            current_task = job_object.tasks[current_task_nb]
                            task_length = current_task.length
                            needed_machine = current_task.machine.id
                            # machine capacity present
                            if machine_days[day][
                                needed_machine
                            ] + task_length <= current_task.machine.capacity(day, cm):
                                can_allocate = True
                                k = 1
                                machine_day = day
                                all_tasks_to_allocate = {machine_day: current_task}
                                while (
                                        can_allocate
                                        and current_task_nb + k < len(job_object.tasks)
                                        and job_object.tasks[
                                            current_task_nb + k
                                        ].directly_after_last
                                ):
                                    machine_day = current_task.machine.next_timestep(
                                        machine_day
                                        + job_object.tasks[
                                            current_task_nb + k
                                            ].free_days_before,
                                        cm,
                                    )
                                    if machine_days[machine_day][
                                        needed_machine
                                    ] + job_object.tasks[
                                        current_task_nb + k
                                    ].length > current_task.machine.capacity(
                                        machine_day, cm
                                    ):
                                        can_allocate = False
                                    all_tasks_to_allocate[machine_day] = job_object.tasks[
                                        current_task_nb + k
                                        ]
                                    k += 1
                                if can_allocate:
                                    for day_allocation in all_tasks_to_allocate:
                                        current_task_allocation = all_tasks_to_allocate[
                                            day_allocation
                                        ]
                                        task_bijection = bijection_task_id[
                                            current_task_allocation.id
                                        ]
                                        task_day_allocation[task_bijection] = day_allocation
                                        machine_days[day_allocation][
                                            current_task.machine.id
                                        ] += current_task_allocation.length
                                        nb_day_left[
                                            i
                                        ] -= current_task_allocation.free_days_before
                                        nb_day_outside_left[
                                            i
                                        ] -= current_task_allocation.free_days_before
                                    current_job_op[i] += len(all_tasks_to_allocate)
                                    if len(all_tasks_to_allocate) > 1:
                                        nb_coupling_day_left[i] -= len(
                                            all_tasks_to_allocate
                                        )
                                    nb_day_left[i] -= len(all_tasks_to_allocate)
                                    if current_job_op[i] == len(job_object.tasks):
                                        number_finished_job += 1
                                        not_finished_job[i] = False
                                    else:
                                        last_task = job_object.tasks[current_job_op[i] - 1]
                                        next_task = job_object.tasks[current_job_op[i]]

                                        task_bijection = bijection_task_id[last_task.id]
                                        min_starts[i] = max(
                                            next_task.earliest_start,
                                            task_day_allocation[task_bijection]
                                            + next_task.free_days_before
                                            + 1,
                                        )
                                        all_task_length[i] = next_task.length
            day += 1

        delay_jobs = np.zeros(len(job_list), dtype=int)
        for i, job in enumerate(job_list):
            if len(job.tasks) > 0:
                delay_jobs[i] = max(
                    0,
                    task_day_allocation[bijection_task_id[job.tasks[-1].id]]
                    - job.deadline,
                )
        obj = sum(
            [
                pm.project(job.project) * pm.pen_per_day * delay_jobs[i]
                for i, job in enumerate(job_list)
            ]
        ) + sum(
            [
                pm.project(job.project) * pm.pen_per_job * (delay_jobs[i] > 0)
                for i, job in enumerate(job_list)
            ]
        )
        # padding all all_job_representation and all_reps tensor with zero to have the same shape[0] size
        # create a mask to remove the padding
        max_len = max([x.shape[0] for x in all_job_representation])
        all_reps = torch.zeros((len(all_job_representation), max_len, all_job_representation[0].shape[1]))
        all_padded_labels = torch.zeros((len(all_labels), max_len), dtype=torch.bool)
        all_padded_weights = torch.zeros((len(all_loss_weights), max_len), dtype=torch.bool)
        mask = torch.zeros((len(all_job_representation), max_len), dtype=torch.bool)
        for i, (x, y) in enumerate(zip(all_job_representation, all_labels)):
            all_reps[i, :x.shape[0], :] = x
            all_padded_labels[i, :y.shape[0]] = y
            all_padded_weights[i, :y.shape[0]] = all_loss_weights[i] + 1
            mask[i, :x.shape[0]] = True
        #print(f'all_reps {all_reps.shape}')
        #print(f'all_padded_labels {all_padded_labels.shape}')
        #print(f'mask {mask.shape}')
        dataset = TensorDataset(all_reps, all_padded_labels, all_padded_weights, mask)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    ex_nn = Network(6, 1)
    number_training_step = 200
    # learning rate scheduler warmup
    optimizer = torch.optim.AdamW(ex_nn.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
    lr_scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=20, num_training_steps=number_training_step
    )
    # criterion multi label cross entropy, higher weight for positive label
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    #criterion = nn.MSELoss(reduction='none')
    for epoch in range(1000):
        acc = 0
        losses = 0
        total_precision = 0
        total_recall = 0
        iter = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0
        for batch in dataloader:
            optimizer.zero_grad()
            reps, labels, mask, weight_loss = batch
            output = ex_nn(reps).squeeze(2)
            loss = criterion(output, labels.float())
            loss = (loss * mask.float() * weight_loss.float()).sum() / mask.float().sum()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            iter += 1
            with torch.no_grad():
                # compute the accuracy
                pred = torch.sigmoid(output) > 0.5
                acc += (pred[mask] == labels[mask]).float().sum() / mask.float().sum()
                losses += loss.item()
                # compute the precision and recall
                tp = ((pred == 1) & (labels == 1)).float().sum()
                fp = ((pred == 1) & (labels == 0)).float().sum()
                fn = ((pred == 0) & (labels == 1)).float().sum()
                tn = ((pred == 0) & (labels == 0)).float().sum()
                total_precision += tp / ((tp + fp) + 1e-8)
                total_recall += tp / ((tp + fn) + 1e-8)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn
        print(f'epoch {epoch} loss {losses / iter} acc {acc / iter} precision {total_precision / iter} recall {total_recall / iter}')
        print(f'tp {total_tp / iter} fp {total_fp / iter} fn {total_fn / iter} tn {total_tn / iter}')
    # save the model
    torch.save(ex_nn.state_dict(), 'pretrained.pt')
    return all_reps, all_padded_labels, mask

