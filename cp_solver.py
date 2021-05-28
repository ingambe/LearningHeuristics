import collections
import multiprocessing as mp
from collections import defaultdict
from math import floor

from ortools.sat.python import cp_model

from instance_reader import *


def hours_to_minutes(hours):
    return int(floor(hours * 60))


def solve_with_cp(task_list: List[Task], machine_list: List[Machine], job_list: List[Job], setup, cm: CapacityManager2, obj_dv=0):
    from time import time
    start = time()
    job_list = sorted(job_list, key=lambda j: j.deadline)

    FAKE_ID = "12345678987654321"

    MAX_DELAY = 20

    model = cp_model.CpModel()

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple('task_type', 'start end interval height')
    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple('assigned_task_type',
                                                'start job index duration')
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)
    machine_bijection = dict()

    # compute meta-data
    oneTimePenalty = setup['objective']['oneTimePenalty']
    penaltyPerDay = setup['objective']['penaltyPerDay']
    default_weight = setup['objective']['jobWeight']
    warm_start = 'warmStart' in setup and setup['warmStart']
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

    project_weight = {a['project']: a['jobWeight'] for a in setup['objective']['projects']}
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
            start_var = model.NewIntVar(task.earliest_start, task.job.tasks[-1].result_processing_day + 10 - task.min_days_after, task.original_id)
            end_var = model.NewIntVar(task.earliest_start,  task.job.tasks[-1].result_processing_day + 10 - task.min_days_after, task.original_id)
            interval_var = model.NewIntervalVar(start_var, duration, end_var, f"interval_{task.original_id}")
            # model.Add(start_var == task.result_processing_day)
            if warm_start:
                model.AddHint(start_var, task.result_processing_day)
            taskType = task_type(start=start_var,
                                 end=end_var,
                                 interval=interval_var,
                                 height=task.length)
            all_tasks[job.id, task.id] = taskType
            machine_to_intervals[task.machine.id].append(taskType)
            machine_bijection[task.machine.id] = task.machine
            all_starts.append(start_var)
            all_ends.append(end_var)
            max_end_date = max(max_end_date, task.result_processing_day + MAX_DELAY + 1)
            machine_max_end_date[task.machine.id] = max(machine_max_end_date[task.machine.id],
                                                        task.result_processing_day + MAX_DELAY + 1)
            if num_task == len(job.tasks) - 1:
                all_final_tasks.append(start_var)

    max_machine_capacity = collections.defaultdict(int)
    waiting_days_machine = collections.defaultdict(list)
    for machine in machine_to_intervals:
        machine_object = machine_bijection[machine]
        for day in range(machine_max_end_date[machine]):
            waiting_days_machine[machine].append(machine_object.next_timestep(day, cm) - (day + 1))
            max_machine_capacity[machine] = max(max_machine_capacity[machine], machine_object.capacity(day, cm))

    for machine in machine_to_intervals:
        machine_task_type = machine_to_intervals[machine]
        intervals = []
        demands = []
        for task_type in machine_task_type:
            intervals.append(task_type.interval)
            demands.append(hours_to_minutes(task_type.height))
        machine_object = machine_bijection[machine]
        for day in range(machine_max_end_date[machine]):
            to_remove = abs(machine_object.capacity(day, cm) - max_machine_capacity[machine])
            if to_remove > 0:
                interval_var = model.NewIntervalVar(day, 1, day + 1, f"{FAKE_ID}_{day}_{machine}")
                intervals.append(interval_var)
                demands.append(hours_to_minutes(to_remove))
        model.AddCumulative(intervals, demands, hours_to_minutes(max_machine_capacity[machine]))

    for job in job_list:
        for i in range(1, len(job.tasks)):
            previous = job.tasks[i - 1]
            task = job.tasks[i]
            waiting_days = model.NewIntVar(0, max(waiting_days_machine[task.machine.id]), "end")
            model.AddElement(all_tasks[job.id, previous.id].start, waiting_days_machine[task.machine.id], waiting_days)
            if task.directly_after_last:
                model.Add(all_tasks[job.id, task.id].start == all_tasks[job.id, previous.id].end + waiting_days)
            else:
                waiting_time = model.NewIntVar(0,
                                               max(max(waiting_days_machine[task.machine.id]), task.free_days_before),
                                               "end")
                model.AddMaxEquality(waiting_time, [model.NewConstant(task.free_days_before), waiting_days])
                model.Add(all_tasks[job.id, task.id].start >= all_tasks[job.id, previous.id].end + waiting_time)

    job_late = [model.NewBoolVar("job_late") for _ in range(number_jobs)]
    weighted_one_time_penality = [model.NewIntVar(0, oneTimePenalty * max_default_weight, "weighted_delay") for _ in
                                  range(number_jobs)]
    sum_weighted_one_time_penality = model.NewIntVar(0, oneTimePenalty * max_default_weight * number_jobs,
                                                     "sum_weighted_one_time_penality")
    model.Add(sum(weighted_one_time_penality) == sum_weighted_one_time_penality)
    for job in job_list:
        if len(job.tasks) > 0:
            index_job = bijection_job_id[job.id]
            last_task = job.tasks[-1]
            last_task_job = all_tasks[job.id, last_task.id]
            model.Add(last_task_job.end <= job.deadline).OnlyEnforceIf(job_late[index_job].Not())
            model.Add(last_task_job.end > job.deadline).OnlyEnforceIf(job_late[index_job])
            model.Add(weighted_one_time_penality[index_job] == (
                    job_late[index_job] * job_weights[index_job] * oneTimePenalty))

    deviation_job = [model.NewIntVar(-(max_task_job * MAX_DELAY), max_task_job * MAX_DELAY, "devation") for _ in
                     range(number_jobs)]
    delay_job = [model.NewIntVar(0, max_task_job * MAX_DELAY, "devation") for _ in range(number_jobs)]
    weighted_delay = [model.NewIntVar(0, max_task_job * MAX_DELAY * max_default_weight, "weighted_delay") for _ in
                      range(number_jobs)]
    sum_weighted_delay = model.NewIntVar(0, max_task_job * MAX_DELAY * max_default_weight * number_jobs,
                                         "sum_weighted_delay")
    model.Add(sum(weighted_delay) == sum_weighted_delay)
    for job in job_list:
        if len(job.tasks) > 0:
            index_job = bijection_job_id[job.id]
            last_task = job.tasks[-1]
            last_task_job = all_tasks[job.id, last_task.id]
            model.Add(deviation_job[index_job] == (last_task_job.end - job.deadline))
            model.AddMaxEquality(delay_job[index_job], [0, deviation_job[index_job]])
            model.Add(weighted_delay[index_job] == (delay_job[index_job] * job_weights[index_job] * penaltyPerDay))

    objective = model.NewIntVar(0, (oneTimePenalty * max_default_weight * number_jobs) + (
            max_task_job * MAX_DELAY * max_default_weight * number_jobs), "objective")
    model.Add(sum_weighted_delay + sum_weighted_one_time_penality == objective)
    model.AddDecisionStrategy(all_final_tasks, cp_model.CHOOSE_FIRST, cp_model.SELECT_MIN_VALUE)
    model.AddDecisionStrategy(all_starts, cp_model.CHOOSE_FIRST, cp_model.SELECT_MIN_VALUE)
    model.Minimize(objective)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = mp.cpu_count()
    solver.parameters.max_time_in_seconds = setup['timeLimit']
    print("Start solving")
    status = solver.Solve(model)

    machine_days = defaultdict(lambda: defaultdict(lambda: 0))
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"We found something {solver.ObjectiveValue()}!")
        print(f"Weighted sum delay {solver.Value(sum_weighted_delay)}")
        print(f"Weighted one time penalty {solver.Value(sum_weighted_one_time_penality)}")
        for job in job_list:
            for task in job.tasks:
                cp_task = all_tasks[job.id, task.id]
                task.solver_result_processing_day = solver.Value(cp_task.start)
                machine_days[task.solver_result_processing_day][task.machine.id] += task.length
            if len(job.tasks) > 0:
                index_job = bijection_job_id[job.id]
                has_delay = solver.Value(job_late[index_job])
                deviation = (job.tasks[-1].solver_result_processing_day + 1) - job.deadline
                delay = max(0, deviation)
                # print(f"delay {delay} cp detect has delay {has_delay} of {solver.Value(delay_job[index_job])}")

        for job in job_list:
            if len(job.tasks) > 0:
                job.solver_delay = max(0, job.tasks[-1].solver_result_processing_day - job.deadline)
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

    from instance_reader import perf_measures
    p = perf_measures(job_list)
    project_weight = {a['project']: a['jobWeight'] for a in setup['objective']['projects']}
    default_job_weight = setup['objective']['jobWeight']
    obj = sum([(project_weight[job.project] if job.project in project_weight else default_job_weight) * \
               setup['objective']['penaltyPerDay'] * job.solver_delay for job in job_list]) \
          + sum([(project_weight[job.project] if job.project in project_weight else default_job_weight) * \
                 setup['objective']['oneTimePenalty'] * job.solver_has_delay for job in job_list])
    p['obj'] = obj + obj_dv
    # assert obj == solver.ObjectiveValue(), f'{obj} computed and {solver.ObjectiveValue()} cp-optimized objectives not equal'
    p['termination_time'] = time() - start
    return dict(machine_days=machine_days, **p)
