import numpy as np

from instance_reader import *


def _compute_job_priority(
    nb_day_deadline, nb_day_left, job_weights, penaltyPerDay, oneTimePenalty
):
    left_over_deadline = nb_day_deadline - nb_day_left
    score = np.zeros(nb_day_left.shape)
    score[left_over_deadline >= 0] = np.exp(
        left_over_deadline[left_over_deadline >= 0]
    ) / (oneTimePenalty + penaltyPerDay)
    score[left_over_deadline < 0] = (
        np.exp(nb_day_left[left_over_deadline < 0]) / penaltyPerDay
    )
    # score[left_over_deadline > 1] = max(score) + left_over_deadline[left_over_deadline > 1]
    score /= job_weights
    return score


def solve_with_greedy(
    task_list: List[Task],
    machine_list: List[Machine],
    job_list: List[Job],
    setup,
    cm,
    obj_dv=0,
):
    machine_days = defaultdict(lambda: defaultdict(lambda: 0))

    bijection_job_id = dict()
    bijection_machine_id = dict()
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
            number_tasks += len(job.tasks)

    pm = PenaltyManager(setup["objective"])
    default_weight = pm.default_weight
    job_weights = [default_weight for _ in range(number_jobs)]
    for job in job_list:
        if len(job.tasks) > 0:
            index_job = bijection_job_id[job.id]
            if job.project in pm.project_weights:
                job_weights[index_job] = pm.project(job.project)

    current_job_op = np.zeros(number_jobs, dtype=int)
    deadlines = np.zeros(number_jobs, dtype=int)
    min_starts = np.zeros(number_jobs, dtype=int)
    nb_day_left = np.zeros(number_jobs, dtype=int)
    for job in job_list:
        if len(job.tasks) > 0:
            job_nb = bijection_job_id[job.id]
            deadlines[job_nb] = job.deadline
            first_task = job.tasks[0]
            i = bijection_job_id[job.id]
            min_starts[i] = max(first_task.earliest_start, first_task.free_days_before)
            for task in job.tasks:
                nb_day_left[job_nb] += 1
                nb_day_left[job_nb] += task.free_days_before

    number_finished_job = 0
    day = 0
    # sorted_job_list = sorted(job_list, key=lambda j: j.deadline - len(j.tasks) - sum([t.free_days_before for t in j.tasks]))
    # order_jobs = [bijection_job_id[job.id] for job in sorted_job_list if len(job.tasks) > 0]
    while number_finished_job < number_jobs:
        priority_jobs = _compute_job_priority(
            deadlines - day, nb_day_left, job_weights, pm.pen_per_day, pm.pen_per_job
        )
        order_jobs = np.argsort(priority_jobs)
        for i in order_jobs:
            job_object = all_jobs[i]
            # check if compatible
            # not finished
            current_task_nb = current_job_op[i]
            if current_task_nb < len(job_object.tasks):
                current_task = job_object.tasks[current_task_nb]
                if min_starts[i] <= day:
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
                                current_task_allocation.result_processing_day = (
                                    day_allocation
                                )
                                machine_days[day_allocation][
                                    current_task.machine.id
                                ] += current_task_allocation.length
                                nb_day_left[
                                    i
                                ] -= current_task_allocation.free_days_before
                            current_job_op[i] += len(all_tasks_to_allocate)
                            nb_day_left[i] -= len(all_tasks_to_allocate)
                            if current_job_op[i] == len(job_object.tasks):
                                number_finished_job += 1
                            else:
                                last_task = job_object.tasks[current_job_op[i] - 1]
                                next_task = job_object.tasks[current_job_op[i]]
                                min_starts[i] = max(
                                    next_task.earliest_start,
                                    last_task.result_processing_day
                                    + next_task.free_days_before
                                    + 1,
                                )
        day += 1

    for job in job_list:
        for task in job.tasks:
            assert task.result_processing_day is not None
        for task_nb in range(1, len(job.tasks)):
            assert (
                job.tasks[task_nb - 1].result_processing_day
                < job.tasks[task_nb].result_processing_day
                - job.tasks[task_nb].free_days_before
            )
        if len(job.tasks) > 0:
            job.greedy_delay = max(
                0, job.tasks[-1].result_processing_day - job.deadline
            )
            job.greedy_has_delay = 1 if job.greedy_delay > 0 else 0
        else:
            job.greedy_delay = 0
            job.greedy_has_delay = 0

    for k1, v1 in machine_days.items():
        for k2, v2 in v1.items():
            cap = [m for m in machine_list if m.id == k2][0].capacity(k1, cm)
            if v2 > cap:
                print(f"Cost more than {cap} for day {k1} and machine {k2} = {v2}")
    p = perf_measures(job_list)
    obj = (
        sum(
            [
                pm.project(job.project) * pm.pen_per_day * job.greedy_delay
                for job in job_list
            ]
        )
        + sum(
            [
                pm.project(job.project) * pm.pen_per_job * job.greedy_has_delay
                for job in job_list
            ]
        )
        + obj_dv
    )
    p["obj"] = obj
    return dict(machine_days=machine_days, **p)


def get_min_obj(
    task_list: List[Task], machine_list: List[Machine], job_list: List[Job], setup, cm
):
    machine_days = defaultdict(lambda: defaultdict(lambda: 0))

    for j in job_list:
        if len(j.tasks) > 0:
            j.min_obj_delay = max(
                0,
                max(0, j.tasks[0].earliest_start)
                - j.deadline
                + len(j.tasks)
                - 1
                + sum([t.free_days_before for t in j.tasks]),
            )
            j.min_obj_has_delay = 1 if j.min_obj_delay > 0 else 0
        else:
            j.min_obj_delay = 0
            j.min_obj_has_delay = 0

    # consider machine downtime
    p = {}
    pm = PenaltyManager(setup["objective"])
    obj = sum(
        [
            pm.project(job.project) * pm.pen_per_day * job.min_obj_delay
            for job in job_list
        ]
    ) + sum(
        [
            pm.project(job.project) * pm.pen_per_job * job.min_obj_has_delay
            for job in job_list
        ]
    )
    p["obj"] = obj
    return dict(machine_days=machine_days, **p)
