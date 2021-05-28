from collections import defaultdict
from typing import List


class CapacityManager2:

    def __init__(self, json_data):
        self.first_day, self.last_day = None, None
        self.machine_capacities = defaultdict(lambda: defaultdict(lambda: 0.0))
        for m in json_data['machines']:
            for ik, k in enumerate(m['capacities']):
                self.machine_capacities[str(m['id'])][ik] = k

    def get_max_capacity(self, machine):
        return min([c for c in self.machine_capacities[str(machine)].values() if c > 0])

    def get_capacity(self, machine, ordinal_day):
        machine = str(machine)
        return self.machine_capacities[machine][ordinal_day]


class Machine:
    def __init__(self, id, external: bool, day_zero):
        self.id = id
        self.external = external
        self.day_zero = day_zero
        assert type(day_zero) is int

    def max_capacity(self, cm):
        return cm.get_max_capacity(self.id)

    def capacity(self, time_step, cm):
        return cm.get_capacity(self.id, time_step + self.day_zero)

    def virtual_t(self, t, cm):
        if self.capacity(t, cm) == 0:
            return None
        else:
            o = 0
            for i in range(t):
                if self.capacity(i, cm) > 0:
                    o += 1
            return o

    def next_timestep(self, t, cm):
        i = 1
        while self.capacity(t + i, cm) == 0:
            i += 1
            if i > 1000:
                raise Exception(f'No capacity in next 1000 day for machine {self.id} and day {t + self.day_zero}')
        return t + i


class Job:
    def __init__(self, id, deadline: int, project: str):
        self.id = id
        self.deadline = deadline

        self.tasks = []
        self.external_timeout = 0

        self.project = project

        self.solver_delay = None

    @property
    def result_delay(self):
        return max(0, self.tasks[-1].result_processing_day - self.deadline) if len(self.tasks) > 0 else 0

    @property
    def min_length(self):
        return sum([t.free_days_before for t in self.tasks]) + len(self.tasks)


class Task:
    def __init__(self, job: Job, machine: Machine, length: int, earliest_start: int, free_days_before: int):
        self.job = job
        self.machine = machine

        self.length = length
        self.earliest_start = earliest_start

        self.result_processing_day = None
        self.solver_result_processing_day = None
        self.directly_after_last = False

        self.free_days_before = free_days_before

        self.id = None
        self.original_id = None
        self.job_id = None

        self.min_days_after = 0


def load_presolved(json_obj):
    tasks = []
    jobs = {}
    machines = {}
    for j in json_obj['jobs']:
        job = Job(j['id'], j['deadline'], 'PROJECT')
        jobs[job.id] = job
    for m in json_obj['machines']:
        machine = Machine(m['id'], m['external'], 0)
        machines[machine.id] = machine
    for t in json_obj['tasks']:
        task = Task(jobs[t['job']], machines[t['machine']], t['length'], t['earliest_start'], t['free_days_before'])
        task.id = t['id']
        task.original_id = str(['id'])
        task.min_days_after = 0
        task.job_id = t['job']
        task.directly_after_last = t['directly_after_last']
        task.earliest_start = t['earliest_start']
        tasks.append(task)
        for t_pre in task.job.tasks:
            t_pre.min_days_after += 1 + task.free_days_before
        jobs[t['job']].tasks.append(task)
    return tasks, machines.values(), jobs.values()


class PenaltyManager:
    def __init__(self, objective):
        self.project_weights = {a['project']: a['jobWeight'] for a in objective['projects']}
        self.default_weight = objective['jobWeight']
        self.pen_per_day = objective['penaltyPerDay']
        self.pen_per_job = objective['oneTimePenalty']

    def project(self, project):
        return self.project_weights[project] if project in self.project_weights else self.default_weight

def perf_measures(job_list: List[Job]):
    sum_delay = sum([job.result_delay for job in job_list])
    max_delay = max([job.result_delay for job in job_list])
    avg_delay = round(sum_delay / len(job_list), 2)
    return dict(sum_delay=sum_delay, max_delay=max_delay, avg_delay=avg_delay)