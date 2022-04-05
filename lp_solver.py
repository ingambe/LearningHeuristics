import multiprocessing
from collections import defaultdict

from pulp import *

from instance_reader import *


class GurobiSolver:
    def __init__(self):
        import gurobipy as gp

        self.m = gp.Model("mip1")

    def int_var(self, min, max, name):
        from gurobipy import GRB

        return self.m.addVar(lb=min, ub=max, vtype=GRB.INTEGER, name=name)

    def float_var(self, min, max, name):
        from gurobipy import GRB

        return self.m.addVar(lb=min, ub=max, vtype=GRB.CONTINUOUS, name=name)

    def constraint(self, what, name):
        return self.m.addConstr(what, name=name)

    def solve(self, time_limit):
        # self.m.Params.MIPGap = 0.01
        self.m.Params.TimeLimit = time_limit
        return self.m.optimize()

    def minimize(self, what):
        from gurobipy import GRB

        return self.m.setObjective(what, GRB.MINIMIZE)

    def value(self, v):
        return v.X

    def set_default_value(self, v, val):
        v.start = val

    def get_objective_value(self):
        return self.m.getObjective().getValue()

    @property
    def mip_gap(self):
        return self.m.MIPGap

    @property
    def mip_gap_abs(self):
        return self.m.getObjective().getValue() - self.m.ObjBound


class SolverInterface:
    def __init__(self):
        self.m = LpProblem("Kostwein", LpMinimize)

    def int_var(self, min, max, name):
        return LpVariable(name, min, max, cat="Integer")

    def constraint(self, what, name=None):
        assert len(name) <= 255, name
        self.m.addConstraint(what, name)

    def solve(self, solver=None):
        return self.m.solve(solver)

    def minimize(self, what):
        self.m += what

    def value(self, v):
        return v.varValue

    def set_default_value(self, v, val):
        v.setInitialValue(val)

    def get_objective_value(self):
        return value(self.m.ObjBound)


def solve_with_lp(
    task_list: List[Task],
    machine_list: List[Machine],
    job_list: List[Job],
    setup,
    cm: CapacityManager2,
    solver_backend=None,
    obj_dv=0,
):
    from time import time

    start = time()
    solver = GurobiSolver()

    machine_time_task_sum = defaultdict(lambda: defaultdict(lambda: []))
    warm_start = "warmStart" in setup and setup["warmStart"]

    # every task(part) is started exactly once
    for task in task_list:
        task.solver_vars_per_timestep = {}
        # start date in real time: consecutive constraints
        task.solver_assigned_start = 0
        # start date in "machine time" (non-working days not computed): no interruption of task
        task.solver_assigned_start_machine_time = 0
        job_length = len(task.job.tasks) + sum(
            [t.free_days_before for t in task.job.tasks]
        )
        for t in range(
            task.earliest_start,
            task.job.tasks[-1].result_processing_day + 10 - task.min_days_after,
        ):
            if task.machine.capacity(t, cm):
                v = solver.int_var(0, 1, f"Task {task.id} start at {t}")
                if warm_start:
                    solver.set_default_value(
                        v, 1 if task.result_processing_day == t else 0
                    )
                machine_time_task_sum[t][task.machine.id].append(
                    {
                        "var": v,
                        "sum": task.length,
                        "machine": task.machine,
                        "t": t,
                    }
                )
                task.solver_vars_per_timestep[t] = v
                task.solver_assigned_start += v * t
                task.solver_assigned_start_machine_time += v * task.machine.virtual_t(
                    t, cm
                )
        solver.constraint(
            sum(task.solver_vars_per_timestep.values()) == 1,
            name=f"Perform task {task.id} once",
        )

    # no more than 24 hrs of work on each day
    for t1 in machine_time_task_sum.values():
        for t in t1.values():
            m: Machine = t[0]["machine"]
            time_step = t[0]["t"]
            solver.constraint(
                sum([a["sum"] * a["var"] for a in t]) <= m.capacity(time_step, cm),
                f"Max work capaciy for machine {m.id} and time step {time_step}",
            )

    # consecutive constraints
    for job in job_list:
        last: Task = None
        for task in job.tasks:
            if last is not None:
                solver.constraint(
                    last.solver_assigned_start
                    <= task.solver_assigned_start - 1 - task.free_days_before,
                    name=f"Consecutive 1 for task {task.id}",
                )
            if task.directly_after_last:
                assert last is not None
                solver.constraint(
                    last.solver_assigned_start_machine_time + 1
                    == task.solver_assigned_start_machine_time,
                    name=f"Consecutive 2 for task {task.id}",
                )
            last = task

    # compute delay
    for job in job_list:
        if len(job.tasks) > 0:
            # assert not job.tasks[-1].machine.external
            big_number = 3000
            job.solver_delay = solver.int_var(0, big_number, f"Job {job.id} delay")
            if warm_start:
                solver.set_default_value(job.solver_delay, job.result_delay)
            solver.constraint(
                job.solver_delay >= job.tasks[-1].solver_assigned_start - job.deadline,
                name=f"delay for job {job.id}",
            )

            job.solver_has_delay = solver.int_var(0, 1, f"Job {job.id} delay")
            if warm_start:
                solver.set_default_value(
                    job.solver_has_delay, 1 if job.result_delay > 0 else 0
                )
            solver.constraint(
                job.solver_has_delay * big_number >= job.solver_delay,
                f"has delay for job {job.id}",
            )
        else:
            job.solver_has_delay = 0
            job.solver_delay = 0

    project_weight = {
        a["project"]: a["jobWeight"] for a in setup["objective"]["projects"]
    }
    default_job_weight = setup["objective"]["jobWeight"]
    solver.minimize(
        sum(
            [
                (
                    project_weight[job.project]
                    if job.project in project_weight
                    else default_job_weight
                )
                * setup["objective"]["penaltyPerDay"]
                * job.solver_delay
                for job in job_list
            ]
        )
        + sum(
            [
                (
                    project_weight[job.project]
                    if job.project in project_weight
                    else default_job_weight
                )
                * setup["objective"]["oneTimePenalty"]
                * job.solver_has_delay
                for job in job_list
            ]
        )
        + obj_dv
    )
    solver.solve(time_limit=setup["timeLimit"])

    machine_days = defaultdict(lambda: defaultdict(lambda: 0))

    for task in task_list:
        for t, v in task.solver_vars_per_timestep.items():
            if int(solver.value(v)) == 1:
                # task.result_processing_day = t
                machine_days[t][task.machine.id] += task.length

    p = perf_measures(job_list)
    p["obj"] = solver.get_objective_value()
    print("Solve objective", solver.get_objective_value())
    p["termination_time"] = time() - start
    p["mip_gap"] = solver.mip_gap
    p["mip_gap_abs"] = solver.mip_gap_abs
    p["num_vars"] = solver.m.getAttr("NumVars")
    p["num_constrs"] = solver.m.getAttr("NumConstrs")
    return dict(machine_days=machine_days, **p)
