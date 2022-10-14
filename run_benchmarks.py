import importlib
import io
import json
from collections import defaultdict
from copy import deepcopy

from instance_reader import CapacityManager2, load_presolved

# Constants

solvers = [
    "greedy_solver:solve_with_greedy",
    #'lp_solver:solve_with_lp',
    #'cp_solver:solve_with_cp',
    "beam_search:beam_search",
    #"train_classification:generate_data",
]

time_limits = [
    60,
]

weeks = [
    2,
]

ref_solver = "greedy_solver:solve_with_greedy"


# generate statistics


def gen_stats(res_to_print):
    with io.open("stats.txt", "w") as f:
        f.write("Time limit\tWeeks\t")
        for solver in solvers:
            f.write(solver + "\t")
        f.write("\n")
        for time_limit in time_limits:
            for week in weeks:
                f.write(f"{time_limit}s\t")
                f.write(f"{week} weeks\t")
                ref = res_to_print[week][time_limit][ref_solver]
                solvers_ = sorted(res_to_print[week][time_limit].keys())
                for solver in solvers_:
                    values = []
                    for res_i, res in enumerate(res_to_print[week][time_limit][solver]):
                        values.append(res / ref[res_i])
                    from statistics import mean

                    if len(values) > 0:
                        f.write(f"{mean(values):.2f}\t")
                    else:
                        f.write("\t")
                f.write("\n")


# run experiments


def main():
    with io.open("results.txt", "w") as output:
        results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))

        for instance in range(1, 6 + 1):
            output.write("******************\n")
            for week in weeks:
                for time_limit in time_limits:
                    fn = f"instance_{week}week_{instance}.json"
                    json_obj = json.load(open(f"data/{fn}", "r"))
                    tasks, machines, jobs = load_presolved(json_obj)
                    cm = CapacityManager2(json_obj)
                    for solver in solvers:
                        info = deepcopy(json_obj["info"])
                        info["timeLimit"] = time_limit
                        info["warmStart"] = True
                        solver_module = importlib.import_module(solver.split(":")[0])
                        solver_fcn = getattr(solver_module, solver.split(":")[1])
                        out = solver_fcn(tasks, machines, jobs, info, cm)
                        if "mip_gap_abs" in out:
                            mg = str(out["mip_gap_abs"])
                            results[week][time_limit][solver + "_mip_gap"].append(
                                out["mip_gap_abs"]
                            )
                        else:
                            mg = ""
                        output.write(
                            solver + " " + fn + " " + str(out["obj"]) + "" + mg + "\n"
                        )
                        output.flush()
                        results[week][time_limit][solver].append(out["obj"])
                        gen_stats(results)


if __name__ == "__main__":
    main()
