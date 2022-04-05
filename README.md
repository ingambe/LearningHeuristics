# Reinforcement Learning of Dispatching Strategies for Large-Scale Industrial Scheduling

This repository contains the code for the paper "Reinforcement Learning of Dispatching Strategies for Large-Scale Industrial Scheduling"

## Install dependencies

```bash
pip install -r requirements.txt
```

If you want to run the linear programming solver, you will need to get and install a Gurobi license.
For academics, you can request a free [one-year Gurobi license](https://www.gurobi.com/academia/academic-program-and-licenses/).

## Run benchmark

The solver, time-limit and instances used are defined in the `run_benchmarks.py` file.
Once you set up the requested experiments parameters, you can simply run the benchmark script:

```bash
python run_benchmarks.py
```

The results are available in the file `results.txt` and the file `stats.txt`.
