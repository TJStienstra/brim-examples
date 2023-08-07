"""Forward dynamics simulation of the Whipple bicycle with an upper body rider model."""
import time

import cloudpickle
import numpy as np

from utilities import Simulator

filename = "data.pkl"
with open(filename, "rb") as f:
    data = cloudpickle.load(f)

print("Initialize simulator")  # noqa: T201
if "simulator" in data and True:
    simulator = data.simulator
    simulator.constants = data.constants
    simulator.initial_conditions = data.initial_conditions
    simulator.controls = data.controls
else:
    start = time.time()
    simulator = Simulator(data.system)
    simulator.constants = data.constants
    simulator.initial_conditions = data.initial_conditions
    simulator.controls = data.controls
    simulator.initialize(check_parameters=False)
    end = time.time()

    data.simulator = simulator
    with open(filename, "wb") as f:
        cloudpickle.dump(data, f)
print("Simulate")  # noqa: T201
time_arr = np.arange(0, 2.5, 0.01)
try:
    simulator.solve(time_arr, solver="dae", rtol=1e-3, atol=1e-6)
    print("Simulated with DAE solver")  # noqa: T201
except ImportError:
    start = time.time()
    simulator.solve((0, time_arr[-1]), t_eval=time_arr)
    end = time.time()
    print("Simulated with ODE solver")  # noqa: T201

data.simulator = simulator
with open(filename, "wb") as f:
    cloudpickle.dump(data, f)
