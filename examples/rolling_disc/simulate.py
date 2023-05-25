"""Script simulating the rolling disc based on the optimization outcome."""

import cloudpickle
from brimopt import Simulator
from scipy.interpolate import CubicSpline

filename = "data.pkl"
with open(filename, "rb") as f:
    data = cloudpickle.load(f)

data_set = data.solution

# Create a simulator object
simulator = Simulator(data.system)
simulator.constants = data.constants
simulator.initial_conditions = dict(zip(data.x, data_set.state[:, 0]))
simulator.controls = {
    fi: CubicSpline(data_set.time, data_set.loads[i, :])
    for i, fi in enumerate(data.controllable_loads)
}
simulator.initialize()
simulator.solve((0, data_set.time[-1]), t_eval=data_set.time)
data.simulator = simulator
with open(filename, "wb") as f:
    cloudpickle.dump(data, f)
