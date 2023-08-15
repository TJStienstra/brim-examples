"""Set the initial conditions for the Whipple bicycle with an upper body rider."""
import os
import sys

import bicycleparameters as bp
import cloudpickle
import numpy as np
from sympy import symbols

data_dir = [os.path.join(pt, "data") for pt in sys.path if pt[-7:] == "brimopt"][0]

with open("data.pkl", "rb") as f:
    data = cloudpickle.load(f)
system = data.system
bicycle = data.model
steer_torque = data.controllable_loads[0]

bike_params = bp.Bicycle("Browser", pathToData=data_dir)
bike_params.parameters.pop("Benchmark")
constants = bicycle.get_param_values(bike_params)
constants.update({
    symbols("g"): 9.81,
    bicycle.symbols["gear_ratio"]: np.random.random() * 1E-10,
    **{sym: np.random.random() * 1E-10 for sym in bicycle.pedals.symbols.values()},
})

initial_conditions = {xi: 0 for xi in system.q[:] + system.u[:]}
initial_conditions.update({
    bicycle.q[3]: 0.2,
    bicycle.q[4]: 0.314,
    bicycle.u[0]: 3,
    bicycle.u[5]: -3 / constants[bicycle.rear_wheel.radius],
})

# Introduce small numerical error on zero values to avoid possible numerical problems
initial_conditions = {xi: np.random.random() * 1E-14 if xval == 0 else xval for xi, xval
                      in initial_conditions.items()}

roll_rate_idx = len(system.q) + system.u[:].index(bicycle.u[3])
max_roll_rate = 0.2
controls = {
    steer_torque: lambda t, x: 10 * max(-1, min(x[roll_rate_idx] / max_roll_rate, 1))
}

data.update({
    "x": system.q.col_join(system.u),
    "constants": constants,
    "initial_conditions": initial_conditions,
    "controllable_loads": tuple(controls.keys()),
    "controls": controls,
})

with open("data.pkl", "wb") as f:
    cloudpickle.dump(data, f)
