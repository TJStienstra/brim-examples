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
steer_torque, disturbance = data.controllable_loads
bike_parametrization = data.bike_parametrization

bike_params = bp.Bicycle(bike_parametrization, pathToData=data_dir)
constants = bicycle.get_param_values(bike_params)
constants.update({
    symbols("g"): 9.81,
})
if bike_parametrization == "Fisher":
    # Rough estimation of missing parameters, most are only used for visualization.
    constants[bicycle.rear_frame.symbols["d4"]] = 0.41
    constants[bicycle.rear_frame.symbols["d5"]] = -0.57
    constants[bicycle.rear_frame.symbols["l_bbx"]] = 0.4
    constants[bicycle.rear_frame.symbols["l_bbz"]] = 0.18
    constants[bicycle.front_frame.symbols["d6"]] = 0.1
    constants[bicycle.front_frame.symbols["d7"]] = 0.3
    constants[bicycle.front_frame.symbols["d8"]] = -0.3
if "k" in bicycle.front_frame.symbols:
    constants[bicycle.front_frame.symbols["d9"]] = \
        constants[bicycle.front_frame.symbols["d3"]] / 2
    # Suspension spring and damper constants are the softest settings provided in:
    # http://dx.doi.org/10.13140/RG.2.2.26063.64162
    constants[bicycle.front_frame.symbols["k"]] = 19.4E3
    constants[bicycle.front_frame.symbols["c"]] = 9E3

initial_conditions = {xi: 0 for xi in system.q[:] + system.u[:]}
initial_conditions.update({
    bicycle.q[3]: 1E-5,
    bicycle.q[4]: 0.314,
    bicycle.u[0]: 3,
    bicycle.u[5]: -3 / constants[bicycle.rear_wheel.radius],
})
if bicycle.front_frame.q:
    bicycle.front_frame.q[0] = 0.002

# Introduce small numerical error on zero values to avoid possible numerical problems
initial_conditions = {xi: np.random.random() * 1E-14 if xval == 0 else xval for xi, xval
                      in initial_conditions.items()}

roll_rate_idx = len(system.q) + system.u[:].index(bicycle.u[3])
max_roll_rate = 0.2
controls = {
    steer_torque: lambda t, x: 10 * max(-1, min(x[roll_rate_idx] / max_roll_rate, 1)),
    disturbance: lambda t, x: (30 + 30 * t) * np.sin(t * 2 * np.pi),
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
