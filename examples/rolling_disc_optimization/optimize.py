"""Script running the trajectory tracking optimization of the rolling disc."""

import cloudpickle
import numpy as np
import pycollo
import sympy as sm
import sympy.physics.mechanics as me
from scipy.interpolate import CubicSpline

from utilities import DataStorage

with open("data.pkl", "rb") as f:
    data = cloudpickle.load(f)

# Definitions of the parameters of the optimization problem
data.duration = 2.5
data.max_torque = 10
data.max_mean_tracking_error = 0.01

data.path = sm.sin(data.model.q[0]) * 1 - data.model.q[1]
data.path_objective = data.path ** 2
data.load_objective = sum(fi ** 2 for fi in data.controllable_loads)
data.use_multi_objective = True
data.path_constraints_type = 0
# 0: Only eoms, enforce nh constraints only for the initial state
# 1: eoms[:len(u_ind)] + nh constraints, enforces nh constraints for all states
# 2: all eoms and nh constraints
data.expected_loads_integrand_value = 0.53
data.aimed_path_integrand_value = data.max_mean_tracking_error ** 2 * data.duration

data.x = data.system.q.col_join(data.system.u)
data.du = sm.Matrix([me.dynamicsymbols(f"d{ui.name}") for ui in data.system.u])
data.eoms = data.system.mass_matrix * data.du - data.system.forcing
data.nh_constraints = data.system.nonholonomic_constraints.xreplace(
    data.system.eom_method.kindiffdict())

data.initial_state_constraints = {data.system.q[0]: 0.0, data.system.q[4]: 0.0}
data.final_state_constraints = {data.system.q[0]: 2 * sm.pi}

guess_data = None

t = me.dynamicsymbols._t

# Problem instantiation
problem = pycollo.OptimalControlProblem(name="Follow sinusoidal path")

# Phase definition
phase = problem.new_phase(name="A")
phase.state_variables = data.x
phase.control_variables = data.du.col_join(sm.Matrix(data.controllable_loads))
phase.state_equations = {
    **{qi: data.system.eom_method.kindiffdict()[qi.diff(t)] for qi in data.system.q},
    **dict(zip(data.system.u, data.du)),
}
if data.path_constraints_type == 0:
    phase.path_constraints = data.eoms
elif data.path_constraints_type == 1:
    phase.path_constraints = (
            data.eoms[:len(data.system.u_ind)] + data.nh_constraints[:])
elif data.path_constraints_type == 2:
    phase.path_constraints = data.eoms.col_join(data.nh_constraints)
else:
    raise ValueError("Invalid path constraints type")
phase.integrand_functions = [data.path_objective, data.load_objective]

# Outbound phase bounds
phase.bounds.initial_time = 0.0
phase.bounds.final_time = data.duration
phase.bounds.state_variables = {
    data.system.q[0]: [-10, 10],
    data.system.q[1]: [-10, 10],
    data.system.q[2]: [-20 * np.pi, 20 * np.pi],
    data.system.q[3]: [-np.pi, np.pi],
    data.system.q[4]: [-20 * np.pi, 20 * np.pi],
    **{ui: [-1000, 1000] for ui in data.system.u},
}
phase.bounds.control_variables = {
    **{dui: [-10_000, 10_000] for dui in data.du},
    **{Ti: [-data.max_torque, data.max_torque] for Ti in data.controllable_loads},
}
phase.bounds.integral_variables = [
    [0, data.duration if data.use_multi_objective else data.aimed_path_integrand_value],
    [0, len(data.controllable_loads) * data.duration * data.max_torque ** 2]]
phase.bounds.initial_state_constraints = data.initial_state_constraints
phase.bounds.final_state_constraints = data.final_state_constraints
phase.bounds.path_constraints = np.array([[0, 0] for _ in phase.path_constraints])

# Outbound phase guess
if guess_data is None:
    guess_data = DataStorage({
        "time": np.array([0, data.duration]),
        "state": np.array([
            [phase.bounds.initial_state_constraints.get(xi, 0) for xi in data.x],
            [phase.bounds.final_state_constraints.get(xi, 0) for xi in data.x]]).T,
        "loads": np.zeros((len(data.controllable_loads), 2)),
        "du": np.zeros((len(data.system.u), 2)),
    })
phase.guess.time = guess_data.time
phase.guess.state_variables = guess_data.state
phase.guess.control_variables = np.concatenate((guess_data.du, guess_data.loads))
phase.guess.integral_variables = np.array([0, 0])

# Problem definitions
if data.use_multi_objective:
    path_weight = data.expected_loads_integrand_value
    loads_weight = data.aimed_path_integrand_value
else:
    path_weight = 0
    loads_weight = 1
problem.objective_function = (path_weight * phase.integral_variables[0] +
                              loads_weight * phase.integral_variables[1])

problem.auxiliary_data = data.constants

periodic_constraints = [
    phase.final_state_variables[state.name] - phase.initial_state_variables[state.name]
    for state in (*data.system.q[1:4], *data.system.u)
]
if data.path_constraints_type == 0:
    problem.endpoint_constraints = periodic_constraints + data.nh_constraints.xreplace(
        {xi: phase.final_state_variables[xi.name] for xi in data.x})[:]
elif data.path_constraints_type in (1, 2):
    problem.endpoint_constraints = periodic_constraints

# Problem bounds
problem.bounds.endpoint_constraints = [0] * len(problem.endpoint_constraints)

# Problem settings
phase.mesh.number_mesh_sections = 20
problem.settings.mesh_tolerance = 1e-3
problem.settings.nlp_tolerance = 1e-7
problem.settings.max_mesh_iterations = 1

# Solve
problem.initialise()
problem.solve()

t_arr = problem.solution._time_[0]
x_arr = problem.solution.state[0]
c_arr = problem.solution.control[0][-len(data.controllable_loads):, :]
p, p_vals = zip(*data.constants.items())
path, load = sm.lambdify((t, data.x, data.controllable_loads, p),
                         (data.path_objective, data.load_objective)
                         )(t_arr, x_arr, c_arr, p_vals)
tracking_term = CubicSpline(t_arr, path).integrate(0, t_arr[-1])
control_term = CubicSpline(t_arr, load).integrate(0, t_arr[-1])
print(f"Tracking term: {tracking_term}")  # noqa: T201
print(f"Control term: {control_term}")  # noqa: T201

data.solution = DataStorage(**{
    "path_weight": path_weight,
    "loads_weight": loads_weight,
    "objective": problem.solution.objective,
    "time": problem.solution._time_[0],
    "state": problem.solution.state[0],
    "du": problem.solution.control[0][:-len(data.controllable_loads), :],
    "loads": problem.solution.control[0][-len(data.controllable_loads):, :],
})

with open("data.pkl", "wb") as f:
    cloudpickle.dump(data, f)
