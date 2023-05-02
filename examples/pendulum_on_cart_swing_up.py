"""Optimal control problem for a pendulum on a cart swing-up.

This is a simple example of an optimal control problem for a pendulum on a cart with n
links. This program has not been optimized in any way, so there is a lot of room for
improvement. However it does nicely show a method on how to use KanesMethod in
combination with Pycollo.
"""
import matplotlib.pyplot as plt
import numpy as np
import pycollo
import sympy as sm
import sympy.physics.mechanics as me
from matplotlib.animation import FuncAnimation
from sympy.physics.mechanics.models import n_link_pendulum_on_cart

n = 2
masses = sm.symbols(f"m:{n + 1}")
lengths = sm.symbols(f"l:{n}")
g, t = sm.symbols("g t")
force = me.dynamicsymbols("F")

kane = n_link_pendulum_on_cart(n, cart_force=True, joint_torques=False)

origin = [p for p in kane.bodies[0].masscenter._pos_dict if p.name == "O"][0]
frame = [f for f in kane.bodies[0].masscenter._vel_dict if f.name == "I"][0]

controllable_loads = [force]

constants = {
    **{mi: 1.0 for mi in masses},
    **{li: 0.5 for li in lengths},
    g: 9.81,
}

F_max = 20.0
d_max = 2.0
d = 1.0
T = 2.0

du = sm.Matrix([me.dynamicsymbols(f"d{ui.name}") for ui in kane.u])
eoms = kane.mass_matrix * du - kane.forcing

# Problem instantiation
problem = pycollo.OptimalControlProblem(name="Pendulum swing-up")

# Phase definition
phase = problem.new_phase(name="A")
phase.state_variables = kane.q.col_join(kane.u)
phase.control_variables = sm.Matrix(controllable_loads).col_join(du)
phase.state_equations = {
    **{qi: kane.kindiffdict()[qi.diff(t)] for qi in kane.q},
    **dict(zip(kane.u, du)),
}
phase.path_constraints = eoms
phase.integrand_functions = [sum(fi ** 2 for fi in controllable_loads)]

# Outbound phase bounds
phase.bounds.initial_time = 0.0
phase.bounds.final_time = T
phase.bounds.state_variables = {
    **{state: [-100, 100] for state in phase.state_variables},
    **{kane.q[0]: [-d_max, d_max]}
}
phase.bounds.control_variables = {
    **{fi: [-F_max, F_max] for fi in controllable_loads},
    **{dui: [-100, 100] for dui in du},
}
phase.bounds.path_constraints = np.array([[0, 0] for _ in phase.path_constraints])
phase.bounds.integral_variables = [[0, 1_000]]
phase.bounds.initial_state_constraints = {kane.q[0]: 0,
                                          **{qi: np.pi for qi in kane.q[1:]},
                                          **{ui: 0 for ui in kane.u}}
phase.bounds.final_state_constraints = {kane.q[0]: d,
                                        **{qi: 0 for qi in kane.q[1:]},
                                        **{ui: 0 for ui in kane.u}}

# Outbound phase guess
phase.guess.time = np.array([phase.bounds.initial_time, phase.bounds.final_time])
state_variables_guess = {
    state: [phase.bounds.initial_state_constraints.get(state, 0),
            phase.bounds.final_state_constraints.get(state, 0)]
    for state in phase.state_variables
}
phase.guess.state_variables = np.array(
    [
        state_variables_guess[state] for state in phase.state_variables
    ]
)
phase.guess.control_variables = np.array(
    [
        [0, 0] for _ in phase.control_variables
    ]
)
phase.guess.integral_variables = np.array([0])

# Problem definitions
problem.objective_function = phase.integral_variables[0]
problem.auxiliary_data = constants

# Problem settings
problem.settings.mesh_tolerance = 1e-3
problem.settings.max_mesh_iterations = 10

# Solve
problem.initialise()
problem.solve()

# CHECK PATH CONSTRAINT
p, p_vals = zip(*constants.items())
eval_path_constraints = sm.lambdify((phase.state_variables, phase.control_variables, p),
                                    phase.path_constraints, cse=True)

plt.subplot(3, 1, 1)
plt.plot(problem.solution._time_[0], problem.solution.state[0][0], marker="x")
plt.ylabel("x (masses)")

plt.subplot(3, 1, 2)
plt.plot(problem.solution._time_[0], problem.solution.state[0][1], marker="x")
plt.ylabel("theta (rad)")

plt.subplot(3, 1, 3)
plt.plot(problem.solution._time_[0], problem.solution.control[0][0], marker="x")
plt.ylabel("force (N)")

coords = [
    p.masscenter.pos_from(origin).to_matrix(frame)[:2] for p in kane.bodies
]
p, p_vals = zip(*constants.items())
eval_coords = sm.lambdify((phase.state_variables, p), coords, cse=True)
coords_solution = np.array([
    eval_coords(problem.solution.state[0][:, i], p_vals)
    for i in range(problem.solution.state[0].shape[1])
])
fig = plt.figure()
ax = fig.add_subplot(aspect="equal")
initial_coords = eval_coords(
    [state_variables_guess[xi][0] for xi in phase.state_variables], p_vals)
final_coords = eval_coords(
    [state_variables_guess[xi][1] for xi in phase.state_variables], p_vals)
ax.plot(*np.array(initial_coords).T, color="red")
ax.plot(*np.array(final_coords).T, color="red")
line, = ax.plot(*coords_solution[0].T)
ax.set_xlim(-d_max, d_max)
max_length = sum(constants[li] for li in lengths)
ax.set_ylim(-0.1 - max_length, 0.1 + max_length)


def animate(i):
    """Animation function."""
    line.set_data(*coords_solution[i].T)
    return line,


n_frames = coords_solution.shape[0]
final_time = problem.solution._time_[0][-1]
fps = 60
if fps * final_time > n_frames:
    fps = int(n_frames / final_time)
ani = FuncAnimation(fig, animate, frames=np.arange(
    0, n_frames, int(n_frames / (fps * final_time))))
ani.save("pendulum.gif", fps=fps)
plt.show()
