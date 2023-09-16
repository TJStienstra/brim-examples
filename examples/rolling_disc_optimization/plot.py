"""Script plotting the results of the rolling disc."""

from copy import copy

import cloudpickle
import matplotlib.pyplot as plt
import numpy as np
import sympy as sm
from brim.utilities.plotting import Plotter
from matplotlib.animation import FuncAnimation
from scipy.interpolate import CubicSpline

with open("data.pkl", "rb") as f:
    data = cloudpickle.load(f)


results = "optimization"  # "optimization" or "simulation"
make_animation = True

if results == "optimization":
    t_arr = data.solution.time
    x_arr = data.solution.state
    du_arr = data.solution.du
    c_arr = data.solution.loads
elif results == "simulation":
    t_arr = data.simulator.t
    if data.simulator.x.shape[0] == len(data.x):
        x_arr = data.simulator.x
    else:
        x_arr = data.simulator.x.T
    c_arr = np.array([
        [data.simulator.controls[fi](ti, x_arr[:, i]) for i, ti in enumerate(t_arr)]
        for fi in data.controllable_loads])
    du_arr = np.array([
        data.simulator.eval_rhs(t_arr[i], x_arr[:, i].flatten())[-len(data.system.u):]
        for i in range(len(t_arr))]).T
elif results in data:
    t_arr = data[results].time
    x_arr = data[results].state
    du_arr = data[results].du
    c_arr = data[results].loads
else:
    raise ValueError("Invalid results type")

x_eval = CubicSpline(t_arr, x_arr.T)

constraints = data.system.holonomic_constraints.col_join(
    data.system.nonholonomic_constraints).xreplace(data.system.eom_method.kindiffdict())
p, p_vals = zip(*data.constants.items())
eval_constraints = sm.lambdify((data.x, p), constraints, cse=True)
constr_arr = eval_constraints(x_arr, p_vals).reshape((len(constraints), len(t_arr)))

eval_eoms = sm.lambdify((data.x, data.du, p, data.controllable_loads), data.eoms,
                        cse=True)
eoms_arr = eval_eoms(x_arr, du_arr, p_vals, c_arr).reshape((len(data.eoms), len(t_arr)))

# Create target path coordinates
q1_path = np.linspace(float(data.initial_state_constraints[data.system.q[0]]),
                      float(data.final_state_constraints[data.system.q[0]]), 1000)
q2_path = sm.lambdify((data.system.q[0],), sm.solve(data.path, data.system.q[1])[0])(
    q1_path)

# Plot target path and solution trajectory in plan view
q1_arr = x_arr[0, :]
q2_arr = x_arr[1, :]
print("Mean tracking error:", abs(  # noqa: T201
    sm.lambdify(data.system.q[:2], data.path)(x_arr[0, :], x_arr[1, :])).mean())

fig, axs = plt.subplots(3, 1, sharex=True, gridspec_kw={"height_ratios": [1.2, 1, 1]},
                        figsize=(6.4, 6.4))
axs[0].plot(q1_path, q2_path, color="k", label="Target")
axs[0].plot(q1_arr, q2_arr, label="Solution")
axs[0].set_ylabel("q2 (m)")
axs[0].legend()
axs[0].set_aspect("equal", adjustable="box")
for i, state in enumerate(data.system.q[2:-1], start=2):
    axs[1].plot(q1_arr, x_arr[i], label=f"{state.name[-state.name[::-1].find('_'):]}")
axs[1].set_ylabel("Angle (rad)")
axs[1].legend()
for i, state in enumerate(data.system.u[:3], start=len(data.system.q)):
    axs[2].plot(q1_arr, x_arr[i], label=f"{state.name[-state.name[::-1].find('_'):]}")
axs[2].set_ylabel("Angular velocity (rad/s)")
axs[2].set_xlabel("q1 (m)")
axs[2].legend(loc="lower right")

plt.figure()
for i, load in enumerate(data.controllable_loads):
    plt.plot(q1_arr, c_arr[i], label=f"{load.name}")
plt.xlabel("q1 (m)")
plt.ylabel("Torque (Nm)")
plt.legend()

fig, axs = plt.subplots(2, 1, sharex=True)
for i in range(len(constraints)):
    axs[0].plot(q1_arr, constr_arr[i, :], label=f"{i}")
axs[0].set_ylabel("Constraint violation")
axs[0].legend()
for i in range(len(data.eoms)):
    axs[1].plot(q1_arr, eoms_arr[i, :], label=f"{i}")
axs[1].set_xlabel("q1 (m)")
axs[1].set_ylabel("EoM violation")
axs[1].legend()

p, p_vals = zip(*data.constants.items())
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))
ax.plot(q1_path, q2_path, np.zeros_like(q1_path), "k", label="Target")
ax.plot(q1_arr, q2_arr, np.zeros_like(q1_arr), "r", label="Solution")
n_frames = 10
plotter = Plotter.from_model(ax, data.model)
plotter.add_point(data.model.tire.contact_point, color="r")
plotter.lambdify_system((data.system.q[:] + data.system.u[:], p))
for i in range(n_frames):
    for artist in plotter.artists:
        artist.set_alpha(i * 1 / (n_frames + 1) + 1 / n_frames)
        ax.add_artist(copy(artist))
    plotter.evaluate_system(x_eval(i / (n_frames - 1) * t_arr[-1]), p_vals)
    plotter.plot()
for artist in plotter.artists:
    artist.set_alpha(1)
    ax.add_artist(copy(artist))
X, Y = np.meshgrid(np.arange(np.pi - 3.6, np.pi + 3.7, 0.3), np.arange(-1.5, 1.6, 0.3))
ax.plot_wireframe(X, Y, np.zeros_like(X), color="k", alpha=0.3, rstride=1, cstride=1)
ax.invert_zaxis()
ax.invert_yaxis()
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.view_init(19, 60)
ax.set_aspect("equal")
ax.axis("off")

if not make_animation:
    plt.show()
    exit()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))
ax.plot(q1_path, q2_path, np.zeros_like(q1_path), "k", label="Target")
plotter = Plotter.from_model(ax, data.model)
cp = plotter.add_point(data.model.tire.contact_point, color="r")
cp.visible = False
sol_line, = ax.plot([], [], [], "r", label="Solution")
plotter.lambdify_system((data.system.q[:] + data.system.u[:], p))
plotter.evaluate_system(x_arr[:, 0].flatten(), p_vals)
plotter.plot()
ax.plot_wireframe(X, Y, np.zeros_like(X), color="k", alpha=0.3, rstride=1, cstride=1)
ax.invert_zaxis()
ax.invert_yaxis()
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.view_init(19, 14)
ax.set_aspect("equal")
ax.axis("off")


def animate(fi):
    """Update the plot for frame i."""
    if fi == 0:
        sol_line.set_data_3d([], [], [])
    plotter.evaluate_system(x_eval(fi / (n_frames - 1) * t_arr[-1]), p_vals)
    sol_line.set_data_3d(*(
        np.append(dat, val) for dat, val in zip(sol_line._verts3d, cp.values[0][0])))
    return *plotter.update(), sol_line,


fps = 30
n_frames = int(fps * t_arr[-1])
ani = FuncAnimation(fig, animate, frames=n_frames, blit=False)
ani.save("animation.gif", dpi=150, fps=fps)
plt.show()
