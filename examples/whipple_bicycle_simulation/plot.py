"""Visualize the results of the simulation."""
from copy import copy

import cloudpickle
import matplotlib.pyplot as plt
import numpy as np
import sympy as sm
from brim.utilities.plotting import Plotter
from matplotlib.animation import FuncAnimation
from scipy.interpolate import CubicSpline
from symmeplot import PlotBody

with open("data.pkl", "rb") as f:
    data = cloudpickle.load(f)

make_animation = True

t_arr = data.simulator.t
x_arr = data.simulator.x
c_arr = np.array([[data.simulator.controls[fi](ti, x_arr[:, i])
                   for i, ti in enumerate(t_arr)] for fi in data.controllable_loads])

# Compute loads from actuators
p, p_vals = zip(*data.constants.items())
x = data.system.q[:] + data.system.u[:]
c = data.controllable_loads
loads = {
    "steer_torque": c_arr[c.index(data.controllable_loads[0]), :],
    "disturbance": c_arr[c.index(data.controllable_loads[1]), :],
}
x_eval = CubicSpline(t_arr, x_arr.T)
c_eval = CubicSpline(t_arr, c_arr.T)

constraints = data.system.holonomic_constraints.col_join(
    data.system.nonholonomic_constraints).xreplace(data.system.eom_method.kindiffdict())
eval_constraints = sm.lambdify((x, p), constraints, cse=True)
constr_arr = eval_constraints(x_arr, p_vals).reshape((len(constraints), len(t_arr)))

qs = {"yaw": data.model.q[2], "roll": data.model.q[3], "steer": data.model.q[6]}
us = {"yaw": data.model.u[2], "roll": data.model.u[3], "steer": data.model.u[6]}
get_q = lambda q_name: x_arr[data.system.q[:].index(qs[q_name]), :]  # noqa: E731
get_u = lambda u_name: x_arr[len(data.system.q) +   # noqa: E731
                             data.system.u[:].index(us[u_name]), :]
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 7))
ax0twin, ax1twin, ax2twin = axs[0].twinx(), axs[1].twinx(), axs[2].twinx()
axs[0].plot(t_arr, loads["disturbance"], color="C0")
axs[0].set_ylabel("Disturbance force (N)", color="C0")
axs[0].tick_params(axis="y", labelcolor="C0")
ax0twin.plot(t_arr, loads["steer_torque"], color="C1")
ax0twin.set_ylabel("Steer torque (Nm)", color="C1")
ax0twin.tick_params(axis="y", labelcolor="C1")
axs[1].plot(t_arr, get_q("yaw"), label="yaw", color="C0")
axs[1].plot(t_arr, get_q("steer"), label="steer", color="C2")
ax1twin.plot(t_arr, get_q("roll"), label="roll", color="C1")
axs[1].set_ylabel("Angle (rad)")
ax1twin.set_ylabel("Angle (rad)", color="C1")
ax1twin.tick_params(axis="y", labelcolor="C1")
ax1twin.legend([plt.Line2D([0], [0], color=f"C{i}") for i in range(3)],
               ["yaw", "roll", "steer"])
axs[2].plot(t_arr, get_u("yaw"), label="yaw", color="C0")
axs[2].plot(t_arr, get_u("steer"), label="steer", color="C2")
ax2twin.plot(t_arr, get_u("roll"), label="roll", color="C1")
axs[2].set_ylabel("Angular velocity (rad/s)")
ax2twin.set_ylabel("Angular velocity (rad/s)", color="C1")
ax2twin.tick_params(axis="y", labelcolor="C1")
ax2twin.legend([plt.Line2D([0], [0], color=f"C{i}") for i in range(3)],
               ["yaw", "roll", "steer"])
fig.tight_layout()

plt.figure()
u1_arr = x_arr[len(data.system.q) + data.system.u[:].index(data.model.u[0]), :]
u2_arr = x_arr[len(data.system.q) + data.system.u[:].index(data.model.u[1]), :]
plt.plot(t_arr, np.sqrt(u1_arr ** 2 + u2_arr ** 2))
plt.xlabel("Time (s)")
plt.ylabel("Forward velocity (m/s)")

if data.model.front_frame.q:
    plt.figure()
    q_comp_arr = x_arr[data.system.q[:].index(data.model.front_frame.q[0]), :]
    plt.plot(t_arr, 1E3 * q_comp_arr)
    plt.xlabel("Time (s)")
    plt.ylabel("Compression (mm)")

plt.figure()
for i in range(len(constraints)):
    plt.plot(t_arr, constr_arr[i, :], label=f"{i}")
plt.xlabel("Time (s)")
plt.ylabel("Constraint violation")
plt.legend()

disturbance_plot_info = {
    "vector": c[1] / c_arr[1, :].max() * data.model.rear_frame.wheel_hub.axis,
    "origin": data.model.rear_frame.saddle.point, "name": "disturbance", "color": "r"}
p, p_vals = zip(*data.constants.items())
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))
n_frames = 6
plotter = Plotter.from_model(ax, data.model)
# plotter.add_vector(**disturbance_plot_info)  # Is not copy compatible.
plotter.lambdify_system((data.system.q[:] + data.system.u[:], c, p))
for plot_object in plotter.plot_objects:
    if isinstance(plot_object, PlotBody):
        plot_object.plot_frame.visible = False
        plot_object.plot_masscenter.visible = False
plotter.evaluate_system(x_arr[:, 0], c_arr[:, 0], p_vals)
for i in range(n_frames):
    for artist in plotter.artists:
        artist.set_alpha(i * 1 / (n_frames + 1) + 1 / n_frames)
        ax.add_artist(copy(artist))
    time = i / (n_frames - 1) * t_arr[-1]
    plotter.evaluate_system(x_eval(time), c_eval(time), p_vals)
    plotter.update()
for artist in plotter.artists:
    artist.set_alpha(1)
    ax.add_artist(copy(artist))
q1_arr = x_arr[data.system.q[:].index(data.model.q[0]), :]
q2_arr = x_arr[data.system.q[:].index(data.model.q[1]), :]
front_contact_coord = data.model.front_tyre.contact_point.pos_from(
    plotter.origin).to_matrix(plotter.inertial_frame)[:2]
eval_fc = sm.lambdify((data.system.q[:] + data.system.u[:], p), front_contact_coord,
                      cse=True)
fc_arr = np.array(eval_fc(x_arr, p_vals))
x_lim = min((fc_arr[0, :].min(), q1_arr.min())), max((fc_arr[0, :].max(), q1_arr.max()))
y_lim = min((fc_arr[1, :].min(), q2_arr.min())), max((fc_arr[1, :].max(), q2_arr.max()))
X, Y = np.meshgrid(np.arange(x_lim[0] - 1, x_lim[1] + 1, 0.5),
                   np.arange(y_lim[0] - 1, y_lim[1] + 1, 0.5))
ax.plot_wireframe(X, Y, np.zeros_like(X), color="k", alpha=0.3, rstride=1, cstride=1)
ax.invert_zaxis()
ax.invert_yaxis()
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.view_init(30, 30)
ax.set_aspect("equal")
ax.axis("off")

if not make_animation:
    plt.show()
    exit()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))
plotter = Plotter.from_model(ax, data.model)
plotter.add_vector(**disturbance_plot_info)
plotter.lambdify_system((data.system.q[:] + data.system.u[:], c, p))
plotter.evaluate_system(x_arr[:, 0], c_arr[:, 0], p_vals)
plotter.plot()
X, Y = np.meshgrid(np.arange(x_lim[0] - 1, x_lim[1] + 1, 0.5),
                   np.arange(y_lim[0] - 1, y_lim[1] + 1, 0.5))
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
    time = fi / (n_frames - 1) * t_arr[-1]
    plotter.evaluate_system(x_eval(time), c_eval(time), p_vals)
    return *plotter.update(),


fps = 30
n_frames = int(fps * t_arr[-1])
ani = FuncAnimation(fig, animate, frames=n_frames, blit=False)
ani.save("animation.gif", dpi=150, fps=fps)
plt.show()
