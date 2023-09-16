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
from sympy.physics.mechanics import dynamicsymbols

with open("data.pkl", "rb") as f:
    data = cloudpickle.load(f)

make_animation = True

t = dynamicsymbols._t
t_arr = data.simulator.t
x = data.system.q[:] + data.system.u[:]
x_arr = data.simulator.x
c = tuple(data.controllable_loads)
c_arr = np.array([[data.simulator.controls[fi](ti, x_arr[:, i])
                  for i, ti in enumerate(t_arr)] for fi in data.controllable_loads])
p, p_vals = zip(*data.constants.items())
x_eval = CubicSpline(t_arr, x_arr.T)
c_eval = CubicSpline(t_arr, c_arr.T)
constraints = data.system.holonomic_constraints.col_join(
    data.system.nonholonomic_constraints).xreplace(data.system.eom_method.kindiffdict())
eval_constraints = sm.lambdify((x, p), constraints, cse=True)
constr_arr = eval_constraints(x_arr, p_vals).reshape((len(constraints), len(t_arr)))

# Compute loads from actuators
loads = {}
eq_angles = {}
for side in ("left", "right"):
    lg = getattr(data.model.rider, f"{side}_shoulder").load_groups[0]
    j = lg.system.joints[0]
    for i, tp in enumerate(("flexion", "adduction", "rotation")):
        eq_angles[f"{side}_shoulder_{tp}"] = lg.symbols[f"q_ref_{tp}"]
        loads[f"{side}_shoulder_{tp}"] = (
                -lg.symbols[f"k_{tp}"] * (j.coordinates[i] - lg.symbols[f"q_ref_{tp}"]
                                          ) - lg.symbols[f"c_{tp}"] * j.speeds[i])
    lg = getattr(data.model.rider, f"{side}_arm").load_groups[0]
    j = lg.system.joints[0]
    loads[f"{side}_elbow"] = (
            -lg.symbols["k"] * (j.coordinates[0] - lg.symbols["q_ref"]) -
            lg.symbols["c"] * j.speeds[0])
    eq_angles[f"{side}_elbow"] = lg.symbols["q_ref"]
lg = data.model.seat.load_groups[0]
j = lg.system.joints[0]
loads["lean"] = (
        -lg.symbols["k"] * (j.coordinates[0] - lg.symbols["q_ref"]) -
        lg.symbols["c"] * j.speeds[0])
eq_angles["lean"] = lg.symbols["q_ref"]
loads = {loc: sm.lambdify((x, p, c), load)(x_arr, p_vals, c_arr)
         for loc, load in loads.items()}
eq_angles = {loc: sm.lambdify((x, p, c), eq_angle)(x_arr, p_vals, c_arr)
             for loc, eq_angle in eq_angles.items()}


def get_arr(sym):
    """Return the array corresponding to the given symbol."""
    if sym in x:
        return x_arr[x.index(sym), :]
    elif sym in c:
        return c_arr[c.index(sym), :]
    raise ValueError(f"Name {sym} not recognized.")


legend_args = (
    [plt.Line2D([0], [0], color=f"C{i}") for i in range(5)] + [
        plt.Line2D([0], [0], color="k", linestyle=":"),
        plt.Line2D([0], [0], color="k")],
    ["lean", "shoulder flexion", "shoulder adduction", "shoulder rotation",
     "elbow flexion", "left", "right"])
plt.figure()
plt.plot(t_arr, loads["lean"], label="lean", color="C0")
for side in ("left", "right"):
    linestyle = ":" if side == "left" else "-"
    for i, dir_name in enumerate(["flexion", "adduction", "rotation"], 1):
        plt.plot(t_arr, loads[f"{side}_shoulder_{dir_name}"],
                 label=dir_name, color=f"C{i}", linestyle=linestyle)
    plt.plot(t_arr, loads[f"{side}_elbow"], label="elbow", color="C4",
             linestyle=linestyle)
plt.xlabel("Time (s)")
plt.ylabel("Torque (Nm)")
plt.legend(*legend_args)


plt.figure()
plt.plot(t_arr, eq_angles["lean"], label="lean", color="C0")
for side in ("left", "right"):
    linestyle = ":" if side == "left" else "-"
    for i, dir_name in enumerate(["flexion", "adduction", "rotation"], 1):
        plt.plot(t_arr, eq_angles[f"{side}_shoulder_{dir_name}"],
                 label=dir_name, color=f"C{i}", linestyle=linestyle)
    plt.plot(t_arr, eq_angles[f"{side}_elbow"], label="elbow", color="C4",
             linestyle=linestyle)
plt.xlabel("Time (s)")
plt.ylabel("Equilibrium angle (rad)")
plt.legend(*legend_args)

plt.figure()
qs = {
    "yaw": data.model.bicycle.q[2],
    "roll": data.model.bicycle.q[3],
    "steer": data.model.bicycle.q[6],
    "lean": data.model.seat.q[0],
    "shoulder flex left": data.model.rider.left_shoulder.q[0],
    "shoulder flex right": data.model.rider.right_shoulder.q[0],
    "elbow flex left": data.model.rider.left_arm.q[0],
    "elbow flex right": data.model.rider.right_arm.q[0],
}
for q_name, q in qs.items():
    plt.plot(t_arr, x_arr[data.system.q[:].index(q), :], label=q_name)
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.legend()
plt.figure()

us = {
    "yaw": data.model.bicycle.u[2],
    "roll": data.model.bicycle.u[3],
    "steer": data.model.bicycle.u[6],
    "lean": data.model.seat.u[0],
    "shoulder flex left": data.model.rider.left_shoulder.u[0],
    "shoulder flex right": data.model.rider.right_shoulder.u[0],
    "elbow flex left": data.model.rider.left_arm.u[0],
    "elbow flex right": data.model.rider.right_arm.u[0],
}
for u_name, u in us.items():
    plt.plot(t_arr, x_arr[len(data.system.q) + data.system.u[:].index(u), :],
             label=u_name)
plt.xlabel("Time (s)")
plt.ylabel("Angular velocity (rad/s)")
plt.legend()

fig, axs = plt.subplots(2, 1, sharex=True)
axs0twin, axs1twin = axs[0].twinx(), axs[1].twinx()
for i, name in [(0, "yaw"), (2, "steer"), (3, "lean")]:
    axs[0].plot(t_arr, x_arr[data.system.q[:].index(qs[name]), :], color=f"C{i}",
                label=name)
    axs[1].plot(t_arr, x_arr[len(data.system.q) + data.system.u[:].index(us[name]), :],
                color=f"C{i}", label=name)
axs0twin.plot(t_arr, x_arr[data.system.q[:].index(qs["roll"]), :], label="roll",
              color="C1")
axs1twin.plot(t_arr, x_arr[len(data.system.q) + data.system.u[:].index(us["roll"]), :],
              label="roll", color="C1")
axs0twin.tick_params(axis="y", labelcolor="C1")
axs1twin.tick_params(axis="y", labelcolor="C1")
axs0twin.set_ylabel("Angle (rad)", color="C1")
axs1twin.set_ylabel("Angular velocity (rad/s)", color="C1")
axs[0].set_ylabel("Angle (rad)")
axs[1].set_ylabel("Angular velocity (rad/s)")
axs[1].set_xlabel("Time (s)")
axs[0].legend([plt.Line2D([0], [0], color=f"C{i}") for i in range(4)],
              ["yaw", "roll", "steer", "lean"], loc="lower left")
fig.tight_layout()

plt.figure()
qs = {
    "lean": (data.model.seat.q[0],
             data.model.seat.load_groups[0].symbols["q_ref"]),
    "shoulder flex left": (
        data.model.rider.left_shoulder.q[0],
        data.model.rider.left_shoulder.load_groups[0].symbols["q_ref_flexion"]),
    "shoulder flex right": (
    data.model.rider.right_shoulder.q[0],
    data.model.rider.right_shoulder.load_groups[0].symbols["q_ref_flexion"]),
    "elbow flex left": (
    data.model.rider.left_arm.q[0],
    data.model.rider.left_arm.load_groups[0].symbols["q_ref"]),
    "elbow flex right": (
    data.model.rider.right_arm.q[0],
    data.model.rider.right_arm.load_groups[0].symbols["q_ref"])
}
for q_name, (q, c_sym) in qs.items():
    plt.plot(t_arr,
             np.array([data.simulator.controls[c_sym](ti, x_arr[:, i])
                       for i, ti in enumerate(t_arr)]) -
             x_arr[data.system.q[:].index(q), :], label=q_name)

plt.xlabel("Time (s)")
plt.ylabel("Angle difference (rad)")
plt.legend()

plt.figure()
u1_arr = x_arr[len(data.system.q) + data.system.u[:].index(data.model.bicycle.u[0]), :]
u2_arr = x_arr[len(data.system.q) + data.system.u[:].index(data.model.bicycle.u[1]), :]
plt.plot(t_arr, np.sqrt(u1_arr ** 2 + u2_arr ** 2))
plt.title("Forward velocity rear wheel")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")

plt.figure()
for i in range(len(constraints)):
    plt.plot(t_arr, constr_arr[i, :], label=f"{i}")
plt.xlabel("Time (s)")
plt.ylabel("Constraint violation")
plt.legend()

dist_idx = c.index(data.disturbance)
disturbance_plot_info = {
    "vector": c[dist_idx] / c_arr[dist_idx, :].max() *
              data.model.bicycle.rear_frame.wheel_hub.axis,
    "origin": data.model.bicycle.rear_frame.saddle.point,
    "name": "disturbance", "color": "r"}
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))
n_frames = 6
plotter = Plotter.from_model(ax, data.model)
plotter.add_vector(**disturbance_plot_info)
plotter.lambdify_system((x, c, p))
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
q1_arr = x_arr[data.system.q[:].index(data.model.bicycle.q[0]), :]
q2_arr = x_arr[data.system.q[:].index(data.model.bicycle.q[1]), :]
front_contact_coord = data.model.bicycle.front_tire.contact_point.pos_from(
    plotter.origin).to_matrix(plotter.inertial_frame)[:2]
eval_fc = sm.lambdify((x, p), front_contact_coord, cse=True)
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
ax.view_init(25, 15)
ax.set_aspect("equal")
ax.axis("off")

if not make_animation:
    plt.show()
    exit()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(13, 13))
plotter = Plotter.from_model(ax, data.model)
plotter.add_vector(**disturbance_plot_info)
plotter.lambdify_system((x, c, p))
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
