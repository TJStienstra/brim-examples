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
from sympy.physics.mechanics import dynamicsymbols

data["x"] = data.system.q.col_join(data.system.u)
data["du"] = sm.Matrix([dynamicsymbols(f"d{ui.name}") for ui in data.system.u])
data["eoms"] = data.system.mass_matrix * data.du - data.system.forcing
eval_funcs = [data.simulator._eval_configuration_constraints,
              data.simulator._eval_velocity_constraints,
              data.simulator._eval_eoms_matrices]
for i, f in enumerate(eval_funcs):
    if hasattr(f, "py_func"):
        eval_funcs[i] = f.py_func
eval_conf, eval_vel, eval_eoms = eval_funcs


def eval_eoms_reshaped(x, du, p, c):
    """Evaluate the equations of motion and reshape the result."""
    mass_matrix, forcing = eval_eoms(x, du, p, c)
    return mass_matrix.reshape((len(x), len(x))), forcing.reshape((len(x),))


make_animation = True

t_arr = data.simulator.t
x_arr = data.simulator.x
c_arr = np.array([[data.simulator.controls[fi](ti, x_arr[:, i])
                  for i, ti in enumerate(t_arr)] for fi in data.controllable_loads])

# Compute loads from actuators
loads = {}
for side in ("left", "right"):
    lg = getattr(data.model.rider, f"{side}_shoulder").load_groups[0]
    j = lg.system.joints[0]
    for i, tp in enumerate(("flexion", "adduction", "rotation")):
        loads[f"{side}_shoulder_{tp}"] = (
                -lg.symbols[f"k_{tp}"] * (j.coordinates[i] - lg.symbols[f"q_ref_{tp}"]
                                          ) - lg.symbols[f"c_{tp}"] * j.speeds[i])
    lg = getattr(data.model.rider, f"{side}_arm").load_groups[0]
    j = lg.system.joints[0]
    loads[f"{side}_elbow"] = (
            -lg.symbols["k"] * (j.coordinates[0] - lg.symbols["q_ref"]) -
            lg.symbols["c"] * j.speeds[0])
lg = data.model.seat_connection.load_groups[0]
j = lg.system.joints[0]
loads["lean"] = (
        -lg.symbols["k"] * (j.coordinates[0] - lg.symbols["q_ref"]) -
        lg.symbols["c"] * j.speeds[0])
p, p_vals = zip(*data.constants.items())
x = data.system.q[:] + data.system.u[:]
c = data.controllable_loads
loads = {loc: sm.lambdify((x, p, c), load)(x_arr, p_vals, c_arr)
         for loc, load in loads.items()}
x_eval = CubicSpline(t_arr, x_arr.T)

constraints = data.system.holonomic_constraints.col_join(
    data.system.nonholonomic_constraints).xreplace(data.system.eom_method.kindiffdict())
eval_constraints = sm.lambdify((data.x, p), constraints, cse=True)
constr_arr = eval_constraints(x_arr, p_vals).reshape((len(constraints), len(t_arr)))

plt.figure()
for name, load_arr in loads.items():
    if name == "lean":
        kwargs = {"color": "C0", "label": "lean"}
    else:
        kwargs = {}
        if name[:4] == "left":
            kwargs["linestyle"] = ":"
        for i, dir_name in enumerate(["flexion", "adduction", "rotation"]):
            if dir_name == name[-len(dir_name):]:
                kwargs["color"] = f"C{i + 1}"
                kwargs["label"] = dir_name
        if name[-5:] == "elbow":
            kwargs["label"] = "elbow"
            kwargs["color"] = "C4"
    plt.plot(t_arr, load_arr, **kwargs)
plt.xlabel("Time (s)")
plt.ylabel("Torque (Nm)")
plt.legend([plt.Line2D([0], [0], color="C0"),
            plt.Line2D([0], [0], color="C1"),
            plt.Line2D([0], [0], color="C2"),
            plt.Line2D([0], [0], color="C3"),
            plt.Line2D([0], [0], color="C4"),
            plt.Line2D([0], [0], color="k", linestyle=":"),
            plt.Line2D([0], [0], color="k")],
           ["lean", "shoulder flexion", "shoulder adduction", "shoulder rotation",
            "elbow flexion", "left", "right"])


plt.figure()
for c, c_func in data.simulator.controls.items():
    if c == data.model.seat_connection.load_groups[0].symbols["q_ref"]:
        kwargs = {"color": "C0", "label": "lean"}
    else:
        kwargs = {}
        if "left" in c.name:
            kwargs["linestyle"] = ":"
        for i, name in enumerate(["flexion", "adduction", "rotation"]):
            if name in c.name and "shoulder" in c.name:
                kwargs["color"] = f"C{i + 1}"
                kwargs["label"] = name
        if "arm" in c.name:
            kwargs["label"] = "elbow"
            kwargs["color"] = "C4"
    plt.plot(t_arr, [c_func(ti, x_arr[:, i]) for i, ti in enumerate(t_arr)], **kwargs)
plt.xlabel("Time (s)")
plt.ylabel("Equilibrium angle (rad)")
plt.legend([plt.Line2D([0], [0], color="C0"),
            plt.Line2D([0], [0], color="C1"),
            plt.Line2D([0], [0], color="C2"),
            plt.Line2D([0], [0], color="C3"),
            plt.Line2D([0], [0], color="C4"),
            plt.Line2D([0], [0], color="k", linestyle=":"),
            plt.Line2D([0], [0], color="k")],
           ["lean", "flexion", "adduction", "rotation", "elbow", "left", "right"])

plt.figure()
qs = {
    "yaw": data.model.bicycle.q[2],
    "roll": data.model.bicycle.q[3],
    "steer": data.model.bicycle.q[6],
    "lean": data.model.seat_connection.q[0],
    "shoulder flex left": data.model.rider.left_shoulder.q[0],
    "shoulder flex right": data.model.rider.right_shoulder.q[0],
    "elbow flex left": data.model.rider.left_arm.q[0],
    "elbow flex right": data.model.rider.right_arm.q[0],
}
for q_name, q in qs.items():
    plt.plot(t_arr, x_arr[data.system.q[:].index(q), :].flatten(), label=q_name)

plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.legend()

plt.figure()
qs = {
    "lean": (data.model.seat_connection.q[0],
             data.model.seat_connection.load_groups[0].symbols["q_ref"]),
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
for q_name, (q, c) in qs.items():
    plt.plot(t_arr,
             np.array([data.simulator.controls[c](ti, x_arr[:, i])
                       for i, ti in enumerate(t_arr)]) -
             x_arr[data.system.q[:].index(q), :].flatten(), label=q_name)

plt.xlabel("Time (s)")
plt.ylabel("Angle difference (rad)")
plt.legend()

plt.figure()
u1_arr = x_arr[len(data.system.q) + data.system.u[:].index(data.model.bicycle.u[0]), :]
u2_arr = x_arr[len(data.system.q) + data.system.u[:].index(data.model.bicycle.u[1]), :]
plt.plot(t_arr, np.sqrt(u1_arr ** 2 + u2_arr ** 2).flatten())
plt.title("Forward velocity rear wheel")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")

plt.figure()
for i in range(len(constraints)):
    plt.plot(t_arr, constr_arr[i, :], label=f"{i}")
plt.xlabel("Time (s)")
plt.ylabel("Constraint violation")
plt.legend()

p, p_vals = zip(*data.constants.items())
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))
n_frames = 6
plotter = Plotter.from_model(ax, data.model)
plotter.lambdify_system((data.system.q[:] + data.system.u[:], p))
for plot_object in plotter.plot_objects:
    if isinstance(plot_object, PlotBody):
        plot_object.plot_frame.visible = False
        plot_object.plot_masscenter.visible = False
plotter.evaluate_system(x_arr[:, 0].flatten(), p_vals)
for i in range(n_frames):
    for artist in plotter.artists:
        artist.set_alpha(i * 1 / (n_frames + 1) + 1 / n_frames)
        ax.add_artist(copy(artist))
    plotter.evaluate_system(
        x_arr[:, int(round(i * (len(t_arr) - 1) / (n_frames - 1)))].flatten(), p_vals)
    plotter.update()
for artist in plotter.artists:
    artist.set_alpha(1)
    ax.add_artist(copy(artist))
q1_arr = x_arr[data.system.q[:].index(data.model.bicycle.q[0]), :]
q2_arr = x_arr[data.system.q[:].index(data.model.bicycle.q[1]), :]
front_contact_coord = data.model.bicycle.front_tyre.contact_point.pos_from(
    plotter.origin).to_matrix(plotter.inertial_frame)[:2]
eval_fc = sm.lambdify((data.system.q[:] + data.system.u[:], p),
                      front_contact_coord, cse=True)
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
plotter.lambdify_system((data.system.q[:] + data.system.u[:], p))
plotter.evaluate_system(x_arr[:, 0].flatten(), p_vals)
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
    plotter.evaluate_system(x_eval(fi / (n_frames - 1) * t_arr[-1]), p_vals)
    return *plotter.update(),


fps = 30
n_frames = int(fps * t_arr[-1])
ani = FuncAnimation(fig, animate, frames=n_frames, blit=False)
ani.save("animation.gif", dpi=150, fps=fps)
plt.show()
