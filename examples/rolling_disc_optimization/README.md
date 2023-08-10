# Trajectory Tracking of a Rolling Disc
The objective in the Optimal Control Problem (OCP) is to find the minimal control
torques to follow a periodic sinusoidal path with a rolling disc. The rolling disc
itself is modeled as a knife-edge wheel subject to pure-rolling on a flat ground. To
control the disc three torques are applied: a driving torque about the rotation axis, a
steer torque about the axis going through the contact point and the center of the disc,
and a roll torque about an axis perpendicular to both the normal of the ground and the
rotation axis.

One can play around with several objective functions. The default objective function is
a multi-objective function that minimizes the control effort and the deviation from the
desired path.
