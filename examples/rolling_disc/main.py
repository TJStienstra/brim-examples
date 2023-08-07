"""Trajectory tracking problem.

Manufactured a proof of concept, where a rolling disc model created using BRiM is tasked
to follow a periodic sinusoidal path.

The objective function is the sum of three squared torques with which the disc is
controlled: one for accelerating and decelerating the disc, one for steering the disc,
and one about the disc's roll axis.

The disc is constrained to follow the path within a certain mean tracking error. The
advantage of applying the path as a constraint is that the objective function stays
single-objective function.
"""
import os
import subprocess

directory = os.path.dirname(__file__)
print("Modeling...")  # noqa: T201
subprocess.call(fr"python {directory}\model.py")
print("Modeling finished.")  # noqa: T201

print("Optimizing...")  # noqa: T201
subprocess.call(fr"python {directory}\optimize.py")
print("Optimization finished.")  # noqa: T201

print("Simulating...")  # noqa: T201
subprocess.call(fr"python {directory}\simulate.py")
print("Simulation finished.")  # noqa: T201

print("Plotting...")  # noqa: T201
subprocess.call(fr"python {directory}\plot.py")
print("Plotting finished.")  # noqa: T201
