"""Create a model of the Whipple bicycle."""
import cloudpickle
from brim import (
    FlatGround,
    KnifeEdgeWheel,
    NonHolonomicTyre,
    RigidFrontFrame,
    RigidRearFrame,
    SimplePedals,
    WhippleBicycle,
)
from sympy import symbols
from sympy.physics.mechanics import Force, dynamicsymbols
from sympy.physics.mechanics._actuator import TorqueActuator

from utilities import DataStorage

# Configure the bicycle rider model
bicycle = WhippleBicycle("bicycle")
bicycle.front_frame = RigidFrontFrame("front_frame")
bicycle.rear_frame = RigidRearFrame("rear_frame")
bicycle.front_wheel = KnifeEdgeWheel("front_wheel")
bicycle.rear_wheel = KnifeEdgeWheel("rear_wheel")
bicycle.front_tyre = NonHolonomicTyre("front_tyre")
bicycle.rear_tyre = NonHolonomicTyre("rear_tyre")
bicycle.pedals = SimplePedals("pedals")  # Used to beautify the visualization
bicycle.ground = FlatGround("ground")

# Define the model
bicycle.define_all()

# Create a sympy system object of the model
system = bicycle.to_system()

# Apply additional forces and torques to the system
g = symbols("g")
system.apply_gravity(-g * bicycle.ground.get_normal(bicycle.ground.origin))
steer_torque = dynamicsymbols("steer_torque")
system.add_actuators(TorqueActuator(steer_torque, bicycle.rear_frame.steer_axis,
                                    bicycle.front_frame.body, bicycle.rear_frame.body))
disturbance = dynamicsymbols("disturbance")
system.add_loads(
    Force(bicycle.rear_frame.saddle, disturbance * bicycle.rear_frame.frame.y))

# The dependent and independent variables need to be specified manually
system.q_ind = [*bicycle.q[:4], *bicycle.q[5:]]
system.q_dep = [bicycle.q[4]]
system.u_ind = [bicycle.u[3], *bicycle.u[5:7]]
system.u_dep = [*bicycle.u[:3], bicycle.u[4], bicycle.u[7]]

# Simple check to see if the system is valid
system.validate_system()
system.form_eoms(constraint_solver="CRAMER")  # LU solve may lead to zero divisions

data = DataStorage(**{
    "model": bicycle,
    "system": system,
    "controllable_loads": [steer_torque, disturbance],
})

with open("data.pkl", "wb") as f:
    cloudpickle.dump(data, f)
