"""Create a model of the Whipple bicycle."""
import cloudpickle
from brim import (
    FlatGround,
    KnifeEdgeWheel,
    NonHolonomicTire,
    RigidFrontFrame,
    RigidRearFrame,
    WhippleBicycle,
)
from brim.bicycle import SuspensionRigidFrontFrame
from sympy import symbols
from sympy.physics.mechanics import Force, TorqueActuator, dynamicsymbols

from utilities import DataStorage

# Choose a bicycle model
# "Browser": Batavus Browser, a Dutch style city bicycle, which fully follows the
#            default Whipple bicycle model.
# "Fisher": Gary Fisher Mountain Bike, a hard-tail mountain bicycle modeled with
#           suspension in the front frame.
bike_parametrization = "Browser"

# Configure the bicycle rider model
bicycle = WhippleBicycle("bicycle")
if bike_parametrization == "Fisher":
    bicycle.front_frame = SuspensionRigidFrontFrame("front_frame")
else:
    bicycle.front_frame = RigidFrontFrame("front_frame")
bicycle.rear_frame = RigidRearFrame("rear_frame")
bicycle.front_wheel = KnifeEdgeWheel("front_wheel")
bicycle.rear_wheel = KnifeEdgeWheel("rear_wheel")
bicycle.front_tire = NonHolonomicTire("front_tire")
bicycle.rear_tire = NonHolonomicTire("rear_tire")
bicycle.ground = FlatGround("ground")

# Define the model
bicycle.define_all()

# Create a sympy system object of the model
system = bicycle.to_system()

# Apply additional forces and torques to the system
g = symbols("g")
system.apply_uniform_gravity(-g * bicycle.ground.get_normal(bicycle.ground.origin))
steer_torque = dynamicsymbols("steer_torque")
system.add_actuators(TorqueActuator(
    steer_torque, bicycle.rear_frame.steer_hub.axis,
    bicycle.front_frame.steer_hub.frame, bicycle.rear_frame.steer_hub.frame))
disturbance = dynamicsymbols("disturbance")
system.add_loads(Force(
    bicycle.rear_frame.saddle.point, disturbance * bicycle.rear_frame.wheel_hub.axis))

# The dependent and independent variables need to be specified manually
system.q_ind = [*bicycle.q[:4], *bicycle.q[5:]]
system.q_dep = [bicycle.q[4]]
system.u_ind = [bicycle.u[3], *bicycle.u[5:7]]
system.u_dep = [*bicycle.u[:3], bicycle.u[4], bicycle.u[7]]
if isinstance(bicycle.front_frame, SuspensionRigidFrontFrame):
    system.add_coordinates(bicycle.front_frame.q[0])
    system.add_speeds(bicycle.front_frame.u[0])

# Simple check to see if the system is valid
system.validate_system()
system.form_eoms(constraint_solver="CRAMER")  # LU solve may lead to zero divisions

data = DataStorage(**{
    "bike_parametrization": bike_parametrization,
    "model": bicycle,
    "system": system,
    "controllable_loads": [steer_torque, disturbance],
})
with open("data.pkl", "wb") as f:
    cloudpickle.dump(data, f)
