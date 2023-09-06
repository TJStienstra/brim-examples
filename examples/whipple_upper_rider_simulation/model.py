"""Create a model of the Whipple bicycle with an upper body rider."""
import cloudpickle
from brim import (
    BicycleRider,
    FixedSacrum,
    FlatGround,
    HolonomicHandGrips,
    KnifeEdgeWheel,
    MasslessCranks,
    NonHolonomicTyre,
    PinElbowStickLeftArm,
    PinElbowStickRightArm,
    PlanarPelvis,
    PlanarTorso,
    Rider,
    RigidFrontFrame,
    RigidRearFrame,
    SideLeanSeat,
    SphericalLeftShoulder,
    SphericalRightShoulder,
    WhippleBicycle,
)
from brim.brim import SideLeanSeatSpringDamper
from brim.rider import PinElbowSpringDamper, SphericalShoulderSpringDamper
from sympy import symbols
from sympy.physics.mechanics import Force, dynamicsymbols

from utilities import DataStorage

# Configure the bicycle rider model
bicycle = WhippleBicycle("bicycle")
bicycle.front_frame = RigidFrontFrame("front_frame")
bicycle.rear_frame = RigidRearFrame("rear_frame")
bicycle.front_wheel = KnifeEdgeWheel("front_wheel")
bicycle.rear_wheel = KnifeEdgeWheel("rear_wheel")
bicycle.front_tyre = NonHolonomicTyre("front_tyre")
bicycle.rear_tyre = NonHolonomicTyre("rear_tyre")
bicycle.cranks = MasslessCranks("cranks")
bicycle.ground = FlatGround("ground")

rider = Rider("rider")
rider.pelvis = PlanarPelvis("pelvis")
rider.torso = PlanarTorso("torso")
rider.left_arm = PinElbowStickLeftArm("left_arm")
rider.right_arm = PinElbowStickRightArm("right_arm")
rider.sacrum = FixedSacrum("sacrum")
rider.left_shoulder = SphericalLeftShoulder("left_shoulder")
rider.right_shoulder = SphericalRightShoulder("right_shoulder")

br = BicycleRider("bicycle_rider")
br.bicycle = bicycle
br.rider = rider
br.seat = SideLeanSeat("seat")
br.hand_grips = HolonomicHandGrips("hand_grips")

left_shoulder_torque = SphericalShoulderSpringDamper("left_shoulder")
right_shoulder_torque = SphericalShoulderSpringDamper("right_shoulder")
left_arm_torque = PinElbowSpringDamper("left_arm")
right_arm_torque = PinElbowSpringDamper("right_arm")
lean_torque = SideLeanSeatSpringDamper("lean")
rider.left_shoulder.add_load_groups(left_shoulder_torque)
rider.right_shoulder.add_load_groups(right_shoulder_torque)
rider.left_arm.add_load_groups(left_arm_torque)
rider.right_arm.add_load_groups(right_arm_torque)
br.seat.add_load_groups(lean_torque)

# Define the model
br.define_connections()
br.define_objects()
bicycle.cranks.symbols["offset"] = rider.pelvis.symbols["hip_width"] / 2
br.define_kinematics()
br.define_loads()
br.define_constraints()

# Create a sympy system object of the model
system = br.to_system()

g = symbols("g")
system.apply_gravity(-g * bicycle.ground.get_normal(bicycle.ground.origin))
disturbance = dynamicsymbols("disturbance")
system.add_loads(Force(
    bicycle.rear_frame.saddle.point, disturbance * bicycle.rear_frame.wheel_hub.axis))

# The dependent and independent variables need to be specified manually
system.q_ind = [
    *bicycle.q[:4], *bicycle.q[5:], rider.left_shoulder.q[1], rider.right_shoulder.q[1],
    *br.seat.q]
system.q_dep = [
    bicycle.q[4], rider.left_shoulder.q[0], rider.left_shoulder.q[2],
    rider.right_shoulder.q[0], rider.right_shoulder.q[2],
    *rider.left_arm.q, *rider.right_arm.q]
system.u_ind = [
    bicycle.u[3], *bicycle.u[5:7],
    rider.left_shoulder.u[1], rider.right_shoulder.u[1], *br.seat.u]
system.u_dep = [
    *bicycle.u[:3], bicycle.u[4], bicycle.u[7],
    rider.left_shoulder.u[0], rider.left_shoulder.u[2],
    rider.right_shoulder.u[0], rider.right_shoulder.u[2],
    *rider.left_arm.u, *rider.right_arm.u]

# Simple check to see if the system is valid
system.validate_system()
system.form_eoms(constraint_solver="CRAMER")

data = DataStorage(**{
    "model": br,
    "system": system,
    "disturbance": disturbance,
})

with open("data.pkl", "wb") as f:
    cloudpickle.dump(data, f)
