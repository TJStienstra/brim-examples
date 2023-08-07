"""Create a model of the Whipple bicycle with an upper body rider."""
import cloudpickle
from brim import (
    BicycleRider,
    FixedPelvisToTorso,
    FlatGround,
    HolonomicHandGrip,
    KnifeEdgeWheel,
    NonHolonomicTyre,
    PinElbowStickLeftArm,
    PinElbowStickRightArm,
    PlanarPelvis,
    PlanarTorso,
    Rider,
    RigidFrontFrame,
    RigidRearFrame,
    SideLeanSeat,
    SimplePedals,
    SphericalLeftShoulder,
    SphericalRightShoulder,
    WhippleBicycle,
)
from brim.brim import SideLeanSeatSpringDamper
from brim.rider import PinElbowSpringDamper, SphericalShoulderSpringDamper
from sympy import symbols

from utilities import DataStorage

# Configure the bicycle rider model
bicycle = WhippleBicycle("bicycle")
bicycle.front_frame = RigidFrontFrame("front_frame")
bicycle.rear_frame = RigidRearFrame("rear_frame")
bicycle.front_wheel = KnifeEdgeWheel("front_wheel")
bicycle.rear_wheel = KnifeEdgeWheel("rear_wheel")
bicycle.front_tyre = NonHolonomicTyre("front_tyre")
bicycle.rear_tyre = NonHolonomicTyre("rear_tyre")
bicycle.pedals = SimplePedals("pedals")
bicycle.ground = FlatGround("ground")

rider = Rider("rider")
rider.pelvis = PlanarPelvis("pelvis")
rider.torso = PlanarTorso("torso")
rider.left_arm = PinElbowStickLeftArm("left_arm")
rider.right_arm = PinElbowStickRightArm("right_arm")
rider.pelvis_to_torso = FixedPelvisToTorso("pelvis_to_torso")
rider.left_shoulder = SphericalLeftShoulder("left_shoulder")
rider.right_shoulder = SphericalRightShoulder("right_shoulder")

br = BicycleRider("bicycle_rider")
br.bicycle = bicycle
br.rider = rider
br.seat_connection = SideLeanSeat("seat_conn")
br.steer_connection = HolonomicHandGrip("steer_conn")

left_shoulder_torque = SphericalShoulderSpringDamper("left_shoulder")
right_shoulder_torque = SphericalShoulderSpringDamper("right_shoulder")
left_arm_torque = PinElbowSpringDamper("left_arm")
right_arm_torque = PinElbowSpringDamper("right_arm")
lean_torque = SideLeanSeatSpringDamper("lean")
rider.left_shoulder.add_load_groups(left_shoulder_torque)
rider.right_shoulder.add_load_groups(right_shoulder_torque)
rider.left_arm.add_load_groups(left_arm_torque)
rider.right_arm.add_load_groups(right_arm_torque)
br.seat_connection.add_load_groups(lean_torque)

# Define the model
br.define_connections()
br.define_objects()
bicycle.pedals.symbols["offset"] = rider.pelvis.symbols["hip_width"] / 2
br.define_kinematics()
br.define_loads()
br.define_constraints()

# Create a sympy system object of the model
system = br.to_system()

g = symbols("g")
system.apply_gravity(-g * bicycle.ground.get_normal(()))

# The dependent and independent variables need to be specified manually
system.q_ind = [
    *bicycle.q[:4], *bicycle.q[5:], rider.left_shoulder.q[1], rider.right_shoulder.q[1],
    *br.seat_connection.q]
system.q_dep = [
    bicycle.q[4], rider.left_shoulder.q[0], rider.left_shoulder.q[2],
    rider.right_shoulder.q[0], rider.right_shoulder.q[2],
    *rider.left_arm.q, *rider.right_arm.q]
system.u_ind = [
    bicycle.u[3], *bicycle.u[5:7],
    rider.left_shoulder.u[1], rider.right_shoulder.u[1], *br.seat_connection.u]
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
})

with open("data.pkl", "wb") as f:
    cloudpickle.dump(data, f)
