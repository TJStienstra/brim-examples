import pytest
from brim import FlatGround, KnifeEdgeWheel, NonHolonomicTyre
from brim.other import RollingDisc
from brimopt.simulator import Simulator
from sympy import symbols
from sympy.physics.mechanics import Torque, dynamicsymbols, inertia


class TestSimulator:
    @pytest.fixture(autouse=True)
    def create_rolling_disc(self) -> None:
        self.rolling_disc = RollingDisc("disc")
        self.rolling_disc.disc = KnifeEdgeWheel("wheel")
        self.rolling_disc.ground = FlatGround("ground")
        self.rolling_disc.tyre = NonHolonomicTyre("tyre")
        self.rolling_disc.define_connections()
        self.rolling_disc.define_objects()
        self.rolling_disc.disc.body.central_inertia = inertia(
            self.rolling_disc.disc.frame, 1, 2, 1
        ) * self.rolling_disc.disc.body.mass * self.rolling_disc.disc.radius ** 2 / 4
        self.rolling_disc.define_kinematics()
        self.rolling_disc.define_loads()
        self.rolling_disc.define_constraints()
        self.system = self.rolling_disc.to_system()
        self.system.apply_gravity(-symbols("g") * self.rolling_disc.ground.get_normal(
            self.rolling_disc.ground.origin))
        self.system.add_loads(
            Torque(self.rolling_disc.disc.frame,
                   dynamicsymbols("T") * self.rolling_disc.disc.rotation_axis))
        self.system.u_ind = self.rolling_disc.u[2:]
        self.system.u_dep = self.rolling_disc.u[:2]
        self.constants = {
            symbols("g"): 9.81,
            self.rolling_disc.disc.body.mass: 1.0,
            self.rolling_disc.disc.radius: 0.3,
        }
        self.controls = {
            dynamicsymbols("T"): lambda t, x: 0.0,
        }
        self.initial_conditions = {
            self.rolling_disc.q[0]: 0.0,
            self.rolling_disc.q[1]: 0.0,
            self.rolling_disc.q[2]: 0.0,
            self.rolling_disc.q[3]: 0.0,
            self.rolling_disc.q[4]: 0.0,
            self.rolling_disc.u[0]: 0.3,
            self.rolling_disc.u[1]: 0.0,
            self.rolling_disc.u[2]: 0.0,
            self.rolling_disc.u[3]: 0.0,
            self.rolling_disc.u[4]: -1.0,
        }

    @pytest.fixture()
    def setup_simulator(self) -> None:
        self.system.form_eoms()
        self.simulator = Simulator(self.system)
        self.simulator.constants = self.constants
        self.simulator.controls = self.controls
        self.simulator.initial_conditions = self.initial_conditions

    def test_set_constants(self, setup_simulator) -> None:
        # Test wrong type
        with pytest.raises(TypeError):
            self.simulator.constants = (symbols("g"), 9.81)
        assert self.simulator.constants == self.constants
        # Test with different values
        constants = {
            self.rolling_disc.disc.radius: 0.5,
            self.rolling_disc.disc.body.mass: 1.3,
            symbols("g"): 10.0,
        }
        self.simulator.constants = constants
        assert self.simulator.constants == constants
        # Test changing constants after initialization
        self.simulator.initialize()
        assert self.simulator.initial_conditions[self.rolling_disc.u[0]] == 0.5
        old_initials = self.simulator.initial_conditions.copy()
        self.simulator.constants = self.constants
        assert self.simulator.constants == self.constants
        assert self.simulator.initial_conditions != old_initials
        assert self.simulator.initial_conditions[self.rolling_disc.u[0]] == 0.3
        # Test with different keys
        self.simulator.constants = {symbols("g"): 9.81}
        assert self.simulator.constants == {symbols("g"): 9.81}
        assert not self.simulator._initialized
