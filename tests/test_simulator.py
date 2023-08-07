import numpy as np
import pytest
from brim import FlatGround, KnifeEdgeWheel, NonHolonomicTyre
from brim.other import RollingDisc
from numpy.testing import assert_allclose
from sympy import symbols
from sympy.physics.mechanics import Torque, dynamicsymbols, inertia

from utilities.simulator import Simulator


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

    def test_set_controls(self, setup_simulator) -> None:
        torque = dynamicsymbols("T")
        # Test wrong type
        with pytest.raises(TypeError):
            self.simulator.controls = (dynamicsymbols("T"), lambda t, x: 0.0)
        with pytest.raises(TypeError):
            self.simulator.controls = {torque: 0.0}
        assert self.simulator.controls[torque](0, np.zeros(10)) == 0.0
        # Test changing controls after initialization
        self.simulator.initialize()
        assert (self.simulator.eval_rhs(0, np.zeros(10)) ** 2).sum() == 0
        controls = {torque: lambda t, x: 1.0}
        self.simulator.controls = controls
        assert self.simulator.controls[torque](0, np.zeros(10)) == 1.0
        assert (self.simulator.eval_rhs(0, np.zeros(10)) ** 2).sum() > 0
        # Test with different keys
        self.simulator.controls = {}
        assert self.simulator.controls == {}
        assert not self.simulator._initialized

    def test_initial_conditions(self, setup_simulator) -> None:
        # Test wrong type
        with pytest.raises(TypeError):
            self.simulator.initial_conditions = (self.rolling_disc.q[0], 0.0)
        assert self.simulator.initial_conditions == self.initial_conditions
        # Test changing initial conditions after initialization
        self.simulator.initialize()
        initial_conditions = {**self.initial_conditions, self.rolling_disc.u[4]: -2.0}
        self.simulator.initial_conditions = initial_conditions
        assert self.simulator.initial_conditions[self.rolling_disc.u[4]] == -2.0
        assert self.simulator.initial_conditions[self.rolling_disc.u[0]] == 0.6
        # Test with different keys
        self.simulator.initial_conditions = {}
        assert self.simulator.initial_conditions == {}
        assert not self.simulator._initialized

    def test_initialize_eoms_not_formed(self) -> None:
        simulator = Simulator(self.system)
        simulator.constants = self.constants
        simulator.controls = self.controls
        simulator.initial_conditions = self.initial_conditions
        with pytest.raises(ValueError):
            simulator.initialize()

    @pytest.mark.parametrize("key", ["constants", "controls", "initial_conditions"])
    def test_initialize_not_set_variable(self, key) -> None:
        self.system.form_eoms()
        simulator = Simulator(self.system)
        for k in ["constants", "controls", "initial_conditions"]:
            if k != key:
                setattr(simulator, k, getattr(self, k))
        with pytest.raises(ValueError):
            simulator.initialize()

    def test_initialize_missing_constant(self, setup_simulator) -> None:
        self.simulator.constants.pop(symbols("g"))
        with pytest.raises(ValueError):
            self.simulator.initialize()

    def test_initialize_missing_control(self, setup_simulator) -> None:
        self.simulator.controls.pop(dynamicsymbols("T"))
        with pytest.raises(ValueError):
            self.simulator.initialize()

    def test_initialize_missing_initial_condition(self, setup_simulator) -> None:
        self.simulator.initial_conditions.pop(self.rolling_disc.q[0])
        with pytest.raises(ValueError):
            self.simulator.initialize()

    def test_initialize(self, setup_simulator) -> None:
        assert not self.simulator._initialized
        self.simulator.initialize()
        assert self.simulator._initialized

    def test_initialize_no_param_check(self, setup_simulator) -> None:
        self.simulator.constants.pop(symbols("g"))
        self.simulator.initialize(False)

    @pytest.mark.parametrize("compile_with_numba", [True, False])
    def test_eval_rhs(self, setup_simulator, compile_with_numba) -> None:
        self.simulator.initialize(False)
        if compile_with_numba:
            self.simulator.compile_with_numba()
        assert_allclose(self.simulator.eval_rhs(0, np.zeros(10)), np.zeros(10))
        x0 = np.array([self.simulator.initial_conditions[xi]
                       for xi in self.system.q.col_join(self.system.u)])
        assert_allclose(self.simulator.eval_rhs(0, x0),
                        [0.3, 0, 0, 0, -1, 0, 0, 0, 0, 0])

    def test_solve_unknown(self, setup_simulator) -> None:
        self.simulator.initialize(False)
        with pytest.raises(ValueError):
            self.simulator.solve((0, 1), solver="unknown")

    @pytest.mark.parametrize("compile_with_numba", [True, False])
    def test_solve_solve_ivp(self, setup_simulator, compile_with_numba) -> None:
        self.simulator.initialize(False)
        if compile_with_numba:
            self.simulator.compile_with_numba()
        t, x = self.simulator.solve((0, 1))
        assert self.simulator.t == t
        assert self.simulator.x == x
