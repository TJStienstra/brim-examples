import pickle
from io import BytesIO
from typing import Any

import pytest
from brim.bicycle import FlatGround, KnifeEdgeWheel, NonHolonomicTyreModel
from brim.other import RollingDisc
from brim.utilities.utilities import to_system
from brimopt import DataManager
from sympy import Function, ImmutableMatrix, Symbol
from sympy.physics.mechanics import (
    Force,
    PrismaticJoint,
    RigidBody,
    Torque,
    dynamicsymbols,
    find_dynamicsymbols,
)
from sympy.physics.mechanics._system import System


class TestDataManager:
    @pytest.fixture()
    def _setup_rolling_disc(self) -> None:
        self.rolling_disc = RollingDisc("sys")
        self.rolling_disc.disc = KnifeEdgeWheel("disc")
        self.rolling_disc.tyre = NonHolonomicTyreModel("tyre")
        self.rolling_disc.ground = FlatGround("ground")
        self.rolling_disc.define_all()
        self.system = to_system(self.rolling_disc)
        self.T = dynamicsymbols("T")
        self.system.add_loads(
            Torque(self.rolling_disc.disc.body, self.T * self.rolling_disc.disc.y)
        )
        self.system.u_ind = self.rolling_disc.u[2:]
        self.system.u_dep = self.rolling_disc.u[:2]
        self.system.form_eoms()
        n = len(self.system.q)
        self.dyn_symbols = {
            **{self.system.q[i]: Symbol(f"q{i}") for i in range(n)},
            **{self.system.q[i].diff(): Symbol(f"dq{i}") for i in range(n)},
            **{self.system.u[i]: Symbol(f"u{i}") for i in range(n)},
            **{self.system.u[i].diff(): Symbol(f"du{i}") for i in range(n)},
            self.T: Symbol("T"),
        }

    def _get_dyn_symbols(self, dyn_symbols: Any) -> Any:
        if dyn_symbols == "search":
            return dyn_symbols
        elif dyn_symbols == "dict":
            return self.dyn_symbols
        elif dyn_symbols == "matrix":
            return ImmutableMatrix(list(self.dyn_symbols.keys()))

    def test_init(self):
        dm = DataManager()
        assert dm.q == ImmutableMatrix()
        assert dm.u == ImmutableMatrix()
        assert dm.q_ind == ImmutableMatrix()
        assert dm.u_ind == ImmutableMatrix()
        assert dm.q_dep == ImmutableMatrix()
        assert dm.u_dep == ImmutableMatrix()
        assert dm.holonomic_constraints == ImmutableMatrix()
        assert dm.nonholonomic_constraints == ImmutableMatrix()
        assert dm.kdes == ImmutableMatrix()
        assert dm.mass_matrix_full == ImmutableMatrix()
        assert dm.forcing_full == ImmutableMatrix()
        assert dm.mass_matrix == ImmutableMatrix()
        assert dm.forcing == ImmutableMatrix()
        assert dm.mass_matrix_kin == ImmutableMatrix()
        assert dm.forcing_kin == ImmutableMatrix()
        f = BytesIO()
        pickle.dump(dm, f)

    @pytest.mark.parametrize("dyn_symbols", ["search", "dict", "matrix"])
    def test_rolling_disc_system_types(self, _setup_rolling_disc, dyn_symbols) -> None:
        dm = DataManager()
        dm.process_system(self.system, self._get_dyn_symbols(dyn_symbols))
        attrs = ["q", "u", "q_ind", "u_ind", "q_dep", "u_dep", "holonomic_constraints",
                 "nonholonomic_constraints", "kdes", "mass_matrix_full", "forcing_full",
                 "mass_matrix", "forcing", "mass_matrix_kin", "forcing_kin"]
        for attr in attrs:
            assert find_dynamicsymbols(getattr(dm, attr)) == set()

    @pytest.mark.parametrize("dyn_symbols", ["search", "dict", "matrix"])
    def test_rolling_disc_system_dump(self, _setup_rolling_disc, dyn_symbols) -> None:
        dm = DataManager()
        dm.process_system(self.system, self._get_dyn_symbols(dyn_symbols))
        f = BytesIO()
        pickle.dump(dm, f)

    @pytest.mark.parametrize("dyn_symbols", ["not_implemented_method", Symbol("T")])
    def test_invalid_dyn_symbols_type(self, _setup_rolling_disc, dyn_symbols) -> None:
        dm = DataManager()
        with pytest.raises(TypeError):
            dm.process_system(self.system, dyn_symbols)

    def test_system_without_eom(self) -> None:
        dm = DataManager()
        with pytest.raises(NotImplementedError):
            dm.process_system(System())

    def test_auto_derivative_name_already_taken(self) -> None:
        dm = DataManager()
        q1, q2, u1, u2 = dynamicsymbols("q1 dq1 u1 du1")
        sys = System()
        sys.q_ind = [q1, q2]
        sys.u_ind = [u1, u2]
        sys.kdes = [q1.diff() - u1, q2.diff() - u2]
        sys.form_eoms()
        dm.process_system(sys)
        assert dm.q[0] != dm.q[1]
        assert dm.u[0] != dm.u[1]
        assert len(dm.kdes[0].free_symbols.difference(dm.q, dm.u)) == 1
        assert len(dm.kdes[1].free_symbols.difference(dm.q, dm.u)) == 1

    def test_not_implemented_function_parsed(self) -> None:
        sys = System.from_newtonian(RigidBody("newtonian"))
        sys.add_bodies()
        q, u = dynamicsymbols("q u")
        f = Function("F")(q, u)
        sys.add_joints(PrismaticJoint("joint", sys.bodies[0], RigidBody("body"), q, u))
        sys.add_loads(Force(sys.bodies[-1], f.diff(dynamicsymbols._t) * sys.x))
        sys.form_eoms()
        with pytest.raises(NotImplementedError):
            dm = DataManager()
            dm.process_system(sys)

    def test_invalid_system_type(self) -> None:
        dm = DataManager()
        with pytest.raises(TypeError):
            dm.process_system(1)
