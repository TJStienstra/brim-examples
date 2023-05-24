"""Module for simulating sympy.physics.mechanics._system.System objects."""
from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from sympy import Basic, Function, lambdify
from sympy.physics.mechanics import (
    KanesMethod,
    dynamicsymbols,
    find_dynamicsymbols,
    msubs,
)
from sympy.physics.mechanics._system import System

__all__ = ["Simulator"]


class Simulator:
    """Simulator for sympy.physics.mechanics._system.System object."""

    def __init__(self, system: System) -> None:
        if not isinstance(system, System):
            raise TypeError(
                f"System should be of type {type(System)} not {type(system)}.")
        self._system = system
        self._constants = None
        self._initial_conditions = None
        self._controls = None
        self._t, self._x = None, None
        self._eval_configuration_constraints = None
        self._eval_velocity_constraints = None
        self._eval_eoms_matrices = None
        self._initialized = False

    @property
    def t(self) -> npt.NDArray[np.float64]:
        """Time array of the simulation."""
        if self._t is None:
            raise ValueError("System has not been integrated yet.")
        return self._t

    @property
    def x(self) -> npt.NDArray[np.float64]:
        """State array of the simulation."""
        if self._x is None:
            raise ValueError("System has not been integrated yet.")
        return self._x

    @property
    def system(self) -> System:
        """System object of the simulator."""
        return self._system

    @property
    def constants(self) -> dict[Basic, float]:
        """Constants of the system."""
        return self._constants

    @constants.setter
    def constants(self, constants: dict[Basic, float]):
        if not isinstance(constants, dict):
            raise TypeError(f"Constants should be of type {type(dict)} not "
                            f"{type(constants)}.")
        self._constants = constants

    @property
    def controls(self) -> dict[Function, Callable[[float], float]]:
        """Controls of the system."""
        return self._controls

    @controls.setter
    def controls(self, controls: dict[Function, Callable[[float], float]]):
        if not isinstance(controls, dict):
            raise TypeError(f"Controls should be of type {type(dict)} not "
                            f"{type(controls)}.")
        for control in controls.values():
            if not callable(control):
                raise TypeError(
                    f"Controls should be of type {type(Callable[[float], float])} not "
                    f"{type(control)}.")
        self._controls = controls

    @property
    def initial_conditions(self) -> dict[Function, float]:
        """Initial conditions of the system."""
        return self._initial_conditions

    @initial_conditions.setter
    def initial_conditions(self, initial_conditions: dict[Function, float]):
        if not isinstance(initial_conditions, dict):
            raise TypeError(f"Initial condintions should be of type "
                            f"{type(dict)} not {type(initial_conditions)}.")
        self._initial_conditions = initial_conditions

    def _solve_configuration_constraints(
            self, q_ind: npt.NDArray[np.float64], q_dep_guess: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:  # type: ignore
        """Solve the configuration constraints for the dependent coordinates."""
        if not self.system.q_dep:
            return np.array([])
        return np.array(fsolve(
            lambda *args: self._eval_configuration_constraints(*args).squeeze(),
            q_dep_guess, (q_ind, self._p_vals)))

    def _solve_velocity_constraints(
            self, q: npt.NDArray[np.float64], u_ind: npt.NDArray[np.float64],
            u_dep_guess: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Solve the velocity constraints for the dependent speeds."""
        if not self.system.u_dep:
            return np.array([])
        return np.array(fsolve(
            lambda *args: self._eval_velocity_constraints(*args).squeeze(),
            u_dep_guess, (q, u_ind, self._p_vals)))

    def solve_initial_conditions(self) -> None:
        """Solve the initial conditions for the dependent coordinates and speeds."""
        if (self._eval_configuration_constraints is None or
                self._eval_velocity_constraints is None):
            raise ValueError("Simulator has not been initialized yet.")
        q_dep_guess = [self.initial_conditions.get(qi, 0.) for qi in self.system.q_dep]
        q_ind = [self.initial_conditions[qi] for qi in self.system.q_ind]
        q_dep = self._solve_configuration_constraints(
            np.array(q_ind), np.array(q_dep_guess))
        q0 = q_ind + list(q_dep)
        u_dep_guess = [self.initial_conditions.get(ui, 0.) for ui in self.system.u_dep]
        u_ind = [self.initial_conditions[ui] for ui in self.system.u_ind]
        u_dep = self._solve_velocity_constraints(
            np.array(q0), np.array(u_ind), np.array(u_dep_guess))
        u0 = u_ind + list(u_dep)
        for qi, q0i in zip(self.system.q, q0):
            self.initial_conditions[qi] = q0i
        for ui, u0i in zip(self.system.u, u0):
            self.initial_conditions[ui] = u0i

    def initialize(self, check_parameters: bool = True) -> None:
        """Initialize the simulator.

        Parameters
        ----------
        check_parameters : bool, optional
            Whether the constants and initial conditions should be checked for
            consistency with the system, by default True.
        """
        if self._initialized:
            raise RuntimeError("Simulator has already been initialized.")
        if self.system.eom_method is None:
            raise ValueError("Equations of motion have not been formed yet.")
        if self.constants is None:
            raise ValueError("Simulator has not been given any constants.")
        if self.initial_conditions is None:
            raise ValueError("Simulator has not been given any initial conditions.")
        if check_parameters:
            free_constants = self.system.mass_matrix_full.free_symbols.union(
                self.system.forcing_full.free_symbols)
            free_dyn = find_dynamicsymbols(self.system.mass_matrix_full).union(
                find_dynamicsymbols(self.system.forcing_full))
            state = set(self.system.q.col_join(self.system.u))
            controls = set(self.controls.keys())

            free_dyn = free_dyn.difference(state, controls)
            free_symbols = free_constants.union(free_dyn).difference(
                {dynamicsymbols._t})
            missing = free_symbols.difference(self.constants.keys())
            if missing:
                raise ValueError(f"Simulator is missing the following constants: "
                                 f"{missing}.")
            missing = set(self.system.q_ind.col_join(self.system.u_ind)).difference(
                self.initial_conditions.keys())
            if missing:
                raise ValueError(f"Simulator is missing the following initial "
                                 f"conditions: {missing}.")

        qdot_to_u = self.system.eom_method.kindiffdict() if isinstance(
            self.system.eom_method, KanesMethod) else {}
        t = dynamicsymbols._t
        self._p, self._p_vals = zip(*self.constants.items())
        self._c, self._c_funcs = zip(*self.controls.items())
        velocity_constraints = msubs(self.system.holonomic_constraints.diff(t).col_join(
            self.system.nonholonomic_constraints), qdot_to_u)
        self._eval_configuration_constraints = lambdify(
            (self.system.q_dep, self.system.q_ind, self._p),
            self.system.holonomic_constraints, cse=True)
        self._eval_velocity_constraints = lambdify(
            (self.system.u_dep, self.system.q, self.system.u_ind, self._p),
            velocity_constraints, cse=True)
        self._eval_eoms_matrices = lambdify(
            (t, self.system.q.col_join(self.system.u), self._p, self._c),
            (self.system.mass_matrix_full, self.system.forcing_full), cse=True)
        self.solve_initial_conditions()
        self._initialized = True

    def eval_rhs(self, t: np.float64, x: npt.NDArray[np.float64]
                 ) -> npt.NDArray[np.float64]:
        """Evaluate the right-hand side of the equations of motion."""
        mass_matrix, forcing = self._eval_eoms_matrices(
            t, x, self._p_vals, [cf(t) for cf in self._c_funcs])
        return np.linalg.solve(mass_matrix, np.squeeze(forcing))

    def solve(self, t_span: tuple[float, float], **kwargs
              ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Simulate the system.

        Parameters
        ----------
        t_span : tuple[float, float]
            The start and end times of the simulation.
        **kwargs
            Keyword arguments to pass to `scipy.integrate.solve_ivp`.
        """
        if not self._initialized:
            raise RuntimeError("Simulator has not been initialized yet.")
        x0 = np.array([self.initial_conditions[xi] for xi in self.system.q.col_join(
            self.system.u)])
        sol = solve_ivp(self.eval_rhs, t_span, x0, **kwargs)
        self._t = sol.t
        self._x = sol.y
        return self.t, self.x