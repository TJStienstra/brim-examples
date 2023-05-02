"""Module containing a datastructure for sympy systems, which can be pickled."""
from __future__ import annotations

from typing import Any

from sympy import Basic, Derivative, Function, ImmutableMatrix, Symbol
from sympy.physics.mechanics import dynamicsymbols, find_dynamicsymbols, msubs
from sympy.physics.mechanics._system import System
from sympy.physics.mechanics.method import _Methods
from sympy.utilities.iterables import iterable

__all__ = ["DataManager"]


class DataManager:
    """Data manager for systems, which is made to be pickled using dill."""

    def __init__(self, system: System, dyn_symbols: Any = "search"):
        self._q_ind = ImmutableMatrix()
        self._q_dep = ImmutableMatrix()
        self._u_ind = ImmutableMatrix()
        self._u_dep = ImmutableMatrix()
        self._kdes = ImmutableMatrix()
        self._hol_coneqs = ImmutableMatrix()
        self._nonhol_coneqs = ImmutableMatrix()
        self._mass_matrix_full = ImmutableMatrix()
        self._forcing_full = ImmutableMatrix()
        dyn_repl = self._get_dynamicsymbols_repl_dict(dyn_symbols, system)
        if isinstance(system, System):
            self._process_system(system, dyn_repl)
        else:
            raise TypeError(f"{system} is of type {type(system)} not of type {System}.")

    @staticmethod
    def _create_symbol_name(symbol: Function | Derivative):
        if symbol.is_Derivative:
            if len(symbol.args) == 2 and symbol.args[1][0] == dynamicsymbols._t:
                return "d" * symbol.args[1][1] + symbol.args[0].name
            else:
                raise NotImplementedError(f"No conversion for {symbol} has been "
                                          f"implemented.")
        return symbol.name

    @staticmethod
    def _get_dynamicsymbols_repl_dict(dyn_symbols: Any, eom_method: _Methods
                                      ) -> dict[Basic, Symbol]:
        if dyn_symbols == "search":
            free_symbols = set().union(
                eom_method.mass_matrix_full.free_symbols,
                eom_method.forcing_full.free_symbols)
            used_names = {sym.name for sym in free_symbols}
            dyn_symbols = set().union(
                find_dynamicsymbols(eom_method.mass_matrix_full),
                find_dynamicsymbols(eom_method.forcing_full),
                eom_method.q,
                eom_method.q.diff(dynamicsymbols._t),
                eom_method.u,
                eom_method.u.diff(dynamicsymbols._t),
            )
            dyn_repl = {}
            for dyn_symbol in dyn_symbols:
                name = DataManager._create_symbol_name(dyn_symbol)
                while name in used_names:
                    name += "_"
                dyn_repl[dyn_symbol] = Symbol(name)
                used_names.add(name)
        elif isinstance(dyn_symbols, dict):
            dyn_repl = dyn_symbols
        elif not isinstance(dyn_symbols, str) and iterable(dyn_symbols):
            dyn_repl = dict(zip(dyn_symbols, [Symbol(DataManager._create_symbol_name(
                sym)) for sym in dyn_symbols]))
        else:
            raise TypeError(f"dyn_symbols must be a dict, iterable or 'search', not "
                            f"{dyn_symbols}.")
        return dyn_repl

    def _process_system(self, system: System, dyn_repl: dict[Basic, Symbol]) -> None:
        """Process a system and store the data in the data manager."""
        self._q_ind = ImmutableMatrix(msubs(system.q_ind, dyn_repl))
        self._q_dep = ImmutableMatrix(msubs(system.q_dep, dyn_repl))
        self._u_ind = ImmutableMatrix(msubs(system.u_ind, dyn_repl))
        self._u_dep = ImmutableMatrix(msubs(system.u_dep, dyn_repl))
        self._kdes = ImmutableMatrix(msubs(system.kdes, dyn_repl))
        self._hol_coneqs = ImmutableMatrix(msubs(
            system.holonomic_constraints, dyn_repl))
        self._nonhol_coneqs = ImmutableMatrix(msubs(
            system.nonholonomic_constraints, dyn_repl))
        self._mass_matrix_full = ImmutableMatrix(msubs(
            system.mass_matrix_full, dyn_repl))
        self._forcing_full = ImmutableMatrix(msubs(system.forcing_full, dyn_repl))

    @property
    def q(self) -> ImmutableMatrix:
        """Matrix of all the generalized coordinates."""
        return self._q_ind.col_join(self._q_dep)

    @property
    def u(self) -> ImmutableMatrix:
        """Matrix of all the generalized speeds."""
        return self._u_ind.col_join(self._u_dep)

    @property
    def q_ind(self) -> ImmutableMatrix:
        """Matrix of the independent generalized coordinates."""
        return self._q_ind

    @property
    def q_dep(self) -> ImmutableMatrix:
        """Matrix of the dependent generalized coordinates."""
        return self._q_dep

    @property
    def u_ind(self) -> ImmutableMatrix:
        """Matrix of the independent generalized speeds."""
        return self._u_ind

    @property
    def u_dep(self) -> ImmutableMatrix:
        """Matrix of the dependent generalized speeds."""
        return self._u_dep

    @property
    def kdes(self) -> ImmutableMatrix:
        """Kinematic differential equations."""
        return self._kdes

    @property
    def holonomic_constraints(self) -> ImmutableMatrix:
        """Matrix with the holonomic constraints as rows."""
        return self._hol_coneqs

    @property
    def nonholonomic_constraints(self) -> ImmutableMatrix:
        """Matrix with the nonholonomic constraints as rows."""
        return self._nonhol_coneqs

    @property
    def mass_matrix_full(self) -> ImmutableMatrix:
        """Full mass matrix."""
        return self._mass_matrix_full

    @property
    def forcing_full(self) -> ImmutableMatrix:
        """Full forcing vector."""
        return self._forcing_full

    @property
    def mass_matrix(self) -> ImmutableMatrix:
        """Mass matrix of only the dynamic part."""
        n = len(self.q)
        return self._mass_matrix_full[n:, n:]

    @property
    def forcing(self) -> ImmutableMatrix:
        """Forcing vector of only the dynamic part."""
        return self._forcing_full[len(self.q):, :]

    @property
    def mass_matrix_kin(self) -> ImmutableMatrix:
        """Mass matrix of only the kinematic part."""
        n = len(self.q)
        return self._mass_matrix_full[:n, :n]

    @property
    def forcing_kin(self) -> ImmutableMatrix:
        """Forcing vector of only the kinematic part."""
        return self._forcing_full[:len(self.q), :]
