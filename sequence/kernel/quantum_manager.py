"""This module defines the quantum manager class, to track quantum states.

The states may currently be defined in two possible ways:
    - KetState
    - DensityMatrix
    - FockDensityMatrix
    - Bell Diagonal

The manager defines an API for interacting with quantum states.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from numpy.typing import NDArray
from threading import Lock
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from .quantum_state import State

from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import gate_sequence_product, Gate
from numpy import cumsum, base_repr
from scipy.sparse import csr_matrix
from scipy.special import binom

from ..components.circuit import Circuit
from .quantum_state import KetState, DensityState, BellDiagonalState, TableauState
from .quantum_utils import *
from ..constants import KET_STATE_FORMALISM, DENSITY_MATRIX_FORMALISM, FOCK_DENSITY_MATRIX_FORMALISM, BELL_DIAGONAL_STATE_FORMALISM, STABILIZER_FORMALISM, TABLEAU_FORMALISM
from ..utils import log
import numpy as np
import stim
from stim import Tableau, TableauSimulator


class QuantumManager(ABC):
    """Class to track and manage quantum states (abstract).

    All states stored are of a single formalism (by default as a ket vector).

    Class Attributes:
        _registry (dict): mapping of formalism names to manager classes.
        _global_formalism_lock (Lock): lock for managing global formalism.
        _global_formalism (str): global formalism.

    Attributes:
        states (dict[int, State]): mapping of state keys to quantum state objects.
        _least_available (int): tracking the total number of quantum states in the quantum network
        truncation (int): maximally allowed number of excited states for elementary subsystems. Default is 1 for qubit.
        dim (int): subsystem Hilbert space dimension. dim = truncation + 1
    """
    _registry: dict = {}
    _global_formalism_lock = Lock()
    _global_formalism: str = KET_STATE_FORMALISM

    def __init__(self, truncation: int = 1):
        self.states: dict[int, "State"] = {}
        self._least_available: int = 0
        self.truncation = truncation
        self.dim = self.truncation + 1

    @classmethod
    def set_global_manager_formalism(cls, formalism: str):
        """Set the global manager formalism.

        Args:
            formalism (str): The formalism to set as the global manager formalism.
        """
        with cls._global_formalism_lock:
            if formalism not in cls._registry:
                raise ValueError(f"Quantum manager '{formalism}' is not registered.")
            cls._global_formalism = formalism

    @classmethod
    def get_active_formalism(cls):
        with cls._global_formalism_lock:
            return cls._global_formalism

    @classmethod
    def clear_active_formalism(cls):
        with cls._global_formalism_lock:
            cls._global_formalism = KET_STATE_FORMALISM

    @classmethod
    def register(cls, name: str, manager_class=None):
        """Register a quantum manager class.

        Args:
            name (str): The name of the quantum manager.
            manager_class (type, optional): The manager class to register.
        """
        if manager_class is not None:
            cls._registry[name] = manager_class
            return None

        def decorator(manager_cls):
            cls._registry[name] = manager_cls
            return manager_cls

        return decorator

    @classmethod
    def create(cls, *args, **kwargs) -> 'QuantumManager':
        """Create a new instance of the quantum manager.
        """
        active_formalism = cls.get_active_formalism()
        if active_formalism not in cls._registry:
            raise ValueError(f"Quantum manager '{active_formalism}' is not registered.")

        return cls._registry[active_formalism](*args, **kwargs)

    @abstractmethod
    def new(self, state) -> int:
        """Method to create a new quantum state.

        Args:
            state: complex amplitudes of new state. Type depends on type of subclass.

        Returns:
            int: key for new state generated.
        """
        pass

    def get(self, key: int) -> "State":
        """Method to get quantum state stored at an index.

        Args:
            key (int): key for quantum state.

        Returns:
            State: quantum state at supplied key.
        """
        return self.states[key]

    @abstractmethod
    def run_circuit(self, circuit: "Circuit", keys: list[int], meas_samp=None):
        """Method to run a circuit on a given set of quantum states.

        Args:
            circuit (Circuit): quantum circuit to apply.
            keys (list[int]): list of keys for quantum states to apply circuit to.
            meas_samp (float): random sample used for measurement.

        Returns:
            dict[int, int]: dictionary mapping qstate keys to measurement results.
        """

        assert len(keys) == circuit.size, "mismatch between circuit size and supplied qubits"
        if len(circuit.measured_qubits) > 0:
            assert meas_samp, "must specify random sample when measuring qubits"

    def _prepare_circuit(self, circuit: "Circuit", keys: list[int]):
        """Prepare the circuit for execution by constructing the necessary state and transformation matrices.
        
        Args:
            circuit (Circuit): quantum circuit to apply.
            keys (list[int]): list of keys for quantum states to apply circuit to.
        """
        old_states = []
        all_keys = []

        # go through keys and get all unique qstate objects
        for key in keys:
            qstate = self.states[key]
            if qstate.keys[0] not in all_keys:
                old_states.append(qstate.state)
                all_keys += qstate.keys

        # construct compound state; order qubits
        new_state = [1]
        for state in old_states:
            new_state = kron(new_state, state)

        # get circuit matrix; expand if necessary
        circ_mat = circuit.get_unitary_matrix()
        if circuit.size < len(all_keys):
            # pad size of circuit matrix if necessary
            diff = len(all_keys) - circuit.size
            circ_mat = kron(circ_mat, identity(2 ** diff))

        # apply any necessary swaps
        if not all([all_keys.index(key) == i for i, key in enumerate(keys)]):
            all_keys, swap_mat = self._swap_qubits(all_keys, keys)
            circ_mat = circ_mat @ swap_mat

        return new_state, all_keys, circ_mat

    @staticmethod
    def _swap_qubits(all_keys: list[int], keys: list[int]):
        """Swap qubits in the circuit.
        
        Args:
            all_keys (list[int]): The list of all qubit keys.
            keys (list[int]): The list of qubit keys to swap.
        """
        swap_circuit = QubitCircuit(N=len(all_keys))
        for i, key in enumerate(keys):
            j = all_keys.index(key)
            if j != i:
                gate = Gate("SWAP", targets=[i, j])
                swap_circuit.add_gate(gate)
                all_keys[i], all_keys[j] = all_keys[j], all_keys[i]
        swap_mat = gate_sequence_product(swap_circuit.propagators()).full()
        return all_keys, swap_mat

    @abstractmethod
    def set(self, keys: list[int], amplitudes: Any) -> None:
        """Method to set quantum state at a given key(s).

        Args:
            keys (list[int]): key(s) of state(s) to change.
            amplitudes (any): Amplitudes to set state to, type determined by type of subclass.
        """

        pass

    def remove(self, key: int) -> None:
        """Method to remove state stored at key.
        
        Args:
            key (int): The key of the state to remove.
        """
        del self.states[key]

    def set_states(self, states: dict):
        """Set multiple quantum states.

        Args:
            states (dict): A dictionary mapping keys to their corresponding quantum states.
        """
        self.states = states


@QuantumManager.register(KET_STATE_FORMALISM)
class QuantumManagerKet(QuantumManager):
    """Class to track and manage quantum states with the ket vector formalism."""

    def __init__(self, **kwargs):
        super().__init__()

    def new(self, state=(complex(1), complex(0))) -> int:
        key = self._least_available
        self._least_available += 1
        self.states[key] = KetState(state, [key])
        return key

    def run_circuit(self, circuit: "Circuit", keys: list[int], meas_samp=None) -> dict[int, int]:
        super().run_circuit(circuit, keys, meas_samp)
        new_state, all_keys, circ_mat = self._prepare_circuit(circuit, keys)

        new_state = circ_mat @ new_state

        if len(circuit.measured_qubits) == 0:
            # set state, return no measurement result
            new_ket = KetState(new_state, all_keys)
            for key in all_keys:
                self.states[key] = new_ket
            return {}
        else:
            # measure state (state reassignment done in _measure method)
            keys = [all_keys[i] for i in circuit.measured_qubits]
            return self._measure(new_state, keys, all_keys, meas_samp)

    def set(self, keys: list[int], amplitudes: list[complex]) -> None:
        super().set(keys, amplitudes)
        new_state = KetState(amplitudes, keys)
        for key in keys:
            self.states[key] = new_state

    def set_to_zero(self, key: int):
        self.set([key], [complex(1), complex(0)])

    def set_to_one(self, key: int):
        self.set([key], [complex(0), complex(1)])

    def _measure(self, state: list[complex], keys: list[int],
                 all_keys: list[int], meas_samp: float) -> dict[int, int]:
        """Method to measure qubits at given keys.

        SHOULD NOT be called individually; only from circuit method (unless for unit testing purposes).
        Modifies quantum state of all qubits given by all_keys.

        Args:
            state (list[complex]): state to measure.
            keys (list[int]): list of keys to measure.
            all_keys (list[int]): list of all keys corresponding to state.
            meas_samp (float): random sample used for measurement result.

        Returns:
            dict[int, int]: mapping of measured keys to measurement results.
        """

        if len(keys) == 1:
            if len(all_keys) == 1:
                prob_0 = measure_state_with_cache_ket(tuple(state))
                if meas_samp < prob_0:
                    result = 0
                else:
                    result = 1

            else:
                key = keys[0]
                num_states = len(all_keys)
                state_index = all_keys.index(key)
                state_0, state_1, prob_0 = measure_entangled_state_with_cache_ket(tuple(state), state_index, num_states)
                if meas_samp < prob_0:
                    new_state = array(state_0, dtype=complex)
                    result = 0
                else:
                    new_state = array(state_1, dtype=complex)
                    result = 1

            all_keys.remove(keys[0])

        else:
            # swap states into correct position
            if not all(
                    [all_keys.index(key) == i for i, key in enumerate(keys)]):
                all_keys, swap_mat = self._swap_qubits(all_keys, keys)
                state = swap_mat @ state

            # calculate meas probabilities and projected states
            len_diff = len(all_keys) - len(keys)
            new_states, probabilities = measure_multiple_with_cache_ket(
                tuple(state), len(keys), len_diff)

            # choose result, set as new state
            for i in range(int(2 ** len(keys))):
                if meas_samp < sum(probabilities[:i + 1]):
                    result = i
                    new_state = new_states[i]
                    break

            for key in keys:
                all_keys.remove(key)

        result_states = [array([1, 0]), array([0, 1])]
        result_digits = [int(x) for x in bin(result)[2:]]
        while len(result_digits) < len(keys):
            result_digits.insert(0, 0)

        for res, key in zip(result_digits, keys):
            # set to state measured
            new_state_obj = KetState(result_states[res], [key])
            self.states[key] = new_state_obj

        if len(all_keys) > 0:
            new_state_obj = KetState(new_state, all_keys)
            for key in all_keys:
                self.states[key] = new_state_obj

        return dict(zip(keys, result_digits))


@QuantumManager.register(DENSITY_MATRIX_FORMALISM)
class QuantumManagerDensity(QuantumManager):
    """Class to track and manage states with the density matrix formalism."""

    def __init__(self, **kwargs):
        super().__init__()

    def new(self,
            state=([complex(1), complex(0)], [complex(0), complex(0)])) -> int:
        key = self._least_available
        self._least_available += 1
        self.states[key] = DensityState(state, [key])
        return key

    def run_circuit(self, circuit: "Circuit", keys: list[int], meas_samp=None) -> dict[int, int]:
        super().run_circuit(circuit, keys, meas_samp)
        new_state, all_keys, circ_mat = super()._prepare_circuit(circuit, keys)

        new_state = circ_mat @ new_state @ circ_mat.conj().T

        if len(circuit.measured_qubits) == 0:
            # set state, return no measurement result
            new_state_obj = DensityState(new_state, all_keys)
            for key in all_keys:
                self.states[key] = new_state_obj
            return {}
        else:
            # measure state (state reassignment done in _measure method)
            keys = [all_keys[i] for i in circuit.measured_qubits]
            return self._measure(new_state, keys, all_keys, meas_samp)

    def set(self, keys: list[int], state: list[list[complex]]) -> None:
        """Method to set the quantum state at the given keys.

        The `state` argument should be passed as list[list[complex]], where each internal list is a row.
        However, the `state` may also be given as a one-dimensional pure state.
        If the list is one-dimensional, will be converted to matrix with the outer product operation.

        Args:
            keys (list[int]): list of quantum manager keys to modify.
            state: quantum state to set input keys to.
        """

        super().set(keys, state)
        new_state = DensityState(state, keys)
        for key in keys:
            self.states[key] = new_state

    def set_to_zero(self, key: int):
        self.set([key], [[complex(1), complex(0)], [complex(0), complex(0)]])

    def set_to_one(self, key: int):
        self.set([key], [[complex(0), complex(0)], [complex(0), complex(1)]])

    def _measure(self, state: list[list[complex]], keys: list[int],
                 all_keys: list[int], meas_samp: float) -> dict[int, int]:
        """Method to measure qubits at given keys.

        SHOULD NOT be called individually; only from circuit method (unless for unit testing purposes).
        Modifies quantum state of all qubits given by all_keys.

        Args:
            state (list[complex]): state to measure.
            keys (list[int]): list of keys to measure.
            all_keys (list[int]): list of all keys corresponding to state.
            meas_samp (float): random sample used for measurement result.

        Returns:
            dict[int, int]: mapping of measured keys to measurement results.
        """

        if len(keys) == 1:
            if len(all_keys) == 1:
                prob_0 = measure_state_with_cache_density(tuple(map(tuple, state)))
                if meas_samp < prob_0:
                    result = 0
                    new_state = [[1, 0], [0, 0]]
                else:
                    result = 1
                    new_state = [[0, 0], [0, 1]]

            else:
                key = keys[0]
                num_states = len(all_keys)
                state_index = all_keys.index(key)
                state_0, state_1, prob_0 = \
                    measure_entangled_state_with_cache_density(tuple(map(tuple, state)), state_index, num_states)
                if meas_samp < prob_0:
                    new_state = array(state_0, dtype=complex)
                    result = 0
                else:
                    new_state = array(state_1, dtype=complex)
                    result = 1

        else:
            # swap states into correct position
            if not all(
                    [all_keys.index(key) == i for i, key in enumerate(keys)]):
                all_keys, swap_mat = self._swap_qubits(all_keys, keys)
                state = swap_mat @ state @ swap_mat.T

            # calculate meas probabilities and projected states
            len_diff = len(all_keys) - len(keys)
            state_to_measure = tuple(map(tuple, state))
            new_states, probabilities = measure_multiple_with_cache_density(
                state_to_measure, len(keys), len_diff)

            # choose result, set as new state
            for i in range(int(2 ** len(keys))):
                if meas_samp < sum(probabilities[:i + 1]):
                    result = i
                    new_state = new_states[i]
                    break

        result_digits = [int(x) for x in bin(result)[2:]]
        while len(result_digits) < len(keys):
            result_digits.insert(0, 0)

        new_state_obj = DensityState(new_state, all_keys)
        for key in all_keys:
            self.states[key] = new_state_obj

        return dict(zip(keys, result_digits))


@QuantumManager.register(FOCK_DENSITY_MATRIX_FORMALISM)
class QuantumManagerDensityFock(QuantumManager):
    """Class to track and manage Fock states with the density matrix formalism."""

    def __init__(self, truncation: int = 1, **kwargs):
        # default truncation is 1 for 2-d Fock space.
        super().__init__(truncation=truncation)

    def new(self, state=None) -> int:
        """Method to create a new state with key

        Args:
            state (str | list[complex] | list[list[complex]]): amplitudes of new state.
                Default value is 'gnd': create zero-excitation state with current truncation.
                Other inputs are passed to the constructor of `DensityState`.
        """

        key = self._least_available
        self._least_available += 1
        if state is None:
            gnd = [1] + [0] * self.truncation
            self.states[key] = DensityState(gnd, [key], truncation=self.truncation)
        else:
            self.states[key] = DensityState(state, [key], truncation=self.truncation)

        return key

    def run_circuit(self, circuit: "Circuit", keys: list[int], meas_samp=None) -> dict[int, int]:
        """Currently the Fock states do not support quantum circuits.
        This method is only to implement abstract method of parent class and SHOULD NOT be called after instantiation.
        """
        raise Exception("run_circuit method of class QuantumManagerDensityFock called")

    def _generate_swap_operator(self, num_systems: int, i: int, j: int):
        """Helper function to generate swapping unitary.

        Args:
            num_systems (int): number of subsystems in state
            i (int): index of first subsystem to swap
            j (int): index of second subsystem to swap

        Returns:
            Array[int]: unitary swapping operator
        """

        size = self.dim ** num_systems
        swap_unitary = zeros((size, size))

        for old_index in range(size):
            old_str = base_repr(old_index, self.dim)
            old_str = old_str.zfill(num_systems)
            new_str = ''.join((old_str[:i], old_str[j], old_str[i + 1:j], old_str[i], old_str[j + 1:]))
            new_index = int(new_str, base=self.dim)
            swap_unitary[new_index, old_index] = 1

        return swap_unitary

    def _prepare_state(self, keys: list[int]):
        """Function to prepare states at given keys for operator application.

        Will take composite quantum state and swap subsystems to correspond with listed keys.
        Should not be called directly, but from method to apply operator or measure state.

        Args:
            keys (list[int]): keys for states to apply operator to.

        Returns:
            tuple(list[list[complex]], list[int]): tuple containing:
                1. new state to apply operator to, with keys swapped to be consecutive.
                2. list of keys corresponding to new state.
        """

        old_states = []
        all_keys = []

        # go through keys and get all unique qstate objects
        for key in keys:
            qstate = self.states[key]
            if qstate.keys[0] not in all_keys:
                old_states.append(qstate.state)
                all_keys += qstate.keys

        # construct compound state
        new_state = [1]
        for state in old_states:
            new_state = kron(new_state, state)

        # apply any necessary swaps to order keys
        if len(keys) > 1:

            # generate desired key order
            start_idx = all_keys.index(keys[0])
            if start_idx + len(keys) > len(all_keys):
                start_idx = len(all_keys) - len(keys)

            for i, key in enumerate(keys):
                i = i + start_idx
                j = all_keys.index(key)
                if j != i:
                    swap_unitary = self._generate_swap_operator(len(all_keys), i, j)
                    new_state = swap_unitary @ new_state @ swap_unitary.T
                    all_keys[i], all_keys[j] = all_keys[j], all_keys[i]

        return new_state, all_keys

    def _prepare_operator(self, all_keys: list[int], keys: list[int], operator) -> NDArray:
        # pad operator with identity
        left_dim = self.dim ** all_keys.index(keys[0])
        right_dim = self.dim ** (len(all_keys) - all_keys.index(keys[-1]) - 1)
        prepared_operator = operator

        if left_dim > 0:
            prepared_operator = kron(identity(left_dim), prepared_operator)
        if right_dim > 0:
            prepared_operator = kron(prepared_operator, identity(right_dim))

        return prepared_operator

    def apply_operator(self, operator: NDArray, keys: list[int]):
        prepared_state, all_keys = self._prepare_state(keys)
        prepared_operator = self._prepare_operator(all_keys, keys, operator)
        new_state = prepared_operator @ prepared_state @ prepared_operator.conj().T
        self.set(all_keys, new_state)

    def set(self, keys: list[int], state: list[list[complex]]) -> None:
        """Method to set the quantum state at the given keys.

        The `state` argument should be passed as list[list[complex]], where each internal list is a row.
        However, the `state` may also be given as a one-dimensional pure state.
        If the list is one-dimensional, will be converted to matrix with the outer product operation.

        Args:
            keys (list[int]): list of quantum manager keys to modify.
            state: quantum state to set input keys to.
        """

        super().set(keys, state)
        new_state = DensityState(state, keys, truncation=self.truncation)
        for key in keys:
            self.states[key] = new_state

    def set_to_zero(self, key: int):
        """set the state to ground (zero) state."""
        gnd = [1] + [0] * self.truncation
        self.set([key], gnd)

    def build_ladder(self):
        """Generate matrix of creation and annihilation (ladder) operators on truncated Hilbert space."""
        truncation = self.truncation
        data = array([sqrt(i + 1) for i in range(truncation)])  # elements in create/annihilation operator matrix
        row = array([i + 1 for i in range(truncation)])
        col = array([i for i in range(truncation)])
        create = csr_matrix((data, (row, col)), shape=(truncation + 1, truncation + 1)).toarray()
        destroy = create.conj().T

        return create, destroy

    def measure(self, keys: list[int], povms: list[NDArray], meas_samp: float) -> int:
        """Method to measure subsystems at given keys in POVM formalism.

        Serves as wrapper for private `_measure` method, performing quantum manager specific operations.

        Args:
            keys (list[int]): list of keys to measure.
            povms: (list[array]): list of POVM operators to use for measurement.
            meas_samp (float): random measurement sample to use for computing resultant state.

        Returns:
            int: measurement as index of matching POVM in supplied tuple.
        """

        new_state, all_keys = self._prepare_state(keys)
        return self._measure(new_state, keys, all_keys, povms, meas_samp)

    def _measure(self, state: list[list[complex]], keys: list[int],
                 all_keys: list[int], povms: list[NDArray], meas_samp: float) -> int:
        """Method to measure subsystems at given keys in POVM formalism.

        Modifies quantum state of all qubits given by all_keys, post-measurement operator determined
        by measurement operators which are chosen as square root of POVM operators.

        Args:
            state (list[list[complex]]): state to measure.
            keys (list[int]): list of keys to measure.
            all_keys (list[int]): list of all keys corresponding to state.
            povms: (list[NDArray]): list of POVM operators to use for measurement.
            meas_samp (float): random measurement sample to use for computing resultant state.

        Returns:
            int: measurement as index of matching POVM in supplied tuple.
        """

        state_tuple = tuple(map(tuple, state))
        povm_tuple = tuple([tuple(map(tuple, povm)) for povm in povms])
        new_state = None
        result = 0

        # calculate meas probabilities and projected states
        if len(keys) == 1:
            if len(all_keys) == 1:
                states, probs = measure_state_with_cache_fock_density(state_tuple, povm_tuple)

            else:
                key = keys[0]
                num_states = len(all_keys)
                state_index = all_keys.index(key)
                states, probs = \
                    measure_entangled_state_with_cache_fock_density(state_tuple, state_index, num_states, povm_tuple,
                                                                    self.truncation)

        else:
            indices = tuple([all_keys.index(key) for key in keys])
            states, probs = \
                measure_multiple_with_cache_fock_density(state_tuple, indices, len(all_keys), povm_tuple,
                                                         self.truncation)

        # calculate result based on measurement sample.
        prob_sum = cumsum(probs)
        for i, (output_state, p) in enumerate(zip(states, prob_sum)):
            if meas_samp < p:
                new_state = output_state
                result = i
                break

        """
        # for potential future work
        result_digits = [int(x) for x in base_repr(result, base=self.dim)[2:]]
        while len(result_digits) < len(keys):
            result_digits.insert(0, 0)

        # assign measured states
        for key, result in zip(keys, result_digits):
            state = [0] * self.dim
            state[result] = 1
            self.set([key], state)
        """

        for key in keys:
            self.states[key] = None  # clear the stored state at key (particle destructively measured)

        # assign remaining state
        if len(keys) < len(all_keys):
            indices = tuple([all_keys.index(key) for key in keys])
            new_state_tuple = tuple(map(tuple, new_state))
            remaining_state = density_partial_trace(new_state_tuple, indices, len(all_keys), self.truncation)
            remaining_keys = [key for key in all_keys if key not in keys]
            self.set(remaining_keys, remaining_state)

        return result

    def _build_loss_kraus_operators(self, loss_rate: float, all_keys: list[int], key: int) -> list[array]:
        """Method to build Kraus operators of a generalized amplitude damping channel.

        This represents the effect of photon loss.

        Args:
            loss_rate (float): loss rate for the quantum channel.
            all_keys (list[int]): list of all keys in affected state.
            key (int): key for subsystem experiencing loss.

        Returns:
            list[array]: list of generated Kraus operators.
        """

        assert 0 <= loss_rate <= 1
        kraus_ops = []

        for k in range(self.dim):
            total_kraus_op = zeros((self.dim ** len(all_keys), self.dim ** len(all_keys)))

            for n in range(k, self.dim):
                coeff = sqrt(binom(n, k)) * sqrt(((1 - loss_rate) ** (n - k)) * (loss_rate ** k))
                single_op = zeros((self.dim, self.dim))
                single_op[n - k, n] = 1
                total_op = self._prepare_operator(all_keys, [key], single_op)
                total_kraus_op += coeff * total_op

            kraus_ops.append(total_kraus_op)

        return kraus_ops

    def add_loss(self, key, loss_rate):
        """Method to apply generalized amplitude damping channel on a *single* subspace corresponding to `key`.

        Args:
            key (int): key for the subspace experiencing loss.
            loss_rate (float): loss rate for the quantum channel.
        """

        prepared_state, all_keys = self._prepare_state([key])
        kraus_ops = self._build_loss_kraus_operators(loss_rate, all_keys, key)
        output_state = zeros(prepared_state.shape, dtype=complex)

        for kraus_op in kraus_ops:
            output_state += kraus_op @ prepared_state @ kraus_op.conj().T

        self.set(all_keys, output_state)


@QuantumManager.register(BELL_DIAGONAL_STATE_FORMALISM)
class QuantumManagerBellDiagonal(QuantumManager):
    """Class to track and manage quantum states with the bell diagonal formalism.

    To be aligned with analytical formulae, we have assumed that successfully generated EPR pair is in Phi+ form.
    And note that the 4 BDS elements are in I, Z, X, Y order.

    * BDS is only used for entanglement distribution (generation, swapping, purification), assuming underlying errors being purely Pauli.
    * All manipulation results can be tracked analytically, without explicit quantum gates / channels / measurements.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def new(self, state=None) -> int:
        """Generates new quantum state key for quantum manager.

        NOTE: since this generates only one state, there will be no corresponding entangled state stored.
        The Bell diagonal state formalism assumes entangled states;
        thus, attempting to call `get` will return an exception until entangled.
        The purpose of this function is thus mainly to avoid state key collisions.

        Args:
            state (Any): to conform to type definition (does nothing).

        Returns:
            int: quantum state key corresponding to state.
        """
        key = self._least_available
        self._least_available += 1
        return key

    def get(self, key: int):
        if key not in self.states:
            raise Exception("Attempt to get Bell diagonal state before entanglement.")

        return super().get(key)

    def set(self, keys: list[int], diag_elems: list[float]) -> None:
        super().set(keys, diag_elems)
        # assert len(keys) == 2, "Bell diagonal states must have 2 keys."
        if len(keys) != 2:
            # raise Warning("bell diagonal quantum manager received invalid set request")  # optional
            for key in keys:
                if key in self.states:
                    self.states.pop(key)
            return
        new_state = BellDiagonalState(diag_elems, keys)
        for key in keys:
            self.states[key] = new_state

    def set_to_noiseless(self, keys: list[int]):
        self.set(keys, [float(1), float(0), float(0), float(0)])

    def run_circuit(self, *args, **kwargs):
        pass


# @QuantumManager.register(STABILIZER_FORMALISM)
# class QuantumManagerStabilizer(QuantumManager):
#     """
#     Quantum manager for stabilizer formalism using Stim with seeded sampling.
    
#     Key design principles:
#     - Each state stores its own Stim circuit and keys
#     - States start as single-qubit (2x2 density matrices)
#     - Manual grouping combines circuits while preserving qubit indices
#     - Efficient Pauli tomography via compiled samplers
#     - Deterministic behavior with seed support
#     """
    
#     def __init__(self, truncation: int = 1, shots: int = 1000, seed: int = None, **kwargs):
#         """
#         Initialize the stabilizer quantum manager with seed support.
        
#         Args:
#             truncation: Dormant parameter for future compatibility
#             shots: Number of samples for Pauli tomography (default 1000)
#             seed: Base seed for deterministic behavior (None for non-deterministic)
#         """
#         super().__init__(truncation=truncation)
#         self.shots = shots
#         self.base_seed = seed
#         self.rng = np.random.default_rng(seed) if seed is not None else None
#         self._seed_counter = 0
#         self.gate_fid = 1.0            # single-qubit gate fidelity (1.0 = no noise)
#         self.two_qubit_gate_fid = 1.0  # two-qubit gate fidelity (1.0 = no noise)
    
    
#     def _get_next_seed(self) -> Optional[int]:
#         """Generate next seed for deterministic sampling."""
#         if self.rng is None:
#             return None
#         self._seed_counter += 1
#         return self.rng.integers(0, 2**31)
    
#     def calculate_dm_from_circuit(self, circuit: stim.Circuit, keys: List[int]) -> np.ndarray:
#         """Calculate the density matrix from the circuit for the given keys."""
#         # Create a tableau simulator with a unique seed for deterministic behavior
#         seed = self._get_next_seed()
#         tableau = stim.TableauSimulator(seed=seed)
#         tableau.do(circuit)
        
#         # Extract the density matrix for the specified keys
#         # Note: Stim does not directly provide density matrices, so we will use the tableau to simulate measurements
#         # and reconstruct the state. For simplicity, we will return the tableau's internal state as a placeholder.
#         # In a full implementation, you would convert the tableau to a density matrix representation.
#         pass
    
    
#     def new(self, state: Optional[stim.Circuit] = None) -> int:
#         """
#         Create a new qubit with optional initial state circuit.
        
#         Args:
#             state: Optional stim.Circuit that prepares the initial state.
#                    If None, creates a qubit in |0⟩ state (empty circuit).
#                    If provided, must be a stim.Circuit object.
        
#         Returns:
#             Key for the new qubit
        
#         Raises:
#             TypeError: If state is not None and not a stim.Circuit
#         """
#         # Validate input type BEFORE incrementing counter
#         if state is not None and not isinstance(state, stim.Circuit):
#             raise TypeError(f"state must be a stim.Circuit or None, got {type(state)}")
        
#         # Only increment if validation passes
#         key = self._least_available
#         self._least_available += 1
        
#         # Use empty circuit if None provided
#         initial_circuit = stim.Circuit() if state is None else state
        
#         # Create single-qubit state with provided or empty circuit
#         state_obj = StabilizerState(
#             original_key=key,
#             keys=[key],
#             circuit=initial_circuit,
#             shots=self.shots,
#             truncation=self.truncation,
#             base_seed=self.base_seed
#         )
        
#         self.states[key] = state_obj
#         return key
    
    
#     def run_circuit(self, circuit, keys: List[int], meas_samp=None) -> Dict[int, int]:
#         """
#         Run a circuit on specified qubits with proper measurement collapse.

#         Behavior matches KetState formalism:
#         - Measurements collapse the state
#         - Measured qubits are SEPARATED from entangled group
#         - Measured qubits get independent collapsed state (|0> or |1>)
#         - Remaining qubits get post-measurement state

#         Args:
#             circuit: stim.Circuit or SeQUeNCe Circuit to apply
#             keys: Keys of qubits that circuit positions map to
#             meas_samp: Random sample [0,1] for measurement determinism (seeds the tableau)
        
#         Returns:
#             Dictionary mapping qubit keys to measurement results
#         """
#         # Convert to Stim circuit if needed
#         if isinstance(circuit, stim.Circuit):
#             stim_circuit = circuit
#         elif hasattr(circuit, 'gates') and hasattr(circuit, 'measured_qubits'):
#             stim_circuit = self._sequence_to_stim(circuit, keys)
#         else:
#             raise TypeError(f"circuit must be stim.Circuit or SeQUeNCe Circuit, got {type(circuit)}")
        
#         # Separate gates from measurements
#         gate_instructions = []
#         measurement_qubits = []  # List of qubit KEYS to measure
        
#         for instruction in stim_circuit:
#             if instruction.name == 'M':
#                 targets = [t.value for t in instruction.targets_copy()]
#                 for target in targets:
#                     if target < len(keys):
#                         measurement_qubits.append(keys[target])
#                     else:
#                         measurement_qubits.append(target)
#             else:
#                 gate_instructions.append(instruction)
        
#         # Group qubits if needed for multi-qubit operations
#         if len(keys) > 1:
#             self.group_qubits(keys)
        
#         if not keys:
#             return {}
        
#         state = self.states[keys[0]]
        
#         # Append gates to circuit
#         for instruction in gate_instructions:
#             gate_name = instruction.name
#             targets = [t.value for t in instruction.targets_copy()]
            
#             # Map circuit indices to actual qubit keys
#             mapped_targets = []
#             for target in targets:
#                 if target < len(keys):
#                     mapped_targets.append(keys[target])
#                 else:
#                     mapped_targets.append(target)
            
#             # Append to circuit
#             gate_args = instruction.gate_args_copy()
#             if gate_args:
#                 state.circuit.append(gate_name, mapped_targets, *gate_args)
#             else:
#                 state.circuit.append(gate_name, mapped_targets)

#             # Inject gate depolarization noise if fidelity < 1.0
#             if gate_name in ('H', 'X', 'Y', 'Z', 'S', 'S_DAG', 'T') and self.gate_fid < 1.0:
#                 state.circuit.append('DEPOLARIZE1', mapped_targets, 1 - self.gate_fid)
#             elif gate_name in ('CX', 'CZ') and self.two_qubit_gate_fid < 1.0:
#                 state.circuit.append('DEPOLARIZE2', mapped_targets, 1 - self.two_qubit_gate_fid)

#             # Invalidate tableau since circuit changed
#             state._tableau = None
        
#         # Handle measurements using TableauSimulator
#         measurement_results = {}
        
#         if measurement_qubits:
#             # CRITICAL: Only create new tableau if one doesn't exist
#             # If tableau already exists, it contains collapsed state from previous measurements
#             if state._tableau is None:
#                 if meas_samp is not None:
#                     seed = int(meas_samp * (2**31 - 1))
#                     state._tableau = stim.TableauSimulator(seed=seed)
#                 else:
#                     state._tableau = stim.TableauSimulator()
                
#                 if state.circuit and len(state.circuit) > 0:
#                     state._tableau.do(state.circuit)
            
#             # Measure each qubit using tableau (collapses state, preserves correlations)
#             for qubit_key in measurement_qubits:
#                 outcome = int(state.tableau.measure(qubit_key))
#                 measurement_results[qubit_key] = outcome
            
#             # Separate measured qubits from the group (matching KetState behavior)
#             remaining_keys = [k for k in state.keys if k not in measurement_qubits]
            
#             # Get the tableau BEFORE we modify states (for remaining qubits to share)
#             shared_tableau = state._tableau
#             shared_circuit = state.circuit.copy()
            
#             # Add measurement collapse to circuit for remaining qubits
#             for qubit_key in measurement_qubits:
#                 outcome = measurement_results[qubit_key]
#                 shared_circuit.append("M", [qubit_key])
#                 shared_circuit.append("R", [qubit_key])
#                 if outcome == 1:
#                     shared_circuit.append("X", [qubit_key])
            
#             # Create independent states for measured qubits
#             for qubit_key in measurement_qubits:
#                 outcome = measurement_results[qubit_key]
                
#                 # Create circuit for collapsed state
#                 collapsed_circuit = stim.Circuit()
#                 if outcome == 1:
#                     collapsed_circuit.append("X", [qubit_key])
                
#                 # Create new independent state
#                 new_state = StabilizerState(
#                     original_key=qubit_key,
#                     keys=[qubit_key],
#                     circuit=collapsed_circuit,
#                     shots=self.shots,
#                     truncation=self.truncation,
#                     base_seed=self.base_seed
#                 )
#                 self.states[qubit_key] = new_state
            
#             # Update remaining entangled qubits - they SHARE the same tableau
#             if remaining_keys:
#                 for key in remaining_keys:
#                     new_state = StabilizerState(
#                         original_key=key,
#                         keys=remaining_keys.copy(),
#                         circuit=shared_circuit,
#                         shots=self.shots,
#                         truncation=self.truncation,
#                         base_seed=self.base_seed
#                     )
#                     # CRITICAL: Share the same collapsed tableau
#                     new_state._tableau = shared_tableau
#                     self.states[key] = new_state
        
#         return measurement_results

#     def _map_stim_targets_preserve_records(self, targets: list, keys: List[int]) -> list:
#         """Map only qubit targets from local circuit indices to actual qstate keys.

#         Non-qubit Stim targets such as rec[-k], sweep bits, and combiners must be
#         preserved exactly for detector conversion to remain valid.
#         """
#         mapped = []
#         for target in targets:
#             if hasattr(target, "is_qubit_target") and target.is_qubit_target:
#                 idx = target.value
#                 if idx < 0:
#                     raise ValueError(f"Unexpected negative qubit target index: {idx}")
#                 mapped.append(keys[idx] if idx < len(keys) else idx)
#             else:
#                 mapped.append(target)
#         return mapped

#     def _count_stim_measurements(self, circuit: stim.Circuit) -> int:
#         count = 0
#         for inst in circuit:
#             if inst.name == "M":
#                 count += sum(1 for t in inst.targets_copy() if getattr(t, "is_qubit_target", False))
#         return count

#     def _postselect_z_measurement_compat(
#         self,
#         tableau: stim.TableauSimulator,
#         qubit_key: int,
#         desired: int,
#     ) -> int:
#         """Force a Z-basis measurement outcome when the Stim API supports it."""
#         desired = int(desired)

#         for method_name in ("postselect_z",):
#             method = getattr(tableau, method_name, None)
#             if method is None:
#                 continue
#             try:
#                 method(qubit_key, desired)
#                 return desired
#             except TypeError:
#                 try:
#                     method(qubit_key, desired_value=desired)
#                     return desired
#                 except TypeError:
#                     pass

#         actual = int(tableau.measure(qubit_key))
#         if actual != desired:
#             raise RuntimeError(
#                 "Unable to force sampled measurement outcome on tableau state. "
#                 "This Stim build does not appear to expose a compatible postselect_z API."
#             )
#         return actual

#     def _commit_sampled_stim_circuit(
#         self,
#         stim_circuit: stim.Circuit,
#         keys: List[int],
#         measurement_bits: np.ndarray,
#     ) -> Dict[int, int]:
#         """Commit a sampled single-shot circuit branch into manager state.

#         This currently supports standard Z-basis M measurements and assumes the
#         appended circuit itself does not rely on probabilistic noise channels.
#         """
#         if not keys:
#             return {}

#         if len(keys) > 1:
#             self.group_qubits(keys)

#         state = self.states[keys[0]]
#         shared_circuit = state.circuit.copy()
#         shared_tableau = state.tableau

#         measurement_results: Dict[int, int] = {}
#         measurement_qubits: List[int] = []
#         measurement_index = 0

#         for inst in stim_circuit:
#             mapped_targets = self._map_stim_targets_preserve_records(inst.targets_copy(), keys)
#             gate_args = inst.gate_args_copy()

#             if inst.name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
#                 continue

#             if inst.name == "M":
#                 for target in mapped_targets:
#                     if not isinstance(target, int):
#                         continue
#                     if measurement_index >= len(measurement_bits):
#                         raise ValueError("Insufficient sampled measurement bits for commit.")
#                     outcome = self._postselect_z_measurement_compat(
#                         shared_tableau,
#                         target,
#                         int(measurement_bits[measurement_index]),
#                     )
#                     measurement_index += 1
#                     measurement_results[target] = outcome
#                     measurement_qubits.append(target)
#                     shared_circuit.append("M", [target])
#                     shared_circuit.append("R", [target])
#                     if outcome == 1:
#                         shared_circuit.append("X", [target])
#                 continue

#             if inst.name in ("MX", "MY", "MR", "MRX", "MRY", "MRZ"):
#                 raise NotImplementedError(
#                     f"Commit mode does not yet support measurement instruction {inst.name}."
#                 )

#             if gate_args:
#                 shared_circuit.append(inst.name, mapped_targets, gate_args)
#                 temp = stim.Circuit()
#                 temp.append(inst.name, mapped_targets, gate_args)
#             else:
#                 shared_circuit.append(inst.name, mapped_targets)
#                 temp = stim.Circuit()
#                 temp.append(inst.name, mapped_targets)

#             # Exact branch commit is incompatible with additional sampled noise here.
#             if inst.name in ('H', 'X', 'Y', 'Z', 'S', 'S_DAG', 'T') and self.gate_fid < 1.0:
#                 raise NotImplementedError(
#                     "commit=True is not implemented for sampled single-qubit gate noise."
#                 )
#             elif inst.name in ('CX', 'CZ') and self.two_qubit_gate_fid < 1.0:
#                 raise NotImplementedError(
#                     "commit=True is not implemented for sampled two-qubit gate noise."
#                 )

#             if len(temp) > 0:
#                 shared_tableau.do(temp)

#         remaining_keys = [k for k in state.keys if k not in measurement_qubits]

#         for qubit_key in measurement_qubits:
#             outcome = measurement_results[qubit_key]
#             collapsed_circuit = stim.Circuit()
#             if outcome == 1:
#                 collapsed_circuit.append("X", [qubit_key])
#             self.states[qubit_key] = StabilizerState(
#                 original_key=qubit_key,
#                 keys=[qubit_key],
#                 circuit=collapsed_circuit,
#                 shots=self.shots,
#                 truncation=self.truncation,
#                 base_seed=self.base_seed,
#             )

#         if remaining_keys:
#             for key in remaining_keys:
#                 new_state = StabilizerState(
#                     original_key=key,
#                     keys=remaining_keys.copy(),
#                     circuit=shared_circuit,
#                     shots=self.shots,
#                     truncation=self.truncation,
#                     base_seed=self.base_seed,
#                 )
#                 new_state._tableau = shared_tableau
#                 self.states[key] = new_state

#         return measurement_results

#     def run_circuit_with_events(
#         self,
#         circuit,
#         keys: List[int],
#         shots: int = 1,
#         seed: Optional[int] = None,
#         append_observables: bool = False,
#         commit: bool = False,
#     ) -> Dict[str, Any]:
#         """Sample same-shot measurements and detector events for a circuit.

#         This method is additive and does not replace run_circuit(). It is intended
#         for FT verification and postselection workflows that need detector-safe
#         handling of rec[-k] targets.

#         Important:
#         - The returned events are sampled from the current state circuit prefix plus
#           the provided circuit.
#         - This method does not perform measurement collapse/separation updates on
#           the manager state. Legacy state evolution remains in run_circuit().
#         """
#         if isinstance(circuit, stim.Circuit):
#             stim_circuit = circuit
#         elif hasattr(circuit, 'gates') and hasattr(circuit, 'measured_qubits'):
#             stim_circuit = self._sequence_to_stim(circuit, keys)
#         else:
#             raise TypeError(f"circuit must be stim.Circuit or SeQUeNCe Circuit, got {type(circuit)}")

#         if not keys:
#             empty = np.zeros((shots, 0), dtype=np.bool_)
#             return {
#                 "measurements": empty,
#                 "detectors": empty,
#                 "observables": empty,
#                 "mapped_circuit": stim.Circuit(),
#             }

#         if len(keys) > 1:
#             self.group_qubits(keys)

#         state = self.states[keys[0]]
#         mapped = state.circuit.copy() if state.circuit is not None else stim.Circuit()

#         for inst in stim_circuit:
#             mapped_targets = self._map_stim_targets_preserve_records(inst.targets_copy(), keys)
#             gate_args = inst.gate_args_copy()
#             if gate_args:
#                 mapped.append(inst.name, mapped_targets, gate_args)
#             else:
#                 mapped.append(inst.name, mapped_targets)

#             # Mirror the existing run_circuit noise model on appended gates.
#             if inst.name in ('H', 'X', 'Y', 'Z', 'S', 'S_DAG', 'T') and self.gate_fid < 1.0:
#                 mapped.append('DEPOLARIZE1', mapped_targets, 1 - self.gate_fid)
#             elif inst.name in ('CX', 'CZ') and self.two_qubit_gate_fid < 1.0:
#                 mapped.append('DEPOLARIZE2', mapped_targets, 1 - self.two_qubit_gate_fid)

#         sampler = mapped.compile_sampler(seed=seed)
#         measurements = sampler.sample(shots=shots)

#         detectors = None
#         observables = None

#         if hasattr(mapped, "compile_m2d_converter"):
#             conv = mapped.compile_m2d_converter()
#             converted = None
#             try:
#                 converted = conv.convert(
#                     measurements=measurements,
#                     append_observables=append_observables,
#                 )
#             except TypeError:
#                 try:
#                     converted = conv.convert(
#                         measurements=measurements,
#                         separate_observables=append_observables,
#                     )
#                 except TypeError:
#                     converted = conv.convert(measurements)
#             except ValueError:
#                 try:
#                     converted = conv.convert(
#                         measurements=measurements,
#                         append_observables=append_observables,
#                     )
#                 except Exception:
#                     converted = conv.convert(measurements)

#             if isinstance(converted, tuple):
#                 if len(converted) > 0:
#                     detectors = converted[0]
#                 if len(converted) > 1:
#                     observables = converted[1]
#             else:
#                 detectors = converted
#                 if append_observables and hasattr(mapped, "num_detectors"):
#                     nd = mapped.num_detectors
#                     if detectors is not None and detectors.shape[1] >= nd:
#                         observables = detectors[:, nd:]
#                         detectors = detectors[:, :nd]

#         result = {
#             "measurements": measurements,
#             "detectors": detectors,
#             "observables": observables,
#             "mapped_circuit": mapped,
#         }

#         if commit:
#             if shots != 1:
#                 raise ValueError("commit=True currently requires shots=1.")
#             prefix_measurements = self._count_stim_measurements(state.circuit)
#             appended_measurements = measurements[0, prefix_measurements:]
#             result["measurement_results"] = self._commit_sampled_stim_circuit(
#                 stim_circuit,
#                 keys,
#                 appended_measurements,
#             )

#         return result

    
#     def _sequence_to_stim(self, circuit, keys: List[int]) -> stim.Circuit:
#         """Convert SeQUeNCe Circuit to Stim Circuit."""
#         stim_circuit = stim.Circuit()
        
#         for gate_info in circuit.gates:
#             gate_name, indices, arg = gate_info
            
#             # Map SeQUeNCe gate names to Stim
#             gate_map = {
#                 'h': 'H',
#                 'x': 'X',
#                 'y': 'Y',
#                 'z': 'Z',
#                 's': 'S',
#                 'sdg': 'S_DAG',
#                 't': 'T',
#                 'cx': 'CX',
#                 'cz': 'CZ',
#             }
            
#             if gate_name in gate_map:
#                 stim_name = gate_map[gate_name]
#                 # Use circuit indices directly (Stim expects 0-based positions)
#                 stim_circuit.append(stim_name, indices)
        
#         # Add measurements
#         for qubit_idx in circuit.measured_qubits:
#             if qubit_idx < len(keys):
#                 stim_circuit.append("M", [qubit_idx])
        
#         return stim_circuit
    
    
#     def group_qubits(self, keys: List[int]) -> None:
#         """
#         Manually group qubits together for joint density matrix computation.

#         Different StabilizerState objects will share the same circuit object.

#         Args:
#             keys: List of qubit keys to group together
#         """
#         if len(keys) <= 1:
#             return

#         # Check if already grouped - they should share the same circuit object
#         states = [self.states[key] for key in keys if key in self.states]
#         if len(states) > 0 and all(s.circuit is states[0].circuit for s in states):
#             return  # Already grouped (share same circuit)
        
#         # Combine circuits from all involved states
#         combined_circuit = stim.Circuit()
#         all_keys = []
        
#         # Collect all unique circuits (by object identity) and their qubits
#         seen_circuit_ids = set()
#         for key in keys:
#             state = self.states[key]
#             circuit_id = id(state.circuit)
            
#             if circuit_id not in seen_circuit_ids:
#                 seen_circuit_ids.add(circuit_id)
#                 all_keys.extend(state.keys)
                
#                 # Copy operations from this state's circuit
#                 for instruction in state.circuit:
#                     gate_args = instruction.gate_args_copy()
#                     targets = [t.value for t in instruction.targets_copy()]
                    
#                     if gate_args:
#                         combined_circuit.append(instruction.name, targets, *gate_args)
#                     else:
#                         combined_circuit.append(instruction.name, targets)
        
#         # Remove duplicates and sort
#         all_keys = sorted(list(set(all_keys)))
        
#         # Create new StabilizerState objects that share the SAME circuit object
#         for key in all_keys:
#             self.states[key] = StabilizerState(
#                 original_key=key,
#                 keys=all_keys,
#                 circuit=combined_circuit,  # Same object for all!
#                 shots=self.shots,
#                 truncation=self.truncation,
#                 base_seed=self.base_seed
#             )
    
    
#     def compute_density_matrix(self, keys: List[int]) -> np.ndarray:
#         """
#         Compute the density matrix for specific qubits on-demand.
        
#         Args:
#             keys: List of qubit keys to compute density matrix for
        
#         Returns:
#             Density matrix as numpy array (2^n × 2^n for n qubits)
#         """
#         if not keys:
#             raise ValueError("Must provide at least one key")
        
#         # Check if all keys exist
#         for key in keys:
#             if key not in self.states:
#                 raise KeyError(f"Key {key} not found in states")
        
#         # Check if all requested qubits share the same circuit
#         states = [self.states[k] for k in keys]
#         if all(s.circuit is states[0].circuit for s in states):
#             # They're grouped, use instance method
#             state = states[0]
#             return state.compute_density_matrix(keys_subset=keys)
        
#         # If not grouped, we need to combine their circuits temporarily
#         # This is a more complex case - for now, raise an error
#         raise ValueError(
#             f"Qubits {keys} are not grouped together. "
#             f"Call group_qubits({keys}) first before computing joint density matrix."
#         )
     
     
#     def set(self, keys: List[int], amplitudes_or_circuit: Union[List[complex], stim.Circuit]) -> None:
#         """
#         Set qubits to a state specified by amplitudes or a Stim circuit.
        
#         The Amplitudes cannot be an arbitrary amplitude because the conversion from amplitudes to circuit is non-trivial for un-standard states. 
#         The amplitudes we can pass in are limited to:
#         - Single qubit states: |0>, |1>, |+>, |->
#         - Two qubit Bell states: |Phi+>, |Phi->, |Psi+>, |Psi->
        
#         Behavior:
#         1. If qubits are in different groups, group them first
#         2. If setting a SUBSET of grouped qubits, ungroup the others first
#         3. REPLACE the circuit entirely with the new operations
#         4. All qubits in keys share the new circuit
        
#         Args:
#             keys: List of qubit keys to set
#             amplitudes_or_circuit: Either list of amplitudes or stim.Circuit
#         """
        
#         # Convert amplitudes to circuit if needed
#         if isinstance(amplitudes_or_circuit, stim.Circuit):
#             circuit = amplitudes_or_circuit
#         else:
#             # Convert amplitudes to circuit
#             amplitudes = amplitudes_or_circuit
#             circuit = stim.Circuit()
            
#             if len(keys) == 1 and len(amplitudes) == 2:
#                 # Single qubit states
#                 plus_state = [1/np.sqrt(2), 1/np.sqrt(2)]
#                 minus_state = [1/np.sqrt(2), -1/np.sqrt(2)]
#                 zero_state = [1, 0]
#                 one_state = [0, 1]

#                 if np.allclose(amplitudes, plus_state):
#                     circuit.append("H", [keys[0]])
#                 elif np.allclose(amplitudes, minus_state):
#                     circuit.append("X", [keys[0]])
#                     circuit.append("H", [keys[0]])
#                 elif np.allclose(amplitudes, one_state):
#                     circuit.append("X", [keys[0]])
#                 # |0> state needs no gates (R will set it to |0>)

#             elif len(keys) == 2 and len(amplitudes) == 4:
#                 # Two qubit Bell states
#                 phi_plus = [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]
#                 phi_minus = [1/np.sqrt(2), 0, 0, -1/np.sqrt(2)]
#                 psi_plus = [0, 1/np.sqrt(2), 1/np.sqrt(2), 0]
#                 psi_minus = [0, 1/np.sqrt(2), -1/np.sqrt(2), 0]

#                 if np.allclose(amplitudes, phi_plus):
#                     circuit.append("H", [keys[0]])
#                     circuit.append("CX", [keys[0], keys[1]])
#                 elif np.allclose(amplitudes, phi_minus):
#                     circuit.append("H", [keys[0]])
#                     circuit.append("CX", [keys[0], keys[1]])
#                     circuit.append("Z", [keys[0]])
#                 elif np.allclose(amplitudes, psi_plus):
#                     circuit.append("H", [keys[0]])
#                     circuit.append("CX", [keys[0], keys[1]])
#                     circuit.append("X", [keys[1]])
#                 elif np.allclose(amplitudes, psi_minus):
#                     circuit.append("H", [keys[0]])
#                     circuit.append("CX", [keys[0], keys[1]])
#                     circuit.append("X", [keys[1]])
#                     circuit.append("Z", [keys[0]])
        
#         # Step 1: Check if qubits need grouping
#         if len(keys) > 1:
#             # Check if all keys share the same circuit
#             states = [self.states[k] for k in keys if k in self.states]
#             if len(states) > 1 and not all(s.circuit is states[0].circuit for s in states):
#                 # Need to group first
#                 self.group_qubits(keys)
        
#         # Step 2: Get the state (all keys should now share the same circuit)
#         state = self.states[keys[0]]
        
#         # Step 3: Check if we're setting a SUBSET of grouped qubits
#         if set(keys) != set(state.keys):
#             # Setting a subset - need to ungroup the others first
#             qubits_not_being_set = [k for k in state.keys if k not in keys]
            
#             for key in qubits_not_being_set:
#                 # Give each unset qubit its own fresh circuit (reset to |0>)
#                 other_state = self.states[key]
#                 other_circuit = stim.Circuit()
#                 other_circuit.append("R", [key])
                
#                 # Create new state for this qubit
#                 other_state.circuit = other_circuit
#                 other_state.keys = [key]
#                 other_state._tableau = None
            
#             # Update state.keys to only include keys being set
#             state.keys = sorted(keys)
#             for key in keys:
#                 self.states[key].keys = sorted(keys)
        
#         # Step 4: Create completely FRESH circuit
#         new_circuit = stim.Circuit()

#         # Add the new operations
#         for instruction in circuit:
#             gate_args = instruction.gate_args_copy()
#             targets = [t.value for t in instruction.targets_copy()]
            
#             if gate_args:
#                 new_circuit.append(instruction.name, targets, *gate_args)
#             else:
#                 new_circuit.append(instruction.name, targets)
        
#         # Step 5: REPLACE the circuit for all qubits in keys
#         for key in keys:
#             if key in self.states:
#                 self.states[key].circuit = new_circuit
#                 self.states[key].keys = sorted(keys)
#                 self.states[key]._tableau = None
        
#     def get_density_matrix(self, key: int) -> np.ndarray:
#         """
#         Get the density matrix for a single qubit on-demand.
        
#         Args:
#             key: Qubit key
        
#         Returns:
#             Density matrix as numpy array (2x2 for single qubit)
#         """
#         if key not in self.states:
#             raise KeyError(f"Key {key} not found in states")
        
#         # Compute on-demand (doesn't cache)
#         return self.compute_density_matrix([key])
    
    
#     def get_circuit(self, key: int) -> stim.Circuit:
#         """Get the Stim circuit for a qubit's state."""
#         if key not in self.states:
#             raise KeyError(f"Key {key} not found")
        
#         return self.states[key].circuit

@QuantumManager.register(TABLEAU_FORMALISM)
class QuantumManagerTableau(QuantumManager):
    """Quantum manager scaffold for tableau-state simulation.

    This class follows SeQUeNCe's `QuantumManager` contract so the formalism
    can be swapped with minimal architectural changes.

    Notes:
        Current implementation supports state creation and assignment. Circuit
        execution is intentionally left as a staged TODO.
    """

    ONE_QUBIT_GATE_TIME_PS = 20_000
    TWO_QUBIT_GATE_TIME_PS = 250_000
    MEASUREMENT_TIME_PS = 500_000
    RESET_TIME_PS = 1_200_000

    def __init__(self, truncation: int = 1, seed: Optional[int] = None, **kwargs):
        """Initialize a tableau manager instance.

        Args:
            truncation (int): Hilbert-space truncation placeholder, retained to
                match the parent API and other formalisms.
            seed (Optional[int]): Base seed used for deterministic child-seed
                generation. If `None`, seeding is disabled.
            gate_fid (float): Single-qubit gate fidelity in [0, 1].
            two_qubit_gate_fid (float): Two-qubit gate fidelity in [0, 1].
            measurement_fid (float): Measurement fidelity in [0, 1].
            **kwargs: Extra keyword arguments accepted for compatibility.

        Returns:
            None.

        Notes:
            The manager uses a counter-based seed derivation strategy for
            reproducible, human-traceable per-state seeds.
        """
        super().__init__(truncation=truncation)
        # Base seed controls deterministic per-state/per-operation seed derivation.
        self.base_seed = seed
    
        self._seed_counter = 0         # Monotonic counter used with `base_seed` to produce unique child seeds.
        self.branch_rng = np.random.default_rng(seed)
        # Default fidelities are noiseless (1.0) unless explicitly overridden.
        self.gate_fid = float(kwargs.get("gate_fid", kwargs.get("single_qubit_gate_fid", 1.0)))
        self.two_qubit_gate_fid = float(kwargs.get("two_qubit_gate_fid", 1.0))
        self.measurement_fid = float(kwargs.get("measurement_fid", 1.0))
        self.initialization_fid = float(kwargs.get("initialization_fidelity", 1.0))
        self.gate_error_channel = str(kwargs.get("gate_error_channel", "pauli")).lower()  # Gate-noise mode: "depolarize" (uniform) or "pauli" (weighted).
        self.pauli_1q_weights = tuple(float(w) for w in kwargs.get("pauli_1q_weights", (1.0, 1.0, 1.0)))  # Relative PAULI_CHANNEL_1 weights in X, Y, Z order.
        raw_pauli_2q_weights = kwargs.get("pauli_2q_weights")
        if raw_pauli_2q_weights is None:
            self.pauli_2q_weights = self._derive_default_pauli_2q_weights(self.pauli_1q_weights)  # Default 2q bias inherits the 1q Pauli bias using an 80/20 single-vs-correlated split.
        else:
            self.pauli_2q_weights = tuple(float(w) for w in raw_pauli_2q_weights)
        if len(self.pauli_1q_weights) != 3:
            raise ValueError("pauli_1q_weights must have 3 entries for X, Y, Z.")
        if len(self.pauli_2q_weights) != 15:
            raise ValueError("pauli_2q_weights must have 15 entries in Stim PAULI_CHANNEL_2 order.")
        pauli_1q_total = float(sum(self.pauli_1q_weights))
        if pauli_1q_total <= 0.0:
            self.pauli_1q_weight_fractions = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        else:
            self.pauli_1q_weight_fractions = tuple(float(weight) / pauli_1q_total for weight in self.pauli_1q_weights)
        pauli_2q_total = float(sum(self.pauli_2q_weights))
        if pauli_2q_total <= 0.0:
            self.pauli_2q_weight_fractions = tuple(1.0 / 15.0 for _ in range(15))
        else:
            self.pauli_2q_weight_fractions = tuple(float(weight) / pauli_2q_total for weight in self.pauli_2q_weights)
        self.last_idle_time_ps_by_key: dict[int, int] = {}  # Last active simulation time per key.
        self.gate_1q_count = 0
        self.gate_2q_count = 0
        self.gate_1q_error_count = 0
        self.gate_2q_error_count = 0
        self.measurement_count = 0
        self.measurement_error_count = 0
        # self.idle_error_counts_by_key: dict[int, int] = {}
        # self.gate_1q_error_counts: dict[tuple[str, int, str], int] = {}
        # self.gate_1q_attempt_counts: dict[tuple[str, int], int] = {}
        # self.gate_2q_error_counts: dict[tuple[str, int, int, str], int] = {}
        # self.gate_2q_attempt_counts: dict[tuple[str, int, int], int] = {}
        # self.measurement_attempt_counts_by_key: dict[int, int] = {}
        # self.measurement_flip_counts_by_key: dict[int, int] = {}
        # self.initialization_flip_counts_by_key: dict[int, int] = {}

    def new(self, state: Optional[Union[TableauState, Tableau, TableauSimulator, stim.Circuit]] = None) -> int:
        """Create and register a new tableau-backed state key.

        Args:
            state (Optional[Union[TableauState, Tableau, TableauSimulator, Circuit]]):
                Optional initializer:
                - None: default seeded single-qubit simulator state.
                - TableauState: copied and rebound to the new key.
                - TableauSimulator: copied and rebound to the new key.
                - Tableau / Circuit: converted to simulator state.
                - Also accepts additional formats (e.g., gate-name strings,
                  stabilizer lists, numpy vectors/matrices, and mapping specs).

        Returns:
            int: Newly allocated state key.

        Raises:
            TypeError: If `state` is not a supported initializer type.
        """
        key = self._least_available
        self._least_available += 1
        if state is None:
            seed = self._next_seed()
            simulator = TableauSimulator(seed=seed)
            simulator.set_num_qubits(1)
            if self.initialization_fid < 1.0:
                flip_probability = max(0.0, min(1.0, 1.0 - self.initialization_fid))
                rng = np.random.default_rng(seed)
                if rng.random() < flip_probability:
                    simulator.x(0)
                    self.initialization_flip_counts_by_key[key] = self.initialization_flip_counts_by_key.get(key, 0) + 1
            self.states[key] = TableauState(state=simulator, keys=[key], seed=seed)
        else:
            self.states[key] = self._initialize_tableau_state(state, [key])
        self.last_idle_time_ps_by_key[key] = 0
        return key

    def set(self, keys: list[int], amplitudes: Any) -> None:
        """Assign a shared tableau state object to one or more keys.

        Args:
            keys (list[int]): State keys that should reference the same state.
            amplitudes (Any): State payload to assign.
                - `TableauState`: copied before assignment.
                - Any other object: wrapped in a new `TableauState`.

        Returns:
            None.

        Examples:
            `qm.set([k0], tableau_obj)`
            `qm.set([k0, k1], tableau_state)`

        Notes:
            As in other SeQUeNCe managers, all provided keys are bound to the
            same underlying state object to represent entanglement/grouping.
        """
        super().set(keys, amplitudes)
        state = self._initialize_tableau_state(amplitudes, list(keys))
        for key in keys:
            self.states[key] = state

    def run_circuit(self, circuit: Union[stim.Circuit, Circuit], keys: list[int], meas_samp=None, inject_gate_error: bool = False) -> dict[int, int]:
        """Execute a Stim or SeQUeNCe circuit on tableau-backed states.

        Args:
            circuit: Stim circuit or SeQUeNCe circuit to execute.
            keys (list[int]): Ordered keys mapped to circuit qubit indices.
            meas_samp: Measurement sample value used by run preparation.
            inject_gate_error: If `True`, execute ideal gates and then apply injected gate-noise channels.

        Returns:
            dict[int, int]: Measurement outcomes keyed by measured state keys.
        """
        circuit = self._to_stim_circuit(circuit)

        measured_qubits: list[int] = []
        saw_measurement = False
        supported_names = {"H", "X", "Y", "Z", "S", "S_DAG", "CX", "CZ", "SWAP", "M", "MX", "MY"}
        instruction_payloads: list[tuple[str, list[int]]] = []

        for instruction in circuit:
            name = instruction.name
            if name not in supported_names:
                raise ValueError(f"Unsupported stim instruction for tableau manager: {name}")

            targets = [int(target.value) for target in instruction.targets_copy()]
            if any(target < 0 or target >= len(keys) for target in targets):
                raise ValueError(f"Stim target out of range for {name}: {targets}")

            if name in {"M", "MX", "MY"}:
                saw_measurement = True
                measured_qubits.extend(targets)
            elif saw_measurement:
                raise ValueError("Tableau manager only supports terminal measurements.")
            instruction_payloads.append((name, targets))

        # Prepare validated inputs, merged/shared topology, and key index mapping.
        meas_samp, state_obj, key_to_local = self._prepare_circuit(len(keys), measured_qubits, keys, meas_samp)
        if state_obj is None:
            return {}

        base_simulator = state_obj.state
        simulator = base_simulator.copy() if hasattr(base_simulator, "copy") else base_simulator

        # Fast path: when gate-noise injection is disabled, batch all non-measurement
        # gates into one Stim circuit and execute it in a single simulator call.
        if not inject_gate_error:
            ideal_circuit = stim.Circuit()
            for name, targets in instruction_payloads:
                if name in {"M", "MX", "MY"}:
                    continue
                for elementary_targets in self._iter_elementary_gate_targets(name, targets):
                    local_targets = [key_to_local[keys[target]] for target in elementary_targets]
                    ideal_circuit.append(name, local_targets)
            if len(ideal_circuit) > 0:
                simulator.do(ideal_circuit)
        else:
            # Slow path: keep per-gate application when noise injection is enabled so
            # each ideal gate can be followed immediately by its noise channel.
            for name, targets in instruction_payloads:
                if name in {"M", "MX", "MY"}:
                    continue

                for elementary_targets in self._iter_elementary_gate_targets(name, targets):
                    circuit_keys = [keys[target] for target in elementary_targets]
                    local_targets = [key_to_local[key] for key in circuit_keys]
                    local_circuit = stim.Circuit()
                    local_circuit.append(name, local_targets)
                    simulator.do(local_circuit)
                    self._apply_gate_error(simulator, name, local_targets, circuit_keys)

        # No measurements: commit a fresh shared state object for all keys.
        if len(measured_qubits) == 0:
            committed_state = TableauState(state=simulator, keys=list(state_obj.keys))
            for key in state_obj.keys:
                self.states[key] = committed_state
            return {}

        # Readout-fidelity noise uses the same toggle as gate-noise injection.
        rng = None
        if inject_gate_error and self.measurement_fid < 1.0:
            rng_seed = int(float(meas_samp) * (2 ** 31 - 1))
            rng = np.random.default_rng(rng_seed)

        # Measure each requested key, report (possibly flipped) bit, and split measured states.
        results: dict[int, int] = {}
        measured_keys: list[int] = []
        for name, targets in instruction_payloads:
            if name not in {"M", "MX", "MY"}:
                continue

            for target in targets:
                measured_key = keys[target]
                local_target = key_to_local[measured_key]
                self.measurement_count += 1

                if name == "MX":
                    simulator.h(local_target)
                elif name == "MY":
                    simulator.s_dag(local_target)
                    simulator.h(local_target)

                physical_bit = int(simulator.measure(local_target))

                reported_bit = physical_bit
                if inject_gate_error and self.measurement_fid < 1.0 and rng is not None and rng.random() > self.measurement_fid:
                    reported_bit ^= 1
                    self.measurement_error_count += 1
                results[measured_key] = reported_bit
                measured_keys.append(measured_key)

                collapsed = TableauSimulator(seed=self._next_seed())
                collapsed.set_num_qubits(1)
                if physical_bit == 1:
                    collapsed.x(0)
                self.states[measured_key] = TableauState(state=collapsed, keys=[measured_key])

        # Physically drop measured qubits so simulator indices stay compact for remaining keys.
        simulator, remaining_keys = self._drop_keys_from_tableau_simulator(simulator, state_obj.keys, measured_keys)

        # Keep unmeasured keys grouped on the shared post-measurement simulator state.
        if remaining_keys:
            remaining_state = TableauState(state=simulator, keys=remaining_keys)
            for key in remaining_keys:
                self.states[key] = remaining_state

        return results

    def _iter_elementary_gate_targets(self, gate_name: str, targets: list[int]) -> list[list[int]]:
        """Split one Stim instruction target list into elementary gate targets.

        Args:
            gate_name: Stim gate name.
            targets: Flat instruction target list in circuit-local indices.

        Returns:
            list[list[int]]: Elementary target groups for one gate application each.
        """
        if gate_name in {"H", "X", "Y", "Z", "S", "S_DAG", "M", "MX", "MY"}:
            return [[target] for target in targets]
        if gate_name in {"CX", "CZ", "SWAP"}:
            if len(targets) % 2 != 0:
                raise ValueError(f"Expected even target count for {gate_name}, got {len(targets)}.")
            return [targets[index:index + 2] for index in range(0, len(targets), 2)]
        raise ValueError(f"Unsupported gate for elementary target split: {gate_name}")

    def get_circuit_duration(self, circuit: Union[stim.Circuit, Circuit]) -> int:
        """Return the estimated execution time of a circuit in picoseconds.

        Args:
            circuit: Stim circuit or SeQUeNCe circuit to estimate.

        Returns:
            int: Estimated circuit duration in picoseconds.
        """
        circuit = self._to_stim_circuit(circuit)
        duration_ps = 0

        for instruction in circuit:
            name = instruction.name
            targets_raw = instruction.targets_copy()
            if any(not getattr(target, "is_qubit_target", False) for target in targets_raw):
                raise RuntimeError(f"Unsupported non-qubit target in duration estimate: {name}")
            target_count = len(targets_raw)

            if name in {"H", "X", "Y", "Z", "S", "S_DAG"}:
                duration_ps += target_count * self.ONE_QUBIT_GATE_TIME_PS
            elif name in {"CX", "CZ", "SWAP"}:
                duration_ps += (target_count // 2) * self.TWO_QUBIT_GATE_TIME_PS
            elif name in {"M", "MX", "MY"}:
                duration_ps += target_count * self.MEASUREMENT_TIME_PS
            else:
                raise RuntimeError(f"Unsupported gate for duration estimate: {name}")

        return int(duration_ps)

    def get_reset_duration(self, num_qubits: int) -> int:
        """Return the estimated reset time in picoseconds.

        Args:
            num_qubits: Number of qubits being reset.

        Returns:
            int: Estimated reset duration in picoseconds.
        """
        if num_qubits < 0:
            raise RuntimeError(f"num_qubits must be >= 0, got {num_qubits}")
        return int(num_qubits * self.RESET_TIME_PS)

    def apply_idling_decoherence(self, keys: list[int], now_ps: int, t1_sec: float, t2_sec: float) -> None:
        """Apply time-based idling decoherence to the provided keys.

        Args:
            keys: Quantum-manager keys to decohere.
            now_ps: Current simulation time in picoseconds.
            t1_sec: Shared T1 time constant in seconds.
            t2_sec: Shared T2 time constant in seconds.

        Returns:
            None.
        """
        if t1_sec >= 1e9 and t2_sec >= 1e9:
            for key in keys:
                self.last_idle_time_ps_by_key[key] = now_ps
            return

        _, state_obj, key_to_local = self._prepare_circuit(len(keys), [], keys, 0.5)
        for key in keys:
            last_ps = self.last_idle_time_ps_by_key.get(key, now_ps)
            idle_sec = (now_ps - last_ps) * 1e-12
            # Skip keys with no positive idle interval since the last watermark update.
            if idle_sec <= 0.0:
                continue

            px = py = (1.0 - np.exp(-idle_sec / t1_sec)) / 4.0
            pz = (1.0 + np.exp(-idle_sec / t1_sec) - 2.0 * np.exp(-idle_sec / t2_sec)) / 4.0
            local = key_to_local[key]
            noise_circuit = stim.Circuit()
            noise_circuit.append("PAULI_CHANNEL_1", [local], [float(px), float(py), float(pz)])
            state_obj.state.do(noise_circuit)
            # sampled_branch = self._sample_pauli_channel_branch(
            #     channel_name="PAULI_CHANNEL_1",
            #     probs=[float(px), float(py), float(pz)],
            #     targets=[local],
            #     source="idle",
            # )
            # if sampled_branch == "X":
            #     state_obj.state.x(local)
            # elif sampled_branch == "Y":
            #     state_obj.state.y(local)
            # elif sampled_branch == "Z":
            #     state_obj.state.z(local)
            # if sampled_branch != "I":
            #     self.idle_error_counts_by_key[key] = self.idle_error_counts_by_key.get(key, 0) + 1
            # log.logger.info(
            #     f"pauli_channel_apply source=idle channel=PAULI_CHANNEL_1 "
            #     f"time_ps={now_ps} key={key} target={local} branch={sampled_branch} "
            #     f"inserted_error={int(sampled_branch != 'I')} "
            #     f"px={float(px):.6e} py={float(py):.6e} pz={float(pz):.6e}"
            # )

        for key in state_obj.keys:
            self.last_idle_time_ps_by_key[key] = now_ps
    
    def set_to_zero(self, key: int | list[int]) -> None:
        """Reset one or more qubits to the |0⟩ computational basis state.

        Args:
            key (int | list[int]): State key or keys of the qubits to reset.

        Returns:
            None.
        """
        if isinstance(key, list):
            for single_key in key:
                self.set_to_zero(single_key)
            return

        seed = self._next_seed()
        simulator = TableauSimulator(seed=seed)
        simulator.set_num_qubits(1)
        if self.initialization_fid < 1.0:
            flip_probability = max(0.0, min(1.0, 1.0 - self.initialization_fid))
            rng = np.random.default_rng(seed)
            if rng.random() < flip_probability:
                simulator.x(0)
                # self.initialization_flip_counts_by_key[key] = self.initialization_flip_counts_by_key.get(key, 0) + 1
        self.states[key] = TableauState(state=simulator, keys=[key], seed=seed)

    def set_to_one(self, key: int) -> None:
        """Reset a single qubit to the |1⟩ computational basis state.

        Args:
            key (int): State key of the qubit to reset.

        Returns:
            None.
        """
        seed = self._next_seed()
        sim = TableauSimulator(seed=seed)
        sim.set_num_qubits(1)
        sim.x(0)
        if self.initialization_fid < 1.0:
            flip_probability = max(0.0, min(1.0, 1.0 - self.initialization_fid))
            rng = np.random.default_rng(seed)
            if rng.random() < flip_probability:
                sim.x(0)
                # self.initialization_flip_counts_by_key[key] = self.initialization_flip_counts_by_key.get(key, 0) + 1
        self.states[key] = TableauState(state=sim, keys=[key], seed=seed)
        self.last_idle_time_ps_by_key[key] = 0

    def remove(self, key: int) -> None:
        """Remove a key and refresh the debug key layout map."""
        super().remove(key)
        self.last_idle_time_ps_by_key.pop(key, None)

    def reset_error_statistics(self) -> None:
        """Clear accumulated error counters for one logical-pair run.

        Args:
            None.

        Returns:
            None.
        """
        self.gate_1q_count = 0
        self.gate_2q_count = 0
        self.gate_1q_error_count = 0
        self.gate_2q_error_count = 0
        self.measurement_count = 0
        self.measurement_error_count = 0

    def get_error_statistics(self) -> dict[str, int | float]:
        """Return per-run gate accounting statistics.

        Args:
            None.

        Returns:
            dict[str, int | float]: One- and two-qubit gate counts, sampled gate-error counts, and measurement counts.
        """
        return {
            "gate_1q_count": self.gate_1q_count,
            "gate_2q_count": self.gate_2q_count,
            "gate_1q_error_count": self.gate_1q_error_count,
            "gate_2q_error_count": self.gate_2q_error_count,
            "measurement_count": self.measurement_count,
            "measurement_error_count": self.measurement_error_count,
        }

    def _next_seed(self) -> Optional[int]:
        """Return next seed value or None if unseeded."""
        # `None` means deterministic seeding is disabled for this manager.
        if self.base_seed is None:
            return None
        # Derive a reproducible child seed and advance the counter.
        seed = int(self.base_seed + self._seed_counter)
        self._seed_counter += 1
        return seed

    def _derive_default_pauli_2q_weights(self, pauli_1q_weights: tuple[float, float, float], single_only_fraction: float = 0.8) -> tuple[float, ...]:
        """Derive default 2q Pauli weights from the configured 1q Pauli bias.

        Args:
            pauli_1q_weights: Relative 1q Pauli weights in X, Y, Z order.
            single_only_fraction: Fraction of 2q faults assigned to one-sided Pauli terms.

        Returns:
            tuple[float, ...]: Relative PAULI_CHANNEL_2 weights in Stim order.
        """
        total = float(sum(pauli_1q_weights))
        if total <= 0.0:
            x_weight = 1.0 / 3.0
            y_weight = 1.0 / 3.0
            z_weight = 1.0 / 3.0
        else:
            x_weight = float(pauli_1q_weights[0]) / total
            y_weight = float(pauli_1q_weights[1]) / total
            z_weight = float(pauli_1q_weights[2]) / total

        correlated_fraction = 1.0 - float(single_only_fraction)
        single_scale = float(single_only_fraction) / 2.0
        return (
            single_scale * x_weight,                    # IX
            single_scale * y_weight,                    # IY
            single_scale * z_weight,                    # IZ
            single_scale * x_weight,                    # XI
            correlated_fraction * x_weight * x_weight,  # XX
            correlated_fraction * x_weight * y_weight,  # XY
            correlated_fraction * x_weight * z_weight,  # XZ
            single_scale * y_weight,                    # YI
            correlated_fraction * y_weight * x_weight,  # YX
            correlated_fraction * y_weight * y_weight,  # YY
            correlated_fraction * y_weight * z_weight,  # YZ
            single_scale * z_weight,                    # ZI
            correlated_fraction * z_weight * x_weight,  # ZX
            correlated_fraction * z_weight * y_weight,  # ZY
            correlated_fraction * z_weight * z_weight,  # ZZ
        )

    def _sample_pauli_channel_branch(self, channel_name: str, probs: list[float], targets: list[int], source: str) -> str:
        """Sample one realized Pauli-channel branch and log it.

        Args:
            channel_name: Either ``PAULI_CHANNEL_1`` or ``PAULI_CHANNEL_2``.
            probs: Non-identity branch probabilities in Stim order.
            targets: Simulator-local target indices.
            source: Short source tag such as ``idle`` or ``gate:CX``.

        Returns:
            str: Sampled Pauli label, e.g. ``I``, ``Z``, ``IX``, or ``ZZ``.
        """
        if channel_name == "PAULI_CHANNEL_1":
            branch_labels = ["I", "X", "Y", "Z"]
            if len(probs) != 3:
                raise ValueError("PAULI_CHANNEL_1 requires 3 probabilities.")
            identity_label = "I"
            identity_prob = max(0.0, 1.0 - sum(probs))
            branch_probs = [identity_prob, probs[0], probs[1], probs[2]]
        elif channel_name == "PAULI_CHANNEL_2":
            branch_labels = [
                "II",
                "IX", "IY", "IZ",
                "XI", "XX", "XY", "XZ",
                "YI", "YX", "YY", "YZ",
                "ZI", "ZX", "ZY", "ZZ",
            ]
            if len(probs) != 15:
                raise ValueError("PAULI_CHANNEL_2 requires 15 probabilities.")
            identity_label = "II"
            identity_prob = max(0.0, 1.0 - sum(probs))
            branch_probs = [identity_prob] + list(probs)
        else:
            raise ValueError(f"Unsupported channel_name '{channel_name}'.")

        total = float(sum(branch_probs))
        if total <= 0.0:
            sampled_branch = identity_label
        else:
            normalized_probs = [prob / total for prob in branch_probs]
            sampled_index = int(self.branch_rng.choice(len(branch_labels), p=normalized_probs))
            sampled_branch = branch_labels[sampled_index]

        # log.logger.info(
        #     f"pauli_channel_sample source={source} channel={channel_name} "
        #     f"targets={targets} branch={sampled_branch} "
        #     f"inserted_error={int(sampled_branch != identity_label)}"
        # )
        return sampled_branch

    def _apply_sampled_pauli_branch(self, simulator: TableauSimulator, targets: list[int], sampled_branch: str) -> None:
        """Apply a sampled Pauli branch directly to the simulator.

        Args:
            simulator: Active simulator to mutate.
            targets: Simulator-local target indices.
            sampled_branch: Sampled Pauli label such as ``I``, ``Z``, ``IX``, or ``ZZ``.

        Returns:
            None.
        """
        if len(sampled_branch) != len(targets):
            raise ValueError("sampled_branch length must match target count.")

        for pauli, target in zip(sampled_branch, targets):
            if pauli == "I":
                continue
            if pauli == "X":
                simulator.x(target)
            elif pauli == "Y":
                simulator.y(target)
            elif pauli == "Z":
                simulator.z(target)
            else:
                raise ValueError(f"Unsupported sampled Pauli '{pauli}'.")

    def _initialize_tableau_state(self, initializer: Union[TableauState, Tableau, TableauSimulator, np.ndarray, list[Union[int, float, complex]], tuple[Union[int, float, complex]]], keys: list[int]) -> TableauState:
        """Create a tableau state from a supported initializer.

        Args:
            initializer: Tableau-compatible initializer.
            keys (list[int]): Keys that should bind to the resulting state.

        Returns:
            TableauState: State bound to `keys`.
        """
        if isinstance(initializer, TableauState):
            state = initializer.copy()
            state.keys = list(keys)
            return state

        if isinstance(initializer, TableauSimulator):
            simulator = initializer.copy() if hasattr(initializer, "copy") else initializer
            return TableauState(state=simulator, keys=list(keys))

        if isinstance(initializer, Tableau):
            tableau = initializer
        elif isinstance(initializer, (np.ndarray, list, tuple)):
            tableau = Tableau.from_state_vector(np.asarray(initializer), endian="little")
        else:
            raise TypeError("Unsupported tableau initializer.")

        if len(tableau) != len(keys):
            raise ValueError(f"Initializer tableau has {len(tableau)} qubits but {len(keys)} keys were supplied.")

        # Load the initializer tableau into a fresh simulator bound to these keys.
        simulator = TableauSimulator(seed=self._next_seed())
        simulator.set_inverse_tableau(tableau.inverse())
        return TableauState(state=simulator, keys=list(keys))

    def _apply_gate_error(self, simulator: TableauSimulator, gate_name: str, targets: list[int], circuit_keys: list[int]) -> None:
        """Apply sampled gate-error noise after ideal gate application.

        Args:
            simulator (TableauSimulator): Active simulator to mutate.
            gate_name (str): Name of gate that was just applied.
            targets (list[int]): Simulator-local target indices for the gate.
            circuit_keys (list[int]): Quantum-manager keys targeted by the gate.

        Returns:
            None.
        """
        name = gate_name.upper()

        if name in {"H", "X", "Y", "Z", "S", "S_DAG"}:
            self.gate_1q_count += 1
            p_error = max(0.0, min(1.0, 1.5 * (1.0 - self.gate_fid)))
            if p_error <= 0.0:
                return
            if self.gate_error_channel == "depolarize":
                probs = [p_error / 3.0, p_error / 3.0, p_error / 3.0]
            elif self.gate_error_channel in {"pauli", "paulierror", "pauli_channel"}:
                probs = [p_error * weight for weight in self.pauli_1q_weight_fractions]
            else:
                raise ValueError("gate_error_channel must be 'depolarize' or 'pauli'.")
            sampled_branch = self._sample_pauli_channel_branch("PAULI_CHANNEL_1", probs, targets, f"gate:{name}")
            if sampled_branch != "I":
                self.gate_1q_error_count += 1
            self._apply_sampled_pauli_branch(simulator, targets, sampled_branch)
            return

        if name in {"CX", "CZ", "SWAP"}:
            self.gate_2q_count += 1
            p_error = min(1.0, 1.25 * (1.0 - self.two_qubit_gate_fid))
            if p_error <= 0.0:
                return
            if self.gate_error_channel == "depolarize":
                probs = [p_error / 15.0] * 15
            elif self.gate_error_channel in {"pauli", "paulierror", "pauli_channel"}:
                probs = [p_error * weight for weight in self.pauli_2q_weight_fractions]
            else:
                raise ValueError("gate_error_channel must be 'depolarize' or 'pauli'.")
            sampled_branch = self._sample_pauli_channel_branch("PAULI_CHANNEL_2", probs, targets, f"gate:{name}")
            if sampled_branch != "II":
                self.gate_2q_error_count += 1
            self._apply_sampled_pauli_branch(simulator, targets, sampled_branch)
            return

    def _to_stim_circuit(self, circuit: Union[stim.Circuit, Circuit]) -> stim.Circuit:
        """Normalize supported circuit input into a Stim circuit.

        Args:
            circuit: Stim circuit or SeQUeNCe circuit.

        Returns:
            stim.Circuit: Equivalent Stim circuit.
        """
        if isinstance(circuit, stim.Circuit):
            return circuit

        if not isinstance(circuit, Circuit):
            raise TypeError(f"circuit must be stim.Circuit or SeQUeNCe Circuit, got {type(circuit)}")

        converted = stim.Circuit()
        gate_map = {
            "h": "H",
            "x": "X",
            "y": "Y",
            "z": "Z",
            "s": "S",
            "sdg": "S_DAG",
            "cx": "CX",
            "cz": "CZ",
            "swap": "SWAP",
        }

        for gate_name, indices, arg in circuit.gates:
            if arg is not None:
                raise ValueError(f"Unsupported gate arg for tableau manager: {gate_name}")
            if gate_name not in gate_map:
                raise ValueError(f"Unsupported gate '{gate_name}' for tableau manager.")
            converted.append(gate_map[gate_name], list(indices))

        for qubit in circuit.measured_qubits:
            converted.append("M", [qubit])
        for qubit in circuit.measured_x_qubits:
            converted.append("MX", [qubit])

        return converted

    def _prepare_circuit(self, num_qubits: int, measured_qubits: list[int], keys: list[int], meas_samp=None) -> tuple[float | None, TableauState | None, dict[int, int]]:
        """Validate run input and prepare shared tableau context.

        Args:
            num_qubits (int): Circuit width.
            measured_qubits (list[int]): Measured qubit indices in circuit order.
            keys (list[int]): Ordered keys mapped to circuit qubit indices.
            meas_samp: Optional measurement sample from caller.

        Returns:
            tuple[float | None, TableauState | None, dict[int, int]]:
                Normalized measurement sample, shared tableau state object, and
                key-to-local index mapping.
        """
        if measured_qubits and meas_samp is None:
            meas_samp = 0.5

        if len(keys) != num_qubits:
            raise ValueError(f"circuit width ({num_qubits}) must equal len(keys) ({len(keys)}).")
        if not keys:
            return meas_samp, None, {}
        if len(set(keys)) != len(keys):
            raise ValueError(f"Duplicate keys are not allowed in run_circuit: {keys}")

        missing_keys = [key for key in keys if key not in self.states]
        if missing_keys:
            raise ValueError(f"Unknown key(s) in run_circuit: {missing_keys}")

        if any(i < 0 or i >= num_qubits for i in measured_qubits):
            raise ValueError(f"Measured qubit index out of range: {measured_qubits}")
        if len(set(measured_qubits)) != len(measured_qubits):
            raise ValueError(f"Duplicate measured qubit indices are not allowed: {measured_qubits}")

        # Collect each distinct shared tableau block touched by the requested keys.
        unique_states: list[TableauState] = []
        seen_state_ids: set[int] = set()
        for key in keys:
            qstate = self.states[key]
            if not isinstance(qstate, TableauState):
                raise ValueError(f"Expected TableauState for key {key}, got {type(qstate)}")
            if qstate.state.num_qubits != len(qstate.keys):
                raise RuntimeError(f"Tableau/key mismatch for state keys: {qstate.keys}")
            if id(qstate) not in seen_state_ids:
                seen_state_ids.add(id(qstate))
                unique_states.append(qstate)

        if len(unique_states) == 1:
            state_obj = unique_states[0]
        else:
            # Merge independent tableau blocks into one shared simulator state.
            merged_keys: list[int] = []
            merged_tableau = None
            for qstate in unique_states:
                merged_keys.extend(qstate.keys)
                block_tableau = qstate.current_tableau()
                merged_tableau = block_tableau if merged_tableau is None else merged_tableau + block_tableau

            if len(set(merged_keys)) != len(merged_keys):
                raise ValueError(f"Merged state contains duplicate keys: {merged_keys}")
            state_obj = self._initialize_tableau_state(merged_tableau, merged_keys)
            for key in merged_keys:
                self.states[key] = state_obj

        # Map manager keys to local qubit indices in the shared tableau block.
        key_to_local = {key: i for i, key in enumerate(state_obj.keys)}
        return meas_samp, state_obj, key_to_local

    def _drop_keys_from_tableau_simulator(self, simulator: TableauSimulator, state_keys: list[int], drop_keys: list[int]) -> tuple[TableauSimulator, list[int]]:
        """Drop arbitrary keys by swapping them to tail and truncating qubits.

        Args:
            simulator (TableauSimulator): Simulator to mutate in place.
            state_keys (list[int]): Current key order mapped to simulator qubit indices.
            drop_keys (list[int]): Keys to remove from simulator state.

        Returns:
            tuple[TableauSimulator, list[int]]: Mutated simulator and remaining key order.
        """
        drop_set = set(drop_keys)
        keep_keys = [key for key in state_keys if key not in drop_set]
        tail_keys = [key for key in state_keys if key in drop_set]
        desired_order = keep_keys + tail_keys
        working_keys = list(state_keys)

        # Permute simulator qubits so kept keys come first and dropped keys are at the tail.
        for i, key in enumerate(desired_order):
            j = working_keys.index(key)
            if i != j:
                simulator.swap(i, j)
                working_keys[i], working_keys[j] = working_keys[j], working_keys[i]

        # Truncate tail qubits to physically shrink simulator state.
        if tail_keys:
            simulator.set_num_qubits(len(keep_keys))

        return simulator, keep_keys
