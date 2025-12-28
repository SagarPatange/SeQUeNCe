"""This module defines the quantum manager class, to track quantum states.

The states may currently be defined in two possible ways:
    - KetState
    - DensityMatrix
    - FockDensityMatrix
    - Bell Diagonal

The manager defines an API for interacting with quantum states.
"""
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from threading import Lock
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from .quantum_state import State

from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import gate_sequence_product, Gate
from numpy import cumsum, base_repr
from scipy.sparse import csr_matrix
from scipy.special import binom

from ..components.circuit import Circuit
from .quantum_state import KetState, DensityState, BellDiagonalState, StabilizerState
from .quantum_utils import *
from ..constants import KET_STATE_FORMALISM, DENSITY_MATRIX_FORMALISM, FOCK_DENSITY_MATRIX_FORMALISM, BELL_DIAGONAL_STATE_FORMALISM, STABILIZER_FORMALISM
import numpy as np
import stim

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


@QuantumManager.register(STABILIZER_FORMALISM)
class QuantumManagerStabilizer(QuantumManager):
    """
    Quantum manager for stabilizer formalism using Stim with seeded sampling.
    
    Key design principles:
    - Each state stores its own Stim circuit and keys
    - States start as single-qubit (2x2 density matrices)
    - Manual grouping combines circuits while preserving qubit indices
    - Efficient Pauli tomography via compiled samplers
    - Deterministic behavior with seed support
    """
    
    def __init__(self, truncation: int = 1, shots: int = 1000, seed: int = None, **kwargs):
        """
        Initialize the stabilizer quantum manager with seed support.
        
        Args:
            truncation: Dormant parameter for future compatibility
            shots: Number of samples for Pauli tomography (default 1000)
            seed: Base seed for deterministic behavior (None for non-deterministic)
        """
        super().__init__(truncation=truncation)
        self.shots = shots
        self.base_seed = seed
        self.rng = np.random.default_rng(seed) if seed is not None else None
        self._seed_counter = 0
    
    
    def _get_next_seed(self) -> Optional[int]:
        """Generate next seed for deterministic sampling."""
        if self.rng is None:
            return None
        self._seed_counter += 1
        return self.rng.integers(0, 2**31)
    
    
    def new(self, state: Optional[stim.Circuit] = None) -> int:
        """
        Create a new qubit with optional initial state circuit.
        
        Args:
            state: Optional stim.Circuit that prepares the initial state.
                   If None, creates a qubit in |0⟩ state (empty circuit).
                   If provided, must be a stim.Circuit object.
        
        Returns:
            Key for the new qubit
        
        Raises:
            TypeError: If state is not None and not a stim.Circuit
        """
        # Validate input type BEFORE incrementing counter
        if state is not None and not isinstance(state, stim.Circuit):
            raise TypeError(f"state must be a stim.Circuit or None, got {type(state)}")
        
        # Only increment if validation passes
        key = self._least_available
        self._least_available += 1
        
        # Use empty circuit if None provided
        initial_circuit = stim.Circuit() if state is None else state
        
        # Create single-qubit state with provided or empty circuit
        state_obj = StabilizerState(
            original_key=key,
            keys=[key],
            circuit=initial_circuit,
            shots=self.shots,
            truncation=self.truncation,
            base_seed=self.base_seed
        )
        
        self.states[key] = state_obj
        return key
    
    
    def run_circuit(self, circuit, keys: List[int], meas_samp=None, compute_dm: bool = True) -> Dict[int, int]:
        """
        Run a circuit on specified qubits with proper measurement collapse.
        
        Behavior matches KetState formalism:
        - Measurements collapse the state
        - Measured qubits are SEPARATED from entangled group
        - Measured qubits get independent collapsed state (|0> or |1>)
        - Remaining qubits get post-measurement state
        
        Args:
            circuit: stim.Circuit or SeQUeNCe Circuit to apply
            keys: Keys of qubits that circuit positions map to
            meas_samp: Random sample [0,1] for measurement determinism (seeds the tableau)
            compute_dm: Whether to recompute density matrix (default True)
        
        Returns:
            Dictionary mapping qubit keys to measurement results
        """
        # Convert to Stim circuit if needed
        if isinstance(circuit, stim.Circuit):
            stim_circuit = circuit
        elif hasattr(circuit, 'gates') and hasattr(circuit, 'measured_qubits'):
            stim_circuit = self._sequence_to_stim(circuit, keys)
        else:
            raise TypeError(f"circuit must be stim.Circuit or SeQUeNCe Circuit, got {type(circuit)}")
        
        # Separate gates from measurements
        gate_instructions = []
        measurement_qubits = []  # List of qubit KEYS to measure
        
        for instruction in stim_circuit:
            if instruction.name == 'M':
                targets = [t.value for t in instruction.targets_copy()]
                for target in targets:
                    if target < len(keys):
                        measurement_qubits.append(keys[target])
                    else:
                        measurement_qubits.append(target)
            else:
                gate_instructions.append(instruction)
        
        # Group qubits if needed for multi-qubit operations
        if len(keys) > 1:
            self.group_qubits(keys)
        
        if not keys:
            return {}
        
        state = self.states[keys[0]]
        
        # Append gates to circuit
        for instruction in gate_instructions:
            gate_name = instruction.name
            targets = [t.value for t in instruction.targets_copy()]
            
            # Map circuit indices to actual qubit keys
            mapped_targets = []
            for target in targets:
                if target < len(keys):
                    mapped_targets.append(keys[target])
                else:
                    mapped_targets.append(target)
            
            # Append to circuit
            gate_args = instruction.gate_args_copy()
            if gate_args:
                state.circuit.append(gate_name, mapped_targets, *gate_args)
            else:
                state.circuit.append(gate_name, mapped_targets)
            
            # Invalidate tableau since circuit changed
            state._tableau = None
        
        # Handle measurements using TableauSimulator
        measurement_results = {}
        
        if measurement_qubits:
            # CRITICAL: Only create new tableau if one doesn't exist
            # If tableau already exists, it contains collapsed state from previous measurements
            if state._tableau is None:
                if meas_samp is not None:
                    seed = int(meas_samp * (2**31 - 1))
                    state._tableau = stim.TableauSimulator(seed=seed)
                else:
                    state._tableau = stim.TableauSimulator()
                
                if state.circuit and len(state.circuit) > 0:
                    state._tableau.do(state.circuit)
            
            # Measure each qubit using tableau (collapses state, preserves correlations)
            for qubit_key in measurement_qubits:
                outcome = int(state.tableau.measure(qubit_key))
                measurement_results[qubit_key] = outcome
            
            # Separate measured qubits from the group (matching KetState behavior)
            remaining_keys = [k for k in state.keys if k not in measurement_qubits]
            
            # Get the tableau BEFORE we modify states (for remaining qubits to share)
            shared_tableau = state._tableau
            shared_circuit = state.circuit.copy()
            
            # Add measurement collapse to circuit for remaining qubits
            for qubit_key in measurement_qubits:
                outcome = measurement_results[qubit_key]
                shared_circuit.append("M", [qubit_key])
                shared_circuit.append("R", [qubit_key])
                if outcome == 1:
                    shared_circuit.append("X", [qubit_key])
            
            # Create independent states for measured qubits
            for qubit_key in measurement_qubits:
                outcome = measurement_results[qubit_key]
                
                # Create circuit for collapsed state
                collapsed_circuit = stim.Circuit()
                if outcome == 1:
                    collapsed_circuit.append("X", [qubit_key])
                
                # Create new independent state
                new_state = StabilizerState(
                    original_key=qubit_key,
                    keys=[qubit_key],
                    circuit=collapsed_circuit,
                    shots=self.shots,
                    truncation=self.truncation,
                    base_seed=self.base_seed
                )
                self.states[qubit_key] = new_state
            
            # Update remaining entangled qubits - they SHARE the same tableau
            if remaining_keys:
                for key in remaining_keys:
                    new_state = StabilizerState(
                        original_key=key,
                        keys=remaining_keys.copy(),
                        circuit=shared_circuit,
                        shots=self.shots,
                        truncation=self.truncation,
                        base_seed=self.base_seed
                    )
                    # CRITICAL: Share the same collapsed tableau
                    new_state._tableau = shared_tableau
                    self.states[key] = new_state
        
        # Optionally recompute density matrix (only if no measurements)
        if compute_dm and not measurement_qubits:
            state = self.states[keys[0]]
            full_dm = state.compute_density_matrix(keys_subset=None)
            for key in state.keys:
                self.states[key].set_density_matrix(full_dm)
        
        return measurement_results
    
    
    def _sequence_to_stim(self, circuit, keys: List[int]) -> stim.Circuit:
        """Convert SeQUeNCe Circuit to Stim Circuit."""
        stim_circuit = stim.Circuit()
        
        for gate_info in circuit.gates:
            gate_name, indices, arg = gate_info
            
            # Map SeQUeNCe gate names to Stim
            gate_map = {
                'h': 'H',
                'x': 'X',
                'y': 'Y',
                'z': 'Z',
                's': 'S',
                'sdg': 'S_DAG',
                't': 'T',
                'cx': 'CX',
                'cz': 'CZ',
            }
            
            if gate_name in gate_map:
                stim_name = gate_map[gate_name]
                # Use circuit indices directly (Stim expects 0-based positions)
                stim_circuit.append(stim_name, indices)
        
        # Add measurements
        for qubit_idx in circuit.measured_qubits:
            if qubit_idx < len(keys):
                stim_circuit.append("M", [qubit_idx])
        
        return stim_circuit
    
    
    def group_qubits(self, keys: List[int]) -> None:
        """
        Manually group qubits together for joint density matrix computation.

        Different StabilizerState objects will share the same circuit object.

        Args:
            keys: List of qubit keys to group together
        """
        if len(keys) <= 1:
            return

        # Check if already grouped - they should share the same circuit object
        states = [self.states[key] for key in keys if key in self.states]
        if len(states) > 0 and all(s.circuit is states[0].circuit for s in states):
            return  # Already grouped (share same circuit)
        
        # Combine circuits from all involved states
        combined_circuit = stim.Circuit()
        all_keys = []
        
        # Collect all unique circuits (by object identity) and their qubits
        seen_circuit_ids = set()
        for key in keys:
            state = self.states[key]
            circuit_id = id(state.circuit)
            
            if circuit_id not in seen_circuit_ids:
                seen_circuit_ids.add(circuit_id)
                all_keys.extend(state.keys)
                
                # Copy operations from this state's circuit
                for instruction in state.circuit:
                    gate_args = instruction.gate_args_copy()
                    targets = [t.value for t in instruction.targets_copy()]
                    
                    if gate_args:
                        combined_circuit.append(instruction.name, targets, *gate_args)
                    else:
                        combined_circuit.append(instruction.name, targets)
        
        # Remove duplicates and sort
        all_keys = sorted(list(set(all_keys)))
        
        # Create new StabilizerState objects that share the SAME circuit object
        for key in all_keys:
            self.states[key] = StabilizerState(
                original_key=key,
                keys=all_keys,
                circuit=combined_circuit,  # Same object for all!
                shots=self.shots,
                truncation=self.truncation,
                base_seed=self.base_seed
            )
    
    
    def compute_density_matrix(self, keys: List[int]) -> np.ndarray:
        """
        Compute the density matrix for specific qubits on-demand.
        
        Args:
            keys: List of qubit keys to compute density matrix for
        
        Returns:
            Density matrix as numpy array (2^n × 2^n for n qubits)
        """
        if not keys:
            raise ValueError("Must provide at least one key")
        
        # Check if all keys exist
        for key in keys:
            if key not in self.states:
                raise KeyError(f"Key {key} not found in states")
        
        # Check if all requested qubits share the same circuit
        states = [self.states[k] for k in keys]
        if all(s.circuit is states[0].circuit for s in states):
            # They're grouped, use instance method
            state = states[0]
            return state.compute_density_matrix(keys_subset=keys)
        
        # If not grouped, we need to combine their circuits temporarily
        # This is a more complex case - for now, raise an error
        raise ValueError(
            f"Qubits {keys} are not grouped together. "
            f"Call group_qubits({keys}) first before computing joint density matrix."
        )
     
     
    def set(self, keys: List[int], amplitudes_or_circuit: Union[List[complex], stim.Circuit], compute_dm: bool = True) -> None:
        """
        Set qubits to a state specified by amplitudes or a Stim circuit.
        
        The Amplitudes cannot be an arbitrary amplitude because the conversion from amplitudes to circuit is non-trivial for un-standard states. 
        The amplitudes we can pass in are limited to:
        - Single qubit states: |0>, |1>, |+>, |->
        - Two qubit Bell states: |Phi+>, |Phi->, |Psi+>, |Psi->
        
        Behavior:
        1. If qubits are in different groups, group them first
        2. If setting a SUBSET of grouped qubits, ungroup the others first
        3. REPLACE the circuit entirely with the new operations
        4. All qubits in keys share the new circuit
        
        Args:
            keys: List of qubit keys to set
            amplitudes_or_circuit: Either list of amplitudes or stim.Circuit
            compute_dm: Whether to recompute density matrix (default True)
        """
        
        # Convert amplitudes to circuit if needed
        if isinstance(amplitudes_or_circuit, stim.Circuit):
            circuit = amplitudes_or_circuit
        else:
            # Convert amplitudes to circuit
            amplitudes = amplitudes_or_circuit
            circuit = stim.Circuit()
            
            if len(keys) == 1 and len(amplitudes) == 2:
                # Single qubit states
                plus_state = [1/np.sqrt(2), 1/np.sqrt(2)]
                minus_state = [1/np.sqrt(2), -1/np.sqrt(2)]
                zero_state = [1, 0]
                one_state = [0, 1]

                if np.allclose(amplitudes, plus_state):
                    circuit.append("H", [keys[0]])
                elif np.allclose(amplitudes, minus_state):
                    circuit.append("X", [keys[0]])
                    circuit.append("H", [keys[0]])
                elif np.allclose(amplitudes, one_state):
                    circuit.append("X", [keys[0]])
                # |0> state needs no gates (R will set it to |0>)

            elif len(keys) == 2 and len(amplitudes) == 4:
                # Two qubit Bell states
                phi_plus = [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]
                phi_minus = [1/np.sqrt(2), 0, 0, -1/np.sqrt(2)]
                psi_plus = [0, 1/np.sqrt(2), 1/np.sqrt(2), 0]
                psi_minus = [0, 1/np.sqrt(2), -1/np.sqrt(2), 0]

                if np.allclose(amplitudes, phi_plus):
                    circuit.append("H", [keys[0]])
                    circuit.append("CX", [keys[0], keys[1]])
                elif np.allclose(amplitudes, phi_minus):
                    circuit.append("H", [keys[0]])
                    circuit.append("CX", [keys[0], keys[1]])
                    circuit.append("Z", [keys[0]])
                elif np.allclose(amplitudes, psi_plus):
                    circuit.append("H", [keys[0]])
                    circuit.append("CX", [keys[0], keys[1]])
                    circuit.append("X", [keys[1]])
                elif np.allclose(amplitudes, psi_minus):
                    circuit.append("H", [keys[0]])
                    circuit.append("CX", [keys[0], keys[1]])
                    circuit.append("X", [keys[1]])
                    circuit.append("Z", [keys[0]])
        
        # Step 1: Check if qubits need grouping
        if len(keys) > 1:
            # Check if all keys share the same circuit
            states = [self.states[k] for k in keys if k in self.states]
            if len(states) > 1 and not all(s.circuit is states[0].circuit for s in states):
                # Need to group first
                self.group_qubits(keys)
        
        # Step 2: Get the state (all keys should now share the same circuit)
        state = self.states[keys[0]]
        
        # Step 3: Check if we're setting a SUBSET of grouped qubits
        if set(keys) != set(state.keys):
            # Setting a subset - need to ungroup the others first
            qubits_not_being_set = [k for k in state.keys if k not in keys]
            
            for key in qubits_not_being_set:
                # Give each unset qubit its own fresh circuit (reset to |0>)
                other_state = self.states[key]
                other_circuit = stim.Circuit()
                other_circuit.append("R", [key])
                
                # Create new state for this qubit
                other_state.circuit = other_circuit
                other_state.keys = [key]
                other_state._tableau = None
            
            # Update state.keys to only include keys being set
            state.keys = sorted(keys)
            for key in keys:
                self.states[key].keys = sorted(keys)
        
        # Step 4: Create completely FRESH circuit
        new_circuit = stim.Circuit()

        # Add the new operations
        for instruction in circuit:
            gate_args = instruction.gate_args_copy()
            targets = [t.value for t in instruction.targets_copy()]
            
            if gate_args:
                new_circuit.append(instruction.name, targets, *gate_args)
            else:
                new_circuit.append(instruction.name, targets)
        
        # Step 5: REPLACE the circuit for all qubits in keys
        for key in keys:
            if key in self.states:
                self.states[key].circuit = new_circuit
                self.states[key].keys = sorted(keys)
                self.states[key]._tableau = None
        
        # Step 6: Optionally recompute density matrix
        if compute_dm:
            # Compute full system density matrix
            full_dm = state.compute_density_matrix(keys_subset=None)
            
            # Update ALL qubits that share this circuit
            for key in keys:
                if key in self.states:
                    qubit_state = self.states[key]
                    qubit_state.set_density_matrix(full_dm)
            
            
    def get_density_matrix(self, key: int) -> np.ndarray:
        """
        Get the density matrix for a single qubit on-demand.
        
        Args:
            key: Qubit key
        
        Returns:
            Density matrix as numpy array (2x2 for single qubit)
        """
        if key not in self.states:
            raise KeyError(f"Key {key} not found in states")
        
        # Compute on-demand (doesn't cache)
        return self.compute_density_matrix([key])
    
    
    def get_circuit(self, key: int) -> stim.Circuit:
        """Get the Stim circuit for a qubit's state."""
        if key not in self.states:
            raise KeyError(f"Key {key} not found")
        
        return self.states[key].circuit