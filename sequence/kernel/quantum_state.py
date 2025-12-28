"""Definition of the quantum state classes.

This module defines the classes used to track quantum states in SeQUeNCe.
These include 2 classes used by a quantum manager, and one used for individual photons:

1. The `KetState` class represents the ket vector formalism and is used by a quantum manager.
2. The `DensityState` class represents the density matrix formalism and is also used by a quantum manager.
3. The `FreeQuantumState` class uses the ket vector formalism, and is used by individual photons (not the quantum manager).
"""

import math
from abc import ABC
from numpy import pi, cos, sin, arange, log, log2
from numpy.random import Generator
import numpy as np
import itertools
import stim
from .quantum_utils import *
from ..constants import EPSILON


def swap_bits(num, pos1, pos2):
    """Swaps bits in num at positions 1 and 2.

    Used by quantum_state.measure_multiple method.
    """

    bit1 = (num >> pos1) & 1
    bit2 = (num >> pos2) & 1
    x = bit1 ^ bit2
    x = (x << pos1) | (x << pos2)
    return num ^ x


class State(ABC):
    """Base class for storing quantum states (abstract).

    Attributes:
        state (any): internal representation of the state, may vary by state type.
        keys (list[int]): list of keys pointing to the state, for use with a quantum manager.
    """

    def __init__(self, **kwargs):
        # potential key word arguments for derived classes, e.g. truncation = d-1 for qudit

        super().__init__()

        self.state = None
        self.keys = []

    def deserialize(self, json_data) -> None:
        self.keys = json_data["keys"]
        self.state = []
        for i in range(0, len(json_data["state"]), 2):
            complex_val = complex(json_data["state"][i],
                                  json_data["state"][i + 1])
            self.state.append(complex_val)

    def serialize(self) -> dict:
        res = {"keys": self.keys}
        state = []
        for cplx_n in self.state:
            if type(cplx_n) is float:
                state.append(cplx_n)
                state.append(0)
            elif isinstance(cplx_n, complex):
                state.append(cplx_n.real)
                state.append(cplx_n.imag)
            else:
                raise ValueError("Unknown type of state")

        res["state"] = state
        return res

    def __str__(self):
        return "\n".join(["Keys:", str(self.keys), "State:", str(self.state)])


class KetState(State):
    """Class to represent an individual quantum state as a ket vector.

    Attributes:
        state (np.array): state vector. Should be of length d ** len(keys), where d is dimension of elementary
            Hilbert space. Default is 2 for qubits.
        keys (list[int]): list of keys (subsystems) associated with this state.
        truncation (int): maximally allowed number of excited states for elementary subsystems.
                Default is 1 for qubit. dim = truncation + 1
    """

    def __init__(self, amplitudes: list[complex], keys: list[int], truncation: int = 1):
        """Constructor for ket state class.

        Args:
            amplitudes
            truncation (int): maximally allowed number of excited states for elementary subsystems.
                Default is 1 for qubit. dim = truncation + 1
        """
        super().__init__()
        self.truncation = truncation
        dim = self.truncation + 1  # dimension of element Hilbert space

        # check formatting
        assert all([abs(a) <= 1 + EPSILON for a in amplitudes]), "Illegal value with abs > 1 in ket vector"
        assert math.isclose(sum([abs(a) ** 2 for a in amplitudes]), 1), "Squared amplitudes do not sum to 1"

        num_subsystems = log(len(amplitudes)) / log(dim)
        assert dim ** int(round(num_subsystems)) == len(amplitudes),\
            "Length of amplitudes should be d ** n, " \
            "where d is subsystem Hilbert space dimension and n is the number of subsystems. " \
            "Actual amplitude length: {}, dim: {}, num subsystems: {}".format(len(amplitudes), dim, num_subsystems)
        num_subsystems = int(round(num_subsystems))
        assert num_subsystems == len(keys),\
            "Length of amplitudes should be d ** n, " \
            "where d is subsystem Hilbert space dimension and n is the number of subsystems. " \
            "Amplitude length: {}, expected subsystems: {}, num keys: {}".format(len(amplitudes), num_subsystems, len(keys))

        self.state = array(amplitudes, dtype=complex)
        self.keys = keys


class DensityState(State):
    """Class to represent an individual quantum state as a density matrix.

    Attributes:
        state (np.array): density matrix values. NxN matrix with N = d ** len(keys), where d is dimension of elementary
            Hilbert space. Default is d = 2 for qubits.
        keys (list[int]): list of keys (subsystems) associated with this state.
        truncation (int): maximally allowed number of excited states for elementary subsystems.
            Default is 1 for qubit. dim = truncation + 1
    """

    def __init__(self, state: list[list[complex]], keys: list[int], truncation: int = 1):
        """Constructor for density state class.

        Args:
            state (list[list[complex]]): density matrix elements given as a list.
                If the list is one-dimensional, will be converted to matrix with outer product operation.
            keys (list[int]): list of keys to this state in quantum manager.
            truncation (int): maximally allowed number of excited states for elementary subsystems.
                Default is 1 for qubit. dim = truncation + 1
        """

        super().__init__()
        self.truncation = truncation
        dim = self.truncation + 1  # dimension of element Hilbert space

        state = array(state, dtype=complex)
        if state.ndim == 1:
            state = outer(state, state.conj())

        # check formatting
        assert abs(trace(array(state)) - 1) < 0.01, "density matrix trace must be 1"
        for row in state:
            assert len(state) == len(row), "density matrix must be square"

        num_subsystems = log(len(state)) / log(dim)
        assert dim ** int(round(num_subsystems)) == len(state), \
            "Length of amplitudes should be d ** n, " \
            "where d is subsystem Hilbert space dimension and n is the number of subsystems. " \
            "Actual amplitude length: {}, dim: {}, num subsystems: {}".format(
                len(state), dim, num_subsystems
            )
        num_subsystems = int(round(num_subsystems))
        assert num_subsystems == len(keys), \
            "Length of amplitudes should be d ** n, " \
            "where d is subsystem Hilbert space dimension and n is the number of subsystems. " \
            "Amplitude length: {}, expected subsystems: {}, num keys: {}".format(
                len(state), num_subsystems, len(keys)
            )

        self.state = state
        self.keys = keys


class FreeQuantumState(State):
    """Class used by photons to track internal quantum states.

    This is an alternative to tracking states in a dedicated quantum manager, which adds simulation overhead.
    It defines several operations, including entanglement and measurement.
    For memories with an internal quantum state and certain photons, such as those stored in a memory or in parallel
    simulation, this class should not be used.
    Quantum states stored in a quantum manager class should be used instead.
    This module uses the ket vector formalism for storing and manipulating states.

    Attributes:
        state (tuple[complex]): list of complex coefficients in Z-basis.
        entangled_states (list[QuantumState]): list of entangled states (including self).
    """

    def __init__(self):
        super().__init__()
        self.state = (complex(1), complex(0))
        self.entangled_states = [self]

    def combine_state(self, another_state: "FreeQuantumState"):
        """Method to tensor multiply two quantum states.

        Arguments:
            another_state (QuantumState): state to entangle current state with.

        Side Effects:
            Modifies the `entangled_states` field for current state and `another_state`.
            Modifies the `state` field for current state and `another_state`.
        """

        entangled_states = self.entangled_states + another_state.entangled_states
        new_state = kron(self.state, another_state.state)
        new_state = tuple(new_state)

        for quantum_state in entangled_states:
            quantum_state.entangled_states = entangled_states
            quantum_state.state = new_state

    def random_noise(self, rng: Generator):
        """Method to add random noise to a single state.

        Chooses a random angle to set the quantum state to (with no phase difference).

        Side Effects:
            Modifies the `state` field.
        """

        # TODO: rewrite for entangled states
        angle = rng.random() * 2 * pi
        self.state = (complex(cos(angle)), complex(sin(angle)))

    # only for use with entangled state
    def set_state(self, state: tuple[complex]):
        """Method to change entangled state of multiple quantum states.

        Args:
            state (tuple[complex]): new coefficients for state.
                Should be 2^n in length, where n is the length of `entangled_states`.

        Side Effects:
            Modifies the `state` field for current and entangled states.
        """

        # check formatting of state
        assert all([abs(a) <= 1.01 for a in state]), "Illegal value with abs > 1 in quantum state"
        assert abs(sum([abs(a) ** 2 for a in state]) - 1) < 1e-5, "Squared amplitudes do not sum to 1"

        num_qubits = log2(len(state))
        assert 2 ** int(round(num_qubits)) == len(state), \
            "Length of amplitudes should be 2 ** n, where n is the number of qubits. " \
            "Actual amplitude length: {}, num qubits: {}".format(
                len(state), num_qubits
            )
        num_qubits = int(round(num_qubits))
        assert num_qubits == len(self.entangled_states), \
            "Length of amplitudes should be 2 ** n, where n is the number of qubits. " \
            "Num qubits in state: {}, num qubits in object: {}".format(
                num_qubits, len(self.entangled_states)
            )

        for qs in self.entangled_states:
            qs.state = state

    # for use with single, unentangled state
    def set_state_single(self, state: tuple[complex]):
        """Method to unentangle and set the state of a single quantum state object.

        Args:
            state (tuple[complex]): 2-element list of new complex coefficients.

        Side Effects:
            Will remove current state from any entangled states (if present).
            Modifies the `state` field of current state.
        """

        for qs in self.entangled_states:
            if qs is not None and qs != self:
                index = qs.entangled_states.index(self)
                qs.entangled_states[index] = None
        self.entangled_states = [self]
        self.state = state

    def measure(self, basis: tuple[tuple[complex]], rng: Generator) -> int:
        """Method to measure a single quantum state.

        Args:
            basis (tuple[tuple[complex]]): measurement basis, given as list of states
                (that are themselves lists of complex coefficients).
            rng (Generator): random number generator for measurement

        Returns:
            int: 0/1 measurement result, corresponding to one basis vector.

        Side Effects:
            Modifies the `state` field for current and any entangled states.
        """

        # handle entangled case
        if len(self.entangled_states) > 1:
            num_states = len(self.entangled_states)
            state_index = self.entangled_states.index(self)
            state0, state1, prob = measure_entangled_state_with_cache(self.state, basis, state_index, num_states)
            if rng.random() < prob:
                new_state = state0
                result = 0
            else:
                new_state = state1
                result = 1
            new_state = tuple(new_state)

        # handle unentangled case
        else:
            prob = measure_state_with_cache(self.state, basis)
            if rng.random() < prob:
                new_state = basis[0]
                result = 0
            else:
                new_state = basis[1]
                result = 1

        # set new state
        # new_state = tuple(new_state)
        for s in self.entangled_states:
            if s is not None:
                s.state = new_state

        return result

    @staticmethod
    def measure_multiple(basis, states, rng: Generator):
        """Method to measure multiple qubits in a more complex basis.

        May be used for bell state measurement.

        Args:
            basis (list[list[complex]]): list of basis vectors.
            states (list[QuantumState]): list of quantum state objects to measure.
            rng (Generator): random number generator for measurement

        Returns:
            int: measurement result in given basis.

        Side Effects:
            Will modify the `state` field of all entangled states.
        """

        # ensure states are entangled
        # (must be entangled prior to calling measure_multiple)
        entangled_list = states[0].entangled_states
        for state in states[1:]:
            assert state in states[0].entangled_states
        # ensure basis and vectors in basis are the right size
        basis_dimension = 2 ** len(states)
        assert len(basis) == basis_dimension
        for vector in basis:
            assert len(vector) == len(basis)

        state = states[0].state

        # move states to beginning of entangled list and quantum state
        pos_state_0 = entangled_list.index(states[0])
        pos_state_1 = entangled_list.index(states[1])
        entangled_list[0], entangled_list[pos_state_0] = entangled_list[pos_state_0], entangled_list[0]
        entangled_list[1], entangled_list[pos_state_1] = entangled_list[pos_state_1], entangled_list[1]
        switched_state = [complex(0)] * len(state)
        for i, coefficient in enumerate(state):
            switched_i = swap_bits(i, pos_state_0, pos_state_1)
            switched_state[switched_i] = coefficient

        state = tuple(switched_state)

        # math for probability calculations
        length_diff = len(entangled_list) - len(states)

        new_states, probabilities = measure_multiple_with_cache(state, basis, length_diff)

        possible_results = arange(0, basis_dimension, 1)
        # result gives index of the basis vector that will be projected to
        res = rng.choice(possible_results, p=probabilities)
        # project to new state, then reassign quantum state and entangled photons
        new_state = new_states[res]
        for state in entangled_list:
            state.quantum_state = new_state
            state.entangled_photons = entangled_list

        return res


class BellDiagonalState(State):
    """Class to represent a 2-qubit EPR pair as Bell diagonal state.

    Has 4 diagonal elements of density matrix in Bell basis.

    Attributes:
        state (np.array): diagonal elements of 2-qubit density matrix in Bell bases. Should be of length 4.
        keys (list[int]): list of keys (subsystems) associated with this state. Should be length 2.
    """

    def __init__(self, diag_elems: list[float], keys: list[int]):
        """Constructor for Bell diagonal state class.

        Args:
            diag_elems (list[float]): 4 diagonal elements of 2-qubit density matrix in Bell bases. 
                Default order: Phi+, Phi-, Psi+, Psi- (i.e. I, Z, X, Y errors).
            keys (list[int]): list of keys to this state in quantum manager. Should be length 2.
        """
        super().__init__()

        # check formatting
        assert all([elem <= 1.001 and elem >= 0 for elem in diag_elems]), \
            "Illegal value with elem > 1 or elem < 0 in density matrix diagonal elements"
        assert abs(sum([elem for elem in diag_elems]) - 1) < 1e-5, \
            "Density matrix diagonal elements do not sum to 1"
        assert len(keys) == 2, "BellDiagonalState density matrix are only supported for 2-qubit entangled states."

        # note: density matrix diagonal elements are guaranteed to be real from Hermiticity
        self.state = array(diag_elems, dtype=float)
        self.keys = keys


class StabilizerState(State):
    """
    Stabilizer state with density matrix representation via Pauli tomography.
    Uses compiled samplers for efficiency and maintains a tableau for exact operations.
    """
    def __init__(self, original_key: int, keys: list[int], circuit: stim.Circuit = None, 
                shots: int = 1000, truncation: int = 1, base_seed: int = None):
        super().__init__()
        self.original_key = original_key
        self.keys = list(keys)
        
        # Validate that original_key is in keys
        if self.original_key not in self.keys:
            raise ValueError(f"original_key {self.original_key} must be in keys {self.keys}")
        self.circuit = circuit if circuit is not None else stim.Circuit()
        self.shots = int(shots)
        self.truncation = truncation
        self.base_seed = base_seed  # Keep this for deterministic seeding
        self.rng = None  # Don't create RNG here
        self._tableau = None
        
        # Compute density matrix without extra RNG calls
        self.state = None
    
    @property
    def tableau(self) -> stim.TableauSimulator:
        """Get tableau simulator, creating it lazily if needed."""
        if self._tableau is None:
            self._tableau = stim.TableauSimulator()
            if self.circuit and len(self.circuit) > 0:
                self._tableau.do(self.circuit)
        return self._tableau
    
    
    def serialize(self) -> dict:
        """Not supported for StabilizerState."""
        raise NotImplementedError(
            "StabilizerState cannot use the base complex-vector serialization. "
            "Persist with a custom stabilizer/circuit schema instead.")
     
        
    def deserialize(self) -> None:
        """Not supported for StabilizerState."""
        raise NotImplementedError(
            "StabilizerState cannot be deserialized from the base complex-vector format. "
            "Load from a custom stabilizer/circuit schema instead.")
       
        
    def set(self, quantum_manager, circuit: stim.Circuit, sampled_keys: list[int] = None, compute_dm: bool = True):
        """
        Set state from a circuit by creating a fresh circuit that resets specified qubits.
        
        BEHAVIOR: Creates completely new circuit (replaces old one) instead of appending.
        Detects subset setting and ungroups other qubits to maintain consistency.
        """
        # Validation 1: sampled_keys cannot be None
        if sampled_keys is None:
            raise ValueError("sampled_keys cannot be None")
        
        # Validation 2: Extract which qubits the circuit operates on
        circuit_qubits = set()
        for instruction in circuit:
            for target in instruction.targets_copy():
                circuit_qubits.add(target.value)
        
        # Validation 3: Circuit must operate on subset of sampled_keys
        if not circuit_qubits.issubset(set(sampled_keys)):
            raise ValueError(
                f"Circuit operates on qubits {sorted(circuit_qubits)} "
                f"but sampled_keys only includes {sampled_keys}. "
                f"Circuit qubits must be subset of sampled_keys."
            )
        
        # Validation 4: original_key must remain in keys (keys never change)
        if self.original_key not in self.keys:
            raise ValueError(
                f"original_key={self.original_key} must be in keys={self.keys}"
            )
        
        # Validation 5: sampled_keys must be subset of current keys
        if not set(sampled_keys).issubset(set(self.keys)):
            raise ValueError(
                f"sampled_keys {sampled_keys} must be subset of current keys {self.keys}"
            )
        
        # ============================================================================
        # CRITICAL FIX: Check if we're setting a SUBSET of grouped qubits
        # If yes, ungroup the other qubits first to maintain consistency
        # ============================================================================
        if set(sampled_keys) != set(self.keys):
            # We're setting a subset - need to ungroup the other qubits first
            # This prevents inconsistent state where circuit doesn't match keys
            
            for key in self.keys:
                if key not in sampled_keys:
                    # This qubit is NOT being set - give it its own circuit
                    other_state = quantum_manager.states[key]
                    
                    # Create a fresh circuit for this qubit (reset to |0⟩)
                    other_circuit = stim.Circuit()
                    other_circuit.append("R", [key])
                    
                    # Assign new circuit and update keys to be solo
                    other_state.circuit = other_circuit
                    other_state.keys = [key]
                    other_state._tableau = None
            
            # Update self.keys to only include sampled_keys
            self.keys = sorted(sampled_keys)
        
        # Step 1: Check if all sampled_keys share the same circuit, if not, group them
        states_to_check = [quantum_manager.states[k] for k in sampled_keys if k in quantum_manager.states]
        if len(states_to_check) > 1 and not all(s.circuit is states_to_check[0].circuit for s in states_to_check):
            # Need to group these qubits first
            quantum_manager.group_qubits(sampled_keys)
        
        # Step 2: Create FRESH circuit that completely replaces the old one
        new_circuit = stim.Circuit()
        
        # Add reset gates for sampled qubits (deduplicated)
        unique_keys = sorted(set(sampled_keys))
        if unique_keys:
            new_circuit.append("R", unique_keys)
        
        # Step 3: Add new circuit operations to the NEW circuit
        for instruction in circuit:
            gate_args = instruction.gate_args_copy()
            targets = [t.value for t in instruction.targets_copy()]
            
            if gate_args:
                new_circuit.append(instruction.name, targets, *gate_args)
            else:
                new_circuit.append(instruction.name, targets)
        
        # Step 4: REPLACE the old circuit for ALL qubits in the (now updated) group
        # After ungrouping above, self.keys now only contains sampled_keys
        for key in self.keys:
            if key in quantum_manager.states:
                quantum_manager.states[key].circuit = new_circuit
        
        # Step 5: Invalidate cached tableau
        self._tableau = None
        
        # Step 6: Optionally recompute density matrix
        if compute_dm:
            self.state = self._compute_density_matrix()
   
    
    def _compute_density_matrix(self) -> np.ndarray:
        """Compute density matrix via Pauli tomography using compiled samplers."""
        k = len(self.keys)
        
        # Pauli operators
        I = np.array([[1, 0], [0, 1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        paulis = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
        
        # Build density matrix via Pauli expansion
        rho = np.zeros((2**k, 2**k), dtype=complex)
        
        for i, pauli_string in enumerate(itertools.product('IXYZ', repeat=k)):
            # Build measurement circuit for this Pauli string
            meas_circuit = self.circuit.copy()
            
            # Add basis rotations and measurements
            measured_qubits = []
            for j, (qubit, pauli) in enumerate(zip(self.keys, pauli_string)):
                if pauli == 'I':
                    continue  # Don't measure identity
                elif pauli == 'X':
                    meas_circuit.append("H", [qubit])
                elif pauli == 'Y':
                    meas_circuit.append("S_DAG", [qubit])
                    meas_circuit.append("H", [qubit])
                # Z needs no rotation
                
                meas_circuit.append("M", [qubit])
                measured_qubits.append(j)
            
            # Estimate expectation value
            if not measured_qubits:
                expectation = 1.0  # All identity operators
            else:
                # Use deterministic seeding - NO RNG CALLS
                if self.base_seed is not None:
                    # Deterministic seed based on base_seed and Pauli string index
                    pauli_hash = hash(pauli_string) % (2**16)
                    seed = (self.base_seed + i + pauli_hash) % (2**31)
                else:
                    seed = None  # Non-deterministic
                
                # Compile sampler with seed
                sampler = meas_circuit.compile_sampler(seed=seed)
                samples = sampler.sample(shots=self.shots)
                
                # Calculate expectation as average parity
                if samples.shape[1] == 0:
                    expectation = 1.0
                else:
                    # Parity: even number of 1s -> +1, odd -> -1
                    parities = np.array([(-1) ** np.sum(row) for row in samples])
                    expectation = np.mean(parities)
            
            # Build tensor product of Pauli matrices
            pauli_op = np.array([[1]], dtype=complex)
            for p in pauli_string:
                pauli_op = np.kron(pauli_op, paulis[p])
            
            rho += expectation * pauli_op
        
        rho /= (2**k)


        # Ensure Hermitian (fix numerical errors)
        rho = (rho + rho.conj().T) / 2
        
        return rho


    def group_qubits(self, quantum_manager, keys_to_group: List[int]) -> None:
        """
        Group this state with other qubits in the quantum manager.
        
        This is a helper method that calls the quantum manager's group_qubits.
        
        Args:
            quantum_manager: Reference to the quantum manager
            keys_to_group: List of qubit keys to group together (must include self.original_key)
        """
        if self.original_key not in keys_to_group:
            raise ValueError(
                f"Cannot group: original_key={self.original_key} must be in keys_to_group={keys_to_group}"
            )
        
        quantum_manager.group_qubits(keys_to_group)
        
        
    def measure(self, qubit_indices: list[int], basis: str = 'Z') -> list[int]:
        """
        Measure qubits, collapsing the state and preserving correlations.
        
        Args:
            qubit_indices: Indices within self.keys to measure
            basis: Measurement basis ('Z', 'X', or 'Y')
        
        Returns:
            List of measurement outcomes (0 or 1)
        """
        results = []
        
        for idx in qubit_indices:
            if idx >= len(self.keys):
                results.append(0)
                continue
            
            qubit = self.keys[idx]
            
            # Apply basis rotation, measure, then undo rotation
            if basis == 'X':
                self.tableau.h(qubit)
                outcome = int(self.tableau.measure(qubit))
                self.tableau.h(qubit)
            elif basis == 'Y':
                self.tableau.s_dag(qubit)
                self.tableau.h(qubit)
                outcome = int(self.tableau.measure(qubit))
                self.tableau.h(qubit)
                self.tableau.s(qubit)
            else:  # Z basis
                outcome = int(self.tableau.measure(qubit))
            
            results.append(outcome)
        
        # Invalidate cached density matrix since state changed
        self.state = None
        
        return results
    
    
    def set_density_matrix(self, density_matrix: np.ndarray) -> None:
        """
        Set the density matrix directly without changing any other attributes.
        
        This is useful when you've computed a density matrix externally and want
        to update the state without affecting original_key, keys, or circuit.
        
        Args:
            density_matrix: Pre-computed density matrix as numpy array
        """
        # Validation: check dimensions match expected size
        expected_dim = 2 ** len(self.keys)
        if density_matrix.shape != (expected_dim, expected_dim):
            raise ValueError(
                f"Density matrix shape {density_matrix.shape} doesn't match "
                f"expected shape ({expected_dim}, {expected_dim}) for {len(self.keys)} qubits"
            )
        
        self.state = density_matrix
        
        
    def compute_density_matrix(self, keys_subset: List[int] = None) -> np.ndarray:
        """
        Compute density matrix for this state, optionally for a subset of qubits.
        
        Args:
            keys_subset: Subset of self.keys to compute density matrix for.
                        If None, computes for all self.keys.
        
        Returns:
            Density matrix as numpy array
        """
        if keys_subset is None:
            # Compute for all qubits in this state
            return self._compute_density_matrix()
        
        # Validate keys_subset
        if not set(keys_subset).issubset(set(self.keys)):
            raise ValueError(
                f"keys_subset {keys_subset} must be subset of state keys {self.keys}"
            )
        
        if len(keys_subset) == len(self.keys):
            # Computing for all qubits anyway
            return self._compute_density_matrix()
        
        # For subset, we need to trace out the other qubits
        # This requires computing full density matrix then tracing out
        full_dm = self._compute_density_matrix()
        
        # Determine which qubits to trace out
        qubits_to_trace = [k for k in self.keys if k not in keys_subset]
        
        # Trace out unwanted qubits
        reduced_dm = self._partial_trace(full_dm, self.keys, qubits_to_trace)
        
        return reduced_dm


    def _partial_trace(self, density_matrix: np.ndarray, all_keys: List[int], 
                    trace_out_keys: List[int]) -> np.ndarray:
        """
        Compute partial trace over specified qubits.
        
        Args:
            density_matrix: Full density matrix
            all_keys: All qubit keys in the density matrix
            trace_out_keys: Which qubits to trace out
        
        Returns:
            Reduced density matrix after tracing out specified qubits
        """
        # Get indices of qubits to keep
        keep_indices = [all_keys.index(k) for k in all_keys if k not in trace_out_keys]
        
        n_qubits = len(all_keys)
        n_keep = len(keep_indices)
        
        # Reshape density matrix to separate each qubit's dimension
        shape = [2] * (2 * n_qubits)
        dm_reshaped = density_matrix.reshape(shape)
        
        # Trace out unwanted qubits
        for qubit_idx in sorted([all_keys.index(k) for k in trace_out_keys], reverse=True):
            # Contract over this qubit's dimension
            dm_reshaped = np.trace(dm_reshaped, axis1=qubit_idx, axis2=qubit_idx + n_qubits)
            n_qubits -= 1
        
        # Reshape back to 2D matrix
        dim = 2 ** n_keep
        return dm_reshaped.reshape(dim, dim)
    
    
    def copy(self):
        """Create a copy of this state."""
        new_state = StabilizerState(
            keys=self.keys.copy(),
            circuit=self.circuit.copy(),
            shots=self.shots,
            truncation=self.truncation,
            base_seed=self.base_seed
        )
        new_state.state = self.state.copy()
        new_state._tableau = None
        return new_state