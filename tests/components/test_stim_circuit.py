import pytest
import numpy as np
import stim
from unittest.mock import patch
from sequence.components.stim_circuit import StabilizerCircuit, StabilizerCircuitError


# ============= Initialization Tests =============

def test_valid_initialization():
    """Test creating circuit with valid number of qubits."""
    circuit = StabilizerCircuit(5)
    assert circuit.num_qubits == 5
    assert isinstance(circuit.circuit, stim.Circuit)
    assert circuit._measured_qubits == []
    assert circuit._key2q == {}


def test_zero_qubits_warning():
    """Test that zero qubits triggers warning."""
    with patch('sequence.utils.log.logger.warning') as mock_warning:
        circuit = StabilizerCircuit(0)
        mock_warning.assert_called_once()


def test_negative_qubits_warning():
    """Test that negative qubits triggers warning."""
    with patch('sequence.utils.log.logger.warning') as mock_warning:
        circuit = StabilizerCircuit(-3)
        mock_warning.assert_called_once()


# ============= Key Binding Tests =============

def test_bind_unbind_key():
    """Test binding and unbinding keys to qubits."""
    circuit = StabilizerCircuit(3)
    
    circuit.bind_key("alice", 0)
    circuit.bind_key("bob", 2)
    
    assert circuit.qubit_for("alice") == 0
    assert circuit.qubit_for("bob") == 2
    
    circuit.unbind_key("alice")
    assert "alice" not in circuit._key2q
    assert circuit.qubit_for("bob") == 2


def test_rebind_key():
    """Test rebinding a key to different qubit."""
    circuit = StabilizerCircuit(3)
    circuit.bind_key("key1", 0)
    circuit.bind_key("key1", 1)  # Rebind
    assert circuit.qubit_for("key1") == 1


def test_measure_keys():
    """Test measuring qubits by key names."""
    circuit = StabilizerCircuit(3)
    circuit.bind_key("q0", 0)
    circuit.bind_key("q1", 1)
    circuit.bind_key("q2", 2)
    
    # Create Bell state on q0, q1
    circuit.h(0)
    circuit.cx(0, 1)
    
    # Mock the measure method to return predictable results
    with patch.object(circuit, 'measure', return_value=[0, 1, 0]):
        results = circuit.measure_keys(["q0", "q1", "q2"])
        assert results == {"q0": 0, "q1": 1, "q2": 0}


# ============= Validation Tests =============

def test_validate_qubit_valid():
    """Test validation passes for valid qubits."""
    circuit = StabilizerCircuit(5)
    circuit._validate_qubit(0)
    circuit._validate_qubit(4)
    # Should not raise


def test_validate_qubit_invalid():
    """Test validation fails for invalid qubits."""
    circuit = StabilizerCircuit(3)
    
    with pytest.raises(StabilizerCircuitError, match="out of range"):
        circuit._validate_qubit(3)
    
    with pytest.raises(StabilizerCircuitError, match="out of range"):
        circuit._validate_qubit(-1)
    
    with pytest.raises(StabilizerCircuitError, match="out of range"):
        circuit._validate_qubit(10)


def test_validate_qubit_gate_target():
    """Test validation accepts stim.GateTarget."""
    circuit = StabilizerCircuit(3)
    target = stim.target_rec(-1)
    circuit._validate_qubit(target)  # Should not raise


def test_validate_probability_valid():
    """Test probability validation for valid values."""
    circuit = StabilizerCircuit(2)
    circuit._validate_probability(0.0)
    circuit._validate_probability(0.5)
    circuit._validate_probability(1.0)


def test_validate_probability_invalid():
    """Test probability validation for invalid values."""
    circuit = StabilizerCircuit(2)
    
    with pytest.raises(StabilizerCircuitError, match="out of"):
        circuit._validate_probability(-0.1)
    
    with pytest.raises(StabilizerCircuitError, match="out of"):
        circuit._validate_probability(1.1)
    
    with pytest.raises(StabilizerCircuitError, match="not numeric"):
        circuit._validate_probability("0.5")


# ============= Single Qubit Gate Tests =============

def test_pauli_gates():
    """Test X, Y, Z gate application."""
    circuit = StabilizerCircuit(2)
    
    circuit.x(0)
    circuit.y(1)
    circuit.z(0)
    
    ops = list(circuit.circuit.flattened_operations())
    assert len(ops) == 3
    assert ops[0] == ("X", [0], [])
    assert ops[1] == ("Y", [1], [])
    assert ops[2] == ("Z", [0], [])


def test_pauli_gates_with_errors():
    """Test Pauli gates with error channels."""
    circuit = StabilizerCircuit(2)
    
    circuit.x(0, error_prob=0.1)
    circuit.y(1, error_prob=0.2)
    circuit.z(0, error_prob=0.3)
    
    ops = list(circuit.circuit.flattened_operations())
    assert len(ops) == 6  # 3 gates + 3 error channels
    assert ops[1] == ("X_ERROR", [0], [0.1])
    assert ops[3] == ("Y_ERROR", [1], [0.2])
    assert ops[5] == ("Z_ERROR", [0], [0.3])


def test_hadamard_gate():
    """Test Hadamard gate."""
    circuit = StabilizerCircuit(2)
    circuit.h(0)
    
    ops = list(circuit.circuit.flattened_operations())
    assert ops[0] == ("H", [0], [])


def test_s_gates():
    """Test S and S† gates."""
    circuit = StabilizerCircuit(2)
    circuit.s(0)
    circuit.s_dag(1)
    
    ops = list(circuit.circuit.flattened_operations())
    assert ops[0] == ("S", [0], [])
    assert ops[1] == ("S_DAG", [1], [])


def test_reset_gate():
    """Test reset operation."""
    circuit = StabilizerCircuit(2)
    circuit.reset(1)
    
    ops = list(circuit.circuit.flattened_operations())
    assert ops[0] == ("R", [1], [])


def test_invalid_qubit_single_gates():
    """Test single qubit gates with invalid indices."""
    circuit = StabilizerCircuit(2)
    
    with pytest.raises(StabilizerCircuitError):
        circuit.x(5)
    
    with pytest.raises(StabilizerCircuitError):
        circuit.h(-1)
    
    with pytest.raises(StabilizerCircuitError):
        circuit.s(2)


# ============= Two Qubit Gate Tests =============

def test_cnot_gate():
    """Test CNOT gate."""
    circuit = StabilizerCircuit(3)
    circuit.cx(0, 1)
    circuit.cx(2, 0)
    
    ops = list(circuit.circuit.flattened_operations())
    assert ops[0] == ("CNOT", [0, 1], [])
    assert ops[1] == ("CNOT", [2, 0], [])


def test_cz_gate():
    """Test CZ gate."""
    circuit = StabilizerCircuit(3)
    circuit.cz(0, 2)
    circuit.cz(1, 2)
    
    ops = list(circuit.circuit.flattened_operations())
    assert ops[0] == ("CZ", [0, 2], [])
    assert ops[1] == ("CZ", [1, 2], [])


def test_two_qubit_gate_validation():
    """Test validation for two-qubit gates."""
    circuit = StabilizerCircuit(2)
    
    with pytest.raises(StabilizerCircuitError):
        circuit.cx(0, 5)  # Target out of range
    
    with pytest.raises(StabilizerCircuitError):
        circuit.cz(-1, 1)  # Control out of range


# ============= Measurement Tests =============

def test_z_basis_measurement():
    """Test Z-basis measurement."""
    circuit = StabilizerCircuit(3)
    circuit.measure(0, basis='Z')
    circuit.measure(1)  # Default is Z
    
    ops = list(circuit.circuit.flattened_operations())
    assert ops[0] == ("M", [0], [])
    assert ops[1] == ("M", [1], [])
    assert circuit.measured == [0, 1]


def test_x_basis_measurement():
    """Test X-basis measurement."""
    circuit = StabilizerCircuit(2)
    circuit.measure(0, basis='X')
    
    ops = list(circuit.circuit.flattened_operations())
    assert ops[0] == ("H", [0], [])  # Rotate to X basis
    assert ops[1] == ("M", [0], [])
    assert circuit.measured == [0]


def test_mixed_basis_measurements():
    """Test measurements in different bases."""
    circuit = StabilizerCircuit(3)
    circuit.measure(0, basis='Z')
    circuit.measure(1, basis='X')
    circuit.measure(2, basis='x')  # Test case insensitivity
    
    assert len(circuit.measured) == 3
    assert circuit.measured == [0, 1, 2]


def test_sample_measurements_empty():
    """Test sampling from circuit with no measurements."""
    circuit = StabilizerCircuit(2)
    circuit.h(0)
    
    results = circuit.sample_measurements(shots=10)
    assert results.shape == (10, 0)


def test_sample_measurements_basic():
    """Test sampling measurement outcomes."""
    circuit = StabilizerCircuit(2)
    circuit.measure(0)
    circuit.measure(1)
    
    results = circuit.sample_measurements(shots=100)
    assert results.shape == (100, 2)
    assert np.all((results == 0) | (results == 1))


# ============= Conditional Operation Tests =============

def test_conditional_gates():
    """Test X_if, Y_if, Z_if methods."""
    circuit = StabilizerCircuit(2)
    circuit.measure(0)
    circuit.x_if(-1, 1)
    circuit.y_if(-1, 1)
    circuit.z_if(-1, 1)
    
    ops = list(circuit.circuit.flattened_operations())
    assert ops[1][0] == "X"
    assert ops[2][0] == "Y"
    assert ops[3][0] == "Z"
    # Check all have measurement record target
    for i in range(1, 4):
        assert isinstance(ops[i][1][0], stim.GateTarget)


def test_rec_method():
    """Test rec method for creating record targets."""
    circuit = StabilizerCircuit(2)
    rec = circuit.rec(-2)
    assert isinstance(rec, stim.GateTarget)


def test_conditional_method():
    """Test generic conditional method."""
    circuit = StabilizerCircuit(3)
    circuit.measure(0)
    circuit.conditional("CNOT", -1, 1, 2)
    
    ops = list(circuit.circuit.flattened_operations())
    assert ops[1][0] == "CNOT"
    assert len(ops[1][1]) == 3  # rec + 2 qubits


def test_conditional_with_invalid_qubit():
    """Test conditional operations with invalid qubits."""
    circuit = StabilizerCircuit(2)
    circuit.measure(0)
    
    with pytest.raises(StabilizerCircuitError):
        circuit.x_if(-1, 5)  # Invalid qubit


# ============= Noise Channel Tests =============

def test_single_qubit_depolarize():
    """Test single-qubit depolarizing channel."""
    circuit = StabilizerCircuit(3)
    circuit.depolarize(1, 0.05)
    
    ops = list(circuit.circuit.flattened_operations())
    assert ops[0] == ("DEPOLARIZE1", [1], [0.05])


def test_two_qubit_depolarize():
    """Test two-qubit depolarizing channel."""
    circuit = StabilizerCircuit(3)
    circuit.depolarize([0, 2], 0.1)
    circuit.depolarize((1, 2), 0.2)  # Test tuple input
    
    ops = list(circuit.circuit.flattened_operations())
    assert ops[0] == ("DEPOLARIZE2", [0, 2], [0.1])
    assert ops[1] == ("DEPOLARIZE2", [1, 2], [0.2])


def test_depolarize_invalid_qubits():
    """Test depolarize with invalid number of qubits."""
    circuit = StabilizerCircuit(4)
    
    with pytest.raises(StabilizerCircuitError, match="1 or 2 qubits"):
        circuit.depolarize([0, 1, 2], 0.1)
    
    with pytest.raises(StabilizerCircuitError, match="1 or 2 qubits"):
        circuit.depolarize([], 0.1)


def test_depolarize_invalid_probability():
    """Test depolarize with invalid probability."""
    circuit = StabilizerCircuit(2)
    
    with pytest.raises(StabilizerCircuitError, match="out of"):
        circuit.depolarize(0, 1.5)
    
    with pytest.raises(StabilizerCircuitError, match="out of"):
        circuit.depolarize(0, -0.1)


def test_pauli_channel_single_qubit():
    """Test single-qubit Pauli channel."""
    circuit = StabilizerCircuit(2)
    probs = (0.1, 0.2, 0.3)  # px, py, pz
    circuit.pauli_channel(0, probs)
    
    ops = list(circuit.circuit.flattened_operations())
    assert ops[0] == ("PAULI_CHANNEL_1", [0], probs)


def test_pauli_channel_two_qubits():
    """Test two-qubit Pauli channel."""
    circuit = StabilizerCircuit(3)
    probs = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15)
    circuit.pauli_channel([0, 2], probs)
    
    ops = list(circuit.circuit.flattened_operations())
    assert ops[0] == ("PAULI_CHANNEL_2", [0, 2], probs)


def test_idle_with_t2():
    """Test idle depolarization with T2 dephasing."""
    circuit = StabilizerCircuit(2)
    circuit.idle(0, time=1e-6, T2=10e-6)
    
    ops = list(circuit.circuit.flattened_operations())
    assert len(ops) == 1
    assert ops[0][0] == "DEPOLARIZE1"
    assert ops[0][1] == [0]
    # Check probability is computed correctly
    expected_p = (4/3) * (1 - np.exp(-1e-6 / 10e-6))
    assert np.isclose(ops[0][2][0], expected_p)


def test_idle_with_t1():
    """Test idle depolarization with T1 relaxation."""
    circuit = StabilizerCircuit(2)
    circuit.idle(1, time=2e-6, T1=20e-6)
    
    ops = list(circuit.circuit.flattened_operations())
    assert len(ops) == 1
    assert ops[0][0] == "DEPOLARIZE1"
    assert ops[0][1] == [1]
    # Check probability is computed correctly
    gamma = 1 - np.exp(-2e-6 / 20e-6)
    expected_p = (2 + gamma - 2 * np.sqrt(1 - gamma)) / 3
    assert np.isclose(ops[0][2][0], expected_p)


def test_idle_no_t1_t2_error():
    """Test idle raises error when neither T1 nor T2 provided."""
    circuit = StabilizerCircuit(2)
    
    with pytest.raises(StabilizerCircuitError, match="T1 or T2"):
        circuit.idle(0, time=1e-6)


def test_idle_two_qubits():
    """Test idle on two qubits."""
    circuit = StabilizerCircuit(3)
    circuit.idle([0, 2], time=1e-6, T2=10e-6)
    
    ops = list(circuit.circuit.flattened_operations())
    assert ops[0][0] == "DEPOLARIZE2"
    assert ops[0][1] == [0, 2]


# ============= Tomography Tests =============

def test_tomography_dm_single_qubit():
    """Test density matrix reconstruction for single qubit."""
    circuit = StabilizerCircuit(2)
    circuit.h(0)  # Create |+⟩ state
    
    rho = circuit.tomography_dm([0], shots=10000)
    
    assert rho.shape == (2, 2)
    assert np.allclose(np.trace(rho), 1.0, atol=0.1)
    # Check if it's close to |+⟩⟨+| state
    expected = 0.5 * np.array([[1, 1], [1, 1]])
    assert np.allclose(rho.real, expected, atol=0.1)


def test_tomography_dm_two_qubits():
    """Test density matrix reconstruction for two qubits."""
    circuit = StabilizerCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)  # Create Bell state
    
    rho = circuit.tomography_dm([0, 1], shots=10000)
    
    assert rho.shape == (4, 4)
    assert np.allclose(np.trace(rho), 1.0, atol=0.1)
    # Check entanglement: should have |00⟩ and |11⟩ components
    assert np.abs(rho[0, 0]) > 0.4  # |00⟩⟨00|
    assert np.abs(rho[3, 3]) > 0.4  # |11⟩⟨11|


def test_tomography_dm_with_direct_pauli():
    """Test tomography using direct Pauli measurements."""
    circuit = StabilizerCircuit(1)
    circuit.x(0)  # |1⟩ state
    
    rho = circuit.tomography_dm([0], shots=5000, use_direct_pauli_meas=True)
    
    expected = np.array([[0, 0], [0, 1]])
    assert np.allclose(rho.real, expected, atol=0.1)


def test_tomography_dm_without_direct_pauli():
    """Test tomography using rotations + Z measurements."""
    circuit = StabilizerCircuit(1)
    circuit.x(0)  # |1⟩ state
    
    rho = circuit.tomography_dm([0], shots=5000, use_direct_pauli_meas=False)
    
    expected = np.array([[0, 0], [0, 1]])
    assert np.allclose(rho.real, expected, atol=0.1)


# ============= State Vector Tests =============

def test_state_vector_basic():
    """Test state vector computation."""
    circuit = StabilizerCircuit(2)
    circuit.h(0)
    
    vec = circuit.state_vector()
    
    assert vec.shape == (4,)
    assert np.allclose(np.abs(vec[0])**2, 0.5)  # |00⟩ component
    assert np.allclose(np.abs(vec[2])**2, 0.5)  # |10⟩ component


def test_state_vector_bell_state():
    """Test state vector for Bell state."""
    circuit = StabilizerCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    
    vec = circuit.state_vector()
    
    assert vec.shape == (4,)
    assert np.allclose(np.abs(vec[0])**2, 0.5)  # |00⟩
    assert np.allclose(np.abs(vec[3])**2, 0.5)  # |11⟩
    assert np.allclose(np.abs(vec[1])**2, 0.0)  # |01⟩
    assert np.allclose(np.abs(vec[2])**2, 0.0)  # |10⟩


# ============= Circuit Management Tests =============

def test_clear_circuit():
    """Test clearing circuit while preserving key bindings."""
    circuit = StabilizerCircuit(3)
    circuit.bind_key("alice", 0)
    circuit.bind_key("bob", 1)
    
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure(2)
    
    circuit.clear_circuit()
    
    assert len(list(circuit.circuit.flattened_operations())) == 0
    assert circuit._measured_qubits == []
    # Key bindings should be preserved
    assert circuit.qubit_for("alice") == 0
    assert circuit.qubit_for("bob") == 1


def test_circuit_copy():
    """Test that circuit can be copied."""
    circuit = StabilizerCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    
    circuit_copy = circuit.circuit.copy()
    
    assert list(circuit_copy.flattened_operations()) == list(circuit.circuit.flattened_operations())


# ============= Quantum Algorithm Tests =============

def test_ghz_state_creation():
    """Test GHZ state creation and measurement correlations."""
    circuit = StabilizerCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    circuit.measure(0)
    circuit.measure(1)
    circuit.measure(2)
    
    results = circuit.sample_measurements(shots=1000)
    
    # All qubits should be perfectly correlated
    assert np.all(results[:, 0] == results[:, 1])
    assert np.all(results[:, 0] == results[:, 2])
    # Should get roughly 50/50 split between 000 and 111
    all_zeros = np.sum(np.all(results == 0, axis=1))
    all_ones = np.sum(np.all(results == 1, axis=1))
    assert all_zeros + all_ones == 1000
    assert 400 < all_zeros < 600  # Statistical test


def test_bell_state_measurement():
    """Test Bell state creation and measurement."""
    circuit = StabilizerCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure(0)
    circuit.measure(1)
    
    results = circuit.sample_measurements(shots=1000)
    
    # Measurements should be perfectly correlated
    assert np.all(results[:, 0] == results[:, 1])
    # Should get roughly 50/50 split
    zeros = np.sum(results[:, 0] == 0)
    assert 400 < zeros < 600


def test_quantum_teleportation_circuit():
    """Test quantum teleportation circuit construction."""
    circuit = StabilizerCircuit(3)
    
    # Prepare Bell pair between qubits 1 and 2
    circuit.h(1)
    circuit.cx(1, 2)
    
    # Prepare state to teleport on qubit 0
    circuit.h(0)
    circuit.s(0)
    
    # Bell measurement on qubits 0 and 1
    circuit.cx(0, 1)
    circuit.h(0)
    circuit.measure(0)
    circuit.measure(1)
    
    # Conditional corrections on qubit 2
    circuit.x_if(-1, 2)  # If qubit 1 measured 1
    circuit.z_if(-2, 2)  # If qubit 0 measured 1
    
    ops = list(circuit.circuit.flattened_operations())
    # Verify circuit has expected structure
    assert any(op[0] == "H" for op in ops)
    assert any(op[0] == "CNOT" for op in ops)
    assert any(op[0] == "M" for op in ops)
    assert circuit.measured == [0, 1]


def test_error_correction_circuit():
    """Test simple error correction circuit with syndrome extraction."""
    circuit = StabilizerCircuit(5)  # 3 data qubits + 2 ancilla
    
    # Encode logical qubit
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    
    # Add some error
    circuit.x(1, error_prob=0.1)
    
    # Syndrome extraction
    circuit.cx(1, 3)
    circuit.cx(2, 3)
    circuit.cx(0, 4)
    circuit.cx(2, 4)
    
    circuit.measure(3)
    circuit.measure(4)
    
    # Error correction based on syndrome
    circuit.conditional("X", -2, 1)  # Correct based on syndrome
    
    ops = list(circuit.circuit.flattened_operations())
    assert any(op[0] == "X_ERROR" for op in ops)
    assert circuit.measured == [3, 4]


# ============= Integration Tests =============

def test_complex_circuit_with_noise():
    """Test complex circuit with multiple noise sources."""
    circuit = StabilizerCircuit(4)
    
    # Initial state preparation
    circuit.h(0)
    circuit.h(1)
    
    # Entangling gates with errors
    circuit.cx(0, 2)
    circuit.cx(1, 3)
    
    # Add various noise channels
    circuit.depolarize([0, 1], 0.01)
    circuit.pauli_channel(2, (0.001, 0.002, 0.003))
    circuit.idle(3, time=1e-6, T1=100e-6)
    
    # More gates
    circuit.cz(2, 3)
    circuit.h(2)
    
    # Measurements
    circuit.measure(0, basis='Z')
    circuit.measure(1, basis='X')
    circuit.measure(2)
    circuit.measure(3)
    
    results = circuit.sample_measurements(shots=100)
    assert results.shape == (100, 4)
    assert np.all((results == 0) | (results == 1))


def test_circuit_with_all_features():
    """Test circuit using all available features."""
    circuit = StabilizerCircuit(4)
    
    # Key binding
    circuit.bind_key("data", 0)
    circuit.bind_key("ancilla", 1)
    
    # Single qubit gates
    circuit.h(circuit.qubit_for("data"))
    circuit.x(circuit.qubit_for("ancilla"))
    circuit.y(2)
    circuit.z(3)
    circuit.s(0)
    circuit.s_dag(1)
    
    # Two qubit gates
    circuit.cx(0, 1)
    circuit.cz(2, 3)
    
    # Reset
    circuit.reset(3)
    
    # Noise
    circuit.depolarize(0, 0.01)
    circuit.pauli_channel([1, 2], (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15))
    
    # Measurement and conditionals
    circuit.measure(0)
    circuit.x_if(-1, 1)
    circuit.y_if(-1, 2)
    circuit.z_if(-1, 3)
    
    # Clear and rebuild
    circuit.clear_circuit()
    circuit.h(circuit.qubit_for("data"))
    
    # Verify key bindings survived clear
    assert circuit.qubit_for("data") == 0
    assert circuit.qubit_for("ancilla") == 1