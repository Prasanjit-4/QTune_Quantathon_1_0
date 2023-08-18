import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit.circuit.library import TwoLocal
from qiskit.opflow import PauliExpectation,AerPauliExpectation
from qiskit.algorithms.optimizers import COBYLA
from qiskit import BasicAer,execute
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.utils import QuantumInstance

a=0.00698131079425246 
b=-0.0004978294000830275
c=+4.664512584628966e-05
d=+0.0004303465157577957
e=+0.5099539391488543
f=+0.5099677387273946
g=+0.5099488492845516
h=+0.5099106232913859
i=+0.5099467089998899
j=+0.5099046167492709


hamiltonian=SparsePauliOp.from_list([("IIIZ",a),("IIZI",b),("IZII",c),("ZIII",d),("IIZZ",e),("IZIZ",f),("IZZI",g),("ZIIZ",h),("ZIZI",i),("ZZII",j)])
h_mat=hamiltonian.to_matrix()

print(hamiltonian.num_qubits)
min_val_for_hamiltonian = NumPyMinimumEigensolver().compute_minimum_eigenvalue(hamiltonian).eigenstate

print(min_val_for_hamiltonian)

ansatz = TwoLocal(hamiltonian.num_qubits, "ry", "cz", reps=3, entanglement="full")
optimizer = COBYLA(maxiter=100)
backend = BasicAer.get_backend('qasm_simulator')
expectation = AerPauliExpectation()
vqe = VQE(ansatz, optimizer, expectation)

quantum_instance = QuantumInstance(backend=backend,shots=1024)

# Run the VQE algorithm to find the ground state energy
result = vqe.compute_minimum_eigenvalue(quantum_instance)

print(result)  # Ground state energy