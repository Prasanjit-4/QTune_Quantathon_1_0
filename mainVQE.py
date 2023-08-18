from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import TwoLocal
from qiskit.algorithms.minimum_eigensolvers import VQE
import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator


num_qubits = 4
ansatz = TwoLocal(num_qubits, "ry", "cz")
optimizer = COBYLA(maxiter=1000)
ansatz.decompose().draw("mpl", style="iqx")
estimator = Estimator()





vqe = VQE(estimator, ansatz, optimizer)
H2_op = SparsePauliOp.from_list([
 ("IIIZ", 0.00698131079425246),
 ("IIZI", -0.0004978294000830275),
 ("IZII", +4.664512584628966e-05),
 ("ZIII", +0.0004303465157577957),
 ("IIZZ", +0.5099539391488543),
 ("IZIZ", +0.5099677387273946 ),
 ("IZZI", +0.5099488492845516),
 ("ZIIZ", +0.5099106232913859),
 ("ZIZI", +0.5099467089998899),
 ("ZZII", +0.5099046167492709)
])

result = vqe.compute_minimum_eigenvalue(H2_op)

plt.xlabel("Iteration")
plt.ylabel("Optimal Parameters")
plt.title("VQE Results")


plt.show()

print(result)




