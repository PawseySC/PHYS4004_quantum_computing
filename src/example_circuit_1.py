from qiskit import QuantumCircuit
import matplotlib.pyplot as plt

qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

fig = qc.draw(output='mpl', fold=-1, scale = 0.25)
fig.savefig('bell_circuit.png', dpi=300, bbox_inches='tight')