# PHYS4004 Quantum Computing Workshop

This repository contains accompanying activities for the **PHYS4004 High Performance Computing** session on quantum computing at Curtin University.


- **`Part_1_Programming_with_Qiskit.ipynb`**
  Introduces quantum circuits, single and two-qubit gates, measurement visualisation and simulation with Qiskit Aer.

- **`Part_2_Solving_Max_Cut_with_QAOA.ipynb`**
  Introduces the Max-Cut problem and builds quantum-solver using the *Quantum Approximate Optimisation Algorithm* (QAOA).


## Suggested Setup

1. **Clone the repository and move into its root directory**

   ```bash
   git clone https://github.com/PawseySC/PHYS4004_quantum_computing.git
   cd PHYS4004_quantum_computing
   ```

2. **Create and activate a virtual environment, then install the Python dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   python -m pip install -r requirements.txt
   ```

3. **Launch JupyterLab**

   ```bash
   python -m jupyterlab
   ```

4. **Open the notebooks**

   Open the notebooks from the Jupyterlab file browser.
