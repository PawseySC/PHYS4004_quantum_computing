import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import qutip as qt
from qutip.bloch import Bloch


def animate_gate(U, name, steps=40, fps=10):

    if name in ("Z", "RZ"):
        U_h = (qt.sigmax() + qt.sigmaz()) / np.sqrt(2)
        psi0 = U_h * qt.basis(2, 0)
    else:
        psi0 = qt.basis(2, 0)

    ops = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    v_start = qt.expect(ops, psi0)
    v_end   = qt.expect(ops, U * psi0)

    vecs = []
    for t in np.linspace(0, 1, steps):
        v = (1 - t) * np.array(v_start) + t * np.array(v_end)
        vecs.append((v / np.linalg.norm(v)).tolist())

    fig = plt.figure(figsize=(1.5,1.7)) 

    b = Bloch(fig = fig)
    b.vector_color = ['darkblue']
    b.point_size   = [30]
    b.sphere_alpha = 0.15
    b.view         = [30, 30]
    b.vector_style = 'fancy'

    #b.xlabel = [
    #    r'$\frac{|0\rangle + |1\rangle}{\sqrt{2}}$',
    #    r'$\frac{|0\rangle - |1\rangle}{\sqrt{2}}$'
    #]
    #b.ylabel = [
    #    r'$\frac{|0\rangle + i|1\rangle}{\sqrt{2}}$',
    #    r'$\frac{|0\rangle - i|1\rangle}{\sqrt{2}}$'
    #]

    b.make_sphere() 

    def update(i):
        b.clear()
        b.add_vectors([vecs[i]]) 
        b.make_sphere() 
        return []

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(vecs),
        interval=1000/fps,
        blit=False
    )

    ani.save(f"bloch_{name}.gif", writer='pillow', fps=fps)
    plt.close(fig)

if __name__ == "__main__":
    theta = np.pi / 2

    U_rx = (-1j * theta/2 * qt.sigmax()).expm()
    U_rz = (-1j * theta/2 * qt.sigmaz()).expm()

    U_h  = (qt.sigmax() + qt.sigmaz()) / np.sqrt(2)

    gates = {
        "X":  qt.sigmax(),
        "Z":  qt.sigmaz(),
        "H":  U_h,
        "RX": U_rx,
        "RZ": U_rz,
    }

    for name, U in gates.items():
        animate_gate(U, name)
