import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.linalg import expm
from scipy.optimize import minimize

def animate_qaoa(statevectors, interval=200, radius_scale=0.2, marker_scale=0.2, edges=False):
    num_frames, N = statevectors.shape
    G = nx.hypercube_graph(int(np.log2(N)))
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.axis('off')
    edge_color = 'gray' if edges else 'white'
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_color, width=2)
    circles, markers, texts = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        c = plt.Circle((x, y), 0, facecolor='white', edgecolor='blue', lw=2, zorder=2)
        ax.add_patch(c)
        circles.append(c)
        m = plt.Circle((x, y), radius_scale * marker_scale, facecolor='blue', zorder=3)
        ax.add_patch(m)
        markers.append(m)
        texts.append(ax.text(x, y, f'{node}', ha='center', va='center', zorder=4))
    def init():
        for c in circles:
            c.radius = 0
        for m in markers:
            m.radius = radius_scale * marker_scale
        return circles + markers + texts
    def update(frame):
        state = statevectors[frame]
        probs = np.abs(state)**2
        max_p = probs.max()
        norm = probs / max_p if max_p > 0 else probs
        phases = np.angle(state)
        for c, p in zip(circles, norm):
            c.radius = p * radius_scale
        for m, node, phi, p in zip(markers, G.nodes(), phases, norm):
            x, y = pos[node]
            r = p * radius_scale
            m.center = (x + r * np.cos(phi), y + r * np.sin(phi))
            m.radius = radius_scale * marker_scale
        return circles + markers + texts
    return FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True, interval=interval)

def compute_mixing(n, initial_state, t1, t2, delta_t):
    G = nx.hypercube_graph(int(np.log2(n)))
    A = nx.to_numpy_array(G)
    eigvals, eigvecs = np.linalg.eigh(A)
    times = np.arange(t1, t2 + delta_t, delta_t)
    coeffs = eigvecs.conj().T @ initial_state
    statevectors = np.array([eigvecs @ (np.exp(-1j * eigvals * t) * coeffs) for t in times])
    return times, statevectors

def phase_shift(n, initial_state, H_diag, t1, t2, delta_t):
    times = np.arange(t1, t2 + delta_t, delta_t)
    phase_factors = np.exp(-1j * np.outer(times, H_diag))
    return times, phase_factors * initial_state[np.newaxis, :]

def cycle_edges(n):
    return [(i, (i + 1) % n) for i in range(n)]

n = 3
edges = cycle_edges(n)
dim = 2**n
H_C = np.diag([sum((i >> u & 1) != (i >> v & 1) for u, v in edges) for i in range(dim)])
X = np.array([[0, 1], [1, 0]])
H_M = sum(np.kron(np.eye(2**i), np.kron(X, np.eye(2**(n - i - 1)))) for i in range(n))
state0 = np.ones(dim) / np.sqrt(dim)

def objective(params):
    gamma, beta = np.split(params, 2)
    state = state0.copy()
    for g, b in zip(gamma, beta):
        state = expm(-1j * g * H_C) @ state
        state = expm(-1j * b * H_M) @ state
    return -np.real(state.conj() @ H_C @ state)

res = minimize(objective, 3 * [0.8, 0.7], method='COBYLA', options={'maxiter': 200})

n = 8
H_diag = H_C.diagonal()
psi0 = np.ones(n, dtype=complex) / np.sqrt(n)
gamma, beta = np.split(res.x, 2)
frames = 25 * 4
writer = writer = FFMpegWriter(fps=25)

for i, (g, b) in enumerate(zip(gamma, beta)):
    _, sv1 = phase_shift(8, psi0, H_diag, 0, g, g / frames)
    anim = animate_qaoa(sv1)
    anim.save(f'qaoa_phase_{i}.mp4', writer=writer)
    _, sv2 = compute_mixing(8, sv1[-1], 0.0, b, b / frames)
    anim = animate_qaoa(sv2, edges=True)
    anim.save(f'qaoa_mix_{i}.mp4', writer=writer)
    psi0 = sv2[-1]
    if i == 0:
        statevectors = np.concat([sv1, sv2], axis=0)
    else:
        statevectors = np.concat([statevectors, sv1, sv2], axis=0)