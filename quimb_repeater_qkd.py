"""
quimb_repeater_qkd.py

Multi-repeater entanglement swapping + E91 & BB84 demos using quimb + numpy.
"""

import numpy as np
import quimb as qu
import random
from math import sqrt

# ------------------------
# Basic states and gates
# ------------------------
ZERO = np.array([1, 0], dtype=complex)
ONE  = np.array([0, 1], dtype=complex)
I2   = np.eye(2, dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)
H = 1/np.sqrt(2) * np.array([[1,1],[1,-1]], dtype=complex)

def kron(*args):
    """Kronecker product wrapper for vectors or matrices."""
    out = args[0]
    for a in args[1:]:
        out = np.kron(out, a)  # <-- use numpy for vectors
    return out

def bell_phi_plus():
    """|Φ+> = (|00> + |11>)/sqrt(2)"""
    return normalize(np.kron(ZERO, ZERO) + np.kron(ONE, ONE))


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n != 0 else v


# ------------------------
# Multi-repeater chain builder
# ------------------------
def build_bell_chain(num_segments):
    """
    Build initial state for a chain of num_segments Bell pairs.
    Each segment is a Bell pair; after swapping internal qubits, endpoints become entangled.
    Qubit ordering: [q0, q1, q2, q3, ..., q_{2*num_segments-1}]
    Pairs: (0,1),(2,3),(4,5),...
    Endpoints for end-to-end entanglement: q0 and q_{2*num_segments-1}
    """
    bell = bell_phi_plus()
    pieces = [bell for _ in range(num_segments)]
    state = pieces[0]
    for p in pieces[1:]:
        state = kron(state, p)
    return normalize(state)

# ------------------------
# State helper utilities
# ------------------------
def nqubits_from_state(state):
    dim = state.size
    return int(np.round(np.log2(dim)))

def idx_bits(index, n):
    return format(index, '0{}b'.format(n))

def project_on_bits(state, fixed_indices):
    """
    Project state (ket vector) onto basis states where qubit positions in fixed_indices
    match specified bits. fixed_indices: dict {pos: '0' or '1'} with pos 0 = leftmost qubit.
    Returns normalized projected state.
    """
    n = nqubits_from_state(state)
    dim = 2**n
    proj = np.zeros_like(state)
    for i in range(dim):
        bits = idx_bits(i, n)
        ok = True
        for pos, bit in fixed_indices.items():
            if bits[pos] != bit:
                ok = False
                break
        if ok:
            proj[i] = state[i]
    return normalize(proj)

# ------------------------
# Simple sampled noise channels
# ------------------------
def apply_depolarizing_sample(state, target, p):
    """
    Sampling-based single-qubit depolarizing on target (0 = leftmost) with prob p.
    This picks one Pauli at random (X/Y/Z) with equal weight when error occurs.
    """
    if random.random() >= p:
        return state
    paulis = [X, Y, Z]
    chosen = random.choice(paulis)
    return apply_single_qubit_gate(state, chosen, target)

def apply_single_qubit_gate(state, gate, target):
    n = nqubits_from_state(state)
    ops = []
    for i in range(n):
        ops.append(gate if i == target else I2)
    U = ops[0]
    for op in ops[1:]:
        U = np.kron(U, op)
    return U @ state

# ------------------------
# Bell measurement (CNOT + H, measure both)
# ------------------------
def bell_measure(state, a, b):
    """
    Perform Bell-state measurement on qubits (a,b), 0-based with a < b and a=leftmost index 0.
    Method: CNOT(a->b), H(a), then measure a and b in computational basis.
    Returns (outcome_string_of_length_2, post_state_vector)
    """
    n = nqubits_from_state(state)
    s = state.copy()
    # CNOT
    s = apply_cnot(s, a, b)
    # H on a
    s = apply_single_qubit_gate(s, H, a)
    # measure a and b: sample outcome according to amplitudes
    probs = np.abs(s)**2
    dim = 2**n
    # build probabilities by summing all basis states that share the same bits at a and b
    counts = {}
    for i in range(dim):
        bits = idx_bits(i, n)
        key = bits[a] + bits[b]
        counts[key] = counts.get(key, 0) + probs[i]
    keys = list(counts.keys())
    weights = np.array([counts[k] for k in keys])
    weights = weights / weights.sum()
    chosen = np.random.choice(keys, p=weights)
    # collapse / project onto chosen bits for a and b
    post = project_on_bits(s, {a: chosen[0], b: chosen[1]})
    return chosen, post

def apply_cnot(state, control, target):
    """Build full CNOT matrix by sparse construction and apply to state."""
    n = nqubits_from_state(state)
    dim = 2**n
    U = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        bits = list(idx_bits(i, n))
        if bits[control] == '1':
            # flip target
            bits[target] = '0' if bits[target] == '1' else '1'
        j = int(''.join(bits), 2)
        U[j, i] = 1
    return U @ state

# ------------------------
# Entanglement swapping / repeater routine
# ------------------------
def perform_swapping_chain(initial_state, num_segments, noise_p=0.0):
    """
    Perform entanglement swapping along the chain to connect qubit 0 and qubit (2*num_segments -1).
    We perform Bell measurements on qubit pairs (1,2), (3,4), ... in that order.
    Each measurement may be followed by Pauli corrections in a real repeater; here we return raw post-measurement state
    and the list of measurement outcomes (strings).
    noise_p: apply depolarizing sampling to the two qubits being measured before the BSM (simulates imperfect channels).
    """
    state = initial_state.copy()
    n = nqubits_from_state(state)
    outcomes = []
    # internal BSMs: for num_segments segments, we have (num_segments - 1) swaps
    # indices to BSM are (1,2), (3,4), ... up to 2*num_segments-3, 2*num_segments-2
    for k in range(num_segments - 1):
        a = 1 + 2*k      # left internal qubit
        b = 2 + 2*k      # right internal qubit
        # apply noise to those qubits (sampling)
        state = apply_depolarizing_sample(state, a, noise_p)
        state = apply_depolarizing_sample(state, b, noise_p)
        out, state = bell_measure(state, a, b)
        outcomes.append(((a,b), out))
        # Note: we do not apply Pauli corrections, so final end-to-end pair may need correction to be Φ+.
    # After all swaps, reduce to endpoints (0 and last)
    final_n = n
    last = 2*num_segments - 1
    # reshape to tensor form and extract subsystem (0,last)
    psi_tensor = state.reshape([2]*final_n)
    # fix middle indices by selecting full slice (we already collapsed measured qubits)
    # take slice psi_tensor[:, :, ..., :] to get 2x2 for endpoints with middle indices fixed to whatever they are now
    # Because we've collapsed measurements, the state already has zero amplitude for inconsistent middle bits.
    # We'll take partial trace by flattening indices 0 and last.
    # One convenient approach: reshape to (2, 2**(n-2), 2) by moving indices
    # Permute axes: bring endpoints to front and back
    perm = [0] + list(range(1, last)) + [last]
    psi_perm = np.transpose(psi_tensor, perm)
    psi_flat = psi_perm.reshape(2, 2**(n-2), 2)
    # To get effective 2-qubit state vector, we need to treat middle subsystem as fixed — but
    # because projections collapsed middle bits, only few entries are nonzero. We'll compute reduced density matrix instead:
    # form density matrix of full state and partial trace out middle qubits
    rho_full = np.outer(state, state.conj())
    # partial trace: we want rho_AB where A=0, B=last
    rho_AB = partial_trace(rho_full, keep=[0, last], dims=[2]*n)
    # try to get a pure-state vector if rank-1; else we'll return density matrix
    # if pure:
    evals, evecs = np.linalg.eigh(rho_AB)
    # pick largest eigenvector
    idx_max = np.argmax(evals)
    vec = evecs[:, idx_max]
    vec = normalize(vec)
    return outcomes, rho_AB, vec

def partial_trace(rho, keep, dims):
    """
    Use quimb's partial trace (robust). 
    keep = list of subsystem indices to keep (0-based, leftmost first).
    """
    n = len(dims)
    # subsystems to trace out
    to_trace = [i for i in range(n) if i not in keep]
    return qu.partial_trace(rho, keep=keep, dims=dims)


# ------------------------
# Fidelity & metrics
# ------------------------
def fidelity_state_to_bell(vec):
    """Fidelity |<Φ+|psi>|^2 for 2-qubit vector psi."""
    bell = bell_phi_plus()
    return np.abs(np.vdot(bell, vec))**2

def fidelity_rho_to_bell(rho):
    """Fidelity = <Φ+| rho |Φ+>"""
    bell = bell_phi_plus()
    return np.real(np.vdot(bell, rho @ bell))

# ------------------------
# E91 simulation (entanglement-based QKD)
# ------------------------
def measure_in_basis(state_vec, basis_angle):
    """
    Measure single-qubit |psi> in basis defined by angle on Bloch sphere in X-Z plane:
    basis states |b0> = cos(theta/2)|0> + sin(theta/2)|1>
                    |b1> = -sin(theta/2)|0> + cos(theta/2)|1>
    Returns 0 or 1 sample and post-measurement state.
    """
    theta = basis_angle
    b0 = np.array([np.cos(theta/2), np.sin(theta/2)], dtype=complex)
    b1 = np.array([-np.sin(theta/2), np.cos(theta/2)], dtype=complex)
    # probabilities when measuring the 2-qubit joint state are handled outside; here we only need single-qubit basis vectors
    return b0, b1

def e91_trial_from_vec(vec2, alice_angle, bob_angle):
    """
    Given 2-qubit pure state vector (2-qubit, length-4), measure each qubit in specified single-qubit basis angle.
    Returns measurement bits (a,b).
    """
    # build measurement projectors
    b0a, b1a = measure_in_basis(None, alice_angle)
    b0b, b1b = measure_in_basis(None, bob_angle)
    # construct joint basis states
    bases = [kron(b0a, b0b), kron(b0a, b1b), kron(b1a, b0b), kron(b1a, b1b)]
    probs = [np.abs(np.vdot(bs, vec2))**2 for bs in bases]
    probs = np.array(probs)
    probs = probs / probs.sum()
    outcome = np.random.choice([0,1,2,3], p=probs)
    a = 0 if outcome in (0,1) else 1
    b = 0 if outcome in (0,2) else 1
    return a, b

def run_e91(num_runs, num_segments, noise_p=0.0):
    """
    Run E91 using repeater to create entangled pairs each run.
    Alice & Bob choose measurement angles from standard E91 set:
    Alice: A1=0, A2=pi/4, A3=pi/2  (example set)
    Bob:   B1=pi/4, B2=pi/2, B3=3pi/4
    We'll compute CHSH S value estimate using subset of angle pairs and raw correlation.
    """
    A_angles = [0.0, np.pi/4, np.pi/2]
    B_angles = [np.pi/4, np.pi/2, 3*np.pi/4]
    # choose CHSH pairs (A2,B1), (A2,B2), (A3,B1), (A3,B2) mapping to angles (pi/4,pi/4),(pi/4,pi/2),(pi/2,pi/4),(pi/2,pi/2)
    S_samples = []
    raw_key_bits = []
    for _ in range(num_runs):
        # create chain and swap
        psi0 = build_bell_chain(num_segments)
        outcomes, rhoAB, vecAB = perform_swapping_chain(psi0, num_segments, noise_p=noise_p)
        # measure in randomly chosen E91 basis indices
        ai = random.randrange(3)
        bi = random.randrange(3)
        a_angle = A_angles[ai]
        b_angle = B_angles[bi]
        a, b = e91_trial_from_vec(vecAB, a_angle, b_angle)
        # For CHSH stats, store correlations for specific index pairs
        S_samples.append((ai, bi, a, b))
        # for raw key: pick runs where ai==0 and bi==0 (example), extract bit when measurement bases chosen appropriately
        if ai == 0 and bi == 0:
            # in E91 sometimes certain basis choices give correlated bits -> here treat equal bits as key bit
            raw_key_bits.append(1 if a==b else 0)
    # estimate simple CHSH S using collected subset matching the 4 CHSH pairs
    # convert S_samples to counts
    # compute expectation E(Ai,Bj) = <(-1)^a * (-1)^b>
    def compute_E(pairA, pairB):
        vals = []
        for (ai, bi, a, b) in S_samples:
            if ai == pairA and bi == pairB:
                vals.append(((1 if a==0 else -1) * (1 if b==0 else -1)))
        return np.mean(vals) if len(vals)>0 else 0.0
    E24 = compute_E(1,0)  # A2,B1
    E25 = compute_E(1,1)  # A2,B2
    E34 = compute_E(2,0)  # A3,B1
    E35 = compute_E(2,1)  # A3,B2
    S = E24 - E25 + E34 + E35
    return {'S_est': S, 'raw_key_len': len(raw_key_bits), 'raw_key': raw_key_bits}

# ------------------------
# BB84 simulation (prepare & measure)
# ------------------------
def bb84_trial(nbits=1000, channel_noise=0.0, eve_p=0.0):
    """
    Simple BB84 with sample-based bit-flip/depolarizing channel. Eve optional intercept-resend with prob eve_p.
    Returns estimated sift fraction and QBER (rough).
    """
    alice_bits = np.random.randint(0,2, size=nbits)
    alice_bases = np.random.randint(0,2, size=nbits)  # 0 = Z, 1 = X (H)
    bob_bases = np.random.randint(0,2, size=nbits)
    bob_results = np.zeros(nbits, dtype=int)
    for i in range(nbits):
        bit = alice_bits[i]
        basis = alice_bases[i]
        # prepare
        q = ZERO.copy() if bit==0 else ONE.copy()
        if basis == 1:
            q = H @ q
        # Eve intercept-resend
        if random.random() < eve_p:
            eve_basis = random.randint(0,1)
            if eve_basis == 1:
                qm = H @ q
            else:
                qm = q
            probs = np.abs(qm)**2
            outcome = np.random.choice([0,1], p=probs)
            q = ZERO.copy() if outcome==0 else ONE.copy()
            if eve_basis == 1:
                q = H @ q
        # channel noise (bit-flip with prob channel_noise)
        if random.random() < channel_noise:
            q = X @ q
        # Bob measures in his basis
        if bob_bases[i] == 1:
            q = H @ q
        probs = np.abs(q)**2
        bob_results[i] = np.random.choice([0,1], p=probs)
    # sift
    sift_idx = np.where(alice_bases == bob_bases)[0]
    if sift_idx.size == 0:
        return {'sift_size':0, 'qber': None, 'key_fraction':0}
    sift_a = alice_bits[sift_idx]
    sift_b = bob_results[sift_idx]
    # use half of sift to estimate QBER
    test_n = max(1, len(sift_a)//4)
    test_idx = np.random.choice(len(sift_a), size=test_n, replace=False)
    qber = np.mean(sift_a[test_idx] != sift_b[test_idx])
    key_fraction = (len(sift_a) - test_n) / nbits
    # if QBER>0.11 declare insecure (naive)
    if qber > 0.11:
        key_fraction = 0
    return {'sift_size': len(sift_a), 'qber': qber, 'key_fraction': key_fraction}

# ------------------------
# Demo runner
# ------------------------
if __name__ == "_main_":
    random.seed(42)
    np.random.seed(42)

    print("=== Multi-repeater entanglement swapping demo ===")
    for segments in [1, 2, 3, 4]:
        psi0 = build_bell_chain(segments)
        outcomes, rhoAB, vecAB = perform_swapping_chain(psi0, segments, noise_p=0.02)
        fid = fidelity_rho_to_bell(rhoAB)
        print(f"Segments={segments:2d}  swaps={segments-1:1d}  BSM_outcomes={outcomes}  Fidelity to Φ+ = {fid:.4f}")

    print("\n=== E91 via repeater demo ===")
    res = run_e91(num_runs=400, num_segments=3, noise_p=0.03)
    print("E91 result example:", res)

    print("\n=== BB84 demo (direct channel) ===")
    bb = bb84_trial(nbits=2000, channel_noise=0.03, eve_p=0.05)
    print("BB84 sample:", bb)
"""
quimb_repeater_animation.py
Animated visualization of entanglement swapping along a repeater chain.
Produces frames and saves an animated GIF: repeater_animation.gif

"""
import numpy as np
import quimb as qu
import matplotlib.pyplot as plt
import imageio
import math
import random
from pathlib import Path

# -------------------------
# Quantum helpers (vector-state style)
# -------------------------
ZERO = np.array([1, 0], dtype=complex)
ONE  = np.array([0, 1], dtype=complex)
I2   = np.eye(2, dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
H = 1/np.sqrt(2) * np.array([[1,1],[1,-1]], dtype=complex)

def kron(*args):
    out = args[0]
    for a in args[1:]:
        out = np.kron(out, a)
    return out

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n != 0 else v

def bell_phi_plus():
    return normalize(kron(ZERO, ZERO) + kron(ONE, ONE))

def nqubits_from_state(state):
    return int(round(math.log2(state.size)))

def apply_cnot(state, control, target):
    n = nqubits_from_state(state)
    dim = 2**n
    U = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        bits = list(format(i, f'0{n}b'))
        if bits[control] == '1':
            bits[target] = '0' if bits[target] == '1' else '1'
        j = int(''.join(bits), 2)
        U[j, i] = 1
    return U @ state

def apply_single(state, gate, target):
    n = nqubits_from_state(state)
    ops = [gate if i == target else I2 for i in range(n)]
    U = ops[0]
    for op in ops[1:]:
        U = np.kron(U, op)
    return U @ state

def project_on_bits(state, fixed):
    # fixed: dict {pos:'0' or '1'}
    n = nqubits_from_state(state)
    dim = 2**n
    proj = np.zeros_like(state)
    for i in range(dim):
        bits = format(i, f'0{n}b')
        ok = True
        for pos, bit in fixed.items():
            if bits[pos] != bit:
                ok = False
                break
        if ok:
            proj[i] = state[i]
    return normalize(proj)

# -------------------------
# Partial trace helper via quimb
# -------------------------
def rho_from_state(state):
    return np.outer(state, state.conj())
def reduce_to_endpoints(rho_full, n, keep=(0,-1)):
    # Convert to quimb array
    keep_indices = [k if k >= 0 else n + k for k in keep]
    dims = [2] * n
    rho_q = qu.qarray(rho_full)  # ✅ use qarray instead of asarray
    rho_red = qu.partial_trace(rho_q, keep=keep_indices, dims=dims)
    return np.asarray(rho_red)


def fidelity_rho_to_bell(rho):
    bell = bell_phi_plus()
    return float(np.real(np.vdot(bell, rho @ bell)))

# -------------------------
# Entanglement swapping simulation engine that records states for animation
# -------------------------
def build_initial_chain(num_segments):
    bells = [bell_phi_plus() for _ in range(num_segments)]
    state = bells[0]
    for b in bells[1:]:
        state = kron(state, b)
    return normalize(state)

def bell_measure_and_collapse(state, a, b):
    # perform CNOT(a->b), H(a), then sample measurement result on a and b and collapse
    s = apply_cnot(state, a, b)
    s = apply_single(s, H, a)
    # compute probs per two-bit outcomes at positions a and b
    n = nqubits_from_state(s)
    dim = 2**n
    probs = {}
    for i in range(dim):
        bits = format(i, f'0{n}b')
        key = bits[a] + bits[b]
        probs[key] = probs.get(key, 0) + abs(s[i])**2
    keys = list(probs.keys())
    weights = np.array([probs[k] for k in keys])
    weights = weights / weights.sum()
    chosen = np.random.choice(keys, p=weights)
    collapsed = project_on_bits(s, {a: chosen[0], b: chosen[1]})
    return chosen, collapsed

def perform_swapping_sequence(num_segments, noise_p=0.0):
    """
    Returns a list of snapshot dicts. Each snapshot contains:
      - 'state' : the full state vector after the step
      - 'step' : descriptive string
      - 'bsm' : tuple ((a,b), outcome) for each performed BSM (empty for initial)
      - 'fidelity': fidelity of endpoints with |Φ+>
    We'll include snapshots: initial, after each BSM.
    """
    snapshots = []
    psi = build_initial_chain(num_segments)
    n = nqubits_from_state(psi)
    # initial snapshot
    rho_full = rho_from_state(psi)
    rho_end = reduce_to_endpoints(rho_full, n, keep=(0,-1))
    fid = fidelity_rho_to_bell(rho_end)
    snapshots.append({'state': psi.copy(), 'step': 'initial', 'bsm': None, 'fidelity': fid})

    # perform sequential BSMs on (1,2), (3,4), ...
    for k in range(num_segments-1):
        a = 1 + 2*k
        b = 2 + 2*k
        # (optional) add simple sampling depolarizing noise on the two qubits before BSM:
        if noise_p > 0.0:
            if random.random() < noise_p:
                psi = apply_single(psi, X, a)
            if random.random() < noise_p:
                psi = apply_single(psi, X, b)
        outcome, psi = bell_measure_and_collapse(psi, a, b)
        rho_full = rho_from_state(psi)
        rho_end = reduce_to_endpoints(rho_full, n, keep=(0,-1))
        fid = fidelity_rho_to_bell(rho_end)
        snapshots.append({'state': psi.copy(), 'step': f'BSM on ({a},{b}) -> {outcome}', 'bsm': ((a,b), outcome), 'fidelity': fid})
    return snapshots

# -------------------------
# Drawing utilities (matplotlib)
# -------------------------
def draw_chain_frame(snap, num_segments, filename=None, figsize=(8,3)):
    """
    Draws a frame for a given snapshot.
    - qubit lines horizontally
    - arcs for Bell pairs: at initial stage show all, later show remaining arcs
    - highlight measured pair when BSM step
    - display fidelity text
    """
    n_qubits = 2 * num_segments
    y_positions = list(range(n_qubits))[::-1]  # place top-to-bottom
    x_start = 0.1
    x_end = 0.9
    xs = np.linspace(x_start, x_end, 5)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0,1)
    ax.set_ylim(-0.5, n_qubits-0.5)
    ax.axis('off')

    # draw qubit horizontal lines and labels
    for i, y in enumerate(y_positions):
        ax.hlines(y, x_start, x_end, lw=2, color='black')
        ax.text(x_start - 0.02, y, f"q{i}", va='center', ha='right', fontsize=10)

    # determine which Bell-pairs are currently present from the state's amplitude pattern:
    # We will visualize pairs (0,1),(2,3),... as arcs initially.
    for seg in range(num_segments):
        qleft = 2*seg
        qright = 2*seg + 1
        # position x for arc center:
        x_center = (x_start + x_end)/2 - 0.2 + 0.4 * (seg / max(1, num_segments-1))
        y_top = y_positions[qleft]
        y_bottom = y_positions[qright]
        y_mid = (y_top + y_bottom) / 2
        # draw a semicircle arc between the two qubit lines (vertical spacing may be 1 step)
        # use a bezier-like semicircle using plot
        arc_x = np.linspace(x_center - 0.05, x_center + 0.05, 50)
        height = abs(y_top - y_bottom) * 0.6
        arc_y = y_mid + height * np.sin(np.pi * (arc_x - (x_center-0.05))/0.1)
        ax.plot(arc_x, arc_y, lw=3, zorder=1)

        # label pair index
        ax.text(x_center, y_mid + height + 0.15, f"pair{seg}", ha='center', fontsize=8)

    # if this snapshot is a BSM, highlight it
    if snap['bsm'] is not None:
        (a,b), outcome = snap['bsm']
        # highlight the measured lines (red) and a marker
        ya = y_positions[a]
        yb = y_positions[b]
        ax.hlines(ya, x_start, x_end, lw=4, color='red', alpha=0.6)
        ax.hlines(yb, x_start, x_end, lw=4, color='red', alpha=0.6)
        ax.text(0.5, (ya+yb)/2 + 0.2, f"BSM on ({a},{b}) result={outcome}", ha='center', fontsize=10, color='red')

    # fidelity display
    fid = snap['fidelity']
    ax.text(0.95, n_qubits - 0.2, f"Fidelity (endpoints, Φ⁺) = {fid:.4f}", ha='right', fontsize=10, bbox=dict(boxstyle='round', fc='wheat', alpha=0.8))

    # step label
    ax.text(0.5, -0.3, snap['step'], ha='center', fontsize=11)

    # save or return image
    fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=150)
        plt.close(fig)
    else:
        return fig

# -------------------------
# Main: produce frames and save GIF
# -------------------------
def make_animation(num_segments=3, noise_p=0.0, out_gif="repeater_animation.gif"):
    snapshots = perform_swapping_sequence(num_segments, noise_p=noise_p)
    tmp_dir = Path("frames_repeater")
    tmp_dir.mkdir(exist_ok=True)
    filenames = []
    for i, snap in enumerate(snapshots):
        fname = tmp_dir / f"frame_{i:03d}.png"
        draw_chain_frame(snap, num_segments, filename=str(fname))
        filenames.append(str(fname))

    # optionally add a couple of repeated final frames
    filenames += [filenames[-1]] * 6

    # save gif
    frames = []
    for fn in filenames:
        frames.append(imageio.v2.imread(fn))
    imageio.mimsave(out_gif, frames, duration=0.8)
    print(f"Animation saved to {out_gif}  (frames in {tmp_dir})")

if __name__ == "_main_":
    random.seed(1)
    np.random.seed(1)
    # choose number of segments: 1 => no swapping, 2 => 1 swap, 3 => 2 swaps, etc.
    make_animation(num_segments=3, noise_p=0.02, out_gif="repeater_animation.gif")
# --- append to quimb_repeater_qkd.py or paste near bottom ---

def calc_bb84(nbits=1000, channel_noise=0.0, eve_p=0.0, return_sift_bits=True):
    """Run BB84 and give a few usable outputs for the simulator."""
    res = bb84_trial(nbits=nbits, channel_noise=channel_noise, eve_p=eve_p)
    output = {
        'sift_size': res['sift_size'],
        'qber': res['qber'],
        'key_fraction': res['key_fraction'],
        # produce example sift bits to show in animation (re-run small trial to get actual sift bits)
        'sift_bits': None
    }
    if return_sift_bits and res['sift_size']>0:
        # quick small re-run to obtain actual sift bits and matching bob bits for animation
        alice_bits = np.random.randint(0,2, size=nbits)
        alice_bases = np.random.randint(0,2, size=nbits)
        bob_bases = np.random.randint(0,2, size=nbits)
        bob_results = np.zeros(nbits, dtype=int)
        for i in range(nbits):
            bit = alice_bits[i]
            basis = alice_bases[i]
            q = ZERO.copy() if bit==0 else ONE.copy()
            if basis == 1:
                q = H @ q
            if random.random() < eve_p:
                eve_basis = random.randint(0,1)
                qm = H @ q if eve_basis==1 else q
                outcome = np.random.choice([0,1], p=np.abs(qm)**2)
                q = ZERO.copy() if outcome==0 else ONE.copy()
                if eve_basis == 1:
                    q = H @ q
            if random.random() < channel_noise:
                q = X @ q
            if bob_bases[i] == 1:
                q = H @ q
            probs = np.abs(q)**2
            bob_results[i] = np.random.choice([0,1], p=probs)
        sift_idx = np.where(alice_bases == bob_bases)[0]
        sift_a = alice_bits[sift_idx]
        sift_b = bob_results[sift_idx]
        output['sift_bits'] = list(zip(sift_a.tolist(), sift_b.tolist()))
    return output

def calc_e91(num_runs=200, num_segments=3, noise_p=0.0):
    """Run your run_e91 but return more usable arrays for animation/visualization."""
    # run_e91 returns S_est, raw_key_len, raw_key
    res = run_e91(num_runs=num_runs, num_segments=num_segments, noise_p=noise_p)
    return {'S_est': res['S_est'], 'raw_key_len': res['raw_key_len'], 'raw_key': res['raw_key']}

def calc_repeater(num_segments=3, noise_p=0.0):
    """Run one swapping chain and return BSM outcomes and endpoint fidelity for animation."""
    psi0 = build_bell_chain(num_segments)
    outcomes, rhoAB, vecAB = perform_swapping_chain(psi0, num_segments, noise_p=noise_p)
    fid = fidelity_rho_to_bell(rhoAB)
    return {'outcomes': outcomes, 'fidelity': fid, 'rhoAB': rhoAB, 'vecAB': vecAB}

# correct main guard:
if __name__ == "__main__":
    # keep your original demo prints if you want to run the module standalone
    random.seed(42)
    np.random.seed(42)
    print("Demo runs (module executed directly).")
    print("E91 example:", calc_e91(num_runs=400, num_segments=3, noise_p=0.03))
    print("BB84 example:", calc_bb84(nbits=2000, channel_noise=0.03, eve_p=0.05))
    print("Repeater example:", calc_repeater(num_segments=3, noise_p=0.02))
