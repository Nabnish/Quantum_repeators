"""
repeater_3d_neon.py

Polished 3D Quantum Repeater visualizer (VPython) driven by quimb-based swapping engine.
- Neon glowing links
- Photon particles with trails
- Pulsing swap flashes
- Camera orbit
- HUD with fidelity, noise, and controls

Controls:
 - Start button / S key        : run full swap sequence (animated)
 - Step button  / Right Arrow  : perform next swap only
 - Reset button / R key        : reset to initial chain
 - Auto toggle button / A key  : toggle automatic stepping
 - Up / Down keys              : increase / decrease noise (0.0 - 0.4)
 - Esc / Q                     : quit VPython window (close tab)
"""

import time
import math
import random as pyrandom
import threading

import numpy as np
import quimb as qu
from vpython import *   # vpython GUI

# -------------------------
# Quantum primitives (kept faithful)
# -------------------------
ZERO = np.array([1, 0], dtype=complex)
ONE  = np.array([0, 1], dtype=complex)
I2   = np.eye(2, dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)
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
    return int(round(np.log2(state.size)))

def idx_bits(index, n):
    return format(index, '0{}b'.format(n))

def apply_single_qubit_gate(state, gate, target):
    n = nqubits_from_state(state)
    ops = []
    for i in range(n):
        ops.append(gate if i == target else I2)
    U = ops[0]
    for op in ops[1:]:
        U = np.kron(U, op)
    return U @ state

def apply_cnot(state, control, target):
    n = nqubits_from_state(state)
    dim = 2**n
    U = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        bits = list(idx_bits(i, n))
        if bits[control] == '1':
            bits[target] = '0' if bits[target] == '1' else '1'
        j = int(''.join(bits), 2)
        U[j, i] = 1
    return U @ state

def apply_depolarizing_sample(state, target, p):
    if pyrandom.random() >= p:
        return state
    paulis = [X, Y, Z]
    chosen = pyrandom.choice(paulis)
    return apply_single_qubit_gate(state, chosen, target)

def bell_measure(state, a, b):
    n = nqubits_from_state(state)
    s = state.copy()
    s = apply_cnot(s, a, b)
    s = apply_single_qubit_gate(s, H, a)
    probs = np.abs(s)**2
    dim = 2**n
    counts = {}
    for i in range(dim):
        bits = idx_bits(i, n)
        key = bits[a] + bits[b]
        counts[key] = counts.get(key, 0) + probs[i]
    keys = list(counts.keys())
    weights = np.array([counts[k] for k in keys])
    weights = weights / weights.sum()
    chosen = np.random.choice(keys, p=weights)
    # projection
    proj = np.zeros_like(s)
    for i in range(dim):
        bits = idx_bits(i, n)
        if bits[a] == chosen[0] and bits[b] == chosen[1]:
            proj[i] = s[i]
    # normalize
    norm = np.linalg.norm(proj)
    if norm != 0:
        proj /= norm
    return chosen, proj

def partial_trace(rho, keep, dims):
    rho_q = qu.qarray(rho)
    return np.asarray(qu.partial_trace(rho_q, keep=keep, dims=dims))

def fidelity_rho_to_bell(rho):
    bell = bell_phi_plus()
    return float(np.real(np.vdot(bell, rho @ bell)))

# -------------------------
# Stepwise swapper class (drives quantum backend)
# -------------------------
class StepwiseSwapper:
    def __init__(self, num_segments=4, noise_p=0.02, rng_seed=None):
        self.num_segments = max(1, int(num_segments))
        self.noise_p = float(noise_p)
        if rng_seed is not None:
            pyrandom.seed(rng_seed)
            np.random.seed(rng_seed)
        self.reset()

    def reset(self):
        bell = bell_phi_plus()
        state = bell
        for _ in range(self.num_segments - 1):
            state = kron(state, bell)
        self.state = normalize(state)
        self.n = nqubits_from_state(self.state)
        self.next_swap_k = 0
        self.done = False
        self._update_endpoint()
        self.last_outcome = None

    def _update_endpoint(self):
        rho_full = np.outer(self.state, self.state.conj())
        last = 2*self.num_segments - 1
        self.rho_AB = partial_trace(rho_full, keep=[0, last], dims=[2]*self.n)
        try:
            evals, evecs = np.linalg.eigh(self.rho_AB)
            idx_max = np.argmax(evals)
            self.vec_AB = normalize(evecs[:, idx_max])
        except Exception:
            self.vec_AB = None
        self.last_fid = fidelity_rho_to_bell(self.rho_AB)

    def step(self):
        if self.done:
            return {'status':'done'}
        if self.next_swap_k >= (self.num_segments - 1):
            self.done = True
            return {'status':'done'}
        a = 1 + 2*self.next_swap_k
        b = 2 + 2*self.next_swap_k
        self.state = apply_depolarizing_sample(self.state, a, self.noise_p)
        self.state = apply_depolarizing_sample(self.state, b, self.noise_p)
        outcome, newstate = bell_measure(self.state, a, b)
        self.state = newstate
        self.next_swap_k += 1
        self._update_endpoint()
        self.last_outcome = ((a, b), outcome)
        snapshot = {
            'status':'ok',
            'performed_bsm':(a,b),
            'outcome': outcome,
            'fidelity': self.last_fid,
            'remaining_swaps': (self.num_segments - 1 - self.next_swap_k)
        }
        if self.next_swap_k >= (self.num_segments - 1):
            self.done = True
        return snapshot

# -------------------------
# VPython scene + polished visuals
# -------------------------
scene.title = "Quantum Repeater — Neon Visualizer"
scene.width = 1100
scene.height = 650
scene.background = vector(0.02, 0.02, 0.03)
scene.forward = vector(0, -0.15, -1)

# Parameters (tweakable)
NUM_SEGMENTS = 4       # default segments (>=1)
NOISE_P = 0.02
PHOTON_SPEED = 3.6     # units/sec
PHOTON_RETENTION = 30  # trail retention
CAMERA_ROT_SPEED = 0.015  # radians per second
DECOHERE_T = 18.0

# Visual element parameters
NODE_R = 0.45
LINK_R = 0.06
PH_R = 0.12

# layout helpers
def layout_positions(n_nodes, spacing=3.2):
    total_len = (n_nodes - 1) * spacing
    start_x = -total_len / 2
    return [vector(start_x + i*spacing, 0, 0) for i in range(n_nodes)]

class VisualNode:
    def __init__(self, name, pos):
        self.name = name
        self.pos = pos
        self.sphere = sphere(pos=pos, radius=NODE_R, color=vector(0.15,0.15,0.2),
                             emissive=True, shininess=0.8)
        self.tag = label(pos=pos + vector(0,-0.9,0), text=name, box=False, height=12, color=color.white)
        self.memory = None
    def set_entangled(self, t):
        self.memory = {'t_stored': t}
        self.sphere.color = vector(0.0,0.9,0.7)  # neon teal
        self.sphere.emissive = True
    def clear_memory(self):
        self.memory = None
        self.sphere.color = vector(0.15,0.15,0.2)
        self.sphere.emissive = False
    def set_decohered(self):
        self.memory = None
        self.sphere.color = vector(0.7,0.15,0.15)  # faded red
        self.sphere.emissive = False

class NeonLink:
    def __init__(self, a_pos, b_pos):
        axis = b_pos - a_pos
        self.cyl = cylinder(pos=a_pos, axis=axis, radius=LINK_R, opacity=0.6, color=vector(0.2,0.6,1.0))
        # glowing overlayed thin brighter cylinder for neon effect
        self.glow = cylinder(pos=a_pos, axis=axis, radius=LINK_R*0.45, color=vector(0.4,0.9,1.0), opacity=0.9)
        self.pulse_t = 0.0
    def pulse(self, strength=1.0, duration=0.6):
        self.pulse_t = time.time() + duration * strength
    def update(self):
        if self.pulse_t > time.time():
            frac = (self.pulse_t - time.time())
            # brightness pulse (simple)
            bright = 0.5 + 0.5 * (frac % 0.6) / 0.6
            col = vector(0.05*bright, 0.6*bright, 1.0*bright)
            self.glow.color = col
            self.glow.opacity = 0.9
        else:
            self.glow.color = vector(0.25,0.8,1.0)
            self.glow.opacity = 0.65

class PhotonParticle:
    def __init__(self, start, end, color_ph=vector(0.8,0.2,1.0)):
        self.start = start
        self.end = end
        self.start_t = time.time()
        dist = mag(end - start)
        self.travel_time = max(0.3, dist / PHOTON_SPEED)
        self.obj = sphere(pos=start, radius=PH_R, color=color_ph, make_trail=True, retain=PHOTON_RETENTION, emissive=True)
    def update(self):
        now = time.time()
        frac = (now - self.start_t) / self.travel_time
        if frac >= 1.0:
            self.obj.visible = False
            return False
        self.obj.pos = self.start + (self.end - self.start) * frac
        return True

# HUD
hud_panel = box(pos=vector(0, -3.4, 0), size=vector(12,0.8,0.1), color=vector(0.02,0.02,0.03), opacity=0.95)
fidelity_w = wtext(text=f"  Fidelity (Φ⁺): --   ")
noise_w = wtext(text=f"  Noise p: {NOISE_P:.3f}   ")
status_w = wtext(text="  Status: idle   ")

# Buttons and controls
def btn_start(b):
    start_sequence()

def btn_step(b):
    do_step()

def btn_reset(b):
    reset_all()

def btn_toggle_auto(b):
    toggle_auto()

button_start = button(text="Start", bind=btn_start)
button_step  = button(text="Step", bind=btn_step)
button_reset = button(text="Reset", bind=btn_reset)
button_auto  = button(text="Auto: OFF", bind=btn_toggle_auto)

# Scene globals
visual_nodes = []
visual_links = []
photons = []
swapper = None
auto_run = False
auto_interval = 0.9

# Build visuals & engine
def build_visuals(num_segments):
    global visual_nodes, visual_links, photons, swapper
    # clear previous visuals
    for vn in visual_nodes:
        try:
            vn.sphere.visible = False
            vn.tag.visible = False
        except: pass
    for link in visual_links:
        try:
            link.cyl.visible = False
            link.glow.visible = False
        except: pass
    for ph in photons:
        try:
            ph.obj.visible = False
        except: pass
    visual_nodes = []
    visual_links = []
    photons = []

    n_nodes = num_segments + 1
    pos_list = layout_positions(n_nodes)
    names = ["Alice"] + [f"R{i+1}" for i in range(num_segments-1)] + ["Bob"]
    for i, nm in enumerate(names):
        vn = VisualNode(nm, pos_list[i])
        visual_nodes.append(vn)
    for i in range(len(visual_nodes)-1):
        a = visual_nodes[i].pos
        b = visual_nodes[i+1].pos
        link = NeonLink(a, b)
        visual_links.append(link)
    # create swapper engine
    swapper = StepwiseSwapper(num_segments=num_segments, noise_p=NOISE_P, rng_seed=42)
    # initial local entanglement: color adjacent nodes green (local Bell pairs)
    tnow = time.time()
    for i in range(len(visual_nodes)-1):
        visual_nodes[i].set_entangled(tnow)
        visual_nodes[i+1].set_entangled(tnow)
    update_hud("--", NOISE_P, "ready")

def update_hud(fid, noiseval, status):
    fidelity_w.text = f"  Fidelity (Φ⁺): {fid if isinstance(fid,str) else f'{fid:.4f}'}   "
    noise_w.text = f"  Noise p: {noiseval:.3f}   "
    status_w.text = f"  Status: {status}   "

# Animation: animate a BSM step visually
def animate_bsm(snapshot):
    (a,b) = snapshot['performed_bsm']
    outcome = snapshot['outcome']
    fid = snapshot['fidelity']
    # map qubit indexes to node indexes (coarse mapping)
    idx_left = min(len(visual_nodes)-1, max(0, a//2))
    idx_right = min(len(visual_nodes)-1, max(0, b//2))
    # pulse links adjacent
    if idx_left < len(visual_links):
        visual_links[idx_left].pulse(strength=1.0)
    if idx_right-1 >= 0 and idx_right-1 < len(visual_links):
        visual_links[idx_right-1].pulse(strength=0.9)
    # create photons moving inward to measurement midpoints
    mid = (visual_nodes[idx_left].pos + visual_nodes[idx_right].pos)/2
    ph1 = PhotonParticle(visual_nodes[idx_left].pos, mid, color_ph=vector(0.2,1.0,0.9))
    ph2 = PhotonParticle(visual_nodes[idx_right].pos, mid, color_ph=vector(1.0,0.4,0.9))
    photons.append(ph1); photons.append(ph2)
    # flash measured nodes briefly
    flash_l = sphere(pos=visual_nodes[idx_left].pos, radius=NODE_R*1.4, color=vector(1,0.9,0.2), opacity=0.45, emissive=True)
    flash_r = sphere(pos=visual_nodes[idx_right].pos, radius=NODE_R*1.4, color=vector(1,0.9,0.2), opacity=0.45, emissive=True)

    # animate until photons reach midpoint
    tstart = time.time()
    while True:
        rate(120)
        # update photons
        for ph in list(photons):
            ok = ph.update()
            if not ok:
                try: photons.remove(ph)
                except: pass
        # update link glow
        for l in visual_links:
            l.update()
        # fade flashes
        elapsed = time.time() - tstart
        flash_l.opacity = max(0, 0.45 - elapsed*1.2)
        flash_r.opacity = max(0, 0.45 - elapsed*1.2)
        if not any([p for p in photons if p.start == visual_nodes[idx_left].pos or p.start == visual_nodes[idx_right].pos]):
            break
    flash_l.visible = False; flash_r.visible = False

    # update node memory mapping: measured nodes cleared, endpoints set entangled
    tnow = time.time()
    if 0 <= idx_left < len(visual_nodes): visual_nodes[idx_left].clear_memory()
    if 0 <= idx_right < len(visual_nodes): visual_nodes[idx_right].clear_memory()
    new_left = idx_left - 1
    new_right = idx_right + 1
    if 0 <= new_left < len(visual_nodes) and 0 <= new_right < len(visual_nodes):
        visual_nodes[new_left].set_entangled(tnow)
        visual_nodes[new_right].set_entangled(tnow)
    update_hud(fid, swapper.noise_p, f"performed BSM {a},{b}")

# Controls: run sequence, step, reset, auto toggle
sequence_thread = None
sequence_lock = threading.Lock()

def sequence_runner():
    global auto_run
    while True:
        with sequence_lock:
            if swapper.done:
                break
            snap = swapper.step()
        if snap.get('status') == 'done':
            break
        animate_bsm(snap)
        time.sleep(auto_interval)
    update_hud(swapper.last_fid, swapper.noise_p, "sequence finished")

def start_sequence():
    global sequence_thread, auto_run
    if sequence_thread and sequence_thread.is_alive():
        return
    update_hud(swapper.last_fid, swapper.noise_p, "running")
    sequence_thread = threading.Thread(target=sequence_runner)
    sequence_thread.daemon = True
    sequence_thread.start()

def do_step():
    if swapper.done:
        update_hud(swapper.last_fid, swapper.noise_p, "done")
        return
    snap = swapper.step()
    if snap.get('status') != 'done':
        animate_bsm(snap)
    else:
        update_hud(swapper.last_fid, swapper.noise_p, "done")

def reset_all():
    global auto_run
    if sequence_thread and sequence_thread.is_alive():
        # we won't forcibly kill thread; user can wait or re-open after finishing
        pass
    swapper.reset()
    tnow = time.time()
    for vn in visual_nodes:
        vn.clear_memory()
    # initial local entanglement coloring
    for i in range(len(visual_nodes)-1):
        visual_nodes[i].set_entangled(tnow)
        visual_nodes[i+1].set_entangled(tnow)
    update_hud("--", swapper.noise_p, "reset")

def toggle_auto():
    global auto_run
    auto_run = not auto_run
    button_auto.text = f"Auto: {'ON' if auto_run else 'OFF'}"
    if auto_run:
        start_sequence()

# keyboard handler
def keydown(evt):
    s = evt.key
    if s == 'w':
        swapper.noise_p = min(0.4, swapper.noise_p + 0.01)
        noise_w.text = f"  Noise p: {swapper.noise_p:.3f}   "
    elif s == 's':
        swapper.noise_p = max(0.0, swapper.noise_p - 0.01)
        noise_w.text = f"  Noise p: {swapper.noise_p:.3f}   "
    elif s == 'v':
        start_sequence()
    elif s == 'd':
        do_step()
    elif s == 'r':
        reset_all()
    elif s == 'a':
        toggle_auto()
    elif s in ('q','escape'):
        # close scene by making main loop stop: VPython typically closes the browser tab manually
        update_hud(swapper.last_fid if hasattr(swapper,'last_fid') else "--", swapper.noise_p if swapper else NOISE_P, "closing")
        # break the main_loop by returning; user can close tab manually
        # No explicit close API available; instruct user
        print("Press browser tab close to exit VPython if needed.")

scene.bind('keydown', keydown)

# camera slow orbit in background (non-blocking by using while loop in main)
def camera_orbit_step(dt):
    theta = CAMERA_ROT_SPEED * dt
    # rotate camera around y axis
    cam = scene.camera
    cam.rotate(angle=theta, axis=vector(0,1,0), origin=vector(0,0,0))

# Build initial visuals and engine
build_visuals(NUM_SEGMENTS)

# main loop: keep UI responsive; update photons and link glow and camera
last_time = time.time()
while True:
    rate(90)
    now = time.time()
    dt = now - last_time
    last_time = now
    # update all photons (remove finished)
    for ph in list(photons):
        ok = ph.update()
        if not ok:
            try: photons.remove(ph)
            except: pass
    # link glow updates
    for l in visual_links:
        l.update()
    # decoherence of memories
    for vn in visual_nodes:
        if vn.memory is not None:
            if time.time() - vn.memory['t_stored'] > DECOHERE_T:
                vn.set_decohered()
    # camera orbit
    camera_orbit_step(dt)
    # if auto-run and no sequence thread, launch
    if auto_run:
        # if no thread or thread finished, start
        if not (sequence_thread and sequence_thread.is_alive()):
            start_sequence()

