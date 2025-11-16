# Quantum Repeater + E91 QKD Visual Simulator (GlowScript / VPython)
# Adds E91 key-generation after Alice-Bob end-to-end entanglement is formed.
# Author: ChatGPT (adapted for GlowScript)

from vpython import *
import random, math

# ---------------------
# PARAMETERS (edit)
# ---------------------
N_repeaters = 3
link_length = 6.0
attempt_interval = 1.0
p_photon_success = 0.6
p_bsm_success = 0.9
p_purify_success = 0.8
memory_decohere_time = 20
photon_speed = 6.0
show_photon_trails = False

# E91 params
e91_round_interval = 0.25   # seconds between measurement rounds once AB entangled
target_raw_key_len = 64     # stop key collection at this length
# ---------------------

scene.width = 1100
scene.height = 550
scene.title = "Quantum Repeater + E91 QKD — click to pause/resume"
paused = False
def toggle_pause(evt):
    global paused
    paused = not paused
scene.bind('click', toggle_pause)

# Nodes (Alice, Repeaters..., Bob)
nodes = []
names = ["Alice"] + [f"R{i+1}" for i in range(N_repeaters)] + ["Bob"]
n_nodes = len(names)
start_x = -(n_nodes - 1) * link_length / 2

for i, nm in enumerate(names):
    pos = vector(start_x + i * link_length, 0, 0)
    s = sphere(pos=pos, radius=0.6, color=color.gray(0.4))
    label(pos=pos + vector(0, 1.1, 0), text=nm, height=12, box=False)
    nodes.append({"name": nm, "sphere": s, "memory": None, "last_attempt": -1e9, "fidelity": 0.0})

# Fibers
for i in range(n_nodes - 1):
    a = nodes[i]["sphere"].pos
    b = nodes[i+1]["sphere"].pos
    cylinder(pos=a, axis=(b - a), radius=0.05, color=color.white * 0.7)

# Legend
legend_y = -2.5
legend_x_center = (start_x + (n_nodes - 1) * link_length / 2)
label(pos=vector(legend_x_center, legend_y, 0), text="Legend", height=14, box=False)
sphere(pos=vector(legend_x_center - 2.0, legend_y - 0.8, 0), radius=0.15, color=color.gray(0.4))
label(pos=vector(legend_x_center - 1.2, legend_y - 0.45, 0), text="Idle", height=10, box=False)
sphere(pos=vector(legend_x_center + 0.3, legend_y - 0.8, 0), radius=0.15, color=color.green)
label(pos=vector(legend_x_center + 1.1, legend_y - 0.45, 0), text="Entangled", height=10, box=False)
sphere(pos=vector(legend_x_center + 2.5, legend_y - 0.8, 0), radius=0.15, color=color.red)
label(pos=vector(legend_x_center + 3.3, legend_y - 0.45, 0), text="Decohered", height=10, box=False)

# Visuals & status
photons = []
def make_photon(a, b, c=color.cyan):
    return {"obj": sphere(pos=a, radius=0.15, color=c, make_trail=show_photon_trails),
            "start": a, "end": b, "t0": t}

status_label = label(pos=vector(0, -3.5, 0), text="", height=14, box=False, color=color.white)
bar = box(pos=vector(0, 2.2, 0), size=vector(0, 0.3, 0.3), color=color.green)
def update_status(msg, fidelity):
    status_label.text = msg
    bar.size.x = max(0.01, fidelity) * 8
    bar.color = color.green if fidelity > 0.7 else color.orange if fidelity > 0.4 else color.red

# QKD UI labels
key_label = label(pos=vector(0, -4.2, 0), text="Raw key: []", height=10, box=False)
qber_label = label(pos=vector(0, -4.6, 0), text="QBER: N/A", height=10, box=False)
chsh_label = label(pos=vector(0, -5.0, 0), text="CHSH S: N/A", height=10, box=False)

# Helper: show node color by memory
def show_state(i):
    n = nodes[i]
    s = n["sphere"]
    mem = n["memory"]
    if mem is None:
        s.color = color.gray(0.4)
    else:
        age = t - mem["stored"]
        if age > memory_decohere_time:
            s.color = color.red
        else:
            s.color = color.green

# Link attempt & entanglement bookkeeping
def attempt_link(i):
    a = nodes[i]["sphere"].pos
    b = nodes[i+1]["sphere"].pos
    photons.append(make_photon(a, b))
    success = random.random() < p_photon_success
    if success:
        fidelity = 0.55 + 0.45 * random.random()  # imperfect entanglement fidelity
        nodes[i]["memory"] = {"partner": i+1, "stored": t}
        nodes[i+1]["memory"] = {"partner": i, "stored": t}
        nodes[i]["fidelity"] = nodes[i+1]["fidelity"] = fidelity
        show_state(i); show_state(i+1)

def purify_if_possible(i):
    if i <= 0 or i >= n_nodes - 1: return
    left = i - 1; right = i + 1
    L = nodes[left]; R = nodes[right]
    if L["memory"] and R["memory"]:
        if random.random() < p_purify_success:
            new_fid = min(1.0, (L["fidelity"] + R["fidelity"]) / 2 + 0.08)
            L["fidelity"] = R["fidelity"] = new_fid

def try_bsm(i):
    if i <= 0 or i >= n_nodes - 1: return
    left = i - 1; right = i + 1
    left_ok = nodes[left]["memory"] and nodes[left]["memory"]["partner"] == i
    right_ok = nodes[right]["memory"] and nodes[right]["memory"]["partner"] == i
    if left_ok and right_ok:
        flash = sphere(pos=nodes[i]["sphere"].pos, radius=0.32, color=color.yellow, opacity=0.6)
        success = random.random() < p_bsm_success
        if success:
            new_fid = (nodes[left]["fidelity"] + nodes[right]["fidelity"]) / 2
            nodes[left]["memory"] = {"partner": right, "stored": t}
            nodes[right]["memory"] = {"partner": left, "stored": t}
            nodes[left]["fidelity"] = nodes[right]["fidelity"] = new_fid
            nodes[i]["memory"] = None
        else:
            nodes[left]["memory"] = nodes[right]["memory"] = nodes[i]["memory"] = None
        rate(10); flash.visible = False
        show_state(left); show_state(right); show_state(i)

def active_links():
    links = []
    for i in range(n_nodes - 1):
        L = nodes[i]["memory"]; R = nodes[i+1]["memory"]
        if not (L and R and L["partner"] == i+1 and R["partner"] == i):
            if t - nodes[i]["last_attempt"] >= attempt_interval:
                links.append(i)
    return links

# ---------------------
# E91 implementation
# ---------------------
# Measurement angle sets (radians)
A_angles = [0.0, math.pi/4, math.pi/2]          # Alice choices
B_angles = [math.pi/4, math.pi/2, 3*math.pi/4]  # Bob choices

# Data accumulators
e91_counts = []        # list of (ai,bi,a,b) tuples
raw_key = []           # collected raw key bits
chsh_acc = {}          # map (ai,bi) -> list of (+1/-1) products for computing E
for ai in range(3):
    for bi in range(3):
        chsh_acc[(ai,bi)] = []

def visibility_from_fidelity(F):
    # approximate visibility V in [-1,1]; for perfect F=1 -> V=1; F=0.5 -> V~0
    return max(0.0, min(1.0, 2*F - 1))

def e91_measure_once(fidelity):
    """
    Simulate one joint measurement on the entangled AB pair with given fidelity.
    We approximate probability of same outcomes as (1 + V*cos(delta))/2 where V ~ 2F-1.
    Returns (ai,bi,a,b) where a,b in {0,1}.
    """
    ai = random.randrange(3)
    bi = random.randrange(3)
    theta_a = A_angles[ai]
    theta_b = B_angles[bi]
    delta = theta_a - theta_b
    V = visibility_from_fidelity(fidelity)
    p_same = 0.5 * (1 + V * math.cos(delta))
    same = random.random() < p_same
    if same:
        # choose 00 or 11 with equal prob
        if random.random() < 0.5:
            a,b = 0,0
        else:
            a,b = 1,1
    else:
        # choose 01 or 10
        if random.random() < 0.5:
            a,b = 0,1
        else:
            a,b = 1,0
    return ai, bi, a, b

def process_e91_result(ai,bi,a,b):
    e91_counts.append((ai,bi,a,b))
    # accumulate CHSH product value (-1)^a * (-1)^b
    val = (1 if a==0 else -1) * (1 if b==0 else -1)
    chsh_acc[(ai,bi)].append(val)
    # choose key extraction rule: (ai==0 and bi==0) -> take Alice's bit as key
    if ai == 0 and bi == 0:
        raw_key.append(a)

def compute_qber_on_matching_basis():
    # compute QBER for rounds where ai==0 and bi==0 (matching key basis)
    pairs = [(ai,bi,a,b) for (ai,bi,a,b) in e91_counts if ai==0 and bi==0]
    if not pairs:
        return None
    errors = sum(1 for (_,_,a,b) in pairs if a != b)
    return errors / len(pairs)

def compute_chsh_S():
    # S = E(A2,B1) - E(A2,B2) + E(A3,B1) + E(A3,B2)
    # map Ai index: A2->1, A3->2 ; B1->0, B2->1
    def E(ai,bi):
        vals = chsh_acc.get((ai,bi), [])
        return sum(vals) / len(vals) if vals else 0.0
    E24 = E(1,0); E25 = E(1,1); E34 = E(2,0); E35 = E(2,1)
    return E24 - E25 + E34 + E35

# E91 driving
e91_running = False
last_e91_time = -1e9

def try_start_e91_if_ready():
    global e91_running, last_e91_time
    A = nodes[0]; B = nodes[-1]
    if A["memory"] and B["memory"] and A["memory"]["partner"] == n_nodes - 1:
        if not e91_running:
            e91_running = True
            last_e91_time = t
    else:
        e91_running = False

# ---------------------
# MAIN LOOP
# ---------------------
t = 0
dt = 0.05
last_log = 0

while True:
    rate(1 / dt)
    if paused: continue

    # move photons
    new_ph = []
    for ph in photons:
        dist = mag(ph["end"] - ph["start"])
        time_travel = dist / photon_speed
        f = (t - ph["t0"]) / time_travel
        if f >= 1:
            ph["obj"].visible = False
        else:
            ph["obj"].pos = ph["start"] + (ph["end"] - ph["start"]) * f
            new_ph.append(ph)
    photons[:] = new_ph

    # link attempts
    for i in active_links():
        nodes[i]["last_attempt"] = t
        attempt_link(i)

    # purification & BSMs
    for idx in range(1, n_nodes - 1):
        purify_if_possible(idx)
        try_bsm(idx)

    # decoherence
    for n in nodes:
        mem = n["memory"]
        if mem and t - mem["stored"] > memory_decohere_time:
            n["memory"] = None
            show_state(nodes.index(n))

    # check AB entanglement and start E91
    try_start_e91_if_ready()

    # run E91 rounds if running
    if e91_running and (t - last_e91_time) >= e91_round_interval and len(raw_key) < target_raw_key_len:
        last_e91_time = t
        # use Alice/Bob fidelity as the current fidelity estimate (take min of their stored fidelities)
        F_A = nodes[0]["fidelity"]; F_B = nodes[-1]["fidelity"]
        F_AB = min(F_A, F_B)
        ai,bi,a,b = e91_measure_once(F_AB)
        process_e91_result(ai,bi,a,b)

    # update UI every 1s
    if t - last_log > 1.0:
        last_log = t
        A = nodes[0]; B = nodes[-1]
        if A["memory"] and B["memory"] and A["memory"]["partner"] == n_nodes - 1:
            # alive AB entanglement
            F_AB = min(A["fidelity"], B["fidelity"])
            update_status(f"✅ Alice & Bob entangled (F={F_AB:.2f}) — collecting E91 rounds...", F_AB)
        else:
            best = max([n["fidelity"] for n in nodes])
            update_status(f"t = {t:.1f}s — distributing entanglement...", best)

        # show key, qber, S
        key_label.text = f"Raw key (len={len(raw_key)}): {''.join(str(b) for b in raw_key[-32:])}"
        qber = compute_qber_on_matching_basis()
        qber_label.text = f"QBER (matching basis): {qber:.3f}" if qber is not None else "QBER: N/A"
        S = compute_chsh_S()
        chsh_label.text = f"CHSH S estimate: {S:.3f}"

        # stop when key target reached
        if len(raw_key) >= target_raw_key_len:
            update_status(f"🔐 Collected raw key ({len(raw_key)} bits). Stop collecting.", F_AB)
            e91_running = False

    t += dt
