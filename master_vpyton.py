# main_vpython.py
from vpython import *
import random
import time
import quimb_repeater_qkd as qr   # your calc module

scene.title = "QKD Simulation (live from quimb calculations)"
scene.width = 900
scene.height = 500
scene.range = 6

# quick UI-ish variable (choose protocol)
PROTOCOL = "E91"   # choose "E91" or "BB84" or "REPEATER"

# --- visual setup (same as your E91 vp code) ---
source_pos = vector(0,0,0)
alice_pos  = vector(-5,0,0)
bob_pos    = vector(5,0,0)

source = sphere(pos=source_pos, radius=0.4, color=color.yellow)
alice = box(pos=alice_pos, size=vector(0.5,1,1), color=color.cyan, opacity=0.6)
bob   = box(pos=bob_pos, size=vector(0.5,1,1), color=color.magenta, opacity=0.6)
label(pos=alice_pos+vector(0,1.2,0), text="Alice", height=14)
label(pos=bob_pos+vector(0,1.2,0), text="Bob", height=14)

alice_text = label(pos=alice_pos+vector(0,-1.5,0), text="", height=14)
bob_text   = label(pos=bob_pos+vector(0,-1.5,0), text="", height=14)

def make_photon(start):
    photon = sphere(pos=start, radius=0.15, color=color.white, make_trail=False)
    return photon

# ----- obtain data from calculation module -----
if PROTOCOL == "E91":
    calc = qr.calc_e91(num_runs=300, num_segments=3, noise_p=0.03)
    # calc['raw_key'] is a list of key bits (0/1) collected when ai==0 and bi==0
    bits_stream = calc.get('raw_key', [])
    if len(bits_stream) == 0:
        # fallback: build pseudo-random correlated bits
        bits_stream = [random.choice([0,1]) for _ in range(200)]
    display_text = f"Protocol=E91  S_est={calc['S_est']:.3f}  key_len={calc['raw_key_len']}"
elif PROTOCOL == "BB84":
    calc = qr.calc_bb84(nbits=2000, channel_noise=0.03, eve_p=0.05)
    # calc['sift_bits'] = list of (alice_bit, bob_bit)
    sift = calc.get('sift_bits', [])
    bits_stream = [a for (a,b) in sift] if sift else [random.choice([0,1]) for _ in range(200)]
    display_text = f"Protocol=BB84  qber={calc['qber']}"
else:  # REPEATER
    calc = qr.calc_repeater(num_segments=3, noise_p=0.02)
    # for repeater you might animate the sequence of BSM outcomes
    bsm_outcomes = calc['outcomes']
    display_text = f"REPEATER fidelity={calc['fidelity']:.4f}"
    # create a simple bit stream from vecAB amplitudes if needed
    bits_stream = [0 if abs(calc['vecAB'][0])>abs(calc['vecAB'][3]) else 1]*50

# show header
label(pos=vector(0,3.8,0), text=display_text, height=14, box=False)

# ----- animation loop using bits_stream -----
idx = 0
while True:
    rate(0.8)
    photon_A = make_photon(source_pos)
    photon_B = make_photon(source_pos)
    target_A = alice_pos
    target_B = bob_pos
    for t in range(100):
        rate(120)
        photon_A.pos = source_pos + (target_A - source_pos) * (t/100)
        photon_B.pos = source_pos + (target_B - source_pos) * (t/100)
    # take next bit(s)
    bit = bits_stream[idx % len(bits_stream)]
    idx += 1
    # E91/BB84 mapping: both get same if entangled/correlated; for BB84 you may show mismatch if bit differs
    if PROTOCOL == "BB84" and sift:
        # for BB84, show real alice/bob pair
        a_bit, b_bit = sift[(idx-1) % len(sift)]
        photon_A.color = color.green if a_bit==1 else color.red
        photon_B.color = color.green if b_bit==1 else color.red
        alice_text.text = f"Alice bit: {a_bit}"
        bob_text.text   = f"Bob bit:   {b_bit}"
    else:
        photon_A.color = color.green if bit==1 else color.red
        photon_B.color = color.green if bit==1 else color.red
        alice_text.text = f"Alice bit: {bit}"
        bob_text.text   = f"Bob bit:   {bit}"

    rate(1)
    photon_A.visible = False
    photon_B.visible = False
