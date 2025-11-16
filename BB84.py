from vpython import *
import random

scene.width = 900
scene.height = 500
scene.title = "BB84 Quantum Key Distribution (Visual Demo)"

# --- Positions ---
alice_pos = vector(-5,0,0)
bob_pos = vector(5,0,0)

# --- Draw Alice and Bob ---
label(pos=alice_pos+vector(0,1,0), text="Alice", height=20)
label(pos=bob_pos+vector(0,1,0), text="Bob", height=20)

# --- Line connecting them ---
curve(pos=[alice_pos, bob_pos], color=color.white)

# --- Function to send one photon ---
def send_photon():
    # Random photon bit (0 or 1)
    bit = random.choice([0,1])

    # Create photon sphere near Alice
    photon = sphere(pos=alice_pos, radius=0.3, color=color.green if bit==0 else color.cyan, make_trail=False)

    # Animate photon traveling to Bob
    for step in range(100):
        rate(60)
        photon.pos.x += 0.1  # moves right

    # Randomly Bob chooses measurement basis
    bob_basis = random.choice(["Z", "X"])
    alice_basis = random.choice(["Z", "X"])

    # Decide if Bob’s measurement matches Alice
    if alice_basis == bob_basis:
        photon.color = color.green   # success
        result = "Match"
    else:
        photon.color = color.red     # mismatch
        result = "Mismatch"

    # Label above the photon after arriving
    label(pos=photon.pos+vector(0,0.7,0),
          text=f"Bit={bit}, A={alice_basis}, B={bob_basis}, {result}",
          height=12, box=False)

# --- Send multiple photons continuously ---
while True:
    send_photon()
    rate(1)   # slow down between photons