import json
import numpy as np
from matplotlib import pyplot as plt
if __name__ == "__main__":
    with open("results/sacred/hok/monster_last_hp/2025-06-24 17:47_monster_lasthp_vdn_env.txt", "r") as f:
        last_hp = f.read()

    last_hp = np.array(eval(last_hp))

    plt.plot(last_hp)
    plt.xlabel("Timestamp")
    plt.ylabel("Last HP")
    plt.ylim(0, 40000)

    plt.savefig("sandbox/results/monster_last_hp2.png")