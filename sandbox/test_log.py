import json
import numpy as np
from matplotlib import pyplot as plt

def load_info(exp_number):
    with open(f"results/sacred/hok/vdn_env/{exp_number}/info.json", "r") as f:
        info = json.load(f)

    last_hp = np.array(info['monster_last_hp_mean'])
    timestamps = np.array(info['monster_last_hp_mean_T'])

    return last_hp, timestamps

exp_numbers = [11, 38, 47]

labels = [
    "Baseline",
    "Mask Monst. HP + Rela. Pos",
    # "Rela. Pos",
    # "Mask Monst. HP",
    "Mask Monst. HP + Rela. Pos + Dist. Penalty",
]

if __name__ == "__main__":

    plt.figure(figsize=(12, 6))

    for exp_number, label in zip(exp_numbers, labels):
        last_hp, timestamps = load_info(exp_number)
        plt.plot(timestamps, last_hp, label=label)

    plt.xlabel("Timestamp")
    plt.ylabel("Monster Last HP")
    plt.ylim(0, 30000)

    plt.legend()
    plt.savefig("sandbox/results/monster_last_hp.png")