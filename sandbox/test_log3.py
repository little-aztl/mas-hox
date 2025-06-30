import json
import numpy as np

if __name__ == "__main__":
    with open("results/sacred/hok/vdn_env/47/info.json", "r") as f:
        data = json.load(f)

    battle_won_mean = np.array(data['battle_won_mean'])
    timestamp = np.array(data['battle_won_mean_T'])

    great_indices = np.argsort(battle_won_mean)[::-1]

    print(timestamp[great_indices[:10]])