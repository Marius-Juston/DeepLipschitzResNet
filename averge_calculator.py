import json
import os
import pprint
from collections import defaultdict

if __name__ == '__main__':
    BASE_PATH = "trained_models/small_model"

    results = defaultdict(lambda: defaultdict(list))

    for dataset in os.listdir(BASE_PATH):
        dataset_path = os.path.join(BASE_PATH, dataset)
        if not os.path.isdir(dataset_path):
            continue

        for file in os.listdir(dataset_path):
            if not file.startswith("certified_seed_") or not file.endswith(".json"):
                continue

            file_path = os.path.join(dataset_path, file)
            with open(file_path, "r") as f:
                data = json.load(f)

            # group by epsilon keys
            for eps, vals in data.items():
                results[dataset][eps].append(vals)

    # compute averages
    averages = {}
    for dataset, eps_data in results.items():
        averages[dataset] = {}
        for eps, vals_list in eps_data.items():
            acc_avg = sum(v["accuracy"] for v in vals_list) / len(vals_list)
            cert_avg = sum(v["certified"] for v in vals_list) / len(vals_list)
            averages[dataset][eps] = {
                "accuracy": acc_avg,
                "certified": cert_avg,
            }

    # pretty print results
    pprint.pprint(averages)

    for dataset, vals in averages.items():
        epsilon = list(map(str, sorted(list(map(int, vals.keys())))))

        result = ' & '.join(f'{vals[v]['certified'] * 100:0.2f}' for v in epsilon)

        val = f'{dataset} | {result}'

        print(val)
