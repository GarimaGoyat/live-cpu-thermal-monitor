"""Small, understandable Apriori implementation for thermal risk patterns."""

from itertools import combinations
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import ROOT, experiment_frame

RESULTS_PATH = ROOT / "data" / "association_rules.csv"


def support(transactions, itemset):
    return sum(itemset.issubset(transaction) for transaction in transactions) / len(transactions)


def apriori(transactions, minimum_support=0.05, maximum_size=3):
    transactions = [set(transaction) for transaction in transactions]
    frequent = {}
    candidates = {frozenset([item]) for transaction in transactions for item in transaction}
    for size in range(1, maximum_size + 1):
        level = {itemset: support(transactions, itemset) for itemset in candidates if support(transactions, itemset) >= minimum_support}
        if not level:
            break
        frequent.update(level)
        items = sorted({item for itemset in level for item in itemset})
        candidates = {frozenset(combo) for combo in combinations(items, size + 1)}
    return frequent


def make_transactions(frame):
    transactions = []
    for _, row in frame.iterrows():
        cpu_band = "CPU_Usage=High" if row.CPU_Usage >= 50 else "CPU_Usage=Medium" if row.CPU_Usage >= 20 else "CPU_Usage=Low"
        gpu_band = "GPU_Temperature=High" if row.GPU_Temperature >= 45 else "GPU_Temperature=Normal"
        transactions.append({cpu_band, gpu_band, f"Risk={row.Temperature_Risk}"})
    return transactions


def run_apriori():
    rules_frame = experiment_frame()[["CPU_Usage", "GPU_Temperature", "Temperature_Risk"]].dropna()
    frequent = apriori(make_transactions(rules_frame), minimum_support=0.05)
    rows = []
    for itemset, itemset_support in frequent.items():
        if len(itemset) < 2:
            continue
        for consequent in itemset:
            antecedent = frozenset(itemset - {consequent})
            antecedent_support = frequent.get(antecedent, 0)
            confidence = itemset_support / antecedent_support if antecedent_support else 0
            rows.append({
                "Antecedent": ", ".join(sorted(antecedent)),
                "Consequent": consequent,
                "Support": itemset_support,
                "Confidence": confidence,
            })
    result = pd.DataFrame(rows).sort_values(["Confidence", "Support"], ascending=False)
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    result.to_csv(RESULTS_PATH, index=False)
    print("Apriori rules saved to", RESULTS_PATH)
    print(result.head(15).to_string(index=False))
    return result


if __name__ == "__main__":
    run_apriori()
