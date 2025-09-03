#!/usr/bin/env python3
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

models = [
    "Logistic Regression",
    "Hybrid: LR + XGBoost",
    "Hybrid: LR + XGB + LGBM",
]
# Fraud-class F1 scores
f1_scores = [0.04, 0.55, 0.52]
colors = ["#8888ff", "#33cc99", "#ffcc66"]

plt.figure(figsize=(8, 4.5))
ax = plt.gca()
ax.bar(models, f1_scores, color=colors, edgecolor="#333333")
ax.set_ylim(0, 0.8)
ax.set_ylabel("F1 (Fraud class)")
ax.set_title("Fraud-class F1 Comparison")
for i, v in enumerate(f1_scores):
    ax.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom', color="#ffffff")
plt.xticks(rotation=15, ha='right')
plt.tight_layout()

out_dir = os.path.join(os.path.dirname(__file__), os.pardir, "images")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.abspath(os.path.join(out_dir, "f1_comparison.png"))
plt.savefig(out_path, dpi=200)
print(out_path)
