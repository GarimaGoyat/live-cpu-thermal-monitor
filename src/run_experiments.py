"""Run all syllabus experiments without changing the live dashboard model."""

from classification import run_classification
from association_rules import run_apriori
from model_comparison import run_experiments as run_regression


if __name__ == "__main__":
    run_regression()
    run_classification()
    run_apriori()