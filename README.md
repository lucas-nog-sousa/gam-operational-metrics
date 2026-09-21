# Finding the SLA Breakpoint with Generalized Additive Models (GAMs)

This directory contains the reproducible Python implementation for analyzing non-linear operational customer churn thresholds using Generalized Additive Models (GAMs) and Splines.

## Problem Statement
Standard linear regressions or rigid logistic models assume every extra minute of wait time penalizes customer retention equally. In reality, human patience exhibits non-linear threshold dynamics. This repository implements a `pygam` pipeline to extract exact inflection points in support logs (e.g., Zendesk, Jira).

## Repository Structure
- `main.py`: Complete Python script for dataset generation, GAM training, summary evaluation, and probability-scale PDP visualization.
- `requirements.txt`: Environment dependencies.
- `assets/`: Generated charts and visualizations.

## Quickstart

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/lucasnogsousa/data-science-ai.git](https://github.com/lucasnogsousa/data-science-ai.git)
   cd data-science-ai/tutorials/02-gam-operational-metrics
