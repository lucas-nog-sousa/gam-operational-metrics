"""
Generalized Additive Models (GAMs) for Non-Linear SLA Breakpoint Analysis.

Author: Lucas Nogueira
Description: Simulates operational support logs (Zendesk) and fits a Logistic GAM 
             using pyGAM to identify critical wait-time thresholds for customer churn.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pygam import LogisticGAM, s


def generate_synthetic_data(n_tickets: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generates synthetic support ticket data with non-linear churn dynamics."""
    np.random.seed(seed)
    
    # Uniform wait time between 1 and 45 minutes
    wait_time = np.random.uniform(1, 45, n_tickets)
    
    # Non-linear log-odds of churn (patience cap at 15 mins, saturation at 30 mins)
    log_odds = -2.5 + 0.4 * np.maximum(0, wait_time - 15) - 0.35 * np.maximum(0, wait_time - 30)
    prob_churn = 1 / (1 + np.exp(-log_odds))
    
    # Binary churn event
    churn = np.random.binomial(1, prob_churn)
    
    return pd.DataFrame({'wait_time': wait_time, 'churn': churn})


def fit_logistic_gam(df: pd.DataFrame) -> LogisticGAM:
    """Fits a Logistic GAM with a smooth spline on wait time."""
    X = df[['wait_time']]
    y = df['churn']
    
    # s(0) applies a spline to the first feature column
    gam = LogisticGAM(s(0)).fit(X, y)
    return gam


def plot_partial_dependence(gam: LogisticGAM, save_path: str = None) -> None:
    """Plots PDP in probability space with 95% CIs and critical window highlight."""
    XX = gam.generate_X_grid(term=0)
    pdp, ci_log_odds = gam.partial_dependence(term=0, X=XX, width=0.95)
    
    # Convert log-odds to absolute probability
    intercept = gam.coef_[-1]
    prob_pdp = 1 / (1 + np.exp(-(pdp + intercept)))
    ci_prob_lower = 1 / (1 + np.exp(-(ci_log_odds[:, 0] + intercept)))
    ci_prob_upper = 1 / (1 + np.exp(-(ci_log_odds[:, 1] + intercept)))
    
    # Plotting
    plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(XX[:, 0], prob_pdp, color='#d62728', linewidth=2.5, label='Churn Probability (GAM)')
    plt.fill_between(XX[:, 0].ravel(), ci_prob_lower, ci_prob_upper, color='#d62728', alpha=0.15, label='95% CI')
    
    # Critical threshold (14-16 minutes)
    plt.axvspan(14, 16, color='#2b5c8f', alpha=0.2, label='Critical Window (14–16 min)')
    plt.axvline(x=15, color='#2b5c8f', linestyle='--', linewidth=1.5, label='Midpoint (~15 min)')
    
    plt.title('Churn Probability vs. Wait Time (Absolute Scale)', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Wait Time (Minutes)', fontsize=11)
    plt.ylabel('Estimated Churn Probability', fontsize=11)
    plt.ylim(0, 1)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Chart saved to {save_path}")
        
    plt.show()


if __name__ == "__main__":
    print("Generating synthetic dataset...")
    df_tickets = generate_synthetic_data()
    
    print("Fitting Logistic GAM...")
    model = fit_logistic_gam(df_tickets)
    
    print("\nModel Summary:")
    print(model.summary())
    
    print("\nGenerating Partial Dependence Plot...")
    plot_partial_dependence(model, save_path="assets/gam_churn_probability.png")
