"""
Visualize the latent regime process and events.
"""
from pathlib import Path

import polars as pl
import matplotlib.pyplot as plt

def main():
    labels = pl.read_parquet("data/raw/labels.parquet")

    # Convert to numpy for plotting
    ts = list(range(len(labels)))  # Use index as x-axis
    vol = labels["vol"].to_numpy()
    liq = labels["liq"].to_numpy()
    regime = labels["regime_id"].to_numpy()

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Plot 1: Volatility with regime coloring
    ax1 = axes[0]
    colors = ['green', 'orange', 'red']
    for i in range(3):
        mask = regime == i
        ax1.scatter([t for t, m in zip(ts, mask) if m],
                    vol[mask], c=colors[i], s=1, label=f'Regime {i}')
    ax1.set_ylabel('Volatility')
    ax1.set_title('Volatility (colored by regime)')
    ax1.axhline(y=vol.mean(), color='black', linestyle='--', alpha=0.5, label='Mean')
    ax1.legend(loc='upper right')

    # Plot 2: Liquidity
    ax2 = axes[1]
    ax2.plot(ts, liq, color='blue', linewidth=0.5)
    ax2.set_ylabel('Liquidity')
    ax2.set_title('Liquidity (drives event rate)')
    ax2.axhline(y=liq.mean(), color='black', linestyle='--', alpha=0.5)

    # Plot 3: Regime over time
    ax3 = axes[2]
    ax3.plot(ts, regime, color='purple', linewidth=0.5)
    ax3.set_ylabel('Regime ID')
    ax3.set_xlabel('Time step')
    ax3.set_title('Regime transitions')
    ax3.set_yticks([0, 1, 2])

    Path("artifacts").mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig("artifacts/latent_process.png", dpi=150)
    print("Saved to artifacts/latent_process.png")
    plt.show()

if __name__ == "__main__":
    main()