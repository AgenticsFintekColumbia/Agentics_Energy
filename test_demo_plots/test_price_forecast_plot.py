#!/usr/bin/env python3
"""Quick test script to visualize the price forecast plot."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Sample realistic day-ahead prices showing arbitrage opportunity
# Low prices at night/early morning, high prices during peak hours
prices = [
    0.08, 0.07, 0.06, 0.06, 0.07, 0.09,  # Hours 0-5: Night (low)
    0.12, 0.15, 0.18, 0.20, 0.19, 0.17,  # Hours 6-11: Morning ramp-up
    0.16, 0.15, 0.17, 0.19, 0.21, 0.24,  # Hours 12-17: Afternoon peak
    0.26, 0.23, 0.19, 0.15, 0.11, 0.09   # Hours 18-23: Evening decline
]

dt_hours = 1.0
T = len(prices)
time_labels = list(range(T))
xlabel = "Hour of Day"

# Calculate price statistics
prices_array = np.array(prices)
mean_price = np.mean(prices_array)
std_price = np.std(prices_array)
min_price = np.min(prices_array)
max_price = np.max(prices_array)
price_spread = max_price - min_price

# Define thresholds for low and high prices
low_threshold = mean_price - 0.3 * std_price
high_threshold = mean_price + 0.3 * std_price

# Output path
out_path = "./test_price_forecast.png"
os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

# ---- Create the plot ----
fig, ax = plt.subplots(figsize=(14, 6))

# Plot the price line
ax.plot(time_labels, prices, linewidth=2.5, color='#2E86AB', marker='o',
        markersize=4, label='Forecasted Price', zorder=3)

# Add horizontal line for mean price
ax.axhline(y=mean_price, color='gray', linestyle='--', linewidth=1.5,
           label=f'Mean Price (${mean_price:.2f}/MWh)', alpha=0.7, zorder=2)

# Shade low price periods (good for charging) in green
for i, price in enumerate(prices):
    if price <= low_threshold:
        ax.axvspan(time_labels[i] - dt_hours/2, time_labels[i] + dt_hours/2,
                  alpha=0.2, color='green', zorder=1)

# Shade high price periods (good for discharging) in red
for i, price in enumerate(prices):
    if price >= high_threshold:
        ax.axvspan(time_labels[i] - dt_hours/2, time_labels[i] + dt_hours/2,
                  alpha=0.2, color='red', zorder=1)

# Add custom legend entries for shaded regions
from matplotlib.patches import Patch
legend_elements = [
    plt.Line2D([0], [0], color='#2E86AB', linewidth=2.5, marker='o',
               markersize=4, label='Forecasted Price'),
    plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5,
               label=f'Mean Price (${mean_price:.2f}/MWh)'),
    Patch(facecolor='green', alpha=0.2, label='Low Price Zones (Charge)'),
    Patch(facecolor='red', alpha=0.2, label='High Price Zones (Discharge)')
]
ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

# Labels and title
ax.set_xlabel(xlabel, fontsize=12)
ax.set_ylabel("Price ($/MWh)", fontsize=12)
ax.set_title("Price Forecast - Arbitrage Potential Analysis", fontsize=14, fontweight='bold')

# Add grid for better readability
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

# Add text box with arbitrage potential summary
textstr = f'Arbitrage Potential:\n'
textstr += f'Price Spread: ${price_spread:.2f}/MWh\n'
textstr += f'Min: ${min_price:.2f} | Max: ${max_price:.2f}\n'
textstr += f'Volatility (σ): ${std_price:.2f}'

props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

fig.tight_layout()
plt.savefig(out_path, dpi=100, bbox_inches="tight")
plt.close(fig)

# Generate caption
num_low = sum(1 for p in prices if p <= low_threshold)
num_high = sum(1 for p in prices if p >= high_threshold)

caption = (
    f"Price forecast showing arbitrage potential. "
    f"Spread: ${price_spread:.2f}/MWh (${min_price:.2f} - ${max_price:.2f}). "
    f"{num_low} low-price periods (green shading - good for charging), "
    f"{num_high} high-price periods (red shading - good for discharging)."
)

print(f"✓ Plot saved to: {out_path}")
print(f"✓ Caption: {caption}")
print("\nPlot generated successfully!")
