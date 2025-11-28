#!/usr/bin/env python3
"""Test script to generate the arbitrage explanation plot."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Sample data: 24-hour day with realistic arbitrage scenario
T = 24
hours = np.arange(T)

# Prices showing clear arbitrage opportunity
prices = np.array([
    0.08, 0.07, 0.06, 0.06, 0.07, 0.09,  # Hours 0-5: Night (low)
    0.12, 0.15, 0.18, 0.20, 0.19, 0.17,  # Hours 6-11: Morning ramp-up
    0.16, 0.15, 0.17, 0.19, 0.21, 0.24,  # Hours 12-17: Afternoon
    0.26, 0.23, 0.19, 0.15, 0.11, 0.09   # Hours 18-23: Evening decline
])

# Battery parameters
capacity = 20.0  # MWh
soc_min_frac = 0.1
soc_max_frac = 0.9
soc_min = soc_min_frac * capacity
soc_max = soc_max_frac * capacity
cmax_MW = 6.0
dmax_MW = 6.0
dt_hours = 1.0

# Simulated charge/discharge schedule (optimized for arbitrage)
charge_MW = np.array([
    6.0, 6.0, 6.0, 6.0, 6.0, 5.0,  # Charge during low prices
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Hold during transition
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Hold
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0   # No charging
])

discharge_MW = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # No discharge
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Hold
    0.0, 0.0, 0.0, 4.0, 5.0, 6.0,  # Discharge during high prices
    6.0, 6.0, 4.0, 0.0, 0.0, 0.0   # Continue then stop
])

# Calculate SoC trajectory
soc_MWh = np.zeros(T)
soc_MWh[0] = 0.5 * capacity  # Start at 50%

for t in range(1, T):
    # Energy change = charge - discharge (simplified, ignoring efficiency)
    energy_change = (charge_MW[t-1] - discharge_MW[t-1]) * dt_hours * 0.95  # ~95% efficiency
    soc_MWh[t] = np.clip(soc_MWh[t-1] + energy_change, soc_min, soc_max)

# Calculate price statistics for zones
mean_price = np.mean(prices)
std_price = np.std(prices)
low_threshold = mean_price - 0.25 * std_price
high_threshold = mean_price + 0.25 * std_price

# Calculate metrics
total_charge_energy = np.sum(charge_MW) * dt_hours
total_discharge_energy = np.sum(discharge_MW) * dt_hours
round_trip_efficiency = (total_discharge_energy / total_charge_energy * 100) if total_charge_energy > 0 else 0

# Estimate objective cost (revenue from discharge - cost of charge)
revenue = np.sum(discharge_MW * prices) * dt_hours
cost = np.sum(charge_MW * prices) * dt_hours
objective_cost = cost - revenue  # Minimizing cost (negative = profit)

out_path = "./test_explanation.png"

# ---- Create 3-panel figure ----
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
fig.suptitle("Battery Arbitrage Strategy Explanation",
             fontsize=16, fontweight='bold', y=0.995)

# ===== PANEL 1: Price Landscape with Decision Zones =====
ax1.plot(hours, prices, linewidth=2.5, color='#2E86AB', marker='o',
         markersize=5, label='Electricity Price', zorder=3)

# Add mean price line
ax1.axhline(y=mean_price, color='gray', linestyle='--', linewidth=1.5,
            label=f'Mean Price (${mean_price:.2f}/MWh)', alpha=0.7)

# Shade decision zones
for i, price in enumerate(prices):
    if price <= low_threshold:
        ax1.axvspan(hours[i] - 0.4, hours[i] + 0.4,
                   alpha=0.25, color='green', zorder=1)
    elif price >= high_threshold:
        ax1.axvspan(hours[i] - 0.4, hours[i] + 0.4,
                   alpha=0.25, color='red', zorder=1)

ax1.set_ylabel("Price ($/MWh)", fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
ax1.legend(loc='upper right', framealpha=0.9, fontsize=10)
ax1.set_title("Price Signals: Green = Low (Charge), Red = High (Discharge)",
              fontsize=11, style='italic', pad=10)

# ===== PANEL 2: Battery Operations (Power Flows) =====
charge_visual = -charge_MW  # negative for visualization
discharge_visual = discharge_MW  # positive

# Create bars
ax2.bar(hours, discharge_visual, width=0.7, color='#E63946',
        label='Discharge (MW)', alpha=0.8, edgecolor='darkred', linewidth=0.5)
ax2.bar(hours, charge_visual, width=0.7, color='#06A77D',
        label='Charge (MW)', alpha=0.8, edgecolor='darkgreen', linewidth=0.5)

# Add zero line
ax2.axhline(y=0, color='black', linewidth=1.2, linestyle='-', alpha=0.7)

# Add capacity limits
ax2.axhline(y=dmax_MW, color='red', linewidth=1,
            linestyle='--', alpha=0.5, label=f'Max Discharge ({dmax_MW} MW)')
ax2.axhline(y=-cmax_MW, color='green', linewidth=1,
            linestyle='--', alpha=0.5, label=f'Max Charge ({cmax_MW} MW)')

ax2.set_ylabel("Power (MW)", fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
ax2.legend(loc='upper right', framealpha=0.9, fontsize=10, ncol=2)
ax2.set_title("Battery Operations: Positive = Discharging, Negative = Charging",
              fontsize=11, style='italic', pad=10)

# ===== PANEL 3: State of Charge with Constraints =====
# Plot SoC bounds as shaded region
ax3.fill_between(hours, soc_min, soc_max, alpha=0.15, color='gray',
                 label=f'Operating Range ({soc_min:.1f}-{soc_max:.1f} MWh)')

# Plot actual SoC trajectory
ax3.plot(hours, soc_MWh, linewidth=3, color='#9B59B6', marker='o',
         markersize=6, label='Actual SoC', zorder=3)

# Add constraint lines
ax3.axhline(y=soc_min, color='orange', linewidth=1.5, linestyle='--',
            alpha=0.7, label=f'Min SoC ({soc_min:.1f} MWh)')
ax3.axhline(y=soc_max, color='darkblue', linewidth=1.5, linestyle='--',
            alpha=0.7, label=f'Max SoC ({soc_max:.1f} MWh)')

# Add initial and final SoC markers
ax3.scatter([0], [soc_MWh[0]], s=150, color='green', marker='o',
            edgecolors='darkgreen', linewidths=2, zorder=4,
            label=f'Initial SoC ({soc_MWh[0]:.1f} MWh)')
ax3.scatter([T-1], [soc_MWh[-1]], s=150, color='red', marker='s',
            edgecolors='darkred', linewidths=2, zorder=4,
            label=f'Final SoC ({soc_MWh[-1]:.1f} MWh)')

ax3.set_xlabel("Hour of Day", fontsize=12, fontweight='bold')
ax3.set_ylabel("State of Charge (MWh)", fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
ax3.legend(loc='upper right', framealpha=0.9, fontsize=10, ncol=2)
ax3.set_title("Battery Energy Level: Must Stay Within Operating Bounds",
              fontsize=11, style='italic', pad=10)
ax3.set_xlim(-0.5, T - 0.5)
ax3.set_xticks(hours[::4])

# Add summary text box
summary_text = f"Summary:\n"
summary_text += f"• Total Charged: {total_charge_energy:.2f} MWh\n"
summary_text += f"• Total Discharged: {total_discharge_energy:.2f} MWh\n"
summary_text += f"• Efficiency: {round_trip_efficiency:.1f}%\n"
summary_text += f"• Objective Cost: ${objective_cost:.2f}"

props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gray', linewidth=1.5)
fig.text(0.02, 0.02, summary_text, fontsize=10, verticalalignment='bottom',
         bbox=props, family='monospace')

plt.tight_layout(rect=[0, 0.08, 1, 0.99])
plt.savefig(out_path, dpi=120, bbox_inches="tight")
plt.close(fig)

print(f"✓ Explanation plot saved to: {out_path}")
print(f"\nPlot features:")
print("  • Panel 1: Price landscape with green (charge) and red (discharge) zones")
print("  • Panel 2: Battery power operations showing charge/discharge schedule")
print("  • Panel 3: SoC evolution with operating constraints and markers")
print(f"\nArbitrage Summary:")
print(f"  • Charged {total_charge_energy:.1f} MWh at avg ${cost/total_charge_energy:.2f}/MWh")
print(f"  • Discharged {total_discharge_energy:.1f} MWh at avg ${revenue/total_discharge_energy:.2f}/MWh")
print(f"  • Net profit: ${-objective_cost:.2f}")
