#!/usr/bin/env python3
"""Quick test script to generate animated optimization results plot."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

# Sample data: 24-hour day
T = 24
hours = np.arange(T)

# Sample prices (same as before)
prices = np.array([
    0.08, 0.07, 0.06, 0.06, 0.07, 0.09,  # Hours 0-5: Night (low)
    0.12, 0.15, 0.18, 0.20, 0.19, 0.17,  # Hours 6-11: Morning
    0.16, 0.15, 0.17, 0.19, 0.21, 0.24,  # Hours 12-17: Afternoon
    0.26, 0.23, 0.19, 0.15, 0.11, 0.09   # Hours 18-23: Evening
])

# Sample battery capacity and SoC (simulated charging at night, discharging at peak)
capacity = 20.0  # MWh
soc_fraction = np.array([
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75,  # Charging at night
    0.80, 0.85, 0.90, 0.90, 0.88, 0.85,  # Morning (some discharge)
    0.82, 0.78, 0.75, 0.70, 0.65, 0.58,  # Afternoon discharge
    0.50, 0.42, 0.38, 0.40, 0.45, 0.50   # Evening recharge
])
soc_MWh = soc_fraction * capacity

# Infer actions from SoC changes
dsoc = np.diff(soc_MWh, prepend=soc_MWh[0])
actions = np.sign(dsoc)

# Derive candlestick data
opens = np.concatenate([[prices[0]], prices[:-1]])
closes = prices
highs = np.maximum(opens, closes)
lows = np.minimum(opens, closes)
eps = 1e-6
highs = np.maximum(highs, lows + eps)

# Output path
out_path = "./test_animated_schedule.gif"  # Use GIF for compatibility

# ---- Create Animation ----
fig, ax1 = plt.subplots(figsize=(14, 6))

# Set up axes
ax1.set_xlim(-0.5, T - 0.5)
ax1.set_ylim(prices.min() * 0.95, prices.max() * 1.05)
ax1.set_xlabel("Hour", fontsize=12)
ax1.set_ylabel("Prices ($/MWh)", fontsize=12)
ax1.set_title("Battery Arbitrage Schedule - Animated", fontsize=14, fontweight='bold')
ax1.grid(alpha=0.25)
ax1.set_xticks(hours[::4])

# Secondary axis for SoC
ax2 = ax1.twinx()
ax2.set_ylim(0, capacity * 1.05)
ax2.set_ylabel("State of Charge (MWh)", color='tab:purple', fontsize=12)
ax2.tick_params(axis='y', labelcolor='tab:purple')

# Storage for artists
candle_patches = []
tick_lines = []
tick_markers = []
soc_line, = ax2.plot([], [], '-o', linewidth=2, markersize=5,
                     color='tab:purple', label='SoC')

# Parameters
pr = np.ptp(prices)
tick_length = 0.06 * pr
candle_width = 0.6

def init():
    return []

def animate(frame):
    # Clear previous elements
    for patch in candle_patches:
        patch.remove()
    for line in tick_lines:
        line.remove()
    for marker in tick_markers:
        marker.remove()
    candle_patches.clear()
    tick_lines.clear()
    tick_markers.clear()

    # Draw up to current frame
    for t in range(frame + 1):
        o, c, h, l = opens[t], closes[t], highs[t], lows[t]
        color = 'tab:green' if c >= o else 'tab:red'

        # Wick
        wick = ax1.vlines(t, l, h, linewidth=1, color=color)
        tick_lines.append(wick)

        # Body
        y = min(o, c)
        height = abs(c - o)
        if height < (0.02 * pr + 1e-9):
            height = 0.02 * pr + 1e-9
        rect = Rectangle((t - candle_width/2, y), candle_width, height,
                       facecolor=color, edgecolor=color, alpha=0.7)
        ax1.add_patch(rect)
        candle_patches.append(rect)

        # Decision ticks
        p = closes[t]
        a = actions[t]
        if a > 0:  # charge
            stem = ax1.vlines(t, p, p + tick_length, linewidth=2, color='tab:green')
            marker = ax1.plot([t], [p + tick_length], marker='^', ms=6,
                            color='tab:green')[0]
            tick_lines.append(stem)
            tick_markers.append(marker)
        elif a < 0:  # discharge
            stem = ax1.vlines(t, p - tick_length, p, linewidth=2, color='tab:red')
            marker = ax1.plot([t], [p - tick_length], marker='v', ms=6,
                            color='tab:red')[0]
            tick_lines.append(stem)
            tick_markers.append(marker)

    # Update SoC line
    soc_line.set_data(hours[:frame+1], soc_MWh[:frame+1])

    return candle_patches + tick_lines + tick_markers + [soc_line]

# Create animation (200ms per frame = 0.2 seconds)
anim = animation.FuncAnimation(
    fig, animate, init_func=init, frames=T,
    interval=200, blit=False, repeat=True
)

# Add legend
price_up = Rectangle((0,0),1,1, color='tab:green', alpha=0.7)
price_dn = Rectangle((0,0),1,1, color='tab:red', alpha=0.7)
stem_up = plt.Line2D([0],[0], color='tab:green', marker='^', linestyle='None')
stem_dn = plt.Line2D([0],[0], color='tab:red', marker='v', linestyle='None')
ax2.legend([price_up, price_dn, stem_up, stem_dn, soc_line],
          ['Bull candle', 'Bear candle', 'Charge', 'Discharge', 'SoC'],
          loc='upper left', frameon=True)

# Save
print("Generating animated plot (this may take a moment)...")
anim.save(out_path, writer='pillow', fps=5, dpi=100)
plt.close(fig)

print(f"✓ Animation saved to: {out_path}")
print(f"✓ Duration: {T} frames at 5 fps = {T/5:.1f} seconds")
print("\nAnimation shows price candles, charge/discharge decisions, and SoC evolution!")
