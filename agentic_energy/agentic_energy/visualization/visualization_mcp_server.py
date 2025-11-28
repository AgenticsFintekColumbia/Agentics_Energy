# agentic_energy/visualization/visualization_mcp_server.py

import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP

from agentic_energy.schemas import SolveResponse, DayInputs, BatteryParams, SolveRequest

mcp = FastMCP("VISUALIZATION")

# ---------- Schemas ----------

class PriceForecastPlotRequest(BaseModel):
    """Inputs needed to visualize price forecast and arbitrage potential."""
    prices: List[float] = Field(..., description="Forecasted prices for the day")
    dt_hours: float = Field(1.0, description="Timestep size in hours")
    title: str = "Price Forecast - Arbitrage Potential"
    out_path: Optional[str] = Field(
        default=None,
        description="Where to save the PNG file. Default: ./plots/price_forecast.png",
    )

class PlotRequest(BaseModel):
    """Inputs needed to draw price vs SoC plot."""
    solve_request: SolveRequest = Field(..., description="Original solve request")
    solve_response: SolveResponse = Field(..., description="Solver output")
    title: str = "Prices vs State of Charge (SoC) Over Time"
    out_path: Optional[str] = Field(
        default=None,
        description="Where to save the PNG file. Default: ./plots/battery_schedule.png",
    )

class PlotResponse(BaseModel):
    image_path: str = Field(..., description="Path to the saved PNG file")
    caption: str = Field(..., description="Short description of what the plot shows")


# ---------- Tools ----------

@mcp.tool()
def plot_price_forecast(plotrequest: PriceForecastPlotRequest) -> PlotResponse:
    """
    Generate a visualization of forecasted prices highlighting arbitrage opportunities.

    Shows:
    - Price levels over time
    - Low price periods (good for charging) in green shading
    - High price periods (good for discharging) in red shading
    - Mean price line for reference
    - Price spread and volatility indicators
    """
    prices = plotrequest.prices
    dt_hours = plotrequest.dt_hours
    T = len(prices)

    # Create time axis based on dt_hours
    if dt_hours == 1.0:
        time_labels = list(range(T))
        xlabel = "Hour of Day"
    elif dt_hours == 0.5:
        time_labels = [i * 0.5 for i in range(T)]
        xlabel = "Hour of Day"
    elif dt_hours == 0.25:
        time_labels = [i * 0.25 for i in range(T)]
        xlabel = "Hour of Day"
    else:
        time_labels = list(range(T))
        xlabel = "Time Step"

    # Calculate price statistics
    prices_array = np.array(prices)
    mean_price = np.mean(prices_array)
    std_price = np.std(prices_array)
    min_price = np.min(prices_array)
    max_price = np.max(prices_array)
    price_spread = max_price - min_price

    # Define thresholds for low and high prices
    # Low: below mean - 0.3*std, High: above mean + 0.3*std
    low_threshold = mean_price - 0.3 * std_price
    high_threshold = mean_price + 0.3 * std_price

    # Prepare output directory
    out_dir = os.path.dirname(plotrequest.out_path) if plotrequest.out_path else "plots"
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    out_path = plotrequest.out_path or os.path.join(out_dir, "price_forecast.png")

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
    ax.set_title(plotrequest.title, fontsize=14, fontweight='bold')

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

    return PlotResponse(image_path=out_path, caption=caption)


@mcp.tool()
def plot_price_soc(plotrequest: PlotRequest) -> PlotResponse:
    """
    Generate an ANIMATED matplotlib plot of price candles vs state-of-charge over time.

    Features:
    - Candlestick-style price visualization
    - Decision indicators (charge/discharge ticks)
    - SoC trajectory on secondary axis
    - Left-to-right animation showing time progression
    - Saves as MP4 video file
    """
    import matplotlib.animation as animation
    from matplotlib.patches import Rectangle

    req = plotrequest.solve_request
    res = plotrequest.solve_response

    prices = np.array(req.day.prices_buy)
    soc = np.array(res.soc)  # length T+1
    decisions = np.array(res.decision) if res.decision else None
    T = len(prices)
    hours = np.arange(T)

    capacity = req.battery.capacity_MWh
    soc_MWh = soc[:-1] * capacity  # drop last SoC to align with hours

    # Derive open/close/high/low for candlesticks
    opens = np.concatenate([[prices[0]], prices[:-1]])
    closes = prices
    highs = np.maximum(opens, closes)
    lows = np.minimum(opens, closes)
    eps = 1e-6
    highs = np.maximum(highs, lows + eps)

    # Infer actions from decisions if available, else from SoC changes
    if decisions is not None:
        actions = decisions
    else:
        dsoc = np.diff(soc_MWh, prepend=soc_MWh[0])
        actions = np.sign(dsoc)

    # Prepare output directory
    out_dir = os.path.dirname(plotrequest.out_path) if plotrequest.out_path else "plots"
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Change extension to .mp4 for animation
    base_path = plotrequest.out_path or os.path.join(out_dir, "battery_schedule.png")
    if base_path.endswith('.png'):
        out_path = base_path.replace('.png', '.mp4')
    else:
        out_path = base_path + '.mp4'

    # ---- Create Animation ----
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Set up axes limits
    ax1.set_xlim(-0.5, T - 0.5)
    ax1.set_ylim(prices.min() * 0.95, prices.max() * 1.05)
    ax1.set_xlabel("Hour", fontsize=12)
    ax1.set_ylabel("Prices ($/MWh)", fontsize=12)
    ax1.set_title(plotrequest.title, fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.25)
    ax1.set_xticks(hours[::max(1, T//12)])  # show every ~12th tick for readability

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

    # Parameters for decision ticks
    pr = np.ptp(prices) if np.ptp(prices) > 0 else 1.0
    tick_length = 0.06 * pr
    candle_width = 0.6

    def init():
        """Initialize animation"""
        return []

    def animate(frame):
        """Draw up to timestep 'frame'"""
        # Clear previous frame's dynamic elements
        for patch in candle_patches:
            patch.remove()
        for line in tick_lines:
            line.remove()
        for marker in tick_markers:
            marker.remove()
        candle_patches.clear()
        tick_lines.clear()
        tick_markers.clear()

        # Draw candles up to current frame
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
            if a > 0:  # charge → green stem up
                stem = ax1.vlines(t, p, p + tick_length, linewidth=2, color='tab:green')
                marker = ax1.plot([t], [p + tick_length], marker='^', ms=6,
                                color='tab:green')[0]
                tick_lines.append(stem)
                tick_markers.append(marker)
            elif a < 0:  # discharge → red stem down
                stem = ax1.vlines(t, p - tick_length, p, linewidth=2, color='tab:red')
                marker = ax1.plot([t], [p - tick_length], marker='v', ms=6,
                                color='tab:red')[0]
                tick_lines.append(stem)
                tick_markers.append(marker)

        # Update SoC line
        soc_line.set_data(hours[:frame+1], soc_MWh[:frame+1])

        return candle_patches + tick_lines + tick_markers + [soc_line]

    # Create animation
    # Frame interval: 200ms = 0.2 seconds per timestep
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=T,
        interval=200, blit=False, repeat=True
    )

    # Add legend
    price_up = Rectangle((0,0),1,1, color='tab:green', alpha=0.7, label='Bull candle')
    price_dn = Rectangle((0,0),1,1, color='tab:red', alpha=0.7, label='Bear candle')
    stem_up = plt.Line2D([0],[0], color='tab:green', marker='^',
                        linestyle='None', label='Charge')
    stem_dn = plt.Line2D([0],[0], color='tab:red', marker='v',
                        linestyle='None', label='Discharge')
    ax2.legend([price_up, price_dn, stem_up, stem_dn, soc_line],
              ['Bull candle', 'Bear candle', 'Charge', 'Discharge', 'SoC'],
              loc='upper left', frameon=True)

    # Save animation - try MP4 first, fall back to GIF if FFMpeg not available
    try:
        writer = animation.FFMpegWriter(fps=5, bitrate=1800)
        anim.save(out_path, writer=writer, dpi=100)
    except (RuntimeError, FileNotFoundError):
        # FFMpeg not available, save as GIF instead
        out_path = out_path.replace('.mp4', '.gif')
        anim.save(out_path, writer='pillow', fps=5, dpi=100)
    plt.close(fig)

    caption = (
        f"Animated arbitrage schedule showing price candles, battery decisions, "
        f"and SoC evolution over {T} hours. Capacity = {capacity:.1f} MWh. "
        f"Green candles/ticks = charging, Red candles/ticks = discharging."
    )

    return PlotResponse(image_path=out_path, caption=caption)


@mcp.tool()
def plot_arbitrage_explanation(plotrequest: PlotRequest) -> PlotResponse:
    """
    Generate a comprehensive 3-panel explanation plot for battery arbitrage strategy.

    Panel 1: Price landscape with decision zones (high/low price periods)
    Panel 2: Battery operations (charge/discharge power in MW)
    Panel 3: State of Charge evolution with min/max constraints

    This plot is designed to support the reasoning agent in explaining the optimization
    results and arbitrage strategy to users.
    """
    req = plotrequest.solve_request
    res = plotrequest.solve_response

    # Extract data
    prices = np.array(req.day.prices_buy)
    charge_MW = np.array(res.charge_MW) if res.charge_MW else np.zeros(len(prices))
    discharge_MW = np.array(res.discharge_MW) if res.discharge_MW else np.zeros(len(prices))
    soc = np.array(res.soc)  # length T+1

    T = len(prices)
    hours = np.arange(T)

    # Battery parameters
    capacity = req.battery.capacity_MWh
    soc_min = req.battery.soc_min * capacity
    soc_max = req.battery.soc_max * capacity
    soc_MWh = soc[:-1] * capacity

    # Calculate price statistics for zones
    mean_price = np.mean(prices)
    std_price = np.std(prices)
    low_threshold = mean_price - 0.25 * std_price
    high_threshold = mean_price + 0.25 * std_price

    # Prepare output directory
    out_dir = os.path.dirname(plotrequest.out_path) if plotrequest.out_path else "plots"
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    base_path = plotrequest.out_path or os.path.join(out_dir, "arbitrage_explanation.png")
    out_path = base_path.replace('.png', '_explanation.png') if 'explanation' not in base_path else base_path

    # ---- Create 3-panel figure ----
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(plotrequest.title or "Battery Arbitrage Strategy Explanation",
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
    # Show charge as negative (going into battery) and discharge as positive (coming out)
    charge_visual = -charge_MW  # negative for visualization
    discharge_visual = discharge_MW  # positive

    # Create stacked bars
    ax2.bar(hours, discharge_visual, width=0.7, color='#E63946',
            label='Discharge (MW)', alpha=0.8, edgecolor='darkred', linewidth=0.5)
    ax2.bar(hours, charge_visual, width=0.7, color='#06A77D',
            label='Charge (MW)', alpha=0.8, edgecolor='darkgreen', linewidth=0.5)

    # Add zero line
    ax2.axhline(y=0, color='black', linewidth=1.2, linestyle='-', alpha=0.7)

    # Add capacity limits as reference lines
    ax2.axhline(y=req.battery.dmax_MW, color='red', linewidth=1,
                linestyle='--', alpha=0.5, label=f'Max Discharge ({req.battery.dmax_MW} MW)')
    ax2.axhline(y=-req.battery.cmax_MW, color='green', linewidth=1,
                linestyle='--', alpha=0.5, label=f'Max Charge ({req.battery.cmax_MW} MW)')

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
    ax3.set_xticks(hours[::max(1, T//24)])

    # Add summary text box with key metrics
    total_charge_energy = np.sum(charge_MW) * req.day.dt_hours
    total_discharge_energy = np.sum(discharge_MW) * req.day.dt_hours
    round_trip_efficiency = (total_discharge_energy / total_charge_energy * 100) if total_charge_energy > 0 else 0

    summary_text = f"Summary:\n"
    summary_text += f"• Total Charged: {total_charge_energy:.2f} MWh\n"
    summary_text += f"• Total Discharged: {total_discharge_energy:.2f} MWh\n"
    summary_text += f"• Efficiency: {round_trip_efficiency:.1f}%\n"
    summary_text += f"• Objective Cost: ${res.objective_cost:.2f}"

    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gray', linewidth=1.5)
    fig.text(0.02, 0.02, summary_text, fontsize=10, verticalalignment='bottom',
             bbox=props, family='monospace')

    plt.tight_layout(rect=[0, 0.08, 1, 0.99])
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    caption = (
        f"Comprehensive arbitrage strategy explanation showing: (1) Price signals with "
        f"charge/discharge zones, (2) Battery power operations over time, and (3) SoC "
        f"evolution within operating constraints. Total charged: {total_charge_energy:.2f} MWh, "
        f"Total discharged: {total_discharge_energy:.2f} MWh, Objective cost: ${res.objective_cost:.2f}."
    )

    return PlotResponse(image_path=out_path, caption=caption)


if __name__ == "__main__":
    mcp.run(transport="stdio")
