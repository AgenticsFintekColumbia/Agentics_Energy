# agentic_energy/visualization/visualization_mcp_server.py

import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP

from agentic_energy.schemas import SolveResponse, DayInputs, BatteryParams, SolveRequest

mcp = FastMCP("VISUALIZATION")

# ---------- Schemas ----------

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


# ---------- Tool ----------

@mcp.tool()
def plot_price_soc(plotrequest: PlotRequest) -> PlotResponse:
    """
    Generate a matplotlib plot of price vs state-of-charge over time.

    - Left y-axis: prices
    - Right y-axis: SoC (MWh)
    """
    req = plotrequest.solve_request
    res = plotrequest.solve_response

    prices = req.day.prices_buy
    soc = res.soc        # length T+1
    T = len(prices)
    hours = list(range(T))

    # SoC in MWh (capacity * soc fraction)
    capacity = req.battery.capacity_MWh
    soc_MWh = [s * capacity for s in soc[:-1]]  # drop last SoC to align with hours

    # Prepare output directory
    out_dir = os.path.dirname(plotrequest.out_path) if plotrequest.out_path else "plots"
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    out_path = plotrequest.out_path or os.path.join(out_dir, "battery_schedule.png")

    # ---- Plot ----
    fig, ax1 = plt.subplots(figsize=(14, 4))

    # Prices
    ax1.plot(hours, prices, linestyle="--")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Prices ($/MWh)")
    ax1.set_title(plotrequest.title)

    # SoC on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(hours, soc_MWh, marker="o")
    ax2.set_ylabel("State of Charge (MWh)")

    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    caption = (
        f"Daily arbitrage schedule: prices on the left axis and battery SoC (MWh) "
        f"on the right axis. Capacity = {capacity:.1f} MWh."
    )

    return PlotResponse(image_path=out_path, caption=caption)


if __name__ == "__main__":
    mcp.run(transport="stdio")
