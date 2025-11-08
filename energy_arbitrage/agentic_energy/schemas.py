from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class MetricStats(BaseModel):
    count: Optional[int] = Field(None, description="Number of valid data points")
    min: Optional[float] = Field(None, description="Minimum value")
    max: Optional[float] = Field(None, description="Maximum value")
    avg: Optional[float] = Field(None, description="Average value")
    median: Optional[float] = Field(None, description="Median value")
    p25: Optional[float] = Field(None, description="25th percentile")
    p75: Optional[float] = Field(None, description="75th percentile")
    std: Optional[float] = Field(None, description="Standard deviation")
    var: Optional[float] = Field(None, description="Variance")

class DateRange(BaseModel):
    start: Optional[str]
    end: Optional[str]

class SummaryStats(BaseModel):
    region: str
    total_records: int
    date_range: DateRange
    prices: Optional[MetricStats]
    consumption: Optional[MetricStats]

class EnergyDataRecord(BaseModel):
    """Base energy data record with common fields across all regions"""
    timestamps: str = Field(description="Timestamp in ISO format")
    prices: Optional[float] = Field(None, description="Energy price at timestamp")
    consumption: Optional[float] = Field(None, description="Energy consumption")
    year: Optional[int] = Field(None, description="Year extracted from timestamp")
    region: Optional[str] = Field(None, description="Energy market region")
    decisions: Optional[float] = Field(None, description = "Decision taken at each time step by the battery - charge (+1), discharge (-1), idle (0)" )

class BatteryParams(BaseModel):
    capacity_kwh: float = Field(100.0, gt=0, description="Battery capacity in kWh")      # C
    soc_init: float = Field(0.5, ge=0, le=1, description="Initial State of Charge (SoC) as fraction of capacity")
    soc_min: float = Field(0.0, ge=0, le=1, description="Minimum State of Charge (SoC) as fraction of capacity")
    soc_max: float = Field(1.0, ge=0, le=1, description="Maximum State of Charge (SoC) as fraction of capacity")
    cmax_kw: float = Field(50, gt=0, description="Maximum charge power rate in kW")
    dmax_kw: float = Field(50, gt=0, description="Maximum discharge power rate in kW")
    eta_c: float = Field(0.95, ge=0, le=1, description="Charge efficiency")
    eta_d: float = Field(0.95, ge=0, le=1, description="Discharge efficiency")
    soc_target: Optional[float] = None          # default: = soc_init

class DayInputs(BaseModel):
    prices_buy: List[float]                      # $/kWh
    demand_kw: List[float]                       # kW
    prices_sell: Optional[List[float]] = None    # if None and export allowed, equals buy
    allow_export: bool = False
    dt_hours: float = 1.0
    prices_buy_forecast: Optional[List[float]] = None
    demand_kw_forecast:  Optional[List[float]] = None
    prices_sell_forecast: Optional[List[float]] = None


class SolveRequest(BaseModel):
    battery: BatteryParams
    day: DayInputs
    solver: Optional[str] = None                 # "MILP","HEURISTIC","RL","RL_TRAIN"
    solver_opts: Optional[Dict] = None

class SolveFromRecordsRequest(BaseModel):
    battery: BatteryParams
    records: List[EnergyDataRecord]
    dt_hours: float = 1.0
    allow_export: bool = False
    solver: Optional[str] = None
    solver_opts: Optional[Dict] = None

class SolveResponse(BaseModel):
    status: str 
    message: Optional[str] = None
    objective_cost: float = Field(..., description="total objective cost i.e. sum of (price_sell times grid_export subtracted from price_buy times grid_import) multiplied by the sample time of operation dt_hours across all timestamps")
    charge_kw: Optional[List[float]] =Field(None, description="Battery charge schedule in kW")
    discharge_kw: Optional[List[float]] = Field(None, description="Battery discharge schedule in kW")
    import_kw: Optional[List[float]] = Field(None, description="Grid import schedule in kW")
    export_kw: Optional[List[float]] = Field(None, description="Grid export schedule in kW")
    soc: Optional[List[float]] = Field(None, description="State of Charge (SoC) over time")
    decision: Optional[List[float]] = Field(None, description="Decision taken at each time step by the battery - charge (+1), discharge (-1), idle (0)")
    confidence: Optional[List[float]] = Field(None, description="Confidence level of each decision (0 to 1)")

# New schemas for forecasting
class ForecastRecord(BaseModel):
    """Single forecast record comparing actual vs predicted"""
    timestamp: str = Field(description="Timestamp of the forecast")
    actual: float = Field(description="Actual observed value")
    predicted: float = Field(description="Predicted value from model")
    error: float = Field(description="Prediction error (predicted - actual)")

class ForecastMetrics(BaseModel):
    """Forecast quality metrics"""
    mse: float = Field(description="Mean Squared Error")
    rmse: float = Field(description="Root Mean Squared Error")
    mae: float = Field(description="Mean Absolute Error")
    num_predictions: int = Field(description="Number of predictions made")

class ForecastResult(BaseModel):
    """Complete forecast result for a target variable"""
    region: str = Field(description="Energy market region")
    target: str = Field(description="Target variable (prices or consumption)")
    start_date: str = Field(description="Forecast start date")
    end_date: str = Field(description="Forecast end date")
    lookback: int = Field(description="Number of historical points used")
    horizon: int = Field(description="Forecast horizon length")
    metrics: ForecastMetrics = Field(description="Forecast quality metrics")
    forecasts: List[ForecastRecord] = Field(description="Individual forecast records")



# from pydantic import BaseModel, Field
# from typing import List, Optional, Dict

# class MetricStats(BaseModel):
#     count: Optional[int] = Field(None, description="Number of valid data points")
#     min: Optional[float] = Field(None, description="Minimum value")
#     max: Optional[float] = Field(None, description="Maximum value")
#     avg: Optional[float] = Field(None, description="Average value")
#     median: Optional[float] = Field(None, description="Median value")
#     p25: Optional[float] = Field(None, description="25th percentile")
#     p75: Optional[float] = Field(None, description="75th percentile")
#     std: Optional[float] = Field(None, description="Standard deviation")
#     var: Optional[float] = Field(None, description="Variance")

# class DateRange(BaseModel):
#     start: Optional[str]
#     end: Optional[str]

# class SummaryStats(BaseModel):
#     region: str
#     total_records: int
#     date_range: DateRange
#     prices: Optional[MetricStats]
#     consumption: Optional[MetricStats]

# Reasoning schemas for decision explanation
class ReasoningRequest(BaseModel):
    """Input schema for the reasoning system"""
    solve_request: SolveRequest = Field(..., description="Original solve request with battery parameters and inputs")
    solve_response: SolveResponse = Field(..., description="Solver response containing decisions and results")
    timestamp_index: int = Field(..., description="Index of the decision to explain")
    context_window: int = Field(default=6, description="Number of timesteps before/after to consider for context")

class ReasoningResponse(BaseModel):
    """Output schema containing the reasoning explanation"""
    explanation: str = Field(..., description="Detailed natural language explanation of the decision")
    key_factors: List[str] = Field(default_factory=list, description="Key factors that influenced the decision")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in the explanation (0-1)")
    supporting_data: Dict[str, float] = Field(
        default_factory=dict,
        description="Relevant numerical data supporting the explanation (e.g., price_delta, soc_margin)"
    )