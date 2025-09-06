from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator, Field, ConfigDict
from typing import Literal, List, Dict, Any, Optional, Tuple
from math import isfinite
import os, io, csv
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="WealthCoach API", version="0.7.0")

# --- CORS for web frontend (tighten allow_origins later to your domain) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Models ----------------
class OtherLoan(BaseModel):
    emi: float
    months_left: int

    @field_validator("emi")
    @classmethod
    def _emi_pos(cls, v):
        if v < 0 or not isfinite(v): raise ValueError("EMI must be non-negative.")
        return v

    @field_validator("months_left")
    @classmethod
    def _months_pos(cls, v):
        if v < 0: raise ValueError("months_left must be >= 0.")
        return v

class UserCashflow(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    monthly_income: float
    fixed_expenses: float
    other_emi: float = Field(default=0.0, alias="other_mi")
    other_loans: List[OtherLoan] = []
    sip_investments: float = 0.0
    buffer_pct: float = 0.10
    cash_savings: float = 0.0

    @field_validator("monthly_income","fixed_expenses","other_emi","sip_investments","buffer_pct","cash_savings")
    @classmethod
    def _non_negative(cls, v):
        if v < 0 or not isfinite(v): raise ValueError("Must be non-negative and finite.")
        return v

class PurchasePlan(BaseModel):
    name: str
    price: float
    down_payment: float = 0.0
    interest_rate_annual: float = 0.12  # accepts 0.085 or 8.5
    tenure_months: int = 60
    recurring_costs: float = 0.0

    @field_validator("price","down_payment","interest_rate_annual","tenure_months","recurring_costs")
    @classmethod
    def _non_negative(cls, v):
        if v < 0 or not isfinite(v): raise ValueError("Must be non-negative and finite.")
        return v

class AffordabilityResponse(BaseModel):
    decision: Literal["BUY_NOW","WAIT","UPGRADE"]
    decision_score: float
    monthly_emi: float
    post_emi_surplus: float
    notes: str
    suggestions: List[str] = []

class ScenarioPlanResult(AffordabilityResponse):
    name: str

class AnalyzeRequest(BaseModel):
    cashflow: UserCashflow
    plan: Optional[PurchasePlan] = None
    plans: Optional[List[PurchasePlan]] = None
    min_emergency_months: int = 6

class AnalyzeResponse(BaseModel):
    results: List[ScenarioPlanResult]

# ---- Forecast models ----
class ForecastRequest(BaseModel):
    cashflow: UserCashflow
    plan: PurchasePlan
    months: int = 12
    salary_growth_pct_pa: float = 0.0
    expense_inflation_pct_pa: float = 0.0
    sip_stepup_pct_pa: float = 0.0
    min_emergency_months: int = 6

    @field_validator("months","salary_growth_pct_pa","expense_inflation_pct_pa","sip_stepup_pct_pa","min_emergency_months")
    @classmethod
    def _non_negative2(cls, v):
        if v < 0: raise ValueError("Must be non-negative.")
        return v

class ForecastRow(BaseModel):
    month: int
    income: float
    fixed_expenses: float
    other_emi: float
    sip_investments: float
    buffer: float
    plan_emi: float
    recurring_costs: float
    surplus: float

class ForecastSummary(BaseModel):
    breakeven_month: Optional[int]
    first_safe_month: Optional[int]
    avg_surplus_12m: float
    avg_surplus_all: float
    decision_today: Literal["BUY_NOW","WAIT","UPGRADE"]
    decision_score_today: float

class ForecastResponse(BaseModel):
    forecast: List[ForecastRow]
    summary: ForecastSummary

# ---- Reverse solver models ----
class ReverseConstraints(BaseModel):
    target_decision: Literal["BUY_NOW","NON_NEGATIVE_SURPLUS"] = "BUY_NOW"  # BUY_NOW=>>=12% surplus
    max_tenure_months: int = 84
    min_down_payment: float = 0.0
    price_range: Optional[List[float]] = None     # [min_price, max_price]
    dp_range: Optional[List[float]] = None        # [min_dp, max_dp]
    optimize_for: Literal["price","down_payment"] = "price"  # which knob to optimize

class ReverseMarket(BaseModel):
    interest_rate_annual: float
    tenure_months: int = 60
    recurring_costs: float = 0.0

class ReverseRequest(BaseModel):
    cashflow: UserCashflow
    constraints: ReverseConstraints
    market: ReverseMarket

class ReverseResponse(BaseModel):
    max_price_today: Optional[float] = None
    required_down_payment_today: Optional[float] = None
    target_price_used: Optional[float] = None
    min_down_payment_for_target_price_today: Optional[float] = None
    safe_in_month: Optional[int] = None
    explanation: str

# ---------------- Helpers ----------------
def normalize_rate(rate: float) -> float:
    return rate / 100.0 if rate >= 1.0 else rate

def emi(principal: float, annual_rate: float, months: int) -> float:
    if principal <= 0 or months <= 0: return 0.0
    r = annual_rate / 12.0
    if r == 0: return principal / months
    return principal * (r * (1 + r) ** months) / ((1 + r) ** months - 1)

def other_emi_total_today(cf: UserCashflow) -> float:
    return cf.other_emi + sum(l.emi for l in cf.other_loans if l.months_left > 0)

def other_emi_for_month(cf: UserCashflow, m: int) -> float:
    return cf.other_emi + sum(l.emi for l in cf.other_loans if l.months_left >= m)

def decision_from_surplus_ratio(ratio: float) -> str:
    buy_threshold = 0.12
    wait_threshold = 0.03
    if ratio >= buy_threshold: return "BUY_NOW"
    if ratio >= wait_threshold: return "WAIT"
    return "WAIT"

def score_from_surplus_ratio(ratio: float) -> float:
    lo, hi = -0.10, 0.25
    x = max(min((ratio - lo) / (hi - lo), 1.0), 0.0)
    return round(100 * x, 1)

def emergency_fund_ok(cf: UserCashflow, min_months: int, using_other_emi: float) -> Tuple[bool, float]:
    monthly_outgo = cf.fixed_expenses + using_other_emi + cf.sip_investments
    required = monthly_outgo * min_months
    return (cf.cash_savings >= required, required)

def suggestions_for_wait(cf: UserCashflow, p: PurchasePlan, post_emi_surplus: float, monthly_emi: float):
    tips = []
    if monthly_emi > 0 and p.tenure_months > 0:
        needed = max(0.0, -post_emi_surplus)
        if needed > 0:
            dp_extra = (needed / monthly_emi) * max(p.price - p.down_payment, 0.0)
            if dp_extra > 0:
                tips.append(f"Increase down payment by ~₹{round(dp_extra, 0):,.0f} to avoid negative cashflow.")
    if p.tenure_months < 84:
        tips.append("Consider extending tenure (e.g., to 72–84 months) to reduce EMI temporarily.")
    if p.recurring_costs > 0:
        tips.append("Trim recurring costs by ₹500–₹1000 (insurance/maintenance) to ease early months.")
    if cf.sip_investments > 0:
        tips.append("Temporarily reduce SIPs by 10–20% until other loans close, then restore.")
    return tips[:3]

# ---------------- Core calc ----------------
def evaluate_plan(cf: UserCashflow, p: PurchasePlan, min_ef_months: int) -> Dict[str, Any]:
    if cf.monthly_income <= 0: raise HTTPException(status_code=400, detail="Income must be > 0")

    principal = max(p.price - p.down_payment, 0.0)
    annual_rate = normalize_rate(p.interest_rate_annual)
    monthly_emi = emi(principal, annual_rate, p.tenure_months)

    other_now = other_emi_total_today(cf)
    buffer_amt = cf.monthly_income * cf.buffer_pct
    must_pay = cf.fixed_expenses + other_now + cf.sip_investments + buffer_amt
    available = cf.monthly_income - must_pay

    post_emi_surplus = available - (monthly_emi + p.recurring_costs)
    surplus_ratio = post_emi_surplus / cf.monthly_income

    decision = decision_from_surplus_ratio(surplus_ratio)
    decision_score = score_from_surplus_ratio(surplus_ratio)

    notes = [f"Buffer kept: {cf.buffer_pct*100:.0f}%."]
    if p.recurring_costs > 0: notes.append(f"Includes plan recurring costs ₹{p.recurring_costs:,.0f}.")
    suggestions = []

    ef_ok, ef_required = emergency_fund_ok(cf, min_ef_months, other_now)
    if not ef_ok:
        decision = "WAIT"
        notes.append(f"Emergency fund low: need ~₹{ef_required:,.0f} (for {min_ef_months} months). Build EF first.")
        suggestions.append("Route bonuses/tax refunds to emergency fund until target is met.")

    if surplus_ratio >= 0.25 and ef_ok:
        decision = "UPGRADE"
        notes.append("High cushion available; consider higher variant or shorter tenure.")

    if decision == "WAIT":
        suggestions.extend(suggestions_for_wait(cf, p, post_emi_surplus, monthly_emi))

    return {
        "decision": decision,
        "decision_score": decision_score,
        "monthly_emi": round(monthly_emi, 2),
        "post_emi_surplus": round(post_emi_surplus, 2),
        "notes": " ".join(notes),
        "suggestions": suggestions
    }

# ---------- Forecast engine ----------
def forecast_cashflow(req: ForecastRequest) -> ForecastResponse:
    cf = req.cashflow
    p = req.plan
    principal = max(p.price - p.down_payment, 0.0)
    annual_rate = normalize_rate(p.interest_rate_annual)
    plan_emi = emi(principal, annual_rate, p.tenure_months)

    g_income  = (1 + req.salary_growth_pct_pa / 100.0) ** (1/12) if req.salary_growth_pct_pa else 1.0
    g_expense = (1 + req.expense_inflation_pct_pa / 100.0) ** (1/12) if req.expense_inflation_pct_pa else 1.0
    g_sip     = (1 + req.sip_stepup_pct_pa / 100.0) ** (1/12) if req.sip_stepup_pct_pa else 1.0

    income = cf.monthly_income
    fixed_exp = cf.fixed_expenses
    sip = cf.sip_investments

    rows: List[ForecastRow] = []
    buy_threshold = 0.12

    for m in range(1, max(1, req.months)+1):
        other_m = other_emi_for_month(cf, m)
        buffer_amt = income * cf.buffer_pct
        surplus = income - (fixed_exp + other_m + sip + buffer_amt + plan_emi + p.recurring_costs)

        rows.append(ForecastRow(
            month=m,
            income=round(income,2),
            fixed_expenses=round(fixed_exp,2),
            other_emi=round(other_m,2),
            sip_investments=round(sip,2),
            buffer=round(buffer_amt,2),
            plan_emi=round(plan_emi,2),
            recurring_costs=round(p.recurring_costs,2),
            surplus=round(surplus,2),
        ))

        income *= g_income
        fixed_exp *= g_expense
        sip *= g_sip

    breakeven = next((r.month for r in rows if r.surplus >= 0), None)
    first_safe = next((r.month for r in rows if (r.surplus / r.income) >= buy_threshold), None)

    today_eval = evaluate_plan(cf, p, req.min_emergency_months)
    avg12 = round(sum(r.surplus for r in rows[:12]) / min(12, len(rows)), 2)
    avg_all = round(sum(r.surplus for r in rows) / len(rows), 2)

    return ForecastResponse(
        forecast=rows,
        summary=ForecastSummary(
            breakeven_month=breakeven,
            first_safe_month=first_safe,
            avg_surplus_12m=avg12,
            avg_surplus_all=avg_all,
            decision_today=today_eval["decision"],
            decision_score_today=today_eval["decision_score"],
        ),
    )

# ---------- Reverse utilities ----------
def surplus_ratio_for(cf: UserCashflow, price: float, dp: float, market: ReverseMarket) -> float:
    p = PurchasePlan(
        name="Target",
        price=price,
        down_payment=dp,
        interest_rate_annual=market.interest_rate_annual,
        tenure_months=market.tenure_months,
        recurring_costs=market.recurring_costs,
    )
    res = evaluate_plan(cf, p, min_ef_months=6)
    return res["post_emi_surplus"] / max(cf.monthly_income, 1e-6)

def earliest_safe_month(cf: UserCashflow, p: PurchasePlan) -> Optional[int]:
    fr = ForecastRequest(
        cashflow=cf, plan=p, months=60,
        salary_growth_pct_pa=0.0, expense_inflation_pct_pa=0.0, sip_stepup_pct_pa=0.0,
        min_emergency_months=6,
    )
    result = forecast_cashflow(fr)
    return result.summary.first_safe_month

def reverse_search(cf: UserCashflow, cons: ReverseConstraints, market: ReverseMarket) -> 'ReverseResponse':
    target_ratio = 0.12 if cons.target_decision == "BUY_NOW" else 0.0
    pr_lo, pr_hi = (cons.price_range or [300000.0, 5000000.0])
    dp_lo, dp_hi = (cons.dp_range or [cons.min_down_payment, max(cons.min_down_payment, 2500000.0)])

    def ok(price: float, dp: float) -> bool:
        return surplus_ratio_for(cf, price, dp, market) >= target_ratio

    # A) Max price today with minimum DP
    max_price_today = None
    if ok(pr_lo, cons.min_down_payment):
        lo, hi = pr_lo, pr_hi
        for _ in range(40):
            mid = (lo + hi) / 2
            if ok(mid, cons.min_down_payment):
                max_price_today = mid
                lo = mid
            else:
                hi = mid

    # B) Min DP today at target price (top of price range)
    target_price_used = pr_hi
    min_dp_today = None
    if ok(target_price_used, dp_hi):
        lo, hi = dp_lo, dp_hi
        for _ in range(40):
            mid = (lo + hi) / 2
            if ok(target_price_used, mid):
                min_dp_today = mid
                hi = mid
            else:
                lo = mid

    # C) If neither feasible, compute earliest safe month for pr_lo with min DP
    if max_price_today is None and min_dp_today is None:
        p = PurchasePlan(
            name="Target", price=pr_lo, down_payment=cons.min_down_payment,
            interest_rate_annual=market.interest_rate_annual,
            tenure_months=market.tenure_months, recurring_costs=market.recurring_costs,
        )
        safe_month = earliest_safe_month(cf, p)
        expl = (f"With down payment ₹{cons.min_down_payment:,.0f}, even ₹{pr_lo:,.0f} "
                f"doesn’t meet the target today. Safe in month {safe_month}.")
        return ReverseResponse(
            max_price_today=None,
            required_down_payment_today=None,
            target_price_used=target_price_used,
            min_down_payment_for_target_price_today=None,
            safe_in_month=safe_month,
            explanation=expl
        )

    # Build combined explanation
    parts = []
    if max_price_today is not None:
        parts.append(f"Max price today ≈ ₹{round(max_price_today,0):,} with DP ₹{cons.min_down_payment:,.0f}.")
    if min_dp_today is not None:
        parts.append(f"At target price ₹{target_price_used:,.0f}, need DP ≈ ₹{round(min_dp_today,0):,} today.")
    expl = " ".join(parts) or "Computed feasible options."

    return ReverseResponse(
        max_price_today=round(max_price_today, 0) if max_price_today is not None else None,
        required_down_payment_today=cons.min_down_payment if max_price_today is not None else None,
        target_price_used=target_price_used,
        min_down_payment_for_target_price_today=round(min_dp_today, 0) if min_dp_today is not None else None,
        safe_in_month=None,
        explanation=expl
    )

# ---------------- Routes ----------------
@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/affordability/analyze", response_model=AnalyzeResponse)
def affordability_analyze(req: AnalyzeRequest):
    plans: List[PurchasePlan] = []
    if req.plan is not None: plans = [req.plan]
    elif req.plans: plans = list(req.plans)
    if not plans: raise HTTPException(status_code=422, detail="Provide either 'plan' or 'plans'.")

    results: List[ScenarioPlanResult] = []
    for p in plans:
        r = evaluate_plan(req.cashflow, p, req.min_emergency_months)
        results.append(ScenarioPlanResult(name=p.name, **r))

    decision_rank = {"UPGRADE":3, "BUY_NOW":2, "WAIT":1}
    results.sort(key=lambda x: (decision_rank.get(x.decision,0), x.decision_score), reverse=True)
    return AnalyzeResponse(results=results)

@app.post("/affordability/preview", response_model=AffordabilityResponse)
def affordability_preview(req: AnalyzeRequest):
    if not req.plan: raise HTTPException(status_code=422, detail="This endpoint needs a single 'plan'.")
    return AffordabilityResponse(**evaluate_plan(req.cashflow, req.plan, req.min_emergency_months))

@app.post("/affordability/scenario", response_model=AnalyzeResponse)
def affordability_scenario(req: AnalyzeRequest):
    if not req.plans: raise HTTPException(status_code=422, detail="This endpoint needs 'plans' array.")
    return affordability_analyze(req)

@app.post("/affordability/forecast", response_model=ForecastResponse)
def affordability_forecast(req: ForecastRequest):
    return forecast_cashflow(req)

# ---- CSV export (same body as /forecast) ----
@app.post("/affordability/forecast.csv")
def forecast_csv(req: ForecastRequest):
    result = forecast_cashflow(req)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["month","income","fixed_expenses","other_emi","sip_investments","buffer","plan_emi","recurring_costs","surplus"])
    for r in result.forecast:
        writer.writerow([r.month, r.income, r.fixed_expenses, r.other_emi, r.sip_investments, r.buffer, r.plan_emi, r.recurring_costs, r.surplus])
    csv_bytes = buf.getvalue().encode("utf-8-sig")
    return Response(content=csv_bytes, media_type="text/csv",
                    headers={"Content-Disposition":"attachment; filename=forecast.csv"})

# ---- Reverse solver ----
@app.post("/planner/reverse", response_model=ReverseResponse)
def planner_reverse(req: ReverseRequest):
    return reverse_search(req.cashflow, req.constraints, req.market)

# --------------- main ----------------
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("APP_HOST","127.0.0.1")
    port = int(os.getenv("APP_PORT","8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)
