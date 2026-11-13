"""Pydantic models for the mortgage underwriting workflow."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Decision = Literal["APPROVED", "CONDITIONAL", "REJECTED", "HUMAN_REVIEW"]
HumanDecision = Literal["APPROVED", "REJECTED"]
WorkflowStage = Literal[
    "STARTED",
    "EXTRACTING_APPLICATION",
    "RETRIEVING_POLICY",
    "RUNNING_ANALYSES",
    "CRITIC_REVIEW",
    "DRAFTING_DECISION",
    "AWAITING_HUMAN_REVIEW",
    "COMPLETED",
]


class CollectionItem(BaseModel):
    """A collection item on a credit report."""

    type: str
    amount: float


class CreditHistory(BaseModel):
    """Applicant credit history summary."""

    bankruptcies: int
    foreclosures: int
    late_payments_12mo: int
    late_payments_24mo: int
    collections: list[CollectionItem]
    inquiries_6mo: int
    oldest_tradeline_years: float
    total_tradelines: int
    credit_notes: str


class EmploymentHistoryEntry(BaseModel):
    """Single employment history entry."""

    employer: str
    position: str
    years: float
    income: float


class IncomeDetails(BaseModel):
    """Detailed income components."""

    base_salary: float
    bonus_2023: float
    bonus_2024: float
    bonus_stable: bool
    employer_confirmation: str


class Employment(BaseModel):
    """Current employment details."""

    employer: str
    position: str
    years: float
    monthly_income: float
    type: str
    employment_gap: str
    gap_explanation: str
    employment_history: list[EmploymentHistoryEntry]
    income_details: IncomeDetails


class Debts(BaseModel):
    """Monthly debt obligations."""

    car_loan: float = 0
    student_loan: float = 0
    credit_cards: float = 0
    personal_loan: float = 0
    total_monthly_debt: float


class RecentDeposit(BaseModel):
    """Recent deposit in asset accounts."""

    date: str
    amount: float
    description: str


class Assets(BaseModel):
    """Applicant assets summary."""

    model_config = ConfigDict(populate_by_name=True)

    checking: float
    savings: float
    liquid_assets_total: float
    retirement_401k: float = Field(alias="401k")
    recent_deposits: list[RecentDeposit]
    deposit_explanations: str
    reserves_months: int


class Loan(BaseModel):
    """Loan request details."""

    amount: float
    down_payment: float
    closing_costs: float
    estimated_payment: float
    property_type: str
    use: str
    monthly_piti: float


class Property(BaseModel):
    """Property collateral details."""

    purchase_price: float
    appraised_value: float
    condition: str
    type: str
    required_repairs: float
    repair_details: str


class MortgageApplication(BaseModel):
    """Full mortgage application data."""

    case_id: str
    name: Optional[str] = None
    ssn: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    credit_score: int
    credit_history: CreditHistory
    employment: Employment
    debts: Debts
    assets: Assets
    loan: Loan
    property: Property
    dti_ratio: float
    expected_decision: Optional[str] = None


class UnderwritingInput(BaseModel):
    """Input to the underwriting workflow."""

    case_id: str
    image_dir: str


class WorkflowProgress(BaseModel):
    """Current stage of the underwriting workflow, exposed via query."""

    stage: WorkflowStage
    detail: str = ""
    updated_at: str


class UnderwritingMetrics(BaseModel):
    """Computed underwriting metrics."""

    dti_ratio: float
    ltv_ratio: float
    monthly_debt: float
    monthly_income: float


class UnderwritingAnalyses(BaseModel):
    """Outputs from specialist agents."""

    credit: str
    income: str
    assets: str
    collateral: str


class DecisionRecommendation(BaseModel):
    """LLM-generated decision recommendation."""

    decision: Decision
    risk_score: int
    memo: str
    conditions: list[str] = Field(default_factory=list)
    human_review_reason: Optional[str] = None


class HumanReviewInput(BaseModel):
    """Human review signal payload."""

    reviewer: str
    decision: HumanDecision
    notes: str


class HumanReviewResult(BaseModel):
    """Recorded human review outcome."""

    reviewer: str
    decision: HumanDecision
    notes: str
    timestamp: str


class HumanReviewPacket(BaseModel):
    """Payload exposed to the human review UI."""

    case_id: str
    display_name: str
    sanitized_applicant: MortgageApplication
    metrics: UnderwritingMetrics
    analyses: UnderwritingAnalyses
    critic_review: str
    decision_recommendation: DecisionRecommendation
    risk_flags: list[str]
    policy_violations: list[str]
    risk_score: int


class UnderwritingOutput(BaseModel):
    """Final workflow output."""

    case_id: str
    sanitized_applicant: MortgageApplication
    metrics: UnderwritingMetrics
    analyses: UnderwritingAnalyses
    critic_review: str
    decision_memo: str
    final_decision: Decision
    risk_score: int
    risk_flags: list[str]
    bias_flags: list[str]
    policy_violations: list[str]
    human_review_required: bool
    human_review: Optional[HumanReviewResult] = None
    timestamp: str


class AgentTask(BaseModel):
    """Activity input for a specialist agent."""

    agent_name: str
    applicant: MortgageApplication
    metrics: UnderwritingMetrics
    policy_context: str


class AgentResult(BaseModel):
    """Activity output for a specialist agent."""

    analysis: str


class SupervisorTask(BaseModel):
    """Activity input for the supervisor agent."""

    applicant: MortgageApplication
    metrics: UnderwritingMetrics
    completed_agents: list[str]
    remaining_agents: list[str]
    risk_flags: list[str]
    policy_context: str


class SupervisorDecision(BaseModel):
    """Supervisor routing decision."""

    next_agent: str
    rationale: str


class ApplicationOcrTask(BaseModel):
    """Activity input for OCR application extraction."""

    case_id: str
    image_dir: str


class CriticTask(BaseModel):
    """Activity input for critic review."""

    applicant: MortgageApplication
    metrics: UnderwritingMetrics
    analyses: UnderwritingAnalyses
    risk_flags: list[str]
    policy_context: str


class CriticResult(BaseModel):
    """Activity output for critic review."""

    review: str


class DecisionTask(BaseModel):
    """Activity input for decision memo drafting."""

    applicant: MortgageApplication
    metrics: UnderwritingMetrics
    analyses: UnderwritingAnalyses
    risk_flags: list[str]
    policy_context: str


class DecisionResult(BaseModel):
    """Activity output for decision memo."""

    recommendation: DecisionRecommendation
    raw_response: str
