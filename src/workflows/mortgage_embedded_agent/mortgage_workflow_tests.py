"""Workflow-level tests with mocked activities."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from temporalio import activity
from temporalio.worker import Worker

from src.workflows.mortgage_embedded_agent.mortgage_models import (
    AgentResult,
    AgentTask,
    ApplicationOcrTask,
    CriticResult,
    CriticTask,
    DecisionRecommendation,
    DecisionResult,
    DecisionTask,
    HumanReviewInput,
    MortgageApplication,
    SupervisorDecision,
    SupervisorTask,
    UnderwritingInput,
)
from src.workflows.mortgage_embedded_agent.mortgage_workflow import MortgageUnderwritingWorkflow

TASK_QUEUE = "test-mortgage-queue"


def _load_case(index: int) -> dict:
    repo_root = Path(__file__).resolve().parents[3]
    data_path = repo_root / "resources" / "mortgage_test_cases.json"
    payload = json.loads(data_path.read_text())
    return payload["test_cases"][index]


@activity.defn(name="retrieve_policy_context")
async def fake_retrieve_policy_context(query: str) -> str:
    return "policy context"


@activity.defn(name="extract_application_from_images")
async def fake_extract_application_from_images(task: ApplicationOcrTask) -> MortgageApplication:
    case = _load_case(1)
    case["case_id"] = task.case_id
    return MortgageApplication(**case)


@activity.defn(name="run_supervisor")
async def fake_run_supervisor(task: SupervisorTask) -> SupervisorDecision:
    next_agent = task.remaining_agents[0] if task.remaining_agents else "decision"
    return SupervisorDecision(next_agent=next_agent, rationale="test routing")


@activity.defn(name="run_agent_analysis")
async def fake_run_agent_analysis(task: AgentTask) -> AgentResult:
    return AgentResult(analysis=f"{task.agent_name} analysis complete")


@activity.defn(name="run_critic_review")
async def fake_run_critic_review(task: CriticTask) -> CriticResult:
    return CriticResult(review="critic ok")


@activity.defn(name="run_decision_memo")
async def fake_run_decision_memo(task: DecisionTask) -> DecisionResult:
    recommendation = DecisionRecommendation(
        decision="CONDITIONAL",
        risk_score=55,
        memo="needs human review",
        conditions=[],
        human_review_reason="test review",
    )
    return DecisionResult(recommendation=recommendation, raw_response="{}")


@pytest.mark.asyncio
async def test_workflow_human_review_signal(client) -> None:
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MortgageUnderwritingWorkflow],
        activities=[
            fake_extract_application_from_images,
            fake_retrieve_policy_context,
            fake_run_supervisor,
            fake_run_agent_analysis,
            fake_run_critic_review,
            fake_run_decision_memo,
        ],
    ):
        case = _load_case(1)
        workflow_input = UnderwritingInput(case_id=case["case_id"], image_dir="/tmp/fake-images")

        handle = await client.start_workflow(
            MortgageUnderwritingWorkflow.run,
            workflow_input,
            id=f"test-{case['case_id']}",
            task_queue=TASK_QUEUE,
        )

        packet = None
        for _ in range(10):
            packet = await handle.query(MortgageUnderwritingWorkflow.get_review_packet)
            if packet is not None:
                break
            await asyncio.sleep(0.1)

        assert packet is not None
        assert packet.decision_recommendation.decision == "CONDITIONAL"
        assert packet.display_name

        progress = await handle.query(MortgageUnderwritingWorkflow.get_progress)
        assert progress.stage == "AWAITING_HUMAN_REVIEW"
        assert progress.detail == "test review"

        await handle.signal(
            MortgageUnderwritingWorkflow.submit_human_review,
            HumanReviewInput(
                reviewer="QA Reviewer",
                decision="APPROVED",
                notes="Approved after review.",
            ),
        )

        result = await handle.result()
        assert result.final_decision == "APPROVED"
        assert result.human_review is not None
        assert result.analyses.credit
        assert result.analyses.income
        assert result.analyses.assets
        assert result.analyses.collateral
        assert "[APPLICANT_NAME]" not in result.decision_memo

        progress = await handle.query(MortgageUnderwritingWorkflow.get_progress)
        assert progress.stage == "COMPLETED"
