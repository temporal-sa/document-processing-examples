"""Temporal workflow for mortgage underwriting."""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from .mortgage_activities import (
        extract_application_from_images,
        retrieve_policy_context,
        run_agent_analysis,
        run_critic_review,
        run_decision_memo,
    )
from .mortgage_models import (
    AgentTask,
    CriticTask,
    DecisionTask,
    DecisionRecommendation,
    HumanReviewInput,
    HumanReviewPacket,
    HumanReviewResult,
    ApplicationOcrTask,
    UnderwritingAnalyses,
    UnderwritingInput,
    UnderwritingOutput,
    WorkflowProgress,
    WorkflowStage,
)
from .mortgage_utils import (
    compute_metrics,
    detect_bias_signals,
    derive_risk_flags,
    format_display_name,
    hard_stop_violations,
    sanitize_pii,
)


@workflow.defn
class MortgageUnderwritingWorkflow:
    """Orchestrate the mortgage underwriting process."""

    def __init__(self) -> None:
        self._human_review: HumanReviewResult | None = None
        self._review_packet: HumanReviewPacket | None = None
        self._progress = WorkflowProgress(stage="STARTED", updated_at=workflow.now().isoformat())

    def _set_stage(self, stage: WorkflowStage, detail: str = "") -> None:
        """Record the current stage so the UI can display progress."""

        self._progress = WorkflowProgress(
            stage=stage,
            detail=detail,
            updated_at=workflow.now().isoformat(),
        )

    @workflow.signal
    async def submit_human_review(self, review: HumanReviewInput) -> None:
        """Receive human review decision and notes."""

        # Capture reviewer input; this unblocks the wait_condition in run().
        self._human_review = HumanReviewResult(
            reviewer=review.reviewer,
            decision=review.decision,
            notes=review.notes,
            timestamp=workflow.now().isoformat(),
        )

    @workflow.query
    def get_review_packet(self) -> HumanReviewPacket | None:
        """Expose the review packet for the human review UI."""

        return self._review_packet

    @workflow.query
    def get_progress(self) -> WorkflowProgress:
        """Expose the current workflow stage for the UI."""

        return self._progress

    @workflow.run
    async def run(self, input_data: UnderwritingInput) -> UnderwritingOutput:
        workflow.logger.info("Starting underwriting for case %s", input_data.case_id)

        # OCR intake: convert uploaded images into a validated MortgageApplication.
        self._set_stage("EXTRACTING_APPLICATION", "Reading loan form images")
        applicant = await workflow.execute_activity(
            extract_application_from_images,
            ApplicationOcrTask(case_id=input_data.case_id, image_dir=input_data.image_dir),
            start_to_close_timeout=timedelta(minutes=2),
        )

        # Sanitize PII for safe LLM usage and compute deterministic metrics.
        sanitized = sanitize_pii(applicant)
        sanitized_payload = sanitized.model_dump(by_alias=True)
        metrics = compute_metrics(applicant)
        risk_flags = derive_risk_flags(applicant, metrics)

        # Retrieve policy snippets in parallel to keep prompts grounded.
        policy_queries = {
            "credit": "credit score minimums, bankruptcy, late payments",
            "income": "income verification requirements, self-employment, DTI limits",
            "assets": "asset verification, reserves requirements, large deposits",
            "collateral": "property condition, appraisal requirements, repairs",
            "decision": "overall underwriting decision thresholds",
        }

        self._set_stage("RETRIEVING_POLICY", "Looking up underwriting policy")
        policy_futures = {
            key: workflow.start_activity(
                retrieve_policy_context,
                query,
                start_to_close_timeout=timedelta(seconds=15),
            )
            for key, query in policy_queries.items()
        }
        policy_context = {key: await fut for key, fut in policy_futures.items()}

        # Run specialist analyses in a fixed, deterministic order.
        analyses = UnderwritingAnalyses(credit="", income="", assets="", collateral="")
        agents = ["credit", "income", "assets", "collateral"]
        for index, agent in enumerate(agents, start=1):
            self._set_stage("RUNNING_ANALYSES", f"{agent.title()} analysis ({index} of {len(agents)})")
            result = await workflow.execute_activity(
                run_agent_analysis,
                AgentTask(
                    agent_name=agent.title(),
                    applicant=sanitized_payload,
                    metrics=metrics,
                    policy_context=policy_context[agent],
                ),
                start_to_close_timeout=timedelta(seconds=60),
            )
            setattr(analyses, agent, result.analysis)

        # Critic pass to check for missed risks or inconsistencies.
        critic_policy = policy_context["decision"]
        self._set_stage("CRITIC_REVIEW", "Checking analyses for missed risks")

        critic_review = await workflow.execute_activity(
            run_critic_review,
            CriticTask(
                applicant=sanitized_payload,
                metrics=metrics,
                analyses=analyses,
                risk_flags=risk_flags,
                policy_context=critic_policy,
            ),
            start_to_close_timeout=timedelta(seconds=60),
        )

        # Decision memo from LLM (structured JSON with decision + memo).
        self._set_stage("DRAFTING_DECISION", "Drafting decision memo")
        decision_result = await workflow.execute_activity(
            run_decision_memo,
            DecisionTask(
                applicant=sanitized_payload,
                metrics=metrics,
                analyses=analyses,
                risk_flags=risk_flags,
                policy_context=critic_policy,
            ),
            start_to_close_timeout=timedelta(seconds=60),
        )
        decision_recommendation = DecisionRecommendation.model_validate(
            decision_result.recommendation.model_dump()
        )
        
        # Re-inject the real name only in the final memo (not in LLM inputs).
        real_name = applicant.name or "[APPLICANT_NAME]"
        memo_with_name = decision_recommendation.memo.replace("[APPLICANT_NAME]", real_name)

        # Scan outputs for potential bias language.
        bias_flags = []
        for text in [
            analyses.credit,
            analyses.income,
            analyses.assets,
            analyses.collateral,
            critic_review.review,
            decision_recommendation.memo,
        ]:
            bias_flags.extend(detect_bias_signals(text))
        bias_flags = sorted(set(bias_flags))

        # Hard-stop policy checks can force a conditional review.
        policy_violations = hard_stop_violations(applicant, metrics)
        if decision_recommendation.decision == "HUMAN_REVIEW":
            decision_recommendation = decision_recommendation.model_copy(
                update={
                    "decision": "CONDITIONAL",
                    "human_review_reason": decision_recommendation.human_review_reason
                    or "LLM requested human review.",
                },
            )
        if policy_violations and decision_recommendation.decision == "APPROVED":
            decision_recommendation = decision_recommendation.model_copy(
                update={
                    "decision": "CONDITIONAL",
                    "human_review_reason": "Policy hard-stop violations require review.",
                },
            )

        final_decision = decision_recommendation.decision
        risk_score = decision_recommendation.risk_score

        # Human review gate: only block when decision is CONDITIONAL.
        human_review_required = final_decision == "CONDITIONAL"
        human_review: HumanReviewResult | None = None

        if human_review_required:
            # Store a sanitized packet for the review UI to query.
            self._review_packet = HumanReviewPacket(
                case_id=input_data.case_id,
                display_name=format_display_name(applicant),
                sanitized_applicant=sanitized_payload,
                metrics=metrics,
                analyses=analyses,
                critic_review=critic_review.review,
                decision_recommendation=decision_recommendation,
                risk_flags=risk_flags,
                policy_violations=policy_violations,
                risk_score=risk_score,
            )
            # Pause until a human reviewer signals the workflow.
            self._set_stage(
                "AWAITING_HUMAN_REVIEW",
                decision_recommendation.human_review_reason or "Conditional decision needs review",
            )
            await workflow.wait_condition(lambda: self._human_review is not None)
            human_review = self._human_review
            final_decision = human_review.decision

        self._set_stage("COMPLETED", f"Final decision: {final_decision}")

        # Emit the final underwriting output for the case.
        return UnderwritingOutput(
            case_id=input_data.case_id,
            sanitized_applicant=sanitized_payload,
            metrics=metrics,
            analyses=analyses,
            critic_review=critic_review.review,
            decision_memo=memo_with_name,
            final_decision=final_decision,
            risk_score=risk_score,
            risk_flags=risk_flags,
            bias_flags=bias_flags,
            policy_violations=policy_violations,
            human_review_required=human_review_required,
            human_review=human_review,
            timestamp=workflow.now().isoformat(),
        )
