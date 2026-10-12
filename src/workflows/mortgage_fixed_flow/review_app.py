"""FastAPI-based human review UI for mortgage underwriting."""

from __future__ import annotations

import asyncio
import html
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from temporalio.client import Client, WorkflowHandle
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.exceptions import TemporalError

from .mortgage_models import HumanReviewInput, UnderwritingInput, UnderwritingOutput, WorkflowProgress
from .mortgage_workflow import MortgageUnderwritingWorkflow

TEMPORAL_ADDRESS = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
TASK_QUEUE = os.environ.get("MORTGAGE_TASK_QUEUE", "mortgage-underwriting")
UPLOAD_ROOT = Path(os.environ.get("UPLOAD_ROOT", "datasets/uploads"))
MANIFEST_PATH = UPLOAD_ROOT / "cases.json"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.client = await Client.connect(
        TEMPORAL_ADDRESS,
        data_converter=pydantic_data_converter,
    )
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        yield
    finally:
        client = getattr(app.state, "client", None)
        close = getattr(client, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result


app = FastAPI(title="Mortgage Human Review", lifespan=lifespan)


def _page(title: str, body: str, refresh_seconds: int | None = None) -> HTMLResponse:
    refresh_meta = ""
    if refresh_seconds:
        refresh_meta = f'<meta http-equiv="refresh" content="{refresh_seconds}">'
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  {refresh_meta}
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f6f7fb;
      --card: #ffffff;
      --text: #111827;
      --muted: #6b7280;
      --border: #e5e7eb;
      --accent: #2563eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: "IBM Plex Sans", "Segoe UI", Arial, sans-serif;
      margin: 0;
      background: var(--bg);
      color: var(--text);
    }}
    .container {{
      max-width: 960px;
      margin: 32px auto;
      padding: 0 20px 40px;
    }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    label {{ display: block; margin-top: 12px; font-weight: 600; }}
    input, select, textarea {{
      width: 100%;
      padding: 10px 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: #fff;
      font-size: 14px;
    }}
    textarea {{ min-height: 120px; }}
    button {{
      margin-top: 12px;
      padding: 10px 14px;
      border-radius: 8px;
      border: none;
      background: var(--accent);
      color: #fff;
      font-weight: 600;
      cursor: pointer;
    }}
    button.secondary {{
      background: #374151;
    }}
    .card {{
      border: 1px solid var(--border);
      padding: 16px;
      border-radius: 12px;
      background: var(--card);
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
      margin-top: 16px;
    }}
    .muted {{ color: var(--muted); }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      overflow-wrap: anywhere;
      background: #f3f4f6;
      padding: 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
      font-family: "IBM Plex Mono", "SFMono-Regular", Consolas, monospace;
      font-size: 13px;
      line-height: 1.5;
    }}
    ul {{ padding-left: 18px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
    }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      background: #e0e7ff;
      color: #3730a3;
      font-size: 12px;
      font-weight: 600;
    }}
    .badge-waiting {{ background: #fef3c7; color: #92400e; }}
    .badge-approved {{ background: #d1fae5; color: #065f46; }}
    .badge-rejected {{ background: #fee2e2; color: #991b1b; }}
    .badge-inactive {{ background: #e5e7eb; color: #374151; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 10px 8px;
      border-bottom: 1px solid var(--border);
      text-align: left;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }}
    .actions {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .alert {{
      border-radius: 10px;
      padding: 12px 14px;
      margin-bottom: 16px;
      border: 1px solid transparent;
      font-weight: 600;
    }}
    .alert-success {{
      background: #ecfdf3;
      color: #065f46;
      border-color: #d1fae5;
    }}
    .alert-error {{
      background: #fef2f2;
      color: #991b1b;
      border-color: #fee2e2;
    }}
  </style>
</head>
<body>
  <div class="container">
    {body}
  </div>
</body>
</html>
"""
    return HTMLResponse(html)


def _workflow_id(case_id: str) -> str:
    return f"mortgage-{case_id}"


async def _get_client() -> Client:
    return app.state.client


def _escape(value: object) -> str:
    return html.escape("" if value is None else str(value))


def _format_percent(value: float) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _format_money(value: float) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "N/A"


def _render_list(items: list[str]) -> str:
    if not items:
        return "<p class=\"muted\">None</p>"
    return "<ul>" + "".join([f"<li>{_escape(item)}</li>" for item in items]) + "</ul>"


def _load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    try:
        payload = json.loads(MANIFEST_PATH.read_text())
        if isinstance(payload, list):
            return payload
    except json.JSONDecodeError:
        return []
    return []


def _write_manifest(entries: list[dict]) -> None:
    MANIFEST_PATH.write_text(json.dumps(entries, indent=2))


def _scan_cases() -> list[dict]:
    if not UPLOAD_ROOT.exists():
        return []
    entries: list[dict] = []
    for path in UPLOAD_ROOT.iterdir():
        if path.name == MANIFEST_PATH.name or not path.is_dir():
            continue
        created_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        entries.append(
            {
                "case_id": path.name,
                "created_at": created_at,
                "image_dir": str(path),
            }
        )
    return entries


def _generate_case_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    suffix = datetime.now(timezone.utc).strftime("%H%M%S")
    return f"MTG-{timestamp}-{suffix}"


def _safe_extension(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".png"):
        return ".png"
    if lower.endswith(".jpg"):
        return ".jpg"
    if lower.endswith(".jpeg"):
        return ".jpeg"
    return ".png"


STAGE_LABELS = {
    "STARTED": "Starting",
    "EXTRACTING_APPLICATION": "Reading application",
    "RETRIEVING_POLICY": "Retrieving policy",
    "RUNNING_ANALYSES": "Analyzing application",
    "CRITIC_REVIEW": "Reviewing analyses",
    "DRAFTING_DECISION": "Drafting decision",
    "AWAITING_HUMAN_REVIEW": "Awaiting human review",
    "COMPLETED": "Finishing up",
}


def _humanize(value: str) -> str:
    return value.replace("_", " ").capitalize()


def _status_badge(status: str, progress: WorkflowProgress | None, decision: str | None) -> str:
    """Colored, human-readable status badge.

    Running cases show their current stage; completed cases show the final decision.
    """
    if status == "RUNNING":
        if progress is None:
            return '<span class="badge">Running</span>'
        label = STAGE_LABELS.get(progress.stage, _humanize(progress.stage))
        css = "badge badge-waiting" if progress.stage == "AWAITING_HUMAN_REVIEW" else "badge"
    elif status == "COMPLETED":
        label = f"Completed – {_humanize(decision)}" if decision else "Completed"
        css = {
            "APPROVED": "badge badge-approved",
            "REJECTED": "badge badge-rejected",
        }.get(decision or "", "badge badge-inactive")
    else:
        # Failed, timed out, terminated, canceled, not found, etc.
        label = _humanize(status)
        css = "badge badge-inactive"
    return f'<span class="{css}">{_escape(label)}</span>'


async def _fetch_decision(handle: WorkflowHandle) -> str | None:
    try:
        result = await handle.result()
    except (TemporalError, ValueError):
        return None
    return result.final_decision


async def _fetch_progress(handle: WorkflowHandle) -> WorkflowProgress | None:
    try:
        return await handle.query(MortgageUnderwritingWorkflow.get_progress)
    except TemporalError:
        return None


async def _fetch_status(client: Client, case_id: str) -> tuple[str, WorkflowProgress | None, str | None]:
    """Return the execution status, the current stage if running, and the decision if completed."""
    handle = client.get_workflow_handle(_workflow_id(case_id), result_type=UnderwritingOutput)
    try:
        description = await handle.describe()
    except TemporalError:
        return "NOT_FOUND", None, None
    if not description.status:
        return "UNKNOWN", None, None
    status = description.status.name.replace("WORKFLOW_EXECUTION_STATUS_", "")
    # Only query running workflows; querying closed ones forces a full replay on the worker.
    progress = await _fetch_progress(handle) if status == "RUNNING" else None
    # Completed results come straight from the server's history, so no worker is needed.
    decision = await _fetch_decision(handle) if status == "COMPLETED" else None
    return status, progress, decision


def _redirect_with_message(case_id: str, message: str | None = None, error: str | None = None) -> RedirectResponse:
    query = []
    if message:
        query.append(f"message={quote(message)}")
    if error:
        query.append(f"error={quote(error)}")
    suffix = f"?{'&'.join(query)}" if query else ""
    return RedirectResponse(url=f"/case/{case_id}{suffix}", status_code=303)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    client = await _get_client()
    entries = _load_manifest()
    if not entries:
        entries = _scan_cases()

    entries_by_case = {entry.get("case_id", ""): entry for entry in entries if entry.get("case_id")}

    # Pull in running and completed workflows, including ones started outside the UI (e.g., demo script).
    try:
        async for workflow in client.list_workflows(
            query=(
                'WorkflowType = "MortgageUnderwritingWorkflow" '
                'and (ExecutionStatus = "Running" or ExecutionStatus = "Completed")'
            ),
            limit=200,
        ):
            workflow_id = workflow.id
            if not workflow_id.startswith("mortgage-"):
                continue
            case_id = workflow_id.replace("mortgage-", "", 1)
            if case_id not in entries_by_case:
                entries_by_case[case_id] = {
                    "case_id": case_id,
                    "created_at": workflow.start_time.isoformat(),
                    "image_dir": "",
                }
    except TemporalError:
        pass

    entries = sorted(entries_by_case.values(), key=lambda item: item.get("case_id", ""))

    # Look up every case concurrently so the page (and its auto-refresh) stays fast.
    statuses = await asyncio.gather(*(_fetch_status(client, entry["case_id"]) for entry in entries))

    rows = []
    for entry, (status, progress, decision) in zip(entries, statuses, strict=True):
        case_id = entry["case_id"]
        rows.append(
            f"<tr>"
            f"<td><a href=\"/case/{_escape(case_id)}\">{_escape(case_id)}</a></td>"
            f"<td>{_escape(entry.get('created_at', ''))}</td>"
            f"<td>{_status_badge(status, progress, decision)}</td>"
            f"</tr>"
        )

    body = f"""
<h1>Mortgage Admin Console</h1>
<p class="muted">Upload a new case or review the status of existing workflows.</p>

<div class="card">
  <h2>Upload Loan Images</h2>
  <form method="post" action="/upload" enctype="multipart/form-data">
    <label>Loan Form Images (PNG/JPG)</label>
    <input type="file" name="files" multiple required />
    <button type="submit">Upload & Start Workflow</button>
  </form>
</div>

<div class="card">
  <h2>Recent Cases</h2>
  <table>
    <thead>
      <tr>
        <th>Case ID</th>
        <th>Created</th>
        <th>Status</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows) if rows else '<tr><td colspan="3" class="muted">No cases yet.</td></tr>'}
    </tbody>
  </table>
</div>
"""
    return _page("Admin Console", body, refresh_seconds=10)


@app.get("/case/", response_class=HTMLResponse)
async def case_redirect(case_id: str) -> HTMLResponse:
    return await case_view(case_id)


@app.get("/case/{case_id}", response_class=HTMLResponse)
async def case_view(case_id: str, message: str | None = None, error: str | None = None) -> HTMLResponse:
    client = await _get_client()
    handle = client.get_workflow_handle(
        _workflow_id(case_id),
        result_type=UnderwritingOutput,
    )
    status_label = "UNKNOWN"
    try:
        description = await handle.describe()
        if description.status:
            status_label = description.status.name.replace("WORKFLOW_EXECUTION_STATUS_", "")
    except TemporalError as exc:
        return _page(
            "Case Not Found",
            f'<p><a href="/">&larr; All cases</a></p>'
            f"<h1>Case {_escape(case_id)}</h1><p>Unable to load workflow: {_escape(exc)}</p>",
        )
    progress = await _fetch_progress(handle) if status_label == "RUNNING" else None
    progress_block = ""
    if progress:
        progress_block = (
            f'<p class="muted">{_escape(progress.detail)} (updated {_escape(progress.updated_at)})</p>'
        )
    final_decision_block = ""
    final_decision: str | None = None
    if status_label == "COMPLETED":
        try:
            result = await handle.result()
            final_decision = result.final_decision
            final_decision_block = f"""
<div class="card">
  <h2>Final Outcome</h2>
  <p>{_status_badge(status_label, None, final_decision)}</p>
  <p><strong>Risk Score:</strong> {_escape(result.risk_score)}</p>
  <p><strong>Human Review Required:</strong> {_escape(result.human_review_required)}</p>
  <p><strong>Decision Memo (Preview):</strong></p>
  <pre>{_escape(result.decision_memo[:800])}</pre>
</div>
"""
        except TemporalError as exc:
            final_decision_block = f"""
<div class="card">
  <h2>Final Outcome</h2>
  <p class="muted">Unable to load final result: {_escape(exc)}</p>
</div>
"""

    alert_block = ""
    if message:
        alert_block = f"<div class=\"alert alert-success\">{_escape(message)}</div>"
    if error:
        alert_block = f"<div class=\"alert alert-error\">{_escape(error)}</div>"

    packet = None
    try:
        packet = await handle.query(MortgageUnderwritingWorkflow.get_review_packet)
    except TemporalError:
        packet = None

    allow_submit = False
    recommendation_block = ""
    details_block = ""
    if packet is None:
        recommendation_block = """
<div class="card">
  <h2>Review Packet</h2>
  <p class="muted">No review packet available yet.</p>
</div>
"""
    else:
        applicant = packet.sanitized_applicant
        metrics = packet.metrics
        analyses = packet.analyses
        recommendation = packet.decision_recommendation
        allow_submit = status_label == "RUNNING" and recommendation.decision == "CONDITIONAL"

        recommendation_block = f"""
<div class="card">
  <h2>Recommendation</h2>
  <p><span class="badge">{_escape(recommendation.decision)}</span></p>
  <p><strong>Risk Score:</strong> {_escape(recommendation.risk_score)}</p>
  <p><strong>Reason:</strong> {_escape(recommendation.human_review_reason or "N/A")}</p>
  <p><strong>Conditions:</strong></p>
  {_render_list(recommendation.conditions)}
  <p><strong>Memo:</strong></p>
  <pre>{_escape(recommendation.memo)}</pre>
</div>
"""

        details_block = f"""
<div class="card">
  <h2>Applicant Summary (Sanitized)</h2>
  <div class="grid">
    <div>
      <p><strong>Name:</strong> {_escape(applicant.name or "N/A")}</p>
      <p><strong>Email:</strong> {_escape(applicant.email or "N/A")}</p>
      <p><strong>Phone:</strong> {_escape(applicant.phone or "N/A")}</p>
      <p><strong>Address:</strong> {_escape(applicant.address or "N/A")}</p>
    </div>
    <div>
      <p><strong>Credit Score:</strong> {_escape(applicant.credit_score)}</p>
      <p><strong>Employer:</strong> {_escape(applicant.employment.employer)}</p>
      <p><strong>Position:</strong> {_escape(applicant.employment.position)}</p>
      <p><strong>Years Employed:</strong> {_escape(applicant.employment.years)}</p>
    </div>
    <div>
      <p><strong>Monthly Income:</strong> {_format_money(applicant.employment.monthly_income)}</p>
      <p><strong>Liquid Assets:</strong> {_format_money(applicant.assets.liquid_assets_total)}</p>
      <p><strong>Reserves (months):</strong> {_escape(applicant.assets.reserves_months)}</p>
      <p><strong>Loan Amount:</strong> {_format_money(applicant.loan.amount)}</p>
    </div>
  </div>
</div>

<div class="card">
  <h2>Metrics</h2>
  <div class="grid">
    <p><strong>DTI Ratio:</strong> {_format_percent(metrics.dti_ratio)}</p>
    <p><strong>LTV Ratio:</strong> {_format_percent(metrics.ltv_ratio)}</p>
    <p><strong>Monthly Debt:</strong> {_format_money(metrics.monthly_debt)}</p>
    <p><strong>Monthly Income:</strong> {_format_money(metrics.monthly_income)}</p>
  </div>
</div>

<div class="card" style="margin-top: 16px;">
  <h2>Analyst Findings</h2>
  <h3>Credit</h3>
  <pre>{_escape(analyses.credit)}</pre>
  <h3>Income</h3>
  <pre>{_escape(analyses.income)}</pre>
  <h3>Assets</h3>
  <pre>{_escape(analyses.assets)}</pre>
  <h3>Collateral</h3>
  <pre>{_escape(analyses.collateral)}</pre>
</div>

<div class="card">
  <h2>Critic Review</h2>
  <pre>{_escape(packet.critic_review)}</pre>
</div>

<div class="card">
  <h2>Risk Flags</h2>
  {_render_list(packet.risk_flags)}
</div>

<div class="card">
  <h2>Policy Violations</h2>
  {_render_list(packet.policy_violations)}
</div>
"""

    body = f"""
<p><a href="/">&larr; All cases</a></p>
<h1>Case {_escape(case_id)}{f" — {_escape(packet.display_name)}" if packet else ""}</h1>
<p class="muted">Workflow ID: {_escape(_workflow_id(case_id))}</p>
{alert_block}
<div class="card">
  <h2>Status</h2>
  <p>{_status_badge(status_label, progress, final_decision)}</p>
  {progress_block}
</div>
{final_decision_block}

{recommendation_block}
{details_block}

{"""
<div class="card">
  <h2>Submit Review</h2>
  <form method="post" action="/submit">
    <input type="hidden" name="case_id" value=\"""" + _escape(case_id) + """\" />
    <label>Reviewer</label>
    <input name="reviewer" placeholder="Senior Underwriter" required />

    <label>Decision</label>
    <select name="decision">
      <option value="APPROVED">APPROVED</option>
      <option value="REJECTED">REJECTED</option>
    </select>

    <label>Notes</label>
    <textarea name="notes" placeholder="Write your review notes..." required></textarea>

    <button type="submit">Submit Review</button>
  </form>
</div>
""" if allow_submit else ""}
"""
    # Don't auto-refresh while the review form is shown, or it wipes the reviewer's input.
    refresh_seconds = 10 if status_label == "RUNNING" and not allow_submit else None
    return _page(f"Case {case_id}", body, refresh_seconds=refresh_seconds)


@app.post("/submit")
async def submit_review(
    case_id: str = Form(...),
    reviewer: str = Form(...),
    decision: str = Form(...),
    notes: str = Form(...),
) -> PlainTextResponse:
    client = await _get_client()
    case_id = case_id.strip()
    reviewer = reviewer.strip()
    decision = decision.strip().upper()
    notes = notes.strip()

    if decision not in {"APPROVED", "REJECTED"}:
        return _redirect_with_message(case_id, error="Invalid decision. Use APPROVED or REJECTED.")

    handle = client.get_workflow_handle(_workflow_id(case_id))

    try:
        description = await handle.describe()
        status = description.status.name.replace("WORKFLOW_EXECUTION_STATUS_", "") if description.status else "UNKNOWN"
        if status != "RUNNING":
            return _redirect_with_message(case_id, error=f"Cannot submit review: workflow status is {status}.")

        packet = await handle.query(MortgageUnderwritingWorkflow.get_review_packet)
        if packet is None:
            return _redirect_with_message(case_id, error="Cannot submit review: no review packet available yet.")
        if packet.decision_recommendation.decision != "CONDITIONAL":
            return _redirect_with_message(case_id, error="Cannot submit review: workflow is not awaiting human review.")

        await handle.signal(
            MortgageUnderwritingWorkflow.submit_human_review,
            HumanReviewInput(reviewer=reviewer, decision=decision, notes=notes),
        )
    except TemporalError as exc:
        return _redirect_with_message(case_id, error=f"Unable to submit review: {exc}")

    return _redirect_with_message(case_id, message=f"Submitted review for {case_id}.")


@app.post("/upload")
async def upload_case(files: list[UploadFile] = File(...)) -> Response:
    if not files:
        return PlainTextResponse("No files uploaded.", status_code=400)

    case_id = _generate_case_id()
    case_dir = UPLOAD_ROOT / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    for index, upload in enumerate(files, start=1):
        ext = _safe_extension(upload.filename or "")
        filename = f"{case_id}_p{index}{ext}"
        target = case_dir / filename
        content = await upload.read()
        target.write_bytes(content)

    client = await _get_client()
    await client.start_workflow(
        MortgageUnderwritingWorkflow.run,
        UnderwritingInput(case_id=case_id, image_dir=str(case_dir)),
        id=_workflow_id(case_id),
        task_queue=TASK_QUEUE,
    )

    entries = _load_manifest()
    created_at = datetime.now(timezone.utc).isoformat()
    entries.append(
        {
            "case_id": case_id,
            "created_at": created_at,
            "image_dir": str(case_dir),
        }
    )
    _write_manifest(entries)

    return RedirectResponse(url=f"/case/{case_id}", status_code=303)
