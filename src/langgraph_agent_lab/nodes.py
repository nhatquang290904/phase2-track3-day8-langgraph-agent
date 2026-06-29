"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, ApprovalDecision, make_event


class Classification(BaseModel):
    route: Literal["simple", "tool", "missing_info", "risky", "error"] = Field(
        description="The workflow route for the support request."
    )
    risk_level: Literal["low", "medium", "high"] = Field(
        description="High only for side-effecting or destructive requests."
    )


def _message_text(response: object) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── TODO(student): implement ALL nodes below ────────────────────────


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM.

    *** MUST use a real LLM call — keyword-only heuristics will lose points. ***

    Use .with_structured_output() or equivalent to get reliable enum classification.
    The LLM should classify into one of: simple, tool, missing_info, risky, error.

    Hints:
    - See llm.py for the get_llm() helper
    - Use Pydantic model or TypedDict with .with_structured_output()
    - Set risk_level to "high" for risky routes, "low" otherwise
    - Priority guide: risky > tool > missing_info > error > simple

    Return: {"route": str, "risk_level": str, "events": [make_event(...)]}
    """
    llm = get_llm()
    classifier = llm.with_structured_output(Classification)
    result = classifier.invoke(
        [
            (
                "system",
                "Classify support tickets into exactly one route. "
                "Use this priority if multiple labels apply: "
                "risky, tool, missing_info, error, simple. "
                "risky means side effects such as refunds, deletes, "
                "cancellations, or sending email. "
                "tool means lookup/search/status requests. missing_info means too vague to act. "
                "error means system failures, timeouts, crashes, or unrecoverable service errors. "
                "simple means general how-to questions answerable without tools.",
            ),
            ("human", state.get("query", "")),
        ]
    )
    route = result.route
    risk_level = "high" if route == "risky" else result.risk_level
    return {
        "route": route,
        "risk_level": risk_level,
        "messages": [f"classify:{route}"],
        "events": [make_event("classify", "completed", "query classified", route=route)],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call.

    Simulate transient failures for error-route scenarios to test retry loops.

    Requirements:
    - Read current attempt count from state
    - If route is "error" and attempt < 2: return error result (string containing "ERROR")
    - Otherwise: return a mock success result string
    - Append result to tool_results list

    Return: {"tool_results": [result_string], "events": [make_event(...)]}
    """
    route = state.get("route", "tool")
    attempt = int(state.get("attempt", 0))
    query = state.get("query", "")
    if route == "error" and attempt < 2:
        result = f"ERROR: transient backend failure on attempt {attempt}"
        event_type = "failed"
    elif route == "risky":
        proposed_action = state.get("proposed_action") or query
        result = f"Approved action prepared for execution: {proposed_action}"
        event_type = "completed"
    else:
        result = f"Tool result for '{query}': mock support data found and verified."
        event_type = "completed"
    return {
        "tool_results": [result],
        "events": [make_event("tool", event_type, "tool executed", attempt=attempt)],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate.

    Check whether the latest tool result is satisfactory or needs retry.

    SHOULD use LLM-as-judge for bonus points. Heuristic (e.g., check for "ERROR" substring)
    is acceptable for base score.

    Requirements:
    - Read the latest entry from tool_results
    - Set evaluation_result to "needs_retry" or "success"
    - This field drives route_after_evaluate conditional edge

    Note: You may need to add 'evaluation_result' to AgentState if not present.

    Return: {"evaluation_result": str, "events": [make_event(...)]}
    """
    latest = (state.get("tool_results") or [""])[-1]
    evaluation_result = "needs_retry" if "ERROR" in latest.upper() else "success"
    return {
        "evaluation_result": evaluation_result,
        "events": [
            make_event(
                "evaluate",
                "completed",
                "tool result evaluated",
                evaluation_result=evaluation_result,
            )
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM.

    *** MUST use a real LLM call — hardcoded strings will lose points. ***

    The LLM should generate a helpful response grounded in available context:
    - tool_results (if any)
    - approval decision (if risky route)
    - original query

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    llm = get_llm(temperature=0.2)
    approval = state.get("approval")
    response = llm.invoke(
        [
            (
                "system",
                "You are a concise support agent. Answer only from the provided context. "
                "If a risky action was approved, mention that approval was recorded. "
                "Do not invent tool data.",
            ),
            (
                "human",
                "\n".join(
                    [
                        f"User query: {state.get('query', '')}",
                        f"Route: {state.get('route', '')}",
                        f"Tool results: {state.get('tool_results', [])}",
                        f"Approval: {approval}",
                    ]
                ),
            ),
        ]
    )
    answer = _message_text(response)
    return {
        "final_answer": answer,
        "events": [make_event("answer", "completed", "final answer generated")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Generate a specific clarification question based on the vague/incomplete query.

    Note: You may need to add 'pending_question' to AgentState if not present.

    Return: {"pending_question": str, "final_answer": str, "events": [make_event(...)]}
    """
    question = (
        "Can you share the specific account, order, or issue details so I can help safely?"
    )
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "clarification requested")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval.

    Describe the proposed action and why it requires approval.

    Note: You may need to add 'proposed_action' to AgentState if not present.

    Return: {"proposed_action": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    proposed_action = f"Review and approve this side-effecting support action: {query}"
    return {
        "proposed_action": proposed_action,
        "events": [make_event("risky_action", "completed", "approval package prepared")],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default behavior: mock approval (approved=True) so tests and CI run offline.
    Extension: if env LANGGRAPH_INTERRUPT=true, use langgraph.types.interrupt() for real HITL.

    Return approval decision and an audit event.
    """
    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        decision = interrupt(
            {
                "proposed_action": state.get("proposed_action"),
                "question": "Approve this risky support action?",
            }
        )
        approved = bool(decision.get("approved", False)) if isinstance(decision, dict) else False
        comment = str(decision.get("comment", "")) if isinstance(decision, dict) else ""
    else:
        approved = True
        comment = "Mock approval for lab execution."
    approval = ApprovalDecision(approved=approved, comment=comment).model_dump()
    return {
        "approval": approval,
        "events": [make_event("approval", "completed", "approval decision recorded")],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt.

    Increment the attempt counter and log the transient failure.

    Requirements:
    - Read current attempt from state, increment by 1
    - Add an error message to errors list
    - Return updated attempt count

    Return: {"attempt": int, "errors": [str], "events": [make_event(...)]}
    """
    attempt = int(state.get("attempt", 0)) + 1
    error = f"Retry attempt {attempt} after transient failure"
    return {
        "attempt": attempt,
        "errors": [error],
        "events": [make_event("retry", "completed", "retry recorded", attempt=attempt)],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded.

    This is the third layer: retry → fallback → dead letter.
    Log the failure and set a final_answer explaining that the request could not be completed.

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    answer = (
        "The request could not be completed after the allowed retry attempts. "
        "It has been escalated for manual review."
    )
    return {
        "final_answer": answer,
        "events": [make_event("dead_letter", "completed", "max retries exhausted")],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END.

    Return: {"events": [make_event("finalize", "completed", "workflow finished")]}
    """
    return {"events": [make_event("finalize", "completed", "workflow finished")]}
