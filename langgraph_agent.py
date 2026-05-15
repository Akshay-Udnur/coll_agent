"""
Personal Loan Collection Agent - LangGraph Implementation
Flow adapted from staged intent/plan/react architecture.
"""
import json
import os
import uuid
from datetime import datetime
from typing import Annotated, Literal
import re

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

import mock_db

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen/qwen3-32b")


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    verified: bool
    loan_no: str
    customer_id: str
    escalated: bool
    conversation_closed: bool
    relevance: str
    pre_plan_action: str
    execution_path: str
    post_memory_action: str
    react_action: str
    reflection_action: str
    response_target: str
    additional_targets: list[str]
    plan_action: str
    policy_ok: bool
    loop_count: int
    memory_context: str


def _json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _last_human_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content or ""
    return ""


def _is_relevant(text: str) -> bool:
    t = text.lower().strip()
    if not t:
        return False
    keys = [
        "loan",
        "payment",
        "due",
        "emi",
        "pay",
        "concession",
        "hardship",
        "human",
        "agent",
        "dob",
        "phone",
        "verify",
        "identity",
        "penalty",
        "calculate",
        "calculated",
        "calculation",
        "amount",
        "total",
        "rs",
        "rupee",
        "why",
        "how",
        "explain",
    ]
    if any(k in t for k in keys):
        return True
    # Treat likely verification/payment payloads as relevant.
    if re.search(r"\b\d{4}-\d{2}-\d{2}\b", t):
        return True
    if re.search(r"\b\d{10}\b", t):
        return True
    return False


def _extract_yyyymm(date_str: str) -> str:
    for fmt in ("%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m")
        except Exception:
            continue
    return ""


def _has_resolution_action(messages: list[BaseMessage]) -> bool:
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name in {
            "payment_link_create",
            "promise_capture",
            "concession_plan_fetching",
        }:
            return True
    return False


@tool
def customer_verify(customer_id: str, dob: str, phone: str) -> str:
    """Verify customer identity using customer id, DOB, and registered phone."""
    customer = mock_db.CUSTOMERS.get(customer_id)
    if customer and customer["dob"] == dob and customer["phone"] == phone:
        return json.dumps({"isverified": True, "customer_id": customer_id, "loan_no": customer["loan_no"]})
    return json.dumps({"isverified": False})


@tool
def loan_document_lookup(loanno: str, loan_start_date: str) -> str:
    """Find loan document id for a loan number and loan start date."""
    loan = mock_db.LOANS.get(loanno)
    if loan and loan["start_date"] == loan_start_date:
        return json.dumps({"loanDocument": loan["loan_document_id"]})
    return json.dumps({"loanDocument": None})


@tool
def loan_document_context_lookup(loanDocument: str) -> str:
    """Return contextual loan details for a loan document id."""
    for loan in mock_db.LOANS.values():
        if loan["loan_document_id"] == loanDocument:
            return json.dumps({"context": loan["loan_details"]})
    return json.dumps({"context": []})


@tool
def fetch_dues(loanno: str) -> str:
    """Fetch overdue amount, due date, and status for a loan."""
    loan = mock_db.LOANS.get(loanno)
    if loan:
        return json.dumps({"amount": loan["amount_due"], "due_date": loan["due_date"], "status": loan["status"]})
    return json.dumps({"error": "Loan not found"})


@tool
def penalty_estimate(loanno: str, proposed_payment_date: str) -> str:
    """Estimate late penalty for a proposed payment date."""
    loan = mock_db.LOANS.get(loanno)
    if not loan:
        return json.dumps({"error": "Loan not found"})
    due = None
    proposed = None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y"):
        if due is None:
            try:
                due = datetime.strptime(loan["due_date"], fmt)
            except Exception:
                pass
        if proposed is None:
            try:
                proposed = datetime.strptime(proposed_payment_date, fmt)
            except Exception:
                pass
    if due is None or proposed is None:
        return json.dumps({"error": "Invalid date format", "expected": "YYYY-MM-DD"})
    days_late = max((proposed - due).days, 0)
    return json.dumps({"days_late": days_late, "estimated_penalty": round(days_late * 25.0, 2), "currency": "INR"})


@tool
def concession_eligibility(loanno: str, customer_details: str, debt_to_asset_ratio: float = 0.6, monthly_shortfall: float = 5000.0) -> str:
    """Check concession eligibility options for a loan."""
    concession = mock_db.CONCESSIONS.get(loanno)
    if concession:
        return json.dumps({"concession": concession["eligibility_options"]})
    return json.dumps({"concession": []})


@tool
def concession_plan_fetching(loanno: str, customerID: str) -> str:
    """Fetch concession/restructure plans for the customer and loan."""
    concession = mock_db.CONCESSIONS.get(loanno)
    if concession:
        return json.dumps({"concession": concession["plans"]})
    return json.dumps({"concession": []})


@tool
def promise_capture(isPTP: bool, due_date: str, amount: float) -> str:
    """Capture a promise-to-pay commitment."""
    if isPTP:
        mock_db.ACTIONS["promises"].append({"due_date": due_date, "amount": amount})
        return json.dumps({"isSuccess": True})
    return json.dumps({"isSuccess": False})


@tool
def payment_pause_tool(from_date: str, to_date: str, isVulnerability: bool, isPause: bool) -> str:
    """Request a payment pause for vulnerable customers."""
    if isPause and isVulnerability:
        mock_db.ACTIONS["payment_pauses"].append({"from": from_date, "to": to_date})
        return json.dumps({"isSuccess": True})
    return json.dumps({"isSuccess": False})


@tool
def payment_link_create(isPayNow: bool) -> str:
    """Create a payment link for immediate payment."""
    if isPayNow:
        link = f"https://pay.example.com/{uuid.uuid4().hex[:8]}"
        mock_db.ACTIONS["payment_links"].append(link)
        return json.dumps({"link": link, "expiration": 3600})
    return json.dumps({"error": "Payment not requested"})


ALL_TOOLS = [
    customer_verify,
    loan_document_lookup,
    loan_document_context_lookup,
    fetch_dues,
    penalty_estimate,
    concession_eligibility,
    concession_plan_fetching,
    promise_capture,
    payment_pause_tool,
    payment_link_create,
]

llm = ChatGroq(api_key=GROQ_API_KEY, model=MODEL_NAME, temperature=0).bind_tools(ALL_TOOLS)

SYSTEM_PROMPT = """You are an AI-powered outbound collections and customer servicing agent for Personal Loans.
Your goal is to collect overdue payments in a compliant, empathetic, and efficient manner.

TOOL CALL RULES:
- Emit valid, complete JSON for tool calls.
- If a required argument is missing, ask a follow-up question.

REQUIRED BEHAVIOR:
- Start each conversation with greeting and identity verification request.
- Introduce yourself as an AI-capable agent from Indian Bank.
- Before verification, do not disclose account-specific details.
- Collect DOB (YYYY-MM-DD) and registered 10-digit phone, then call customer_verify.
- If only one verification field is provided, ask for the missing field.
- After verification, call fetch_dues and explain amount, due date, and status.

RESOLUTION:
- Prioritize same-month repayment.
- If same-month payment is not possible, discuss concession flow and mention penalties/late charges.
- Use concession_eligibility and concession_plan_fetching for hardship flow.
- If customer confirms "pay now" (or equivalent), call payment_link_create with isPayNow=true immediately and share the link.

ESCALATION:
- Only for abuse or explicit human-agent request.

CLOSURE:
- Append <END_CALL> only when the conversation is complete.
"""


def intent_relevance_gate_node(state: AgentState) -> AgentState:
    t = _last_human_text(state["messages"])
    if not t.strip():
        # Initial turn has no user message yet; continue flow for greeting.
        return {"relevance": "relevant"}
    # Once the call is in progress (verified or loan context resolved),
    # keep follow-up customer questions in-flow unless truly empty.
    if state.get("verified") or state.get("loan_no"):
        return {"relevance": "relevant"}
    return {"relevance": "relevant" if _is_relevant(t) else "irrelevant"}


def irrelevant_response_node(state: AgentState) -> AgentState:
    return {"messages": [AIMessage(content="I can help with your personal loan dues, payment options, or concessions. Please share what you need.")]}


def intent_pre_plan_gate_node(state: AgentState) -> AgentState:
    if not state.get("verified"):
        return {"pre_plan_action": "decide"}
    if state.get("loop_count", 0) > 2:
        return {"pre_plan_action": "plan"}
    return {"pre_plan_action": "decide"}


def plan_proposal_node(state: AgentState) -> AgentState:
    plan_text = (
        "Plan: verify identity if pending, then fetch dues, prioritize same-month payment, "
        "else evaluate concession and penalties, then confirm next commitment."
    )
    return {"messages": [AIMessage(content=plan_text)], "plan_action": "continue"}


def intent_execution_path_node(state: AgentState) -> AgentState:
    if state.get("memory_context"):
        return {"execution_path": "need_tool"}
    if state.get("verified"):
        return {"execution_path": "need_tool"}
    return {"execution_path": "need_memory"}


def memory_retrieve_node(state: AgentState) -> AgentState:
    snippets = []
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage) and msg.name in {"promise_capture", "payment_link_create", "concession_plan_fetching"}:
            snippets.append(f"{msg.name}:{msg.content}")
        if len(snippets) >= 3:
            break
    memory = " | ".join(snippets) if snippets else "No prior commitments recorded."
    return {"memory_context": memory}


def intent_post_memory_plan_gate_node(state: AgentState) -> AgentState:
    if "No prior commitments" in state.get("memory_context", ""):
        return {"post_memory_action": "react"}
    return {"post_memory_action": "plan"}


def react_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
    try:
        response = llm.invoke(messages)
    except Exception:
        # Keep graph alive on transient provider/network failures.
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I am unable to reach the model service right now due to a connection issue. "
                        "Please retry in a few seconds."
                    )
                )
            ],
            "react_action": "respond_end",
        }
    txt = response.content or ""
    updates: dict = {"messages": [response], "react_action": "respond_end"}
    if hasattr(response, "tool_calls") and response.tool_calls:
        updates["react_action"] = "act"
    if "human agent" in txt.lower():
        updates["escalated"] = True
    if "<END_CALL>" in txt:
        cleaned = txt.replace("<END_CALL>", "").strip()
        updates["messages"] = [AIMessage(content=cleaned)]
        # Guard closure so greeting/early turns cannot end the conversation.
        if state.get("verified") and _has_resolution_action(state["messages"]):
            updates["conversation_closed"] = True
    return updates


def collection_reflect_node(state: AgentState) -> AgentState:
    loop = state.get("loop_count", 0) + 1
    updates: dict = {"loop_count": loop, "reflection_action": "complete", "policy_ok": True}

    if loop >= 12:
        # Do not auto-escalate for loop count alone. Keep conversation alive and
        # push the agent back to a direct action question.
        updates["reflection_action"] = "retry_react"
        updates["messages"] = [
            AIMessage(
                content=(
                    "Please confirm your preferred next step: pay now, set a payment date, "
                    "or explore concession options."
                )
            )
        ]
        return updates

    # Policy: if out-of-month promise captured, enforce concession path note
    due_month = ""
    proposed_date = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage) and msg.name == "fetch_dues":
            due_month = _extract_yyyymm(_json(msg.content).get("due_date", ""))
            break
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage) and msg.name == "promise_capture":
            proposed_date = _json(msg.content).get("due_date", "")
            break
    if due_month and proposed_date and _extract_yyyymm(proposed_date) != due_month:
        updates["policy_ok"] = False
        updates["reflection_action"] = "retry_plan"
        updates["messages"] = [AIMessage(content="Policy check: out-of-month payment needs concession guidance and penalty disclosure.")]
        return updates

    if not state.get("conversation_closed"):
        updates["reflection_action"] = "complete"
    return updates


def relevant_response_node(state: AgentState) -> AgentState:
    target = "customer"
    additional = []
    if state.get("escalated"):
        target = "human_agent"
        additional.append("handoff_required")
    return {"response_target": target, "additional_targets": additional}


def tool_execution_node():
    return ToolNode(ALL_TOOLS)


def route_after_relevance(state: AgentState) -> Literal["irrelevant_response", "memory_retrieve"]:
    if state.get("relevance") in {"irrelevant", "empty"}:
        return "irrelevant_response"
    return "memory_retrieve"


def route_after_pre_plan(state: AgentState) -> Literal["plan_proposal", "react"]:
    return "plan_proposal" if state.get("pre_plan_action") == "plan" else "react"


def route_after_execution_path(state: AgentState) -> Literal["memory_retrieve", "react"]:
    return "memory_retrieve" if state.get("execution_path") == "need_memory" else "react"


def route_after_post_memory(state: AgentState) -> Literal["plan_proposal", "react"]:
    return "plan_proposal" if state.get("post_memory_action") == "plan" else "react"


def route_after_react(state: AgentState) -> Literal["tool_execution", "reflect", "escalation"]:
    if state.get("escalated"):
        return "escalation"
    if state.get("react_action") == "act":
        return "tool_execution"
    return "reflect"


def route_after_plan(state: AgentState) -> Literal["tool_execution", "reflect"]:
    if state.get("plan_action") == "propose":
        return "tool_execution"
    return "reflect"


def route_after_reflect(state: AgentState) -> Literal["react", "plan_proposal", "relevant_response"]:
    action = state.get("reflection_action")
    if action == "retry_react":
        return "react"
    if action == "retry_plan":
        return "plan_proposal"
    return "relevant_response"


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("intent_relevance_gate", intent_relevance_gate_node)
    builder.add_node("irrelevant_response", irrelevant_response_node)
    builder.add_node("pre_plan_gate", intent_pre_plan_gate_node)
    builder.add_node("plan_proposal", plan_proposal_node)
    builder.add_node("execution_path", intent_execution_path_node)
    builder.add_node("memory_retrieve", memory_retrieve_node)
    builder.add_node("post_memory_plan_gate", intent_post_memory_plan_gate_node)
    builder.add_node("react", react_node)
    builder.add_node("tool_execution", tool_execution_node())
    builder.add_node("reflect", collection_reflect_node)
    builder.add_node("relevant_response", relevant_response_node)
    builder.add_node("escalation", lambda s: {"escalated": True})

    builder.add_edge(START, "intent_relevance_gate")
    builder.add_conditional_edges("intent_relevance_gate", route_after_relevance, {"irrelevant_response": "irrelevant_response", "memory_retrieve": "memory_retrieve"})
    builder.add_edge("irrelevant_response", END)
    builder.add_edge("memory_retrieve", "pre_plan_gate")
    builder.add_conditional_edges("pre_plan_gate", route_after_pre_plan, {"plan_proposal": "plan_proposal", "react": "react"})
    builder.add_conditional_edges("react", route_after_react, {"tool_execution": "tool_execution", "reflect": "reflect", "escalation": "escalation"})
    builder.add_edge("tool_execution", "react")
    builder.add_conditional_edges("plan_proposal", route_after_plan, {"tool_execution": "tool_execution", "reflect": "reflect"})
    builder.add_conditional_edges("reflect", route_after_reflect, {"react": "react", "plan_proposal": "plan_proposal", "relevant_response": "relevant_response"})
    builder.add_edge("relevant_response", END)
    builder.add_edge("escalation", END)
    return builder.compile()


graph = build_graph()


def initial_state(case_ref: str, customer_context: dict | None = None) -> AgentState:
    customer_context = customer_context or {}
    customer_name = customer_context.get("name", "")
    customer_id = customer_context.get("customer_id", "")
    loan_no = customer_context.get("loan_no", "")
    system_with_context = (
        SYSTEM_PROMPT
        + f"\n\n[SYSTEM CONTEXT - NOT FOR DISCLOSURE]: case_ref='{case_ref}', customer_id='{customer_id}', customer_name='{customer_name}', loan_no='{loan_no}'."
    )
    return {
        "messages": [SystemMessage(content=system_with_context)],
        "verified": False,
        "loan_no": loan_no,
        "customer_id": customer_id,
        "escalated": False,
        "conversation_closed": False,
        "relevance": "",
        "pre_plan_action": "decide",
        "execution_path": "need_tool",
        "post_memory_action": "react",
        "react_action": "respond_end",
        "reflection_action": "complete",
        "response_target": "customer",
        "additional_targets": [],
        "plan_action": "continue",
        "policy_ok": True,
        "loop_count": 0,
        "memory_context": "",
    }


def get_last_ai_text(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return ""
