"""
Personal Loan Collection Agent — LangGraph Implementation
Uses: LangGraph StateGraph + ChatGroq (Qwen3-32B) + LangChain tools
"""
import os
import json
import uuid
from typing import Annotated, Literal
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

import mock_db

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "qwen/qwen3-32b")

# ── State ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages:    Annotated[list, add_messages]  # full conversation history
    verified:    bool                           # has identity been verified?
    loan_no:     str                            # resolved after verification
    customer_id: str                            # resolved after verification
    escalated:   bool                           # human escalation requested?

# ── LangChain Tools ───────────────────────────────────────────────────────────
@tool
def customer_verify(customer_id: str, dob: str, phone: str) -> str:
    """Verify the customer's identity using their Date of Birth (YYYY-MM-DD) and
    registered Phone number. Returns isverified, customer_id, and loan_no."""
    customer = mock_db.CUSTOMERS.get(customer_id)
    if customer and customer["dob"] == dob and customer["phone"] == phone:
        return json.dumps({
            "isverified": True,
            "customer_id": customer_id,
            "loan_no": customer["loan_no"]
        })
    return json.dumps({"isverified": False})


@tool
def loan_document_lookup(loanno: str, loan_start_date: str) -> str:
    """Find the exact loan document for a given loan number and start date (YYYY-MM-DD)."""
    loan = mock_db.LOANS.get(loanno)
    if loan and loan["start_date"] == loan_start_date:
        return json.dumps({"loanDocument": loan["loan_document_id"]})
    return json.dumps({"loanDocument": None})


@tool
def loan_document_context_lookup(loanDocument: str) -> str:
    """Lookup details from the loan document (RAG simulation)."""
    for loan in mock_db.LOANS.values():
        if loan["loan_document_id"] == loanDocument:
            return json.dumps({"context": loan["loan_details"]})
    return json.dumps({"context": []})


@tool
def fetch_dues(loanno: str) -> str:
    """Fetch the overdue amount, due date, and payment status for a loan."""
    loan = mock_db.LOANS.get(loanno)
    if loan:
        return json.dumps({
            "amount": loan["amount_due"],
            "due_date": loan["due_date"],
            "status": loan["status"]
        })
    return json.dumps({"error": "Loan not found"})


@tool
def concession_eligibility(
    loanno: str,
    customer_details: str,
    debt_to_asset_ratio: float = 0.6,
    monthly_shortfall: float = 5000.0
) -> str:
    """Check concession/hardship eligibility for a customer. Use default values for
    debt_to_asset_ratio (0.6) and monthly_shortfall (5000) if the customer hasn't provided them."""
    concession = mock_db.CONCESSIONS.get(loanno)
    if concession:
        return json.dumps({"concession": concession["eligibility_options"]})
    return json.dumps({"concession": []})


@tool
def concession_plan_fetching(loanno: str, customerID: str) -> str:
    """Fetch the available concession/restructure plans for the customer."""
    concession = mock_db.CONCESSIONS.get(loanno)
    if concession:
        return json.dumps({"concession": concession["plans"]})
    return json.dumps({"concession": []})


@tool
def promise_capture(isPTP: bool, due_date: str, amount: float) -> str:
    """Capture a Promise to Pay (PTP) from the customer."""
    if isPTP:
        mock_db.ACTIONS["promises"].append({"due_date": due_date, "amount": amount})
        return json.dumps({"isSuccess": True})
    return json.dumps({"isSuccess": False})


@tool
def payment_pause_tool(
    from_date: str,
    to_date: str,
    isVulnerability: bool,
    isPause: bool
) -> str:
    """Request a payment pause for a vulnerable customer."""
    if isPause and isVulnerability:
        mock_db.ACTIONS["payment_pauses"].append({"from": from_date, "to": to_date})
        return json.dumps({"isSuccess": True})
    return json.dumps({"isSuccess": False})


@tool
def payment_link_create(isPayNow: bool) -> str:
    """Generate a payment link (60-minute expiry) for the customer to pay now."""
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
    concession_eligibility,
    concession_plan_fetching,
    promise_capture,
    payment_pause_tool,
    payment_link_create,
]

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=MODEL_NAME,
    temperature=0,
).bind_tools(ALL_TOOLS)

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an AI-powered outbound collections and customer servicing agent for Personal Loans.
Your goal is to collect overdue payments in a compliant, empathetic, and efficient manner.

## TOOL CALL RULES (CRITICAL — follow exactly):
- ONLY emit valid, complete JSON when calling tools. NEVER truncate tool arguments mid-way.
- NEVER add trailing commas or incomplete JSON.
- If you are unsure of a required argument value, ask the customer a follow-up question instead of calling the tool.
- Always complete the full JSON structure before emitting a tool call.

## CONVERSATION FLOW (NOT STRICTLY SEQUENTIAL)
- Run the conversation naturally like a human collections representative.
- You do NOT need to follow a rigid numbered order.
- Adapt to what the customer says and ask only the next most relevant question.

## REQUIRED BEHAVIOR
- Always start a new conversation with a natural greeting, confirm account-holder identity,
  and request Date of Birth (YYYY-MM-DD) plus registered phone number for verification.
- In the opening line, clearly introduce yourself as an AI-capable agent from Indian Bank.
- Before verification, do not discuss dues, concessions, plans, or any account-specific detail.
- Identity verification is mandatory before sharing account-specific details:
  ask for Date of Birth (YYYY-MM-DD) and registered phone number, then call `customer_verify`.
- When calling `customer_verify`, always pass the selected `customer_id` from system context.
- If verification fails, apologise and offer one more attempt.
- After verification, call `fetch_dues` with loan_no and clearly explain due amount/date/status.
- For resolution:
  - Pay now: call `payment_link_create(isPayNow=True)` and share link.
  - Pay later: collect date + amount and call `promise_capture`.
  - Hardship: show empathy, call `concession_eligibility` then `concession_plan_fetching`;
    call `payment_pause_tool` only when vulnerability criteria are met.
- For loan document questions: use `loan_document_lookup` and `loan_document_context_lookup`.
- If a requested concession is not available in eligible options/plans, do NOT escalate.
  Continue negotiation by offering closest available alternatives and re-confirm customer preference.
- If `loan_document_lookup` returns no document for the provided start date, do NOT escalate to human.
  Ask for confirmation/correction of the loan start date and retry lookup, up to 4 attempts total.
  If still unavailable after 4 attempts, continue the conversation with available account actions
  (payment now, promise to pay, hardship/concession flow) without escalation.

## ESCALATION — ONLY for:
- Customer uses abusive language.
- Customer explicitly requests a human agent.
- Say: "I will now connect you with a human agent. Please hold." then stop.
- DO NOT escalate for any other reason.

## CRITICAL:
- NEVER use internal DOB/phone not stated by the customer in the conversation.
- Always wait for tool results before continuing.
- Be empathetic, professional, concise.
- Maximum tool usage: each individual tool can be called at most 4 times per conversation.
- If a tool has already been called 4 times and another call is attempted for that tool, escalate to human agent.

## PER-TOOL CALL CAPS (PER CONVERSATION)
- `customer_verify`: max 4 calls
- `fetch_dues`: max 4 calls
- `loan_document_lookup`: max 4 calls
- `loan_document_context_lookup`: max 4 calls
- `concession_eligibility`: max 4 calls
- `concession_plan_fetching`: max 4 calls
- `payment_pause_tool`: max 4 calls
- `promise_capture`: max 4 calls
- `payment_link_create`: max 4 calls
"""

# ── Graph Nodes ───────────────────────────────────────────────────────────────
def agent_node(state: AgentState) -> AgentState:
    """Main LLM reasoning node."""
    messages = state["messages"]

    # Inject system prompt if not already present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    # Handle repeated loan-document lookup failures deterministically before LLM call.
    loan_lookup_attempts = 0
    last_lookup_failed = False
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == "loan_document_lookup":
            loan_lookup_attempts += 1
            try:
                result = json.loads(msg.content)
                if not result.get("loanDocument"):
                    last_lookup_failed = True
                else:
                    last_lookup_failed = False
            except Exception:
                last_lookup_failed = True

    if last_lookup_failed and loan_lookup_attempts < 4:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I could not find the loan document with that start date. "
                        "Please re-check and share the loan start date again in YYYY-MM-DD format."
                    )
                )
            ],
            "escalated": False,
        }

    if last_lookup_failed and loan_lookup_attempts >= 4:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I still could not locate the loan document after multiple checks. "
                        "No problem, we can continue with available options now. "
                        "Would you like to pay now, set a payment date, or discuss assistance options?"
                    )
                )
            ],
            "escalated": False,
        }

    response = llm.invoke(messages)

    # Enforce hard tool-call cap: max 4 calls per tool per conversation.
    tool_call_counts: dict[str, int] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name:
            tool_call_counts[msg.name] = tool_call_counts.get(msg.name, 0) + 1

    blocked_tool_names: list[str] = []
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            tname = tc.get("name")
            if tname and tool_call_counts.get(tname, 0) >= 4:
                blocked_tool_names.append(tname)

    if blocked_tool_names:
        blocked_text = ", ".join(sorted(set(blocked_tool_names)))
        response = AIMessage(
            content=(
                f"I have reached the allowed tool usage limit for {blocked_text}. "
                "I will now connect you with a human agent. Please hold."
            )
        )

    # Extract verification info from the latest tool messages if not yet verified
    updates: dict = {"messages": [response]}
    if not state.get("verified"):
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage) and msg.name == "customer_verify":
                try:
                    result = json.loads(msg.content)
                    if result.get("isverified"):
                        updates["verified"]    = True
                        updates["loan_no"]     = result.get("loan_no", "")
                        updates["customer_id"] = result.get("customer_id", "")
                except Exception:
                    pass
                break

    # Detect escalation in the model's reply
    content = response.content or ""
    if "human agent" in content.lower() or "connect you with" in content.lower():
        # Escalate on explicit handoff language, including tool-limit exceed conditions.
        updates["escalated"] = True

    return updates


def human_escalation_node(state: AgentState) -> AgentState:
    """Terminal node when human escalation is triggered."""
    return {"escalated": True}


# ── Routing ───────────────────────────────────────────────────────────────────
def route_after_agent(state: AgentState) -> Literal["tools", "escalation", "__end__"]:
    last = state["messages"][-1]
    if state.get("escalated"):
        return "escalation"
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "__end__"


# ── Build Graph ───────────────────────────────────────────────────────────────
def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("agent",      agent_node)
    builder.add_node("tools",      ToolNode(ALL_TOOLS))
    builder.add_node("escalation", human_escalation_node)

    builder.add_edge(START,       "agent")
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "escalation": "escalation", "__end__": END}
    )
    builder.add_edge("tools", "agent")      # after tools → back to agent
    builder.add_edge("escalation", END)

    return builder.compile()


graph = build_graph()


# ── Initial state factory ─────────────────────────────────────────────────────
def initial_state(case_ref: str, customer_context: dict | None = None) -> AgentState:
    """Create a fresh state for a new outbound call."""
    customer_context = customer_context or {}
    customer_name = customer_context.get("name", "")
    customer_id = customer_context.get("customer_id", "")
    loan_no = customer_context.get("loan_no", "")

    system_with_context = (
        SYSTEM_PROMPT
        + f"\n\n[SYSTEM CONTEXT — NOT FOR DISCLOSURE]: "
        f"Internal case reference is '{case_ref}'. "
        f"Customer id is '{customer_id}'. Customer name is '{customer_name}'. "
        f"Loan number is '{loan_no}'. "
        f"Use the customer name from this context in your greeting. "
        f"Do NOT mention internal IDs. Do NOT pre-fill DOB or phone — ask the customer."
    )
    return {
        "messages":    [SystemMessage(content=system_with_context)],
        "verified":    False,
        "loan_no":     "",
        "customer_id": "",
        "escalated":   False,
    }


# ── Convenience helper ────────────────────────────────────────────────────────
def get_last_ai_text(state: AgentState) -> str:
    """Extract the last assistant text message from state."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return ""
