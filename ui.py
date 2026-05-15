"""
Personal Loan Collection Agent - Streamlit UI (LangGraph edition)
"""
import json

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import mock_db
from langgraph_agent import build_graph, get_last_ai_text, initial_state


def _merge_state(base: dict, update: dict) -> dict:
    merged = dict(base)
    for key, value in update.items():
        if key == "messages":
            merged.setdefault("messages", [])
            merged["messages"].extend(value)
        else:
            merged[key] = value
    return merged


def _execution_graph_dot(active_node: str | None, events: list[dict]) -> str:
    completed = {event["node"] for event in events if event["event"] == "node_complete"}
    nodes = [
        "intent_relevance_gate",
        "irrelevant_response",
        "pre_plan_gate",
        "plan_proposal",
        "memory_retrieve",
        "react",
        "tool_execution",
        "reflect",
        "relevant_response",
        "escalation",
        "END",
    ]
    lines = [
        "digraph G {",
        'rankdir=LR; bgcolor="transparent"; nodesep=0.35; ranksep=0.55; splines=spline;',
        'node [shape=box style="rounded,filled" fontname="Arial" fontsize=10 fillcolor="#f4f4f5"];',
    ]
    for node in nodes:
        fill = "#f4f4f5"
        if node in completed:
            fill = "#d1fae5"
        if active_node == node:
            fill = "#bfdbfe"
        lines.append(f'"{node}" [fillcolor="{fill}"];')
    lines.extend(
        [
            '"intent_relevance_gate" -> "irrelevant_response";',
            '"intent_relevance_gate" -> "memory_retrieve";',
            '"irrelevant_response" -> "END";',
            '"memory_retrieve" -> "pre_plan_gate";',
            '"pre_plan_gate" -> "plan_proposal";',
            '"pre_plan_gate" -> "react";',
            '"react" -> "tool_execution";',
            '"tool_execution" -> "react";',
            '"react" -> "reflect";',
            '"react" -> "escalation";',
            '"plan_proposal" -> "tool_execution";',
            '"plan_proposal" -> "reflect";',
            '"reflect" -> "react";',
            '"reflect" -> "plan_proposal";',
            '"reflect" -> "relevant_response";',
            '"relevant_response" -> "END";',
            '"escalation" -> "END";',
            "}",
        ]
    )
    return "\n".join(lines)


def _render_notes(events: list[dict], limit: int = 20) -> None:
    for event in reversed(events[-limit:]):
        if event["event"] == "tool_call":
            st.code(
                f"{event['node']} -> tool: {event['tool_name']} {json.dumps(event['args'])}",
                language="json",
            )
        else:
            st.write(f"- `{event['node']}` completed")


def _stream_invoke(graph, current_state: dict, graph_slot, notes_slot) -> tuple[dict, list[dict]]:
    turn_events: list[dict] = []
    rebuilt_state = dict(current_state)
    active_node = None

    try:
        for chunk in graph.stream(current_state, stream_mode="updates"):
            for node_name, node_update in chunk.items():
                active_node = node_name
                turn_events.append({"event": "node_complete", "node": node_name})
                rebuilt_state = _merge_state(rebuilt_state, node_update)
                for msg in node_update.get("messages", []):
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            turn_events.append(
                                {
                                    "event": "tool_call",
                                    "node": node_name,
                                    "tool_name": tool_call["name"],
                                    "args": tool_call["args"],
                                }
                            )

            graph_slot.graphviz_chart(_execution_graph_dot(active_node, turn_events))
            with notes_slot.container():
                st.caption("Live execution")
                _render_notes(turn_events, limit=10)
    except Exception as exc:
        rebuilt_state.setdefault("messages", [])
        rebuilt_state["messages"].append(
            AIMessage(
                content=(
                    "Agent execution failed due to a temporary model connection issue. "
                    "Please try again."
                )
            )
        )
        turn_events.append({"event": "node_complete", "node": "execution_error"})
        with notes_slot.container():
            st.caption("Live execution")
            st.error(f"Execution error: {type(exc).__name__}")

    turn_events.append({"event": "node_complete", "node": "END"})
    graph_slot.graphviz_chart(_execution_graph_dot("END", turn_events))

    return rebuilt_state, turn_events


st.set_page_config(page_title="Collection Agent (LangGraph)", page_icon="Phone")
st.title("Outbound Collection AI Agent")
st.caption("Powered by LangGraph, Groq, and Qwen3-32B")

st.sidebar.header("Agent Context")
st.sidebar.write("Select a customer to initiate an outbound call:")

customer_options = {cid: customer["name"] for cid, customer in mock_db.CUSTOMERS.items()}
selected_cid = st.sidebar.selectbox(
    "Customer",
    options=[None] + list(customer_options.keys()),
    index=0,
    format_func=lambda item: "Select customer..." if item is None else customer_options[item],
)

if "is_running" not in st.session_state:
    st.session_state.is_running = False

if selected_cid is None:
    st.info("Select a customer from the sidebar to start the conversation.")
    st.stop()

if "current_customer" not in st.session_state or st.session_state.current_customer != selected_cid:
    st.session_state.current_customer = selected_cid
    st.session_state.chat_display = []
    st.session_state.exec_events = []
    st.session_state.lg_state = None
    st.session_state.lg_graph = build_graph()

    selected_customer = mock_db.CUSTOMERS[selected_cid]
    state = initial_state(
        selected_cid,
        {
            "customer_id": selected_cid,
            "name": selected_customer["name"],
            "loan_no": selected_customer["loan_no"],
        },
    )
    with st.spinner("Agent is initiating the call..."):
        state = st.session_state.lg_graph.invoke(state)

    st.session_state.lg_state = state
    greeting = get_last_ai_text(state)
    if greeting:
        st.session_state.chat_display.append(("assistant", greeting))

if "exec_events" not in st.session_state:
    st.session_state.exec_events = []

st.sidebar.subheader("Tool and Agent Execution")
sidebar_graph = st.sidebar.empty()
sidebar_notes = st.sidebar.empty()
sidebar_graph.graphviz_chart(_execution_graph_dot(None, st.session_state.exec_events))
with sidebar_notes.container():
    st.caption("Tool calls and node execution notes (latest first)")
    _render_notes(st.session_state.exec_events, limit=20)

for role, content in st.session_state.chat_display:
    with st.chat_message(role):
        st.markdown(content)

state_view = st.session_state.get("lg_state") or {}
conversation_blocked = False

if state_view.get("escalated"):
    st.error("Escalated to Human Agent. This conversation has ended.")
    conversation_blocked = True

if state_view.get("conversation_closed"):
    st.success("Conversation ended by agent.")
    conversation_blocked = True

if prompt := st.chat_input(
    "Type your response as the customer...",
    disabled=st.session_state.is_running or conversation_blocked,
):
    st.session_state.chat_display.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    current_state = st.session_state.lg_state
    current_state["messages"].append(HumanMessage(content=prompt))

    live_graph = sidebar_graph
    live_notes = sidebar_notes
    st.session_state.is_running = True
    try:
        with st.spinner("Agent thinking..."):
            new_state, turn_events = _stream_invoke(
                st.session_state.lg_graph,
                current_state,
                live_graph,
                live_notes,
            )
    finally:
        st.session_state.is_running = False

    st.session_state.lg_state = new_state
    st.session_state.exec_events.extend(turn_events)
    if new_state.get("escalated"):
        st.session_state.exec_events.append({"event": "node_complete", "node": "escalation"})
        sidebar_graph.graphviz_chart(_execution_graph_dot("escalation", st.session_state.exec_events))

    messages = new_state["messages"]
    last_human_idx = max(index for index, msg in enumerate(messages) if isinstance(msg, HumanMessage))
    for msg in messages[last_human_idx + 1 :]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tool_call in msg.tool_calls:
                with st.chat_message("assistant", avatar=":material/build:"):
                    st.write(f"Tool called: `{tool_call['name']}`")
                    st.json(tool_call["args"])
        elif isinstance(msg, ToolMessage):
            with st.chat_message("assistant", avatar=":material/download:"):
                st.write(f"Tool result (`{msg.name}`):")
                try:
                    st.json(json.loads(msg.content))
                except Exception:
                    st.write(msg.content)

    reply = get_last_ai_text(new_state)
    if reply:
        st.session_state.chat_display.append(("assistant", reply))
        with st.chat_message("assistant"):
            st.markdown(reply)

    st.rerun()
