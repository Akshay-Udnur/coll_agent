"""
Personal Loan Collection Agent — Streamlit UI (LangGraph edition)
"""
import streamlit as st
import json
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

import mock_db
from langgraph_agent import build_graph, initial_state, get_last_ai_text

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Collection Agent (LangGraph)", page_icon="📞")
st.title("📞 Outbound Collection AI Agent")
st.caption("Powered by LangGraph · Groq · Qwen3-32B")

# ── Sidebar: customer selector ─────────────────────────────────────────────────
st.sidebar.header("📋 Agent Context")
st.sidebar.write("Select a customer to initiate an outbound call:")

customer_options = {cid: c["name"] for cid, c in mock_db.CUSTOMERS.items()}
selected_cid = st.sidebar.selectbox(
    "Customer",
    options=[None] + list(customer_options.keys()),
    index=0,
    format_func=lambda x: "Select customer..." if x is None else customer_options[x]
)

# Show live state for debugging
if st.sidebar.checkbox("🔍 Show Agent State"):
    if "lg_state" in st.session_state:
        s = st.session_state.lg_state
        st.sidebar.json({
            "verified":    s.get("verified"),
            "loan_no":     s.get("loan_no"),
            "customer_id": s.get("customer_id"),
            "escalated":   s.get("escalated"),
        })

if st.sidebar.checkbox("🗄️ View Captured Actions"):
    st.sidebar.json(mock_db.ACTIONS)

if selected_cid is None:
    st.info("Select a customer from the sidebar to start the conversation.")
    st.stop()

# ── State init: new session or new customer ────────────────────────────────────
if "current_customer" not in st.session_state or st.session_state.current_customer != selected_cid:
    st.session_state.current_customer = selected_cid
    st.session_state.chat_display = []          # [(role, content)] for rendering
    st.session_state.lg_state = None
    st.session_state.lg_graph = build_graph()   # fresh graph per selected customer

    # Build initial LangGraph state and run the first agent step (agent greeting)
    selected_customer = mock_db.CUSTOMERS[selected_cid]
    state = initial_state(
        selected_cid,
        {
            "customer_id": selected_cid,
            "name": selected_customer["name"],
            "loan_no": selected_customer["loan_no"],
        },
    )
    with st.spinner("📡 Agent is initiating the call..."):
        state = st.session_state.lg_graph.invoke(state)

    st.session_state.lg_state = state
    greeting = get_last_ai_text(state)
    if greeting:
        st.session_state.chat_display.append(("assistant", greeting))

# ── Render existing chat ───────────────────────────────────────────────────────
for role, content in st.session_state.chat_display:
    with st.chat_message(role):
        st.markdown(content)

# ── Show escalation banner if triggered ───────────────────────────────────────
if st.session_state.get("lg_state", {}).get("escalated"):
    st.error("🔴 Escalated to Human Agent — This conversation has ended.")
    st.stop()

# ── Chat input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Type your response as the customer..."):
    # Show user message
    st.session_state.chat_display.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Inject the user message into LangGraph state and run the graph
    current_state = st.session_state.lg_state
    current_state["messages"].append(HumanMessage(content=prompt))

    with st.spinner("🤖 Agent thinking..."):
        new_state = st.session_state.lg_graph.invoke(current_state)

    st.session_state.lg_state = new_state

    # Display tool calls that happened in this turn
    msgs = new_state["messages"]
    last_human_idx = max(
        i for i, m in enumerate(msgs) if isinstance(m, HumanMessage)
    )
    for msg in msgs[last_human_idx + 1:]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                with st.chat_message("assistant", avatar="⚙️"):
                    st.write(f"**Tool called:** `{tc['name']}`")
                    st.json(tc["args"])
        elif isinstance(msg, ToolMessage):
            with st.chat_message("assistant", avatar="📥"):
                st.write(f"**Tool result** (`{msg.name}`):")
                try:
                    st.json(json.loads(msg.content))
                except Exception:
                    st.write(msg.content)

    # Final agent reply
    reply = get_last_ai_text(new_state)
    if reply:
        st.session_state.chat_display.append(("assistant", reply))
        with st.chat_message("assistant"):
            st.markdown(reply)

    # Escalation check after response
    if new_state.get("escalated"):
        st.error("🔴 Escalated to Human Agent — This conversation has ended.")
        st.stop()

    st.rerun()
