"""
Real Estate Lead Qualifier – Streamlit + Airtable (logs every message)
"""

import os
import uuid
from datetime import datetime, timezone
from typing import Annotated, TypedDict
from dotenv import load_dotenv

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from tavily import TavilyClient
from pyairtable import Api

load_dotenv()

# ─────────────────────────────────────────────
# Airtable setup
# ─────────────────────────────────────────────
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME")

airtable = None
if AIRTABLE_API_KEY and AIRTABLE_BASE_ID:
    try:
        api = Api(AIRTABLE_API_KEY)
        airtable = api.table(AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME)
    except Exception as e:
        st.warning(f"Airtable connection failed: {e}")

def log_message_to_airtable(session_id: str, role: str, content: str, lead_name: str = "", phone: str = ""):
    """Log a single message to Airtable."""
    if not airtable:
        return
    try:
        airtable.create({
            "Session ID": session_id,
            "Role": role,
            "Content": content,
            "Lead Name": lead_name or "",
            "Phone": phone or "",
            "Timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        print(f"Failed to log to Airtable: {e}")

# ─────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def research_market(query: str) -> str:
    """Research neighborhoods, schools, market conditions, or general real estate info."""
    try:
        response = tavily.search(
            query=query,
            search_depth="basic",
            max_results=4,
            include_answer=True
        )
        if response.get("answer"):
            return response["answer"]
        results = [f"- {r['title']}: {r['content'][:250]}..." for r in response.get("results", [])[:3]]
        return "\n".join(results) if results else "No good results found."
    except Exception as e:
        return f"Research error: {e}"

@tool
def log_to_crm(status: str, notes: str) -> str:
    """Log the lead to CRM. Only call when the lead is clearly qualified."""
    print(f"\n=== CRM LOG ===\nStatus: {status}\nNotes: {notes}\n===============\n")
    return "Logged to CRM successfully."

@tool
def notify_agent(message: str) -> str:
    """Notify the human real estate agent for Hot leads."""
    print(f"\n🔔 NOTIFY AGENT:\n{message}\n")
    return "Notification sent to agent."

tools = [research_market, log_to_crm, notify_agent]

# ─────────────────────────────────────────────
# LangGraph setup
# ─────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0.2,
    max_tokens=800,
)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are a friendly, professional real estate lead qualification assistant helping a licensed agent.

Your goals:
1. Have a natural conversation to qualify the lead.
2. Collect: name, phone, budget, timeline, preferred area, property type, pre-approval status, and motivation.
3. NEVER re-ask for information the lead has already given you. Acknowledge what you know.
4. Ask only 1–2 questions at a time.
5. When you have enough information (especially phone + budget + timeline + area), summarize and offer to connect them with the agent or schedule a call.
6. Use the research tool when the lead asks about neighborhoods, schools, or market conditions.
7. Only call log_to_crm and notify_agent when the lead is clearly ready (Hot) or the conversation is ending.

Rules:
- Be warm and helpful.
- Never give legal, tax, or appraisal advice.
- Follow Fair Housing guidelines.
- If the lead changes location or key details, acknowledge the change and adapt.
"""

def agent_node(state: AgentState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

graph = builder.compile()

# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="Real Estate Lead Qualifier", page_icon="🏠")
st.title("🏠 Real Estate Lead Qualifier")
st.caption("AI assistant that helps qualify real estate leads")

# Session ID (stays the same for the whole browser session)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Chat input
if prompt := st.chat_input("Type the lead's message here..."):
    # --- Log & show user message ---
    st.session_state.messages.append(HumanMessage(content=prompt))
    log_message_to_airtable(
        session_id=st.session_state.session_id,
        role="user",
        content=prompt
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Run agent ---
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = graph.invoke({"messages": st.session_state.messages})
            ai_message = result["messages"][-1]
            response_text = ai_message.content
            st.markdown(response_text)

    # --- Log & save assistant message ---
    st.session_state.messages.append(ai_message)
    log_message_to_airtable(
        session_id=st.session_state.session_id,
        role="assistant",
        content=response_text
    )