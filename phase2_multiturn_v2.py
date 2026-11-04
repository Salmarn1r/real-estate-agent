"""
Phase 2 Multi-turn v2 – Simplified & Stable Real Estate Lead Qualifier
Uses Groq + Tavily. Focuses on conversation quality over heavy structured extraction.
"""

import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from tavily import TavilyClient

load_dotenv()

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
    """Log the lead to CRM. Only call this when the lead is clearly qualified or you are ending the conversation."""
    print(f"\n=== CRM LOG ===\nStatus: {status}\nNotes: {notes}\n===============\n")
    return "Logged to CRM successfully."

@tool
def notify_agent(message: str) -> str:
    """Notify the human real estate agent. Use only for Hot leads that are ready for a call."""
    print(f"\n🔔 NOTIFY AGENT:\n{message}\n")
    return "Notification sent to agent."

tools = [research_market, log_to_crm, notify_agent]

# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────
llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0.2,
    max_tokens=900,
)
llm_with_tools = llm.bind_tools(tools)

# ─────────────────────────────────────────────
# System Prompt (the most important part)
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────
def agent_node(state: AgentState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# ─────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# ─────────────────────────────────────────────
# Graph
# ─────────────────────────────────────────────
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

graph = builder.compile()

# ─────────────────────────────────────────────
# Interactive Loop
# ─────────────────────────────────────────────
def run_conversation():
    print("Real Estate Lead Qualifier v2 (type 'quit' to exit)\n")
    
    state = {"messages": []}
    
    # Initial message
    first = input("Paste the initial lead message (or press Enter for sample):\n> ").strip()
    if not first:
        first = (
            "Hi, I'm looking for a 3-bedroom house in the $450-550k range in North Austin. "
            "Hoping to move in the next 3-4 months. Pre-approved for up to $520k. "
            "My name is Sarah Johnson, email sarah.j@email.com"
        )
    
    state["messages"].append(HumanMessage(content=first))
    
    while True:
        # Run the graph
        result = graph.invoke(state)
        state = result
        
        # Print the latest AI message
        last_msg = state["messages"][-1]
        print("\n" + "="*50)
        print("AGENT:")
        print(last_msg.content)
        print("="*50)
        
        # Get user reply
        user_input = input("\nYour reply (or 'quit'): ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nConversation ended.")
            break
        
        state["messages"].append(HumanMessage(content=user_input))

if __name__ == "__main__":
    run_conversation()