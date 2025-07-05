from typing_extensions import TypedDict
from pydantic import BaseModel
import os
from os import getenv
from langchain.chat_models import init_chat_model
from typing import Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
load_dotenv()

class State(TypedDict):
    """
    A state dictionary that holds the state of the graph
    """
    user_message: str
    is_coding_question: bool
    ai_message: str


class DetectCallResponse(BaseModel):
    """A response model schema for detecting coding queries """
    is_question_ai: bool

class SolveCodingResponse(BaseModel):
    """A response model schema for solving coding queries"""
    answer: str


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def detect_query(state: State):
    """ To Detect if the query is a coding query or not """

    SYSTEM_PROMPT = """
    YOU are an AI assistant who is expert in recognizing whether the given user query is a coding query or not.
    Your job is to detect if the user's query is related to coding question or not.
    Return the response in specified JSON boolean only."""

    user_message = state.get("user_message")
    if not user_message:
        return {"is_coding_question": False, "ai_message": "Please provide a query."}
    
   
    llm = init_chat_model(
        model="google_genai:gemini-2.0-flash-exp",  # Fixed model name
        temperature=0.1,
        api_key=GOOGLE_API_KEY,
    )
    
   
    result = llm.with_structured_output(DetectCallResponse).invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ])

    
    state["is_coding_question"] = result.is_question_ai

    return state

# Route edge is a conditional that decides the flow of the graph
def route_edge(state: State) -> Literal["solve_coding_question", "solve_simple_question"]:
    """A route edge that decides the next node to be called based on the state"""
     
    is_coding_question = state.get("is_coding_question")
    if is_coding_question:
        return "solve_coding_question"
    else:
        return "solve_simple_question"

def solve_coding_question(state: State):
    """To solve the coding question using the LLM"""
    user_message = state.get("user_message")

    SYSTEM_PROMPT = """You are an AI assistant who is expert in solving coding questions"""

    llm = init_chat_model(
        model="google_genai:gemini-2.0-flash-exp",
        temperature=0.1,
        api_key=GOOGLE_API_KEY,
    )
    
 
    result = llm.with_structured_output(SolveCodingResponse).invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ])

    state["ai_message"] = result.answer

    return state

def solve_simple_question(state: State):
    """To solve the simple user query using the LLM. Your job is to chat with user"""
    user_message = state.get("user_message")

    SYSTEM_PROMPT = """You are an AI assistant who is expert in solving simple questions"""
    
    # Fixed: Corrected the LLM initialization (removed extra comma)
    llm = init_chat_model(
        model="google_genai:gemini-2.0-flash-exp",
        temperature=0.1,
        api_key=GOOGLE_API_KEY,
    )

    # Fixed: For simple questions, we don't need structured output
    result = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ])
    
    # Fixed: Corrected the result parsing for simple text response
    state["ai_message"] = result.content

    return state

   
graph_builder = StateGraph(State)

graph_builder.add_node("detect_query", detect_query)
graph_builder.add_node("solve_coding_question", solve_coding_question)
graph_builder.add_node("solve_simple_question", solve_simple_question)

# Fixed: Removed the route_edge node as it's not a processing node
graph_builder.add_edge(START, "detect_query")
graph_builder.add_conditional_edges("detect_query", route_edge)

graph_builder.add_edge("solve_coding_question", END)
graph_builder.add_edge("solve_simple_question", END)

graph = graph_builder.compile()

# Use the graph 
def call_graph():
    state = {
        "user_message": "Can you explain What is PyDantic and what is BaseModel in PyDantic?",
        "is_coding_question": False,
        "ai_message": ""
    }

    result = graph.invoke(state)
    print("Final Result:", result)

if __name__ == "__main__":
    call_graph()

