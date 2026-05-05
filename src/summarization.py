import operator
import os
from typing import TypedDict, Annotated, List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import Send
from langgraph.graph import StateGraph, START, END

load_dotenv()

model = ChatOpenAI(
    model="openrouter/owl-alpha",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)


class State(TypedDict):
    contents: List[str]
    summaries: Annotated[List[str], operator.add]
    final_summary: str


def map_summarize(state: dict):
    content = state["content"]

    prompt = f"""
    Summarize this meeting transcript chunk.
    Extract key decisions and action items.

    Transcript:
    {content}
    """

    response = model.invoke([HumanMessage(content=prompt)])

    return {"summaries": [response.content]}


def map_summaries(state: State):
    return [Send("map_summarize", {"content": c}) for c in state["contents"]]


def reduce_summaries(state: State):
    joined = "\n\n".join(state["summaries"])

    prompt = f"""
    Combine the following partial summaries into a final structured meeting summary.
    Return the answer in the Russian language.
    
    Include:
    - key decisions
    - action items
    - important discussion points

    Partial summaries:
    {joined}
    """

    response = model.invoke([HumanMessage(content=prompt)])

    return {"final_summary": response.content}


builder = StateGraph(State)

builder.add_node("map_summarize", map_summarize)
builder.add_node("reduce", reduce_summaries)

builder.add_conditional_edges(START, map_summaries, ["map_summarize"])
builder.add_edge("map_summarize", "reduce")
builder.add_edge("reduce", END)

graph = builder.compile()
