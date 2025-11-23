from typing import TypedDict, List, Annotated, Union
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    retrieved_docs: List[str]
    final_answer: str
