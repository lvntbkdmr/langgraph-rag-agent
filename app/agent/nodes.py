from app.agent.state import AgentState
from app.rag.retriever import retriever
from app.llm.qwen import qwen_client
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import logging

logger = logging.getLogger(__name__)

def retrieval(state: AgentState) -> dict:
    query = state['messages'][-1].content
    docs = retriever.retrieve(query)
    # Format docs as strings
    doc_contents = [f"Source: {doc.metadata.get('source', 'unknown')}\nContent: {doc.page_content}" for doc in docs]
    return {"retrieved_docs": doc_contents}

def response_generation(state: AgentState) -> dict:
    retrieved_docs = state.get('retrieved_docs', [])
    messages = state['messages']
    
    context = "\n\n".join(retrieved_docs)
    
    system_prompt = """You are a strict code reviewer specializing in coding standards. 
Carefully analyze the provided code against the coding standards in the context.
Be specific about violations and cite the relevant standards.
Provide clear, actionable feedback."""

    if context:
        system_prompt += f"\n\nContext:\n{context}"

    # Prepare messages for Qwen
    # We need to convert LangChain messages to OpenAI format
    openai_messages = [{"role": "system", "content": system_prompt}]
    
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        openai_messages.append({"role": role, "content": msg.content})

    response = qwen_client.chat_completion(openai_messages)
    content = response.choices[0].message.content
    
    return {"final_answer": content, "messages": [AIMessage(content=content)]}
