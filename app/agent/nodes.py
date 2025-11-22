from app.agent.state import AgentState
from app.rag.retriever import retriever
from app.llm.qwen import qwen_client
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import logging

logger = logging.getLogger(__name__)

def mode_detector(state: AgentState) -> dict:
    last_message = state['messages'][-1].content.lower()
    
    code_review_keywords = [
        "code review", "coding standard", "complies", "check function", 
        "meets requirements", "review", "check", "compliance", 
        "style guide", "violate"
    ]
    
    if any(keyword in last_message for keyword in code_review_keywords):
        mode = "code_review"
    else:
        mode = "general"
        
    logger.info(f"Detected mode: {mode}")
    return {"mode": mode}

def rag_decision(state: AgentState) -> dict:
    # Simple logic: code_review always needs RAG, general might not but we'll enable it for now
    # to be helpful with uploaded docs.
    # In a more complex system, we could use an LLM call to decide.
    return {} # Pass through, decision is implicit in graph edges or we just always retrieve if docs exist

def retrieval(state: AgentState) -> dict:
    query = state['messages'][-1].content
    docs = retriever.retrieve(query)
    # Format docs as strings
    doc_contents = [f"Source: {doc.metadata.get('source', 'unknown')}\nContent: {doc.page_content}" for doc in docs]
    return {"retrieved_docs": doc_contents}

def response_generation(state: AgentState) -> dict:
    mode = state['mode']
    retrieved_docs = state.get('retrieved_docs', [])
    messages = state['messages']
    
    context = "\n\n".join(retrieved_docs)
    
    if mode == "code_review":
        system_prompt = """You are a strict code reviewer specializing in coding standards. 
Carefully analyze the provided code against the coding standards in the context.
Be specific about violations and cite the relevant standards.
Provide clear, actionable feedback."""
    else:
        system_prompt = """You are a helpful technical assistant with access to company documentation.
Answer questions clearly and cite sources when using specific document information."""

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
