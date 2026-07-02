from analyst.graph.state import ResearchState
from analyst.retrieval.vector_store import VectorStore
from analyst.config import settings

from langchain_core.documents import Document
from langgraph.graph.state import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

GENERATE_SYSTEM_PROMPT = """You are a financial research analyst. Answer the user's question \
using ONLY the context chunks provided below. Do not use any outside knowledge.
 
Rules:
- After every claim, add an inline citation in this exact format:
  [TICKER | FILING_TYPE | DATE | SECTION — HEADING]
  Example: [AAPL | 10-K | 2023-09-30 | mda — Liquidity and Capital Resources]
- If the context does not contain enough information to answer, say exactly:
  "The available filings do not contain sufficient information to answer this question."
- Do not guess, infer beyond what is stated, or fabricate figures.
- Be concise but complete — answer what was asked, nothing more.
 
Context chunks:
{context}"""

def format_retireved_into_context(chunks: list[dict]) -> str:

    formatted = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        header = (
            f"[{i}] {meta.get('ticker','?')} | "
            f"{meta.get('filing_type','?')} | "
            f"{meta.get('filing_date','?')} | "
            f"{meta.get('section','?')} — {meta.get('heading','?')}"
        )
        formatted.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(formatted)

def docs_to_chunks(docs: list[Document]) -> list[dict]:
    return [
        {"text": doc.page_content, "metadata": doc.metadata}
        for doc in docs
    ]

def retrieve(state: ResearchState, store: VectorStore):
    
    question = state["question"]
    retriever = store.as_retriever()
    docs = retriever.invoke(question)
    chunks = docs_to_chunks(docs)

    return {
        "retrieved_chunks": chunks,
        "retrieval_query": question
    }

def generate(state: ResearchState, llm):
    
    chunks = state["retrieved_chunks"]
    
    question = state["question"]
    context = format_retireved_into_context(chunks)

    messages = [
        SystemMessage(content=GENERATE_SYSTEM_PROMPT.format(context=context)),
        HumanMessage(content=question),
    ]

    response = llm.invoke(messages)
    return {
        "final_answer": response.content
    }
    

def build_graph():

    load_dotenv()
    builder = StateGraph(ResearchState)
    store = VectorStore()
    llm = init_chat_model(
        model=f"groq:{settings.GROQ_MODEL_LARGE}",
        temperature=0
    )

    # Node definitions
    def retrieval_node(state: ResearchState):
        return retrieve(state, store)

    def generate_node(state: ResearchState):
        return generate(state, llm)

    builder.add_node("retrieve", retrieval_node)
    builder.add_node("generate", generate_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    compiled = builder.compile()

    return compiled

if __name__ == "__main__":
    question = "What are apple's main risk factors?"
    graph = build_graph()
    result = graph.invoke({
        "question":           question,
        "plan":               "",
        "needs_qualitative":  True,
        "needs_quantitative": False,
        "retrieved_chunks":   [],
        "narrative_answer":   "",
        "retrieval_query":    "",
        "retry_count":        0,
        "quantitative_answer": "",
        "critic_feedback":    "",
        "critic_approved":    False,
        "final_answer":       "",
        "human_feedback":     "",
        "human_approved":     False,
    })

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result["final_answer"])
