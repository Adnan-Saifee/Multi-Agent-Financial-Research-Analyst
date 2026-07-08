from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, ToolMessage, HumanMessage

from typing import Optional

from analyst.retrieval.vector_store import VectorStore
from analyst.config import settings
from analyst.log import print_verbose
from analyst.graph.state import ResearchState

from dotenv import load_dotenv

ESCALATION_SIGNAL = "ESCALATE_TO_QUANTITATIVE"
# MAX_ITERATIONS = 20

QUALITATIVE_AGENT_PROMPT = """You are a financial research analyst specializing in narrative 
analysis of SEC filings (10-K and 10-Q). Your job is to find and synthesize relevant 
narrative information to answer questions about companies.
 
You have access to tools to search through SEC filing text. Use them strategically:
 
1. Start with search_filings to get context to answer a given question, and 
   include relevant tickers or sections mentioned in the query as arguments.
2. After retrieving chunks (context), use grade_chunks to evaluate if they are sufficient (returns True) or insufficient (returns False).
3. If chunks are insufficient, try to carefully edit the search query, still keeping the relevant information
   in the original question intact, and use this with search_filings to get better context. 
4. Once you have sufficient chunks, write a comprehensive answer with inline citations.
 
Citation format — after every claim write:
[TICKER | FILING_TYPE | DATE | SECTION — HEADING]
Example: [AAPL | 10-K | 2023-09-30 | mda — Liquidity and Capital Resources]
 
Important rules:
- Only use information from retrieved chunks — never use outside knowledge
- If after multiple searches you cannot find enough information, say so explicitly
- If the question is purely about specific financial figures/numbers with no narrative 
   component, use escalate_to_quantitative (only available in qualitative-only mode)
- Be thorough but concise — answer what was asked, nothing more"""


def make_tools(
    store: VectorStore,
    detected_tickers: list[str],
    include_escalate_quantitative: bool,
    grading_llm
):
    tools = []

    @tool
    def search_filings(query: str, tickers: Optional[list[str]] = None, section: Optional[str] = None) -> str:
        """
        Used to get context from SEC filings to answer questions about different companies.
        The data is extracted from MD&A and Risk Factor sections of 10-K and 10-Q Filings.

        Args:
            query: The query used to retrieve context using Semantic Similarity searching.
            tickers: Optional list of ticker symbols to scope search e.g. ["AAPL", "MSFT"].
            section: Optional section filter e.g. "mda" or "risk_factors"
        """
        effective_tickers = tickers or detected_tickers

        filters = {}
        if effective_tickers:
            if len(effective_tickers) == 1:
                filters["ticker"] = effective_tickers[0]
            else:
                filters["$or"] = [{"ticker": t} for t in effective_tickers]
        
        if section:
            if filters:
                filters = {"$and": [filters, {"section": section}]}
            else:
                filters = {"section": section}
        
        print_verbose(search_filings, message=f"{filters}")

        chunks = store.query(query, 3, filters=filters if filters else None)

        if not chunks:
            print_verbose(search_filings, error=True, message="No chunks were retrieved")
            return "No relevant chunks of data were retrieved."
        
        formatted = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk["metadata"]
            header = (
                f"[{i}] {meta.get('ticker','?')} | "
                f"{meta.get('filing_type','?')} | "
                f"{meta.get('filing_date','?')} | "
                f"{meta.get('section','?')} — {meta.get('heading','?')} "
                f"(score: {chunk['score']:.3f})"
            )
            formatted.append(f"{header}\n{chunk['text']}")
            
        print_verbose(search_filings, message=f"{"\n\n---\n\n".join(formatted)}")

        return "\n\n---\n\n".join(formatted)
    
    @tool
    def grade_chunks(question: str, context: str) -> str:
        """
        Evaluate whether the retrieved chunks are sufficient to answer the question.
        Returns SUFFICIENT if the context is relevant, INSUFFICIENT if not.

        Args:
            question: The original research question asked by the user
            context: The context (retrieved chunks) from search_filings to be graded
        """
        
        prompt = """You are an expert grader that analyzes whether given context is relevant to a question.
        Grade leniently — if the context contains any useful information toward answering the question, grade it as sufficient.

        Respond ONLY with a clear ONE-WORD answer, either with TRUE or FALSE:
        If suffcient answer with: "TRUE"
        If insuffcient answer with: "FALSE"

        Question:
        {question}

        Context:
        {context}"""

        response = grading_llm.invoke([
            HumanMessage(content=prompt.format(question=question, context=context))
        ])

        if "true" in response.content.lower():
            print_verbose(grade_chunks, local_verbose=True, message="Retrieved chunks PASSED relevancy test.")
            return "SUFFICIENT: The retrieved context contains relevant information. Proceed to write your final answer."
        else:
            print_verbose(grade_chunks, error=True, message="Retrieved chunks FAILED relevancy test.")
            return "INSUFFICIENT: The retrieved context does not adequately answer the question. Try a different search query with search_filings."
    
    tools = [search_filings, grade_chunks]

    if include_escalate_quantitative:
        @tool
        def escalate_to_quantitative(reason: str) -> str:
            """
            Signal that this question requires exact financial figures from tables, NOT narrative text.
 
            Args:
                reason: Brief explanation of why quantitative data is needed
            """
            return f"{ESCALATION_SIGNAL}: {reason}"

        tools.append(escalate_to_quantitative)
    
    return tools


def qualitative_node(question: str, needs_quant: bool, llm, store: VectorStore, detected_tickers: list[str] | None = None):
    escalate_to_quant = not needs_quant
    tools = make_tools(store, detected_tickers or [], escalate_to_quant, grading_llm=llm)

    agent = create_agent(
        llm,
        tools=tools,
        system_prompt=QUALITATIVE_AGENT_PROMPT
    )

    agent_question = question
    if detected_tickers:
        # Fixed the nested double quote syntax error here
        agent_question = f"{question}\n\nDetected Tickers: {', '.join(detected_tickers)}"
    
    response = agent.invoke(
        {"messages": [{"role": "user", "content": agent_question}]}
    )

    messages = response.get("messages", [])
    escalated = False

    # Check ToolMessages or AIMessage contents for the escalation string
    for message in messages:
        if isinstance(message, ToolMessage) and ESCALATION_SIGNAL in str(message.content):
            print_verbose(qualitative_node, "Escalating to Quantitative Agent", True)
            escalated = True
            break
    
    if escalated:
        return {
            "needs_quantitative": True,
            "needs_qualitative": False,
            "narrative_answer": "",
            "retrieved_chunks": []
        }
    
    # Get latest message that is not a tool call execution
    final_answer = ""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content and not getattr(message, "tool_calls", None):
            final_answer = message.content
            break

    if not final_answer:
        final_answer = "The available filings do not contain sufficient information to answer this question."
        print_verbose(qualitative_node, error=True, message="[QualitativeAgent] No final answer found in agent messages.")
        
    return {
        "narrative_answer": final_answer,
        "retrieved_chunks": []
    }


if __name__ == "__main__":
    load_dotenv()
    from pprint import pprint
    
    # question = "Why did Microsoft's cloud revenue grow so fast in 2025?"
    question = "What was Apple's gross margin in fiscal year 2023?"

    # Initialize your LLM config
    llm = init_chat_model(model="openai/gpt-oss-120b", model_provider="groq")
    grading_llm = init_chat_model(model="openai/gpt-oss-20b", model_provider="groq")
    store = VectorStore()
    response = qualitative_node(question=question, needs_quant=False, llm=grading_llm, store=store)
    pprint(response)