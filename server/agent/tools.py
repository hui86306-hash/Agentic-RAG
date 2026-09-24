from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.retrievers import ArxivRetriever
import os
import dashscope


def format_pdf_hits(docs):
    lines = ["PDF RAG Results:"]

    for i, d in enumerate(docs, 1):
        snippet = d.page_content.replace("\n", " ")[:400]
        lines.append(
            f"{i}. (page {d.metadata.get('page', 0) + 1}) {snippet}"
        )

    return "\n".join(lines)


def rerank_documents(query, documents, top_n=3):
    if not documents:
        return []

    texts = [doc.page_content for doc in documents]

    response = dashscope.TextReRank.call(
        model="gte-rerank-v2",
        query=query,
        documents=texts,
        top_n=top_n,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"DashScope rerank failed: "
            f"{response.code} - {response.message}"
        )

    results = response.output.results

    reranked_docs = []

    for result in results:
        index = result["index"]
        reranked_docs.append(documents[index])

    return reranked_docs


def make_pdf_tool(retriever):
    @tool("search_pdf")
    def search_pdf(query: str) -> str:
        """Search the uploaded PDF, rerank results, and return the most relevant chunks."""

        # 第一阶段：向量检索 Top 10
        hits = retriever.invoke(query)

        if not hits:
            return "PDF RAG Results:\nNo relevant content found."

        # 第二阶段：DashScope Rerank Top 3
        reranked_docs = rerank_documents(
            query=query,
            documents=hits,
            top_n=3,
        )

        return format_pdf_hits(reranked_docs)

    return search_pdf


def make_web_tool(serper_api_key: str):
    serper = GoogleSerperAPIWrapper(api_key=serper_api_key)

    @tool("search_web")
    def search_web(query: str) -> str:
        """Search the web using Serper."""

        results = serper.results(query)
        organic = results.get("organic", [])

        out = ["Web Search Results:"]

        for r in organic[:5]:
            out.append(
                f"- {r.get('title')}: {r.get('snippet')}"
            )

        return "\n".join(out)


    return search_web


def make_arxiv_tool():
    arxiv = ArxivRetriever(max_results=3)

    @tool("search_arxiv")
    def search_arxiv(query: str) -> str:
        """Search arXiv for scientific papers."""

        papers = arxiv.invoke(query)

        out = ["arXiv Results:"]

        for p in papers:
            out.append(
                p.metadata.get("title", "")
            )

        return "\n".join(out)


    return search_arxiv


def build_tools(retriever):
    tools = []

    if retriever:
        tools.append(
            make_pdf_tool(retriever)
        )

    tools.append(make_arxiv_tool())

    return tools