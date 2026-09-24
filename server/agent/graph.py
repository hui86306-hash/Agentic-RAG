from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)

from server.agent.state import AgentState



def build_agent(model_name: str, dashscope_api_key: str, tools):

    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=dashscope_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    llm_with_tools = llm.bind_tools(tools)

    SYSTEM_PROMPT = """
    You are an agentic RAG assistant.

    Rules:

    1. The uploaded PDF is the primary knowledge source.
    2. The PDF retrieval result will be provided before answering the user's question.
    3. Answer questions about the uploaded PDF based only on the retrieved content.
    4. Do not claim that you cannot access the uploaded PDF.
    5. Do not invent information that is not supported by the retrieved content.
    6. When answering based on the PDF, always cite the source page number in the format:
       [Source: page X]
    7. Only cite page numbers that actually appear in the retrieved PDF results.
    8. If the retrieved PDF content is insufficient to answer the question, clearly say:
       "根据当前 PDF 内容，无法找到足够的信息回答这个问题。"
    9. Do not guess or fabricate missing information or source pages.
    10. Web or arXiv tools may be used when additional information is needed.
    """

    # 找到 PDF 检索工具
    pdf_tool = next(
        (t for t in tools if t.name == "search_pdf"),
        None
    )

    # --------------------------------
    # 1. 强制 PDF 检索节点
    # --------------------------------
    def pdf_retrieve_node(state: AgentState):

        messages = state["messages"]

        # 找到当前最新的用户问题
        user_message = None

        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                user_message = message
                break

        if user_message is None or pdf_tool is None:
            return {"messages": []}

        query = user_message.content

        # 直接执行 search_pdf
        result = pdf_tool.invoke(query)

        # 构造一个工具调用结果
        tool_call_id = "forced_pdf_search"

        tool_call_message = ToolMessage(
            content=result,
            tool_call_id=tool_call_id,
            name="search_pdf",
        )

        return {
            "messages": [
                tool_call_message
            ]
        }

    # --------------------------------
    # 2. LLM 节点
    # --------------------------------
    def llm_node(state: AgentState):

        messages = state["messages"]

        # 添加系统提示词
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [
                SystemMessage(content=SYSTEM_PROMPT)
            ] + messages

        response = llm_with_tools.invoke(messages)

        return {
            "messages": [response]
        }

    # --------------------------------
    # 3. 创建 Graph
    # --------------------------------
    graph = StateGraph(AgentState)

    graph.add_node(
        "pdf_retrieve",
        pdf_retrieve_node
    )

    graph.add_node(
        "llm",
        llm_node
    )

    graph.add_node(
        "tools",
        ToolNode(tools)
    )

    # 用户进入后：
    # 先强制 PDF 检索
    graph.set_entry_point("pdf_retrieve")

    # PDF 检索完成后交给 LLM
    graph.add_edge(
        "pdf_retrieve",
        "llm"
    )

    # LLM 如果主动要求其他工具
    graph.add_conditional_edges(
        "llm",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )

    # 其他工具执行完，再回 LLM
    graph.add_edge(
        "tools",
        "llm"
    )

    return graph.compile()