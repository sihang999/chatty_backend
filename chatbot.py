import os
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from update_index import get_verctor_store
# -----------------------------------------------------------------------------------------

def model_call(input_message: str, thread_id: str):

    # -----------------------------------------------------------------------------------------
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    vector_store = get_verctor_store()
    # -----------------------------------------------------------------------------------------
    #store conversations in local DB
    import sqlite3
    os.makedirs("state_db", exist_ok=True)
    db_path = "state_db/example.db"
    conn = sqlite3.connect(db_path, check_same_thread = False)
    memory = SqliteSaver(conn)
    # -----------------------------------------------------------------------------------------

    # ExtendedMessagesState based MessageState
    class State(MessagesState):
        summary: str
    # -----------------------------------------------------------------------------------------

    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        """Retrieve information related to a query."""
        # Retrieve relevant documents
        docs = vector_store.similarity_search(query, k=3)
        # retrieved_docs = vector_store.similarity_search(query, k=2)
        retrieved_docs = docs
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    # -----------------------------------------------------------------------------------------

    # Step 1/Node 1: Generate an AIMessage that may include a tool-call to be sent.
    def query_or_respond(state: State):
        # Get summary if it exists

        summary = state.get("summary", "")
        system_message = ("You are an assistant representing Sihang Wu. "
            "For questions unrelated to Sihang Wu, politely redirect the focus back to topics related to him. "
            "If current context lacks sufficient information, you must invoking a tool again."
        )
        # If there is summary, then we add it
        if summary:
            
            # Add summary to system message
            system_message = ("You are an assistant representing Sihang Wu. "
                            f"And this is the summary of conversation earlier: {summary} "
                            "\n"
                            "For questions unrelated to Sihang Wu, politely redirect the focus back to topics related to him. "
                            "\n"
                            "If current context lacks sufficient information,, you must invoking a tool again."
            )
            # Append summary to any newer messages
            messages = [SystemMessage(system_message)] + state["messages"]

        else:
            messages = [SystemMessage(system_message)] + state["messages"]


        """Generate tool call for retrieval or respond."""
        llm_with_tools = llm.bind_tools([retrieve])
        response = llm_with_tools.invoke(messages)
        # MessagesState appends messages to state instead of overwriting
        return {"messages": [response]}

    # -----------------------------------------------------------------------------------------

    # Step 2/Node 2: Execute the retrieval.
    tools = ToolNode([retrieve])

    # -----------------------------------------------------------------------------------------
    # Step 3/Node 3: Generate a response using the retrieved content.
    def generate(state: State):
        """Generate answer."""
        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
            #get the last tool message
        tool_messages = recent_tool_messages[::-1]

        # Format into prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages)

        #get summary
        summary = state.get("summary", "")
        # If there is summary, then we add it
        if summary:
            # Add summary to system message
            system_message_summary = f"Summary of conversation earlier: {summary}"
            system_message_content = (
                "You are an assistant for respresenting Sihang Wu. "
                "Use the following pieces of Retrieved context and summary of conversation earlier to answer "
                "the questionn in a concise and organized manner. If you don't know the answer, say that you "
                "don't know."
                "\n\n"
                f"Retrieved context written by Sihang Wu: {docs_content}"
                "\n\n"
                f"{system_message_summary}"
            )
        else :
            system_message_content = (
            "You are an assistant for respresenting Sihang Wu. "
            "Use the following pieces of retrieved context to answer "
            "the question in a concise and organized manner. If you don't know the answer, say that you "
            "don't know."
            "\n\n"
            f"{docs_content}"
        )

        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        # Run
        response = llm.invoke(prompt)

        # return the last AImesaage（not list)，graph will add it to the state['messages']
        return {"messages": [response]}


    # -----------------------------------------------------------------------------------------

    # Step 4/Node 4: Execute the retrieval.
    def summarize_conversation(state: State):
        
        # check the conversation length
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human")
            or (message.type == "ai" and not message.tool_calls)
        ]
        # check the tools messages length
        recent_tool_messages = []
        recent_tool_call_messages = []
        for message in state["messages"]:
            if message.type == "ai" and message.tool_calls:
                recent_tool_call_messages.append(message)
        for message in state["messages"]:
            if message.type == "tool":
                recent_tool_messages.append(message)

        # First, we get any existing summary
        summary = state.get("summary", "")

        # If there are more than six messages, then we summarize the conversation
        if len(conversation_messages) > 6:

            # Create our summarization prompt 
            if summary:
                
                # A summary already exists
                summary_message = (
                    f"This is summary of the conversation to date: {summary}\n\n"
                    "Extend the summary by taking into account the new messages above:"
                )
                
            else:
                summary_message = "Summarize the conversation above in one concise paragraph:"

            messages = conversation_messages + [HumanMessage(content=summary_message)]
            # get the new summary
            response = llm.invoke(messages)

            # Delete all but the 2 most recent messages and 1 too message
            if len(recent_tool_messages) > 1:
                delete_messages = [RemoveMessage(id=m.id) for m in conversation_messages[:-2]] + [RemoveMessage(id=m.id) for m in recent_tool_call_messages[:-1]] + [RemoveMessage(id=m.id) for m in recent_tool_messages[:-1]]
            else :
                delete_messages = [RemoveMessage(id=m.id) for m in conversation_messages[:-2]]
            return {"summary": response.content, "messages": delete_messages}
        
        # only keep 1 tool message
        elif len(recent_tool_messages) > 1:
            delete_messages = [RemoveMessage(id=m.id) for m in recent_tool_call_messages[:-1]] + [RemoveMessage(id=m.id) for m in recent_tool_messages[:-1]]
            return {"messages": delete_messages}
        else:
            # nothing is changed
            return {"summary": summary}
    # -----------------------------------------------------------------------------------------

    # Determine whether to end or summarize or tool the conversation/conditional edge
    def should_summarize_or_tool_or_end(state: State):
        """Return the next node to execute."""

        # messages = state["messages"]
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human")
            or (message.type == "ai" and not message.tool_calls)
        ]

        # first check if the last message is AI tool calls, if it is just directly call tool
        last_message = state["messages"][-1]
        if last_message.type == "ai" and last_message.tool_calls:
            result = tools_condition(state)
            return result
            
        # If there are more than six messages, then we summarize the conversation
        if len(conversation_messages) > 6:
            #这里的if-else决定返回到哪一个node
            return "summarize_conversation"
        else:
            # print(state)
            result = tools_condition(state)
            return result

    # -----------------------------------------------------------------------------------------

    graph_builder = StateGraph(State)

    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)
    graph_builder.add_node(summarize_conversation)


    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        should_summarize_or_tool_or_end,
        {END: END, "tools": "tools","summarize_conversation": "summarize_conversation"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("summarize_conversation", END)

    graph_builder.add_edge("generate", "summarize_conversation")

    graph = graph_builder.compile(checkpointer=memory)

    # # create the graph picture
    # image_data = graph.get_graph().draw_mermaid_png()
    # with open("graph_image.png", "wb") as f:
    #     f.write(image_data)
    # print("already stores graph_image.png")

    # -----------------------------------------------------------------------------------------
    for step in graph.stream(
        {"messages": [{"role": "user", "content": input_message}]},
        stream_mode="values",
        config = {"configurable": {"thread_id": f"{thread_id}"}}
    ):
        final_response=step["messages"][-1].content
        # print(step[0].content, end="|")

    # just print the final message content
    return final_response
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# 我现在的graph最后会输出两个AI message（因为summarize这个节点的增加），但是没有关系，并不影响速度。
# stream_mode="updates",有两种，updates只会在输入内容之后，输出后面调用的节点及相应的内容。很难定位到最后的AI结果
# values 会在每个节点之后，返回整个state，和invoke一样，拿最后一个就是AI的结果
# messages 会只会输出过程中的整个AI message，不同步骤的ai messages都会被输出在一起。
