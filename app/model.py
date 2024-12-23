from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, RemoveMessage
import app.routes

# set openai API env on you local computer, it will be useless if you host the app in the server
# from dotenv import load_dotenv
# load_dotenv()


class State(MessagesState):
    summary: str
model = ChatOpenAI(model="gpt-4o", temperature=0.5)


def get_langgraph():
    def call_model(state: State):
        # Get summary if it exists
        summary = state.get("summary", "")
        # If there is summary, then we add it
        if summary:
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        f"{app.routes.system_prompt_fmt}. Summary: {summary}",
                    ),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            # Append summary to any newer messages
            message_prompt = prompt_template.invoke(state)
        else:
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        f"{app.routes.system_prompt_fmt}",
                    ),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            # Append summary to any newer messages
            message_prompt = prompt_template.invoke(state)
        # print("---------------------------input----------------------------------\n")
        # print(message_prompt)
        response = model.invoke(message_prompt)
        return {"messages": response}

    # Define the logic to how to summarize conversation, which will be used for a "summarize_conversation" node
    def summarize_conversation(state: State):
        summary = state.get("summary", "")
        # Create our summarization prompt, completed by  openai
        if summary:
            # A summary already exists
            summary_message = (
                f"This is summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )
        else:
            summary_message = "Create a summary of the conversation above:"
        # Add prompt to our history
        messages = state["messages"] + [HumanMessage(content=summary_message)]
        response = model.invoke(messages)
        # Delete all but the 2 most recent messages
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        return {"summary": response.content, "messages": delete_messages}

    # Determine whether to end or summarize the conversation/conditional edge
    def should_continue(state: State):
        """Return the next node to execute."""
        messages = state["messages"]
        # If there are more than six messages, then we summarize the conversation
        if len(messages) > 6:
            return "summarize_conversation"
        # Otherwise we can just end
        return END

    # Define a new graph
    workflow = StateGraph(State)
    workflow.add_node("conversation", call_model)
    workflow.add_node(summarize_conversation)

    # Set the entrypoint as conversation
    workflow.add_edge(START, "conversation")
    workflow.add_conditional_edges("conversation", should_continue)
    workflow.add_edge("summarize_conversation", END)

    # Compile
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    return graph

graph = get_langgraph()