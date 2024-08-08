import logging

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, MessageGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from src.chat_bot import agent, agent_instance
from src.simulated_user import customer, customer_instance
from src.constants import MAX_TURNS, TERMINATION_KEYWORD

logger = logging.getLogger(__name__)


def create_workflow(config):
    logger.info("Creating workflow")
    chat_bot = agent(config['model_name'])
    simulated_user = customer(config)

    graph = MessageGraph()

    graph.add_node("user", lambda messages: customer_instance(messages, simulated_user))
    graph.add_node("chat_bot", lambda messages: agent_instance(messages, chat_bot))

    graph.set_entry_point("chat_bot")  # Change this back to "chat_bot"

    graph.add_edge("chat_bot", "user")
    graph.add_conditional_edges(
        "user",
        conversation_decider,
        {
            "end": END,
            "continue": "chat_bot",
        },
    )

    memory = SqliteSaver.from_conn_string(config['sqlite_conn_string'])
    workflow = graph.compile(checkpointer=memory)
    logger.info("Workflow created successfully")
    return workflow


def conversation_decider(messages):
    if isinstance(messages, list) and len(messages) > 0:
        last_message = messages[-1]
        if isinstance(last_message, (AIMessage, HumanMessage)):
            if len(messages) > MAX_TURNS or last_message.content.strip().upper() == TERMINATION_KEYWORD:
                logger.info("Conversation should end")
                return "end"
    logger.debug("Conversation should continue")
    return "continue"
