import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)


def customer(config):
    logger.info("Creating simulated user")
    system_prompt_template = config['system_prompt_template']
    instructions = config['user_instructions']
    model_name = config['model_name']

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_template),
        MessagesPlaceholder(variable_name="messages"),
    ]).partial(name=config['user_name'], instructions=instructions)

    model = Ollama(model=model_name)
    simulated_user = prompt | model
    logger.info("Simulated user created successfully")
    return simulated_user


def customer_instance(messages, customer_object):
    logger.debug(f"Simulated user node processing messages: {messages}")
    new_messages = role_decider(messages)
    response = customer_object.invoke({"messages": new_messages})
    if isinstance(response, str):
        return HumanMessage(content=response)
    else:
        return HumanMessage(content=response.content)


def role_decider(messages):
    new_messages = []
    for m in messages:
        if isinstance(m, AIMessage):
            new_messages.append(HumanMessage(content=m.content))
        else:
            new_messages.append(AIMessage(content=m.content))
    return new_messages
