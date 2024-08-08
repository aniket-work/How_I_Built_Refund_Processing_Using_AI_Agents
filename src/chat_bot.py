import logging
from typing import List
from langchain_community.llms import Ollama
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


def agent(model_name: str):
    logger.info(f"Creating chat bot with model: {model_name}")
    return ChatBot(model_name)


class ChatBot:
    def __init__(self, model_name: str):
        self.llm = Ollama(model=model_name)
        logger.info(f"ChatBot initialized with model: {model_name}")

    def generate_response(self, messages: List[dict]) -> AIMessage:
        logger.debug(f"Generating response for messages: {messages}")
        system_message = {
            "role": "system",
            "content": "You are a customer_object support agent for a product company.",
        }
        messages = [system_message] + messages

        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        response = self.llm.invoke(prompt)
        # logger.info(f"Generated response: {response}")
        return AIMessage(content=response)


def agent_instance(messages, chat_bot):
    logger.debug(f"Chat bot node processing messages: {messages}")
    agent_response = chat_bot.generate_response([{"role": m.type, "content": m.content} for m in messages])
    return agent_response
