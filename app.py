import json
import logging
from langchain_core.messages import HumanMessage, AIMessage
from src.graph import create_workflow
from src.utils import setup_logging
from src.constants import CONFIG_PATH, LOGGING_CONFIG


def main():
    # Setup logging
    setup_logging(LOGGING_CONFIG)
    logger = logging.getLogger(__name__)

    logger.info("Starting the application")

    # Load configuration
    try:
        with open(CONFIG_PATH, 'r') as config_file:
            config = json.load(config_file)
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        return

    try:
        workflow = create_workflow(config)
        logger.info("Workflow created successfully")
        print("-" * 50)
        initial_message = [HumanMessage(content=config['initial_message'][0]['content'])]
        print(f"Customer: {initial_message}")
        print("-" * 50)
        for output in workflow.stream(initial_message, config['graph_config']):
            if output and 'end' not in output:
                for key, message in output.items():
                    if isinstance(message, (HumanMessage, AIMessage)):
                        sender = "Customer" if key == "user" else "Agent"
                        logger.info(f"{sender}: {message.content}")
                        print(f"{sender}: {message.content}")
                        print("-" * 50)
            elif 'end' in output:
                logger.info("Conversation ended")
                print("Conversation ended")
                break
    except Exception as e:
        logger.error(f"An error occurred during workflow execution: {str(e)}")
        print(f"An error occurred: {str(e)}")

    logger.info("Conversation completed")
    print("Conversation completed")


if __name__ == "__main__":
    main()