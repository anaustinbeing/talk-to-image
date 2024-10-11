
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

sys_msg = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
"""



def initialize_agent_with_tools(tools, openai_api_key):
    """Initializes the agent with tools and memory."""
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0, model_name='gpt-4o-mini')
    
    # Conversational memory
    memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
    )

    # Initialize agent with tools and memory
    agent = initialize_agent(
        agent='chat-conversational-react-description',
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=3,
        early_stopping_method='generate',
        memory=memory
    )

    new_prompt = agent.agent.create_prompt(
        system_message=sys_msg,
        tools=tools
    )

    agent.agent.llm_chain.prompt = new_prompt
    return agent
