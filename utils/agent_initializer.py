
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

sys_msg = """Assistant is a smart program created by OpenAI to help with many different tasks. It can answer simple questions or dive deep into more complex topics. Since it's a language model, it understands the text it gets and replies in a way that sounds natural and makes sense.

Assistant keeps learning and improving all the time. It can handle a lot of information and give accurate answers. It can also create its own responses based on what you ask, making it great for explanations and discussions.

In short, Assistant is a useful tool for answering questions and having conversations on a wide range of subjects. Whether you need specific information or just want to chat, it's ready to help."""



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
