import os
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from langchain.agents import initialize_agent
from tools.CircumferenceTool import CircumferenceTool
from tools.PythagorasTool import PythagorasTool


load_dotenv()

# initialize LLM (we use ChatOpenAI because we'll later define a `chat` agent)
llm = ChatOpenAI(
    temperature=0,
    model_name='gpt-3.5-turbo'
)
# initialize conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

# if we have multiple tools, one with single input and the other with multiple, we get errors, so we need to consider one tool at time
tools = [
    CircumferenceTool(),
    #PythagorasTool()
]

# initialize agent with tools
agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory
)

response=agent("can you calculate the circumference of a circle that has a radius of 7.81mm")
# > Entering new AgentExecutor chain...
# {
#     "action": "Final Answer",
#     "action_input": "The circumference of a circle with a radius of 7.81mm is approximately 49.03mm."
# }
# > Finished chain.
# {'input': 'can you calculate the circumference of a circle that has a radius of 7.81mm',
#  'chat_history': [],
#  'output': 'The circumference of a circle with a radius of 7.81mm is approximately 49.03mm.'}
print(response)
#The answer is pretty close but not quite right. We can see in the outputs above that the model decided not to use the circumference calculator tool, that is because it is (falsely) confident in it's ability to answer the question.

# the following is the system template; we need to modify this so that it can use our tools
print(agent.agent.llm_chain.prompt.messages[0].prompt.template)

# we tell the assistant that is TERRIBLE in MATH so that it can use our tool
sys_msg = """Assistant is a large language model trained by OpenAI.
Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.
Unfortunately, Assistant is terrible at maths. When provided with math questions, no matter how simple, assistant always refers to it's trusty tools and absolutely does NOT try to answer math questions by itself
Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
"""

new_prompt = agent.agent.create_prompt(
    system_message=sys_msg,
    tools=tools
)

agent.agent.llm_chain.prompt = new_prompt

agent("can you calculate the circumference of a circle that has a radius of 7.81")
# > Entering new AgentExecutor chain...
# ```json
# {
#     "action": "Circumference calculator",
#     "action_input": "7.81"
# }
# ```
# Observation: 49.071677249072565
# Thought:```json
# {
#     "action": "Final Answer",
#     "action_input": "The circumference of a circle with a radius of 7.81mm is approximately 49.07mm."
# }
# ```
# > Finished chain.
# {'input': 'can you calculate the circumference of a circle that has a radius of 7.81mm',
#  'chat_history': [HumanMessage(content='can you calculate the circumference of a circle that has a radius of 7.81mm', additional_kwargs={}),
#   AIMessage(content='The circumference of a circle with a radius of 7.81mm is approximately 49.03mm.', additional_kwargs={})],
#  'output': 'The circumference of a circle with a radius of 7.81mm is approximately 49.07mm.'}
print(response)

tools = [
    #CircumferenceTool(),
    PythagorasTool()
]

new_prompt = agent.agent.create_prompt(
    system_message=sys_msg,
    tools=tools
)

agent.agent.llm_chain.prompt = new_prompt
agent.tools = tools

agent("If I have a triangle with two sides of length 51cm and 34cm, what is the length of the hypotenuse?")
# > Entering new AgentExecutor chain...
# {
#     "action": "Hypotenuse calculator",
#     "action_input": {
#         "adjacent_side": 34,
#         "opposite_side": 51
#     }
# }
# Observation: 61.29437168288782
# Thought:{
#     "action": "Final Answer",
#     "action_input": "The length of the hypotenuse is approximately 61.29cm."
# }

# > Finished chain.
# {'input': 'If I have a triangle with two sides of length 51cm and 34cm, what is the length of the hypotenuse?',
#  'chat_history': [HumanMessage(content='can you calculate the circumference of a circle that has a radius of 7.81mm', additional_kwargs={}),
#   AIMessage(content='The circumference of a circle with a radius of 7.81mm is approximately 49.03mm.', additional_kwargs={}),
#   HumanMessage(content='can you calculate the circumference of a circle that has a radius of 7.81mm', additional_kwargs={}),
#   AIMessage(content='The circumference of a circle with a radius of 7.81mm is approximately 49.07mm.', additional_kwargs={})],
#  'output': 'The length of the hypotenuse is approximately 61.29cm.'}