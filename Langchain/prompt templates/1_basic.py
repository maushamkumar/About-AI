from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate
# Load environment variables from .env
load_dotenv()

# Create a Groq llama3-8b-8192 model 
model = ChatGroq(model="llama3-8b-8192")

# Part -1 Create a ChatPromptTemplate using a template string
# template = "Tell me a joke about {topic}"
# prompt_template = ChatPromptTemplate.from_template(template)

# print("----Prompt from Template----")
# prompt_template = ChatPromptTemplate.invoke({"topic": "cats"})
# print


# PART-2: Prompt with Multiple Placeholders 
template_multiple = """You are a helpful assistant. 
Humman: Tell me a {adjective} story about a {animal}. 
Assistant:"""

prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
prompt = prompt_multiple.invoke({'adjective': "funny", "animal": "panda"})
print("\n--- Prompt with Multiple Placeholder ---\n")
print(prompt)


# PART 3: Prompt with System and Human Messages (Using Tuples)
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."), 
    ("Human", "Tell me {joke_count} jokes.")
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
print("\n---- Prompt with System and Humman Messages (Tuple) ----\n")
print(prompt)


# Extra Information about part 3. 
# This does work: 
messages = [
    ("system", "you are a comedian who tells jokes about {topic}."), 
    HumanMessage(content="Tell me 3 jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers"})
print("\n---- Prompt with System and Human Messages (Tuple) ---\n")
print(prompt)