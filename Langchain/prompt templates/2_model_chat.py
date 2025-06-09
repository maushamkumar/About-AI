from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
# Load environment variables from .env
load_dotenv()

# Create a Groq llama3-8b-8192 model 
model = ChatGroq(model="llama3-8b-8192")

# PART 1: Create a ChatPromptTemplate using a template string 
# print("----Prompt from Template----")
# template = "Tell me a joke about {topic}."
# prompt_template = ChatPromptTemplate.from_template(template)

# prompt = prompt_template.invoke({"topic" : "cats"})
# result = model.invoke(prompt)
# print(result.content)


# PART-2: Prompt with Multiple Placeholders 
# template_multiple = """You are a helpful assistant. 
# Humman: Tell me a {adjective} story about a {animal}. 
# Assistant:"""

# prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
# prompt = prompt_multiple.invoke({'adjective': "funny", "animal": "panda"})

# result = model.invoke(prompt)
# print(result.content)

# PART 3: Prompt with System and Human Messages (Using Tuples)
print("\n --- Prompt with System and Human Messages (Tuples) ---\n")
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."), 
    ("human", "Tell me {joke_count} jokes.")
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})

result = model.invoke(prompt)
print(result.content)