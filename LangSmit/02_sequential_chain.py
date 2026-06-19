from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os 

os.environ['LANGCHAIN_PROJECT'] = "Sequential-LLM"

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

llm_model = "llama-3.1-8b-instant"
model1 = ChatGroq(
    model=llm_model,
    temperature=0,
)
model2 = ChatGroq(
    model=llm_model,
    temperature=0,
)

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

config = {
    'tags': ['llm app', 'report generation', 'summarization'],
    'metadata': {'model1': 'same model', 'model2': "doing it to learn langsmith", 'parse': 'stringoutput'}
}

result = chain.invoke({'topic': 'Unemployment in India'}, config=config)

print(result)


