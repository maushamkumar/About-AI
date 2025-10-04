```
{'input_types': {},
 'input_variables': [],

 'messages': [HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], input_types={}, partial_variables={}, template="Translate the text that is delimited by triple backticks into a style that is \nAmerican English in a calm and respectful tone \n\n\n. text: ```\nHello, My name  mausham kumar. I'm learning  Langchain. \nSo, I get an internship.\n```\n"), additional_kwargs={})],

 'metadata': None,
 'name': None,
 'optional_variables': [],
 'output_parser': None,
 'partial_variables': {},
 'tags': None,
 'validate_template': False}

 ```

# Langchain prompt template configuration 
- Each field defines how your prompt behaves, what variables it expects, and what metadata is attached. 

## 1. Input Variable 
- This tells langchain what data types each input variable should have. 
- For exampel, If your expect a `user_name` (string) and `age` (int), it might look like:

`input_types = {'user_name': str, 'age': int}`

 