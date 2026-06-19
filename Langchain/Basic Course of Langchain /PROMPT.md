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

- In our case it's `{}`, meaning this template **doesn't take any external inputs** Everything is hardcoded. 

## 2. Input Variabel 
- These are **placeholder** inside your prompt text (like `{name}` or `{text}`) that you fill when you call the template. 
- Example 
```python
template = "Translate this: {text}"
input_variables = ["text"]
```
- Yours is an empty `[]`, which means there is **there are no placeholder** the whole text static

## messages: [HumanMessagePromptTemplate(...)]
- Langchain's `ChatPromptTemplate` can hold multiple messages - e.g one from a system, one from a human, one from an AI. 
- Here, it has **one human message**, created from a PromptTemplate. 
- Inside that message
    - template -> is the actual text you provided 

    `Translate the text that is delimited by triple backticks ...`

    - `input_variables` inside this prompt → also empty (`[]`), confirming it’s fully static.

## Metadata: None
 - Optional field to attach extra into (like a source, author, or description)
 - You can use it for tracking prompts in production (e.g., `"metadata": {"version": 2, "topic": "translation"}`).

## Name: None 
- You can give a prompt a name (e.g., `"translation_prompt"`).
- It helps if you manage multiple prompts or chains 

## Optional_variable: []
- Variable that are optional to fill - Langchain won't throw an error if they're missing. 
- Yours is empty, meaning every variable (if any existed) would be required. 


## output_parser: None 
- Determines how the model's output should be parsed. 
- Example: Extracting a JSON field, or splitting lines into list elements. 
- You don't have one defined - so raw text is returned. 

## partial_variables: {}
- These are **pre-filled variables** that are locked in advance.
- Think of it as giving default values:

```python
partial_variables = {"language": "English"}
```
- Yours is empty — everything (if it existed) would need to be passed when you call the prompt.

## tags: None 
- tags are used for tracking or categorizing prompts (like "translation", "debug", or "internal").

- None here, so this prompt isn’t tagged.

## validate_template: False
- When True, LangChain checks your template for missing variables or syntax issues.

- It’s False here, meaning validation is skipped.