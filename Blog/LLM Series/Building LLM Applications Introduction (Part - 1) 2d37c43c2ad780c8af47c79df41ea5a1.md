# Building LLM Applications: Introduction (Part - 1)

Created: December 24, 2025 6:46 PM

![Preview image](https://miro.medium.com/v2/resize:fit:700/0*JnLO2hzuwwd9_Q3i.png)

# **Building LLM Applications: Introduction (Part 1)**

## **Learn Large Language Models ( LLM ) through the lens of a Retrieval Augmented Generation ( RAG ) Application.**

![Vipra Singh](https://miro.medium.com/v2/resize:fill:88:88/1*LDjQS3c-G1gsojOf24ijGg@2x.jpeg)

[**Vipra Singh**Follow](https://medium.com/@vipra_singh)

androidstudio·January 9, 2024 (Updated: March 17, 2024)·Free: Yes

### **Posts in this series**

1. [***Introduction](https://medium.com/@vipra_singh/building-llm-applications-introduction-part-1-1c90294b155b#4d28) ( This Post )***
2. [***Data Preparation***](https://medium.com/@vipra_singh/building-llm-applications-data-preparation-part-2-b7306d224245)
3. [***Sentence Transformers***](https://medium.com/@vipra_singh/building-llm-applications-sentence-transformers-part-3-a9e2529f99c1)
4. [***Vector Database***](https://medium.com/@vipra_singh/building-llm-applications-vector-database-part-4-2bb29e7c798d)
5. [***Search & Retrieval***](https://medium.com/@vipra_singh/building-llm-applications-retrieval-search-part-5-c83a7004037d)
6. [***LLM***](https://medium.com/@vipra_singh/building-llm-applications-large-language-models-part-6-ea8bd982bdee)
7. [***Open-Source Chatbots***](https://medium.com/@vipra_singh/building-llm-applications-open-source-chatbots-part-7-1ca9c3653175)
8. ***Evaluation***
9. ***Fine-Tuning Embedding Models***
10. ***Fine-Tuning LLMs***
11. ***Serving LLMs***

### **Table of Contents**

**· [Posts in this series](https://freedium.cfd/https://medium.com/@vipra_singh/building-llm-applications-introduction-part-1-1c90294b155b#7e9b) · [What is Retrieval Augmented Generation?](https://freedium.cfd/https://medium.com/@vipra_singh/building-llm-applications-introduction-part-1-1c90294b155b#4d28) · [Why RAG?](https://freedium.cfd/https://medium.com/@vipra_singh/building-llm-applications-introduction-part-1-1c90294b155b#4abe) · [High-Level RAG Architecture](https://freedium.cfd/https://medium.com/@vipra_singh/building-llm-applications-introduction-part-1-1c90294b155b#0bb7) · [Wrap-Up](https://freedium.cfd/https://medium.com/@vipra_singh/building-llm-applications-introduction-part-1-1c90294b155b#36d4)**

![None](https://miro.medium.com/v2/resize:fit:700/0*JnLO2hzuwwd9_Q3i.png)

[RAG Architecture](https://gradientflow.substack.com/p/best-practices-in-retrieval-augmented)

Greetings!

In my recent exploration of Language Model (LLM) applications, I've been captivated by the significant role of Retrieval Augmented Generation (RAG). Understanding the end-to-end RAG architecture, from its conceptualization to its deployment over the cloud, can be quite challenging.

To address this, I'm thrilled to announce a forthcoming series of detailed blogs where I will unravel the intricacies of LLM through the lens of RAG application, providing a comprehensive understanding of each stage of the RAG pipeline along with practical, hands-on experience.

I aim to make the complex world of LLM more accessible to everyone, ensuring that each stage is covered in detail.

Join me on this enlightening journey as we delve into the potential of LLM applications through the comprehensive study of RAG.

### [**What is Retrieval Augmented Generation?**](https://llmstack.ai/blog/retrieval-augmented-generation)

If you have been looking up data in a vector store or some other database and passing relevant info to LLM as context when generating output, you are already doing retrieval augmented generation. Retrieval augmented generation or RAG for short is the architecture [popularized by Meta](https://ai.meta.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/) in 2020 that aims to improve the performance of LLMs by passing relevant information to the model along with the question/task details.

### [**Why RAG?**](https://llmstack.ai/blog/retrieval-augmented-generation)

LLMs are trained on large corpora of data and can answer any questions or complete tasks using their parameterized memory. These models have knowledge cutoff dates depending on when they were last trained. When asked a question out of its knowledge base or about events that happened after the knowledge cutoff date, there is a high chance that the model will hallucinate. Researchers at Meta discovered that by [providing relevant information about the task at hand](https://ai.meta.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/), the model's performance at completing the task improves significantly.

For example, if the model is being asked about an event that happened after the cutoff date, providing information about this event as context and then asking the question will help the model answer the question correctly. Because of the limited context window length of LLMs, we can only pass the most relevant knowledge for the task at hand. The quality of the data we add in the context influences the quality of the response that the model generates. There are multiple techniques that ML practitioners use in different stages of an RAG pipeline to improve LLM's performance.

### **High-Level RAG Architecture**

LangChain has an example of [RAG](https://python.langchain.com/docs/use_cases/question_answering/#quickstart) in its smallest (but not simplest) form:

A typical RAG application has two main components:

**Indexing**: A pipeline for ingesting data from a source and indexing it. This usually happens offline.

**Retrieval and Generation**: The actual RAG chain, which takes the user query at run time and retrieves the relevant data from the index, then passes that to the model.

The most common full sequence from raw data to answer looks like this:

**Indexing**

1. **Load:** First we need to load our data. This is done with DocumentLoaders.
2. **Split:** Text splitters break large Documents into smaller chunks. This is useful both for indexing data and for passing it into a model, since large chunks are harder to search over and won't fit in a model's finite context window.
3. **Store:** We need somewhere to store and index our splits so that they can later be searched. This is often done using a VectorStore and Embeddings model.

[None](https://miro.medium.com/v2/resize:fit:700/0*14W5qzX5b42-xouB)

[RAG](https://python.langchain.com/docs/use_cases/question_answering/#quickstart)

**Retrieval**: Given a user input, relevant splits are retrieved from storage using a Retriever.

**Generation**: A ChatModel / LLM produces an answer using a prompt that includes the question and the retrieved data

[None](https://miro.medium.com/v2/resize:fit:700/0*mw1Hy3iOZeRZyH2A)

[Retrieval and Generation](https://python.langchain.com/docs/use_cases/question_answering/#quickstart)

```jsx
Copyfrom langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator
loader = WebBaseLoader("https://www.promptingguide.ai/techniques/rag")
index = VectorstoreIndexCreator().from_loaders([loader])
index.query("What is RAG?")
```

With these five lines, we get a description of RAG, but the code is heavily abstracted, so it's difficult to understand what's happening: We fetch the contents of a web page (our knowledge base for this example).

1. We process the source contents and store them in a knowledge base (in this case, a vector database).
2. We input a prompt, LangChain finds bits of information from the knowledge base and passes both prompt and knowledge base results to the LLM.

While this script is helpful for prototyping and understanding the main beats of using RAG, it's not all that useful for moving beyond that stage because you don't have much control. Let's discuss what goes into implementation.

### **Wrap-Up**

In this first part of the LLM Apps series, we dissected the vital role of Retrieval Augmented Generation (RAG) in enhancing Large Language Models (LLMs). From understanding the motivation behind RAG to exploring the components of an LLM application architecture, we unveiled the layers that make these applications tick.

We delved into the intricacies of knowledge base retrieval, emphasizing the importance of ETL pipelines and tools like Unstructured, LlamaIndex, and LangChain's Document loaders. The significance of maintaining an updated knowledge base was highlighted, with a nod to efficient document indexing processes.

Join us in the upcoming blogs as we continue our journey through the remaining stages of the RAG pipeline. From User Query Processing to Front-End Development, we'll provide hands-on insights to demystify LLM applications. Together, let's unlock the full potential of LLMs, making the complex accessible to all in the realm of natural language understanding and generation. Stay tuned for more!

### **Thank you for reading!**

If this guide has enhanced your understanding of Python and Machine Learning:

- Please show your support with a clap 👏 or several claps!
- Your claps help me create more valuable content for our vibrant Python or ML community.
- Feel free to share this guide with fellow Python or AI / ML enthusiasts.
- Your feedback is invaluable — it inspires and guides my future posts.