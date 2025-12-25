# Building LLM Applications: Search & Retrieval (Part 5)

Created: December 25, 2025 6:59 PM

# **Building LLM Applications: Search & Retrieval (Part 5)**

## **Learn Large Language Models ( LLM ) through the lens of a Retrieval Augmented Generation ( RAG ) Application.**

![Vipra Singh](https://miro.medium.com/v2/resize:fill:88:88/1*LDjQS3c-G1gsojOf24ijGg@2x.jpeg)

[**Vipra Singh**Follow](https://medium.com/@vipra_singh)

androidstudio·January 28, 2024 (Updated: March 17, 2024)·Free: Yes

### **Posts in this Series**

1. [***Introduction***](https://medium.com/@vipra_singh/building-llm-applications-introduction-part-1-1c90294b155b#4d28)
2. [***Data Preparation***](https://medium.com/@vipra_singh/building-llm-applications-data-preparation-part-2-b7306d224245)
3. [***Sentence Transformers***](https://medium.com/@vipra_singh/building-llm-applications-sentence-transformers-part-3-a9e2529f99c1)
4. [***Vector Database***](https://medium.com/@vipra_singh/building-llm-applications-vector-database-part-4-2bb29e7c798d)
5. [***Search & Retrieval](https://medium.com/@vipra_singh/building-llm-applications-retrieval-search-part-5-c83a7004037d) ( This Post )***
6. [***LLM***](https://medium.com/@vipra_singh/building-llm-applications-large-language-models-part-6-ea8bd982bdee)
7. [***Open-Source Chatbots***](https://medium.com/@vipra_singh/building-llm-applications-open-source-chatbots-part-7-1ca9c3653175)
8. ***Evaluation***
9. ***Fine-Tuning Embedding Models***
10. ***Fine-Tuning LLMs***
11. ***Serving LLMs***

### **Table of Contents**

· [Posts in this Series](https://freedium.cfd/https://medium.com/@vipra_singh/building-llm-applications-retrieval-search-part-5-c83a7004037d#4b09) · [Introduction](https://freedium.cfd/https://medium.com/@vipra_singh/building-llm-applications-retrieval-search-part-5-c83a7004037d#3d12) · [Issues with Search & Retrieval](https://freedium.cfd/https://medium.com/@vipra_singh/building-llm-applications-retrieval-search-part-5-c83a7004037d#b422) · [Optimizing Search & Retrieval](https://freedium.cfd/https://medium.com/@vipra_singh/building-llm-applications-retrieval-search-part-5-c83a7004037d#021d) · [Types of Search](https://freedium.cfd/https://medium.com/@vipra_singh/building-llm-applications-retrieval-search-part-5-c83a7004037d#8ff6) · [Semantic Search](https://freedium.cfd/https://medium.com/@vipra_singh/building-llm-applications-retrieval-search-part-5-c83a7004037d#cb7a) ∘ [Background](https://freedium.cfd/https://medium.com/@vipra_singh/building-llm-applications-retrieval-search-part-5-c83a7004037d#aeac) ∘ [Symmetric vs. Asymmetric Semantic Search](https://freedium.cfd/https://medium.com/@vipra_singh/building-llm-applications-retrieval-search-part-5-c83a7004037d#ac0d) · [Retrieval Algorithms](https://freedium.cfd/https://medium.com/@vipra_singh/building-llm-applications-retrieval-search-part-5-c83a7004037d#c1e4) ∘ [Similarity Search (Vanilla Search) & Maximum Marginal Relevance(MMR)](https://freedium.cfd/https://medium.com/@vipra_singh/building-llm-applications-retrieval-search-part-5-c83a7004037d#1418) · [Retrieve & Re-Rank](https://freedium.cfd/https://medium.com/@vipra_singh/building-llm-applications-retrieval-search-part-5-c83a7004037d#ef92) ∘ [Retrieve & Re-Rank Pipeline](https://freedium.cfd/https://medium.com/@vipra_singh/building-llm-applications-retrieval-search-part-5-c83a7004037d#cb2a) ∘ [Retrieval: Bi-Encoder](https://freedium.cfd/https://medium.com/@vipra_singh/building-llm-applications-retrieval-search-part-5-c83a7004037d#3305) ∘ [Re-Ranker: Cross-Encoder](https://freedium.cfd/https://medium.com/@vipra_singh/building-llm-applications-retrieval-search-part-5-c83a7004037d#5dc2) ∘ [Example Scripts](https://freedium.cfd/https://medium.com/@vipra_singh/building-llm-applications-retrieval-search-part-5-c83a7004037d#336f) ∘ [Pre-trained Bi-Encoders (Retrieval)](https://freedium.cfd/https://medium.com/@vipra_singh/building-llm-applications-retrieval-search-part-5-c83a7004037d#0e75) ∘ [Pre-trained Cross-Encoders (Re-Ranker)](https://freedium.cfd/https://medium.com/@vipra_singh/building-llm-applications-retrieval-search-part-5-c83a7004037d#5e69) · [Evaluation of Information Retrieval](https://freedium.cfd/https://medium.com/@vipra_singh/building-llm-applications-retrieval-search-part-5-c83a7004037d#e1ac) ∘ [Example](https://freedium.cfd/https://medium.com/@vipra_singh/building-llm-applications-retrieval-search-part-5-c83a7004037d#957f) ∘ [Actual vs. Predicted](https://freedium.cfd/https://medium.com/@vipra_singh/building-llm-applications-retrieval-search-part-5-c83a7004037d#b602) ∘ [Recall@K](https://freedium.cfd/https://medium.com/@vipra_singh/building-llm-applications-retrieval-search-part-5-c83a7004037d#d6ff) ∘ [Mean Reciprocal Rank (MRR)](https://freedium.cfd/https://medium.com/@vipra_singh/building-llm-applications-retrieval-search-part-5-c83a7004037d#4dc1) ∘ [Mean Average Precision (MAP)](https://freedium.cfd/https://medium.com/@vipra_singh/building-llm-applications-retrieval-search-part-5-c83a7004037d#3a29) ∘ [Normalized Discounted Cumulative Gain (NDCG@K)](https://freedium.cfd/https://medium.com/@vipra_singh/building-llm-applications-retrieval-search-part-5-c83a7004037d#cd8b) · [Wrap Up!](https://freedium.cfd/https://medium.com/@vipra_singh/building-llm-applications-retrieval-search-part-5-c83a7004037d#06c1)

![None](https://miro.medium.com/v2/resize:fit:700/1*OwSFM9iAzQJ16JWxp1VAiA.png)

Image by Author

Greetings! Let's kick off our exploration into the search for pertinent data within the RAG Application.

When a user inputs a query, the process involves tokenizing the user query and performing embedding using the identical model utilized for embedding raw data. Subsequently, relevant chunks are extracted from the knowledge base, guided by their similarity to the user's query.

![None](https://miro.medium.com/v2/resize:fit:700/0*Rtc59-h618e9LhBt.png)

[Vector DB](https://dev.to/sfoteini/image-vector-similarity-search-with-azure-computer-vision-and-postgresql-12f7)

This blog will take a comprehensive look into the intricacies of the search process. The below figure shows where exactly "Search" fits in the entire RAG Pipeline.

![None](https://miro.medium.com/v2/resize:fit:700/0*nwvJoBeCe78uCmXe.png)

[Retrieval Augmented Generation reduces the likelihood of hallucinations by providing domain-specific information through an LLM's context window.](https://www.pinecone.io/learn/retrieval-augmented-generation/)

### **Introduction**

Let's consider the common scenario of developing a customer support chatbot using an LLM. Usually, teams possess a wealth of product documentation, which includes a vast amount of unstructured data detailing their product, frequently asked questions, and use cases.

This data is broken down into pieces through a process called "chunking." After the data is broken down, each chunk is assigned a unique identifier and embedded into a high-dimensional space within a vector database. This process leverages advanced natural language processing techniques to understand the context and semantic meaning of each chunk.

When a customer's question comes in, the LLM uses a retrieval algorithm to quickly identify and fetch the most relevant chunks from the vector database. This retrieval is based on the semantic similarity between the query and the chunks, not just keyword matching.

![None](https://miro.medium.com/v2/resize:fit:700/0*KWY-CaUJERKyhHQr.png)

Image By [Arize](https://arize.com/blog-course/introduction-to-retrieval-augmented-generation/)

The picture above shows how search and retrieval is used with a prompt template (Ɗ' in the above image) to generate a final LLM prompt context. The above view is the search and retrieval LLM use case in its simplest form: a document is broken into chunks, these chunks are embedded into a vector store, and the search and retrieval process pulls on this context to shape LLM output.

This approach offers several advantages. First, it significantly reduces the time and computational resources required for the LLM to process large amounts of data, as it only needs to interact with the relevant chunks instead of the entire documentation.

Second, it allows for real-time updates to the database. As product documentation evolves, the corresponding chunks in the vector database can be easily updated. This ensures that the chatbot always provides the most up-to-date information.

Finally, by focusing on semantically relevant chunks, the LLM can provide more precise and contextually appropriate responses, leading to improved customer satisfaction.

### **Issues with Search & Retrieval**

While the search and retrieval method greatly enhances the efficiency and accuracy of LLMs, it's not without potential pitfalls. Identifying these issues early can prevent them from impacting user experience.

One such challenge arises when a user inputs a query that doesn't closely match any chunks in the vector store. The system looks for a needle in a haystack but finds no needle at all. This lack of match, often caused by unique or highly specific queries, can leave the system to draw on the "most similar" chunks available — ones that aren't entirely relevant.

In turn, this leads to a subpar response from the LLM. Since the LLM depends on the relevance of the chunks to generate responses, the lack of an appropriate match could result in an output that's tangentially related or even completely unrelated to the user's query.

![None](https://miro.medium.com/v2/resize:fit:700/0*W4N0hqUHhxq638-y.png)

Image By [Arize](https://arize.com/blog-course/introduction-to-retrieval-augmented-generation/)

Irrelevant or subpar responses from the LLM can frustrate users, lowering their satisfaction and ultimately causing them to lose trust in the system and product as a whole.

Monitoring three main things can help prevent these issues:

**Query Density (Drift)**: Query density refers to how well user queries are covered by the vector store. If query density drifts significantly, it signals that our vector store may not be capturing the full breadth of user queries, resulting in a shortage of closely associated chunks. Regularly monitoring query density enables us to spot these gaps or shortcomings. With this insight, we can augment the vector store by incorporating more relevant chunks or refining the existing ones, improving the system's ability to fetch data in response to user queries.

**Ranking Metrics**: These metrics evaluate how well the search and retrieval system is performing in terms of selecting the most relevant chunks. If the ranking metrics indicate a decline in performance, it's a signal that the system's ability to distinguish between relevant and irrelevant chunks might need refinement.

**User Feedback**: Encouraging users to provide feedback on the quality and relevance of the LLM's responses helps gauge user satisfaction and identify areas for improvement. Regular analysis of this feedback can point out patterns and trends, which can then be used to adjust your application as necessary.

### **Optimizing Search & Retrieval**

Optimization of search and retrieval processes should be a constant endeavor throughout the lifecycle of your LLM-powered application, from the building phase through to post-production.

During the building phase, attention should be given to developing a robust testing and evaluation strategy. This approach allows us to identify potential issues early on and optimize our strategies, forming a solid foundation for the system.

Key areas to focus on include:

- **Chunking Strategy**: Evaluating how information is broken down and processed during this stage can help highlight areas for improvement in performance.
- **Retrieval Performance**: Assessing how well the system retrieves information can indicate if we need to employ different tools or strategies, such as context ranking or HYDE.

Upon release, optimization efforts should continue as we enter the post-production phase. Even after launch, with a well-defined evaluation strategy, we can proactively identify any emerging issues and continue to improve our model's performance. Consider approaches like:

- **Expanding our Knowledge Base**: Adding documentation can significantly improve our system's response quality. An expanded data set allows our LLM to provide more accurate and tailored responses.
- **Refining Chunking Strategy**: Further modifying the way information is broken down and processed can lead to marked improvements.
- **Enhancing Context Understanding**: Incorporating an extra 'context evaluation' step helps the system incorporate the most relevant context into the LLM's response.

Specifics on these and other strategies for continuous optimization will be detailed in the following sections of this course. Remember, the goal is to create a system that not only meets users' needs at launch but also evolves with them over time.

### **Types of Search**

We have to remember that vector databases are not the panacea of search — they are very good at *semantic* search, but in many cases, traditional keyword search can yield more relevant results and increase user satisfaction. Why is that? It's largely to do with the fact that ranking based on metrics like cosine similarity causes results that have a higher similarity score to appear above partial matches that may contain specific input keywords, reducing their relevance to the end user.

However, pure keyword search also has its limitations — in case the user enters a term that is semantically similar to the stored data (but is not exact), potentially useful and relevant results are not returned. As a result of this trade-off, real-world use cases for search & retrieval demand a combination of keyword and vector searches, **of which vector databases form a key component** (because they house the embeddings, enabling semantic similarity search and can scale to very large datasets).

To summarize the points above:

- **Keyword search**: Finds relevant, useful results when the user *knows* what they're looking for and expects results that match exact phrases in their search terms. Does **not** require vector databases.
- **Vector search**: Finds relevant results when the user *doesn't* know what exactly they're looking for. Requires a vector database.
- **Hybrid (keyword + vector) search**: Typically combines candidate results from full-text keyword and vector searches and re-ranks them using cross-encoder models. Requires both a document database and a vector database.

This can be effectively visualized per the diagram below:

![None](https://miro.medium.com/v2/resize:fit:700/0*kMGnmG0np2Tx9JDw.png)

[Search](https://thedataquarry.com/posts/vector-db-2/)

### **Semantic Search**

Semantic search seeks to improve search accuracy by understanding the content of the search query. In contrast to traditional search engines which only find documents based on lexical matches, semantic search can also find synonyms.

### **Background**

The idea behind semantic search is to embed all entries in our corpus, whether they be sentences, paragraphs, or documents, into a vector space.

At search time, the query is embedded into the same vector space and the closest embeddings from our corpus are found. These entries should have a high semantic overlap with the query.

![None](https://miro.medium.com/v2/resize:fit:700/0*WjSxNR9xyHrOJEfx.png)

### **Symmetric vs. Asymmetric Semantic Search**

A **critical distinction** for our setup is *symmetric* vs. *asymmetric semantic search*:

- For **symmetric semantic search,** our query and the entries in our corpus are of about the same length and have the same amount of content. An example would be searching for similar questions: Our query could for example be *"How to learn Python online?"* and we want to find an entry like *"How to learn Python on the web?"*. For symmetric tasks, we could potentially flip the query and the entries in our corpus.
- For **asymmetric semantic search**, we usually have a **short query** (like a question or some keywords), and we want to find a longer paragraph answering the query. An example would be a query like *"What is Python"* and we wand to find the paragraph *"Python is an interpreted, high-level and general-purpose programming language. Python's design philosophy …"*. For asymmetric tasks, flipping the query and the entries in our corpus usually does not make sense.

We must choose **the right model** for our type of task.

Suitable models for **symmetric semantic search**: [Pre-Trained Sentence Embedding Models](https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models)

Suitable models for **asymmetric semantic search**: [Pre-Trained MS MARCO Models](https://www.sbert.net/docs/pretrained-models/msmarco-v3.html)

### **Retrieval Algorithms**

### [**Similarity Search (Vanilla Search) & Maximum Marginal Relevance(MMR)**](https://medium.com/tech-that-works/maximal-marginal-relevance-to-rerank-results-in-unsupervised-keyphrase-extraction-22d95015c7c5)

When it comes to retrieving documents, the majority of methods will do a similarity metric like cosine similarity, euclidean distance, or dot product. All of these will return documents that are most similar to our query/question.

However, what if we want similar documents that are also diverse from each other? That is where [Maximum Marginal Relevance (MMR)](https://community.fullstackretrieval.com/retrieval-methods/maximum-marginal-relevance-mmr) steps in.

The goal is to take into account how similar retrieved documents *are to each other* when determining which to return. In theory, we should have a well-rounded, diverse set of documents.

In case of **unsupervised learning**, let's say our final keyPhrases are ranked like **`Good Product, Great Product, Nice Product, Excellent Product, Easy Install, Nice UI, Light weight etc.`** But there is an issue with this approach, all the phrases like **`good product, nice product, excellent product`** are similar and define the same property of the product and are ranked higher. Suppose we have a space to show just 5 key phrases, in that case, we don't want to show all these similar phrases.

We want to properly utilize this limited space such that the information displayed by the Keyphrases about the documents is diverse enough. Similar types of phrases should not dominate the whole space and users can see a variety of information about the document.

1. **Remove redundant phrases using cosine similarity**
2. **Re-ranking the key phrases using MMR**

![None](https://miro.medium.com/v2/resize:fit:700/0*We1fR0HMfPV4evit.png)

MMR

Above are two widely used Retrieval methods. Other methods like *Multi Query Retrieval, Long-Context Reorder, Multi-Vector Retriever, Parent Document Retriever, Self-Querying, Time-weighted Vector Store Retrieval* are some of the advanced Retrieval strategies that we will cover in a separate blog post.

Now let's discuss the Retrieval and Re-ranking pipeline below and let's see how it enhances the results.

### **Retrieve & Re-Rank**

In [Semantic Search](https://www.sbert.net/examples/applications/semantic-search/README.html) we have shown how to use Sentence Transformer to compute embeddings for queries, sentences, and paragraphs and how to use this for semantic search.

For complex search tasks, for example, for question-answering retrieval, the search can significantly be improved by using **Retrieve & Re-Rank**.

### **Retrieve & Re-Rank Pipeline**

A pipeline for information retrieval / question-answering retrieval that works well is the following. All components are provided and explained in this article:

![None](https://miro.medium.com/v2/resize:fit:700/0*_jinKUFSeXsOb0Rp.png)

By [SBERT](https://www.sbert.net/examples/applications/retrieve_rerank/README.html)

Given a search query, we first use a **retrieval system** that retrieves a large list of e.g. 100 possible hits that are potentially relevant for the query. For the retrieval, we can use either lexical search, e.g. with ElasticSearch, or we can use dense retrieval with a bi-encoder.

However, the retrieval system might retrieve documents that are not that relevant to the search query. Hence, in the second stage, we use a **re-ranker** based on a **cross-encoder** that scores the relevancy of all candidates for the given search query.

The output will be a ranked list of hits we can present to the user.

### **Retrieval: Bi-Encoder**

Lexical search looks for literal matches of the query words in our document collection. It will not recognize synonyms, acronyms or spelling variations. In contrast, semantic search (or dense retrieval) encodes the search query into vector space and retrieves the document embeddings that are close in vector space.

![None](https://miro.medium.com/v2/resize:fit:700/0*ByYZ9FesWqy7gKwp.png)

Semantic search overcomes the shortcomings of lexical search and can recognize synonyms and acronyms. Have a look at the [semantic search article](https://www.sbert.net/examples/applications/semantic-search/README.html) for different options to implement semantic search.

### **Re-Ranker: Cross-Encoder**

The retriever has to be efficient for large document collections with millions of entries. However, it might return irrelevant candidates.

A re-ranker based on a Cross-Encoder can substantially improve the final results for the user. The query and a possible document are passed simultaneously to the transformer network, which then outputs a single score between 0 and 1 indicating how relevant the document is for the given query.

![None](https://miro.medium.com/v2/resize:fit:700/0*N6ydV7IWJOCPUHs4.png)

By [SBERT](https://www.sbert.net/examples/applications/retrieve_rerank/README.html)

The advantage of Cross-Encoders is the higher performance, as they perform attention across the query and the document.

Scoring thousands or millions of (query, document)-pairs would be rather slow. Hence, we use the retriever to create a set of e.g. 100 possible candidates which are then re-ranked by the Cross-Encoder.

![None](https://miro.medium.com/v2/resize:fit:700/0*PB1LuUvsUh2PMQ9_.png)

[Cross Encoding Re-Ranking High-Level Flow](https://levelup.gitconnected.com/3-query-expansion-methods-implemented-using-langchain-to-improve-your-rag-81078c1330cd)

### **Example Scripts**

- [**retrieve_rerank_simple_wikipedia.ipynb**](https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/retrieve_rerank/retrieve_rerank_simple_wikipedia.ipynb) [ [Colab Version](https://colab.research.google.com/github/UKPLab/sentence-transformers/blob/master/examples/applications/retrieve_rerank/retrieve_rerank_simple_wikipedia.ipynb) ]: This script uses the smaller [Simple English Wikipedia](https://simple.wikipedia.org/wiki/Main_Page) as a document collection to provide answers to user questions/search queries. First, we split all Wikipedia articles into paragraphs and encode them with a bi-encoder. If a new query/question is entered, it is encoded by the same bi-encoder and the paragraphs with the highest cosine-similarity are retrieved (see [semantic search](https://www.sbert.net/examples/applications/semantic-search/README.html)). Next, the retrieved candidates are scored by a Cross-Encoder re-ranker and the 5 passages with the highest score from the Cross-Encoder are presented to the user.
- [**in_document_search_crossencoder.py](https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/retrieve_rerank/in_document_search_crossencoder.py):** If have only have a small set of paragraphs, we don't have the retrieval stage. This is for example the case if we want to perform a search within a single document. In this example, take the Wikipedia article about Europe and split it into paragraphs. Then, the search query/question and all paragraphs are scored using the Cross-Encoder re-ranker. The most relevant passages for the query are returned.

### **Pre-trained Bi-Encoders (Retrieval)**

The bi-encoder produces embeddings independently for our paragraphs and our search queries. We can use it like this:

```makefile
Copyfrom sentence_transformers import SentenceTransformer
model = SentenceTransformer('model_name')

docs = ["My first paragraph. That contains information", "Python is a programming language."]
document_embeddings = model.encode(docs)

query = "What is Python?"
query_embedding = model.encode(query)
```

For more details on how to compare the embeddings, please visit [semantic search](https://www.sbert.net/examples/applications/semantic-search/README.html).

We provide pre-trained models based on:

- **MS MARCO:** 500k real user queries from Bing search engine. See [MS MARCO models](https://www.sbert.net/docs/pretrained-models/msmarco-v3.html)

### **Pre-trained Cross-Encoders (Re-Ranker)**

For pre-trained models, we can refer: [MS MARCO Cross-Encoders](https://www.sbert.net/docs/pretrained-models/ce-msmarco.html)

### **Evaluation of Information Retrieval**

Evaluation of information retrieval (IR) systems is critical to making well-informed design decisions. From search to recommendations, evaluation measures are paramount to understanding what does and does not work in retrieval.

Evaluation measures for IR systems can be split into *two* categories: *online* or *offline* metrics.

**Online metrics** are captured during actual usage of the IR system when it is *online*. These consider user interactions like whether a user clicked on a recommended show from Netflix or if a particular link was clicked from an email advertisement (the click-through rate or CTR). There are many online metrics, but they all relate to some form of user interaction.

**Offline metrics** are measured in an isolated environment before deploying a new IR system. These look at whether a particular set of *relevant* results are returned when retrieving items with the system.

![None](https://miro.medium.com/v2/resize:fit:700/0*YgUlKE13iMxiEVP_.png)

By [Pinecone](https://www.pinecone.io/learn/offline-evaluation/)

Evaluation measures can be categorized as either offline or online metrics. Offline metrics can be further divided into order-unaware or order-aware, which we will explain soon.

Organizations often use *both* offline and online metrics to measure the performance of their IR systems. It begins, however, with offline metrics to predict the system's performance *before deployment*.

We will focus on the most useful and popular offline metrics:

- Recall@K
- **M**ean **R**eciprocal **R**ank (MRR)
- **M**ean **A**verage **P**recision@K (MAP@K)
- **N**ormalized **D**iscounted **C**umulative **G**ain (NDCG@K)

These metrics are deceptively simple yet provide invaluable insight into the performance of IR systems.

We can use one or more of these metrics in different evaluation stages. During the development of Spotify's podcast search; *Recall@K* (using *K*=1) was used during training on "evaluation batches", and after training, [both](https://www.pinecone.io/learn/spotify-podcast-search/#:~:text=Spotify%20details%20their%20full%2Dretrieval%20setting%20metrics%20as%20using%20Recall%4030%20and%20MRR%4030%2C%20performed%20both%20on%20queries%20from%20the%20eval%20set%20and%20on%20their%20curated%20dataset.) [*Recall@K*](https://www.pinecone.io/learn/spotify-podcast-search/#:~:text=Spotify%20details%20their%20full%2Dretrieval%20setting%20metrics%20as%20using%20Recall%4030%20and%20MRR%4030%2C%20performed%20both%20on%20queries%20from%20the%20eval%20set%20and%20on%20their%20curated%20dataset.) [and](https://www.pinecone.io/learn/spotify-podcast-search/#:~:text=Spotify%20details%20their%20full%2Dretrieval%20setting%20metrics%20as%20using%20Recall%4030%20and%20MRR%4030%2C%20performed%20both%20on%20queries%20from%20the%20eval%20set%20and%20on%20their%20curated%20dataset.) [*MRR](https://www.pinecone.io/learn/spotify-podcast-search/#:~:text=Spotify%20details%20their%20full%2Dretrieval%20setting%20metrics%20as%20using%20Recall%4030%20and%20MRR%4030%2C%20performed%20both%20on%20queries%20from%20the%20eval%20set%20and%20on%20their%20curated%20dataset.)* (using *K*=30) were used with a much larger evaluation set.

For now, understand that Spotify was able to predict system performance *before* deploying anything to customers. This allowed them to deploy successful A/B tests and significantly increase podcast engagement.

We have two more subdivisions for these metrics; *order-aware* and *order-unaware*. This refers to whether the order of results impacts the metric score. If so, the metric is *order-aware*. Otherwise, it is *order-unaware*.

### **Example**

Throughout the article, we will be using a *very* small dataset of eight images. In reality, this number is likely to be millions or more.

![None](https://miro.medium.com/v2/resize:fit:700/0*CUuV01AnElluVcas.png)

[Example query and ranking of the eight possible results.](https://www.pinecone.io/learn/offline-evaluation/)

If we were to search for *"cat in a box"*, we may return something like the above. The numbers represent the relevance *rank* of each image as predicted by the IR system. Other queries would yield a different order of results.

![None](https://miro.medium.com/v2/resize:fit:700/0*mEQDRyh-T6XHqG18.png)

[Example query and ranking with actual relevant results highlighted.](https://www.pinecone.io/learn/offline-evaluation/)

We can see that results *2*, *4*, *5*, and *7* are *actual relevant* results. The other results are *not* relevant as they show cats *without* boxes, boxes *without* cats, or a dog.

### **Actual vs. Predicted**

When evaluating the performance of the IR system, we will be comparing *actual* vs. *predicted* conditions, where:

- **Actual condition** refers to the true label of every item in the dataset. These are *positive* (*p*) if an item is relevant to a query or *negative* (*n*) if an item is *ir*relevant to a query.
- **Predicted condition** is the *predicted* label returned by the IR system. If an item is returned, it is predicted as being *positive* (*p*^) and, if it is not returned, is predicted as a *negative* (*n^*).

From these actual and predicted conditions, we create a set of outputs from which we calculate all of our offline metrics. Those are the true/false positives and true/false negatives.

The *positive* results focus on what the IR system returns. Given our dataset, we ask the IR system to return *two* items using the *"cat in a box"* query. If it returns an *actual relevant* result this is a *true positive* (*pp*^); if it returns an irrelevant result, we have a *false positive* (*np*^).

![None](https://miro.medium.com/v2/resize:fit:700/0*s31jEXecbtCPmc6u.png)

By [Pinecone](https://www.pinecone.io/learn/offline-evaluation/)

For *negative* results, we must look at what the IR system *does not* return. Let's query for two results. Anything that *is relevant* but is *not* returned is a *false negative* (*pn*^). Irrelevant items that were *not* returned are *true negatives* (*nn*^).

With all of this in mind, we can begin with the first metric.

### **Recall@K**

*Recall@K* is one of the most interpretable and popular offline evaluation metrics. It measures how many relevant items were returned (*pp*^) against how many relevant items exist in the entire dataset (*pp*^+*pn*^).

![None](https://miro.medium.com/v2/resize:fit:700/1*FmxzUgAbpQo6l9c1bbcf-A.png)

The *K* in this and all other offline metrics refers to the number of items returned by the IR system. In our example, we have a total number of *N = 8* items in the entire dataset, so *K* can be any value between [1 ,…, *N*].

![None](https://miro.medium.com/v2/resize:fit:700/0*fisyHFyD3fabsGuL.png)

[With recall@2 we return the predicted top K = 2 most relevant results.](https://www.pinecone.io/learn/offline-evaluation/)

When *K = 2*, our *recall@2* score is calculated as the number of *returned* relevant results over the total number of relevant results in the *entire dataset*. That is:

![None](https://miro.medium.com/v2/resize:fit:700/1*N4G19ImLMaVIXuQBojiQzg.png)

With recall@K, the score improves as *K* increases and the scope of returned items increases.

![None](https://miro.medium.com/v2/resize:fit:700/0*qpAQ8Q82lpu4sYKL.png)

[With recall@K we will see the score increase as K increases and more positives (whether true or false) are returned.](https://www.pinecone.io/learn/offline-evaluation/)

We can calculate the same recall@K score easily in Python. For this, we will define a function named **recall** that takes lists of *actual conditions* and *predicted conditions*, a *K* value, and returns a recall@K score.

```python
Copy# recall@k function
def recall(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = round(len(act_set & pred_set) / float(len(act_set)), 2)
    return result
```

Using this, we will replicate our eight-image dataset with *actual relevant* results in rank positions *2*, *4*, *5*, and *7*.

```python
Copyactual = ["2", "4", "5", "7"]
predicted = ["1", "2", "3", "4", "5", "6", "7", "8"]
for k in range(1, 9):
    print(f"Recall@{k} = {recall(actual, predicted, k)}")
```

Output :

```
CopyRecall@1 = 0.0
Recall@2 = 0.25
Recall@3 = 0.25
Recall@4 = 0.5
Recall@5 = 0.75
Recall@6 = 0.75
Recall@7 = 1.0
Recall@8 = 1.0
```

**Pros and Cons**

Recall@K is undoubtedly one of the most easily interpretable evaluation metrics. We know that a perfect score indicates that all relevant items are being returned. We also know that a smaller *k* value makes it harder for the IR system to score well with recall@K.

Still, there are disadvantages to using *recall@K*. By increasing *K* to *N* or near *N*, we can return a perfect score every time, so relying solely on recall@K can be deceptive.

Another problem is that it is an *order-unaware metric*. That means if we used recall@4 and returned one relevant result at rank *one*, we would score the same as if we returned the same result at rank *four*. Clearly, it is better to return the actual relevant result at a higher rank, but recall@K *cannot* account for this.

### **Mean Reciprocal Rank (MRR)**

The **M**ean **R**eciprocal **R**ank (MRR) is an *order-aware metric*, which means that, unlike recall@K, returning an actual relevant result at rank *one* scores better than at rank *four*.

Another differentiator for MRR is that it is calculated based on multiple queries. It is calculated as:

![None](https://miro.medium.com/v2/resize:fit:700/1*rLf_uesqvPqzewSdwhlmbw.png)

*Q* is the number of queries, *q* is a specific query, and *rank-q* is the rank of the first *actual relevant* result for query *q*. We will explain the formula step-by-step.

Using our last example where a user searches for *"cat in a box"*. We add two more queries, giving us *Q*=3.

![None](https://miro.medium.com/v2/resize:fit:700/0*vnB23XDv6KUwvLR0.png)

[We perform three queries while calculating the MRR score.](https://www.pinecone.io/learn/offline-evaluation/)

We calculate the rank reciprocal 1/*rankq* for each query *q*. For the first query, the first actual relevant image is returned at position *two*, so the rank reciprocal is 1/2. Let's calculate the reciprocal rank for all queries:

![None](https://miro.medium.com/v2/resize:fit:700/1*sxOuOmpKIevI6nzKm5JYvw.png)

Next, we sum all of these reciprocal ranks for queries *q*=[1,…, *Q*] (e.g., all three of our queries):

![None](https://miro.medium.com/v2/resize:fit:700/1*GKeVTsYJq0GZSVJXGdAvPQ.png)

As we are calculating the **mean** reciprocal rank (**M**RR), we must take the average value by dividing our total reciprocal ranks by the number of queries *Q*:

![None](https://miro.medium.com/v2/resize:fit:700/1*W-6_Bi8-gbDZWGBKvT_AzA.png)

Now let's translate this into Python. We will replicate the same scenario where *Q*=3 using the same *actual relevant* results.

```python
Copy# relevant results for query #1, #2, and #3
actual_relevant = [
    [2, 4, 5, 7],
    [1, 4, 5, 7],
    [5, 8]
]
# number of queries
Q = len(actual_relevant)

# calculate the reciprocal of the first actual relevant rank
cumulative_reciprocal = 0
for i in range(Q):
    first_result = actual_relevant[i][0]
    reciprocal = 1 / first_result
    cumulative_reciprocal += reciprocal
    print(f"query #{i+1} = 1/{first_result} = {reciprocal}")
# calculate mrr
mrr = 1/Q * cumulative_reciprocal
# generate results
print("MRR =", round(mrr,2))
```

Output :

```
Copyquery #1 = 1/2 = 0.5
query #2 = 1/1 = 1.0
query #3 = 1/5 = 0.2
MRR = 0.57
```

And as expected, we calculate the same MRR score of *0.57*.

**Pros and Cons**

MRR has its own unique set of advantages and disadvantages. It is *order-aware*, a massive advantage for use cases where the rank of the first relevant result is important, like chatbots or [question-answering](https://www.pinecone.io/learn/series/nlp/question-answering/).

On the other hand, we consider the rank of the *first* relevant item, but no others. That means for use cases where we'd like to return multiple items like recommendations or search engines, MRR is not a good metric. For example, if we'd like to recommend ~10 products to a user, we ask the IR system to retrieve 10 items. We could return just one *actual relevant* item in rank one and no other relevant items. Nine of ten irrelevant items is a terrible result, but MRR would score a perfect *1.0*.

Another *minor* disadvantage is that MRR is less readily interpretable compared to a simpler metric like recall@K. However, it is still more interpretable than many other evaluation metrics.

### **Mean Average Precision (MAP)**

![1000049807.png](Building%20LLM%20Applications%20Search%20&%20Retrieval%20(Part/1000049807.png)

![1000049808.png](Building%20LLM%20Applications%20Search%20&%20Retrieval%20(Part/1000049808.png)

![1000049809.png](Building%20LLM%20Applications%20Search%20&%20Retrieval%20(Part/1000049809.png)

![1000049810.png](Building%20LLM%20Applications%20Search%20&%20Retrieval%20(Part/1000049810.png)

![1000049811.png](Building%20LLM%20Applications%20Search%20&%20Retrieval%20(Part/1000049811.png)

```python
# initialize variables
actual = [
    [2, 4, 5, 7],
    [1, 4, 5, 7],
    [5, 8]
]
Q = len(actual)
predicted = [1, 2, 3, 4, 5, 6, 7, 8]
k = 8
ap = []

# loop through and calculate AP for each query q
for q in range(Q):
    ap_num = 0
    # loop through k values
    for x in range(k):
        # calculate precision@k
        act_set = set(actual[q])                                                                                                                                   
        pred_set = set(predicted[:x+1])
        precision_at_k = len(act_set & pred_set) / (x+1)
        # calculate rel_k values
        if predicted[x] in actual[q]:
            rel_k = 1
        else:
            rel_k = 0
        # calculate numerator value for ap
        ap_num += precision_at_k * rel_k
    # now we calculate the AP value as the average of AP
    # numerator values
    ap_q = ap_num / len(actual[q])
    print(f"AP@{k}_{q+1} = {round(ap_q,2)}")
    ap.append(ap_q)
# now we take the mean of all ap values to get mAP
map_at_k = sum(ap) / Q
# generate results
print(f"mAP@{k} = {round(map_at_k, 2)}")
```

### Normalized Discounted Cumulative Gain (NDCG@K)

![1000049813.png](Building%20LLM%20Applications%20Search%20&%20Retrieval%20(Part/1000049813.png)

![1000049814.png](Building%20LLM%20Applications%20Search%20&%20Retrieval%20(Part/1000049814.png)

![1000049816.png](Building%20LLM%20Applications%20Search%20&%20Retrieval%20(Part/1000049816.png)

![1000049817.png](Building%20LLM%20Applications%20Search%20&%20Retrieval%20(Part/1000049817.png)

```python
from math import log2
# initialize variables
relevance = [0, 7, 2, 4, 6, 1, 4, 3]
K = 8
dcg = 0
# loop through each item and calculate DCG
for k in range(1, K+1):
    rel_k = relevance[k-1]
    # calculate DCG
    dcg += rel_k / log2(1 + k)
```

![1000049818.png](Building%20LLM%20Applications%20Search%20&%20Retrieval%20(Part/1000049818.png)

![1000049819.png](Building%20LLM%20Applications%20Search%20&%20Retrieval%20(Part/1000049819.png)

```python
# sort items in 'relevance' from most relevant to less relevant
ideal_relevance = sorted(relevance, reverse=True)

print(ideal_relevance)

idcg = 0
# as before, loop through each item and calculate *Ideal* DCG
for k in range(1, K+1):
    rel_k = ideal_relevance[k-1]
    # calculate DCG
    idcg += rel_k / log2(1 + k)
```

![1000049820.png](Building%20LLM%20Applications%20Search%20&%20Retrieval%20(Part/1000049820.png)

```python
dcg = 0
idcg = 0

for k in range(1, K+1):
    # calculate rel_k values
    rel_k = relevance[k-1]
    ideal_rel_k = ideal_relevance[k-1]
    # calculate dcg and idcg
    dcg += rel_k / log2(1 + k)
    idcg += ideal_rel_k / log2(1 + k)
    # calcualte ndcg
    ndcg = dcg / idcg
```

![1000049821.png](Building%20LLM%20Applications%20Search%20&%20Retrieval%20(Part/1000049821.png)

![1000049822.png](Building%20LLM%20Applications%20Search%20&%20Retrieval%20(Part/1000049822.png)

![1000049823.png](Building%20LLM%20Applications%20Search%20&%20Retrieval%20(Part/1000049823.png)

![1000049824.png](Building%20LLM%20Applications%20Search%20&%20Retrieval%20(Part/1000049824.png)