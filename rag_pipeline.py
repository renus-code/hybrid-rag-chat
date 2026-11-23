import os

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

DATA_DIR = "./new_articles/"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "openai/gpt-oss-20b"
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"
RETRIEVE_K = 20
FETCH_K = 40
RERANK_TOP_N = 4

PROMPT_TEMPLATE = """You are a helpful assistant.

Answer the question using ONLY the context below.

Rules:
1. Do not use outside knowledge.
2. If the context does not have enough information, say exactly:
   "I don't have enough information to answer."
3. Be concise and factual.
4. Cite sources using the [Source: filename] tags in the context.

Context:
{context}

Question:
{question}

Answer:"""


def load_and_split():
    loader = DirectoryLoader(
        path=DATA_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True},
    )
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return docs, splitter.split_documents(docs)


def build_retriever():
    docs, chunks = load_and_split()

    embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL, batch_size=16)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    vector_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVE_K, "fetch_k": FETCH_K},
    )

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = RETRIEVE_K

    hybrid = EnsembleRetriever(retrievers=[bm25, vector_retriever], weights=[0.5, 0.5])

    FlashrankRerank.model_rebuild()
    reranker = FlashrankRerank(model=RERANK_MODEL, top_n=RERANK_TOP_N)

    retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=hybrid,
    )
    return retriever, len(docs), len(chunks)


def format_docs(docs):
    parts = []
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        parts.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_chain():
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatGroq(
        model=LLM_MODEL,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )
    return prompt | llm | StrOutputParser()


def answer_question(question, retriever, chain):
    docs = retriever.invoke(question)
    answer = chain.invoke({"context": format_docs(docs), "question": question})
    return answer, docs