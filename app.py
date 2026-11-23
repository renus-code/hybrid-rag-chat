import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from rag_pipeline import build_retriever, build_chain, answer_question

st.title("Hybrid RAG Chat")
st.write("Ask a question about the articles. Answers use hybrid search + reranking.")


# Build the search index only once (so the app stays fast)
@st.cache_resource
def load_pipeline():
    retriever, n_files, n_chunks = build_retriever()
    chain = build_chain()
    return retriever, chain


retriever, chain = load_pipeline()

question = st.text_input("Your question:")

if st.button("Ask") and question:
    answer, docs = answer_question(question, retriever, chain)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources used")
    for doc in docs:
        st.text(doc.page_content)