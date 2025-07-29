import os
import streamlit as st
from dotenv import load_dotenv

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding 
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core import VectorStoreIndex, StorageContext  
from llama_index.core import load_index_from_storage


load_dotenv()

INDEX_DIR = 'wiki_rag'
 
PAGES = [
    "Artificial intelligence",
    "Machine learning",
    "Deep learning",
    "Natural language processing"
    # "Computer vision",
    # "Neural network",
    # "Artificial general intelligence",
    # "Turing test",
    # "Expert system",
    # "Reinforcement learning",
    # "Computer science",
    # "Algorithm",
    # "Data structure",
    # "Computational complexity theory",
    # "Software engineering",
    # "Operating system",
    # "Compiler",
    # "Data science",
    # "Big data",
    # "Data mining",
    # "Predictive analytics",
    # "Statistical learning",
    # "K-means clustering",
    # "Principal component analysis",
    # "Data visualization",
    # "Information retrieval",
    # "Knowledge discovery in databases",
    # "Supervised learning",
    # "Unsupervised learning",
    # "Feature engineering"
]

@st.cache_resource
def get_index():
    if os.path.isdir(INDEX_DIR):
        storage = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        return load_index_from_storage(storage)  
    
    docs = WikipediaReader().load_data(pages=PAGES, auto_suggest=False)      
    embedding_model = OpenAIEmbedding(model='text-embedding-3-small')
    index = VectorStoreIndex.from_documents(docs, embed_model=embedding_model)
    index.storage_context.persist(persist_dir=INDEX_DIR)
    
    return index

@st.cache_resource
def get_query_engine():
    index = get_index()
    
    llm = OpenAI(model='gpt-4o-mini', temperature=0)
    
    return index.as_query_engine(llm=llm, similarity_top_k=3)


def main():
    st.title('Wikipedia Simple RAG Application')
    
    question = st.text_input('Ask a question')
    
    if st.button('Submit') and question:
        with st.spinner('Thinking.....'):
            qa = get_query_engine()
            response = qa.query(question)
            
        st.subheader("Answer")
        st.write(response.response)
        
        st.subheader('Retreived context')
        
        for src in response.source_nodes:
            st.markdown(src.node.get_context())
            
if __name__ == "__main__":
    main()