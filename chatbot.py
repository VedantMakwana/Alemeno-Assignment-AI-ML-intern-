import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
import PyPDF2

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def create_vector_store(pdf_paths, vector_store_path):
    """Create a FAISS vector store from PDF documents"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cuda'}
    )
    
    data = ""
    for path in pdf_paths:
        data += extract_text_from_pdf(path)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(data)
    
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local(vector_store_path)
    return vector_store

def load_vector_store(vector_store_path):
    """Load the FAISS vector store"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cuda'}
    )
    return FAISS.load_local(vector_store_path, embeddings,allow_dangerous_deserialization=True)

def initialize_chain(vector_store):
    """Initialize the conversational chain"""
    llm = OllamaLLM(model="llama3.1",
        temperature=0.5,  # Lower temperature for more focused responses
        top_k=10,  # Limit token consideration
        top_p=0.3,  # More focused sampling
        )
    
    # Initialize memory with output_key
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",  # Specify which output to store
        return_messages=True
    )
    
    # Create chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        return_source_documents=True,
        verbose=True  # Add this for debugging
    )
    
    return chain

def main():
    st.title("Document Chat Assistant")
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.header("Document Upload")
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            pdf_paths = []
            for file in uploaded_files:
                path = f"temp_{file.name}"
                with open(path, "wb") as f:
                    f.write(file.getvalue())
                pdf_paths.append(path)
            
            vector_store_path = "vector_store"
            if not os.path.exists(vector_store_path):
                with st.spinner("Processing documents..."):
                    vector_store = create_vector_store(pdf_paths, vector_store_path)
                st.success("Documents processed successfully!")
            else:
                vector_store = load_vector_store(vector_store_path)
            
            chain = initialize_chain(vector_store)
            st.session_state.chain = chain
            
            # Clean up temporary files
            for path in pdf_paths:
                os.remove(path)

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to know about the documents?"):
        if not hasattr(st.session_state, 'chain'):
            st.error("Please upload documents first!")
            return

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chain({"question": prompt})
                    response_text = response['answer']
                    
                    # Display source documents if available
                    if 'source_documents' in response:
                        response_text += "\n\nSources:"
                        for idx, doc in enumerate(response['source_documents'], 1):
                            response_text += f"\n{idx}. {doc.page_content[:200]}..."
                    
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()