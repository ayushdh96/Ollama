import os
import warnings
import ollama  # ✅ Import Ollama for LLM inference

# Suppress OpenSSL warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Path to the PDF folder
FOLDER_PATH = "./data/"

# System Prompt for the LLM
SYSTEM_PROMPT = (
    "You are a smart AI designed to assist Mr. Dhoundiyal. "
    "Your task is to answer questions based only on the provided context. "
    "If the answer is not in the context, say 'I don't know'."
)

# Step 1: Load All PDFs from the Folder
def load_documents(folder_path):
    all_documents = []
    
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            print(f"📄 Loading {file_path}")
            pdf_loader = PyPDFLoader(file_path)
            all_documents.extend(pdf_loader.load())  # ✅ Append text from each PDF

    return all_documents  # ✅ Now contains all PDFs' text

# Step 2: Split All Documents into Chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(documents)

# Step 3: Convert Chunks to Embeddings & Store in FAISS (Keeps All PDFs)
def create_vector_store(all_docs):
    embeddings = OllamaEmbeddings(model="llama3.2")

    # ✅ Store ALL document chunks in FAISS in a single index
    vector_store = FAISS.from_documents(all_docs, embeddings)
    
    return vector_store

# Step 4: Retrieve Most Relevant Chunks Based on Query
def retrieve_context(vector_store, query, top_k=3):
    query_embedding = OllamaEmbeddings(model="llama3.2").embed_query(query)  # Convert query to embeddings
    retriever = vector_store.similarity_search_by_vector(query_embedding, k=top_k)  # Find top k matches
    context = "\n\n".join([doc.page_content for doc in retriever])  # Extract relevant text
    return context

# Step 5: Generate Answer Using Context + Query
def generate_answer(context, query):
    if not context.strip():
        return "I don't know."

    full_prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nUser Query: {query}"
    response = ollama.generate(model="llama3.2", prompt=full_prompt)  # Get model response
    return response["response"]  # Extract final answer

# Step 6: CLI Interaction for Querying
def main():
    print("\n🚀 Processing all PDFs in the folder and building the FAISS index...")

    # ✅ Load all PDFs
    documents = load_documents(FOLDER_PATH)
    if not documents:
        print("❌ No valid PDF content found.")
        return

    # ✅ Split all PDFs into chunks together
    splitted_docs = split_documents(documents)

    # ✅ Store all chunks in FAISS in a single index
    vector_store = create_vector_store(splitted_docs)

    print("\n✅ Setup complete! Ollama CLI is ready. Type 'exit' to quit.")

    while True:
        try:
            user_query = input("\nMr. Dhoundiyal, enter your query: ")
            if user_query.lower() == "exit":
                print("\n👋 Exiting... Have a great day, Mr. Dhoundiyal!")
                break
            
            retrieved_context = retrieve_context(vector_store, user_query)  # Retrieve relevant info
            response = generate_answer(retrieved_context, user_query)  # Generate response

            print("\n💡 Response:\n", response)
        
        except KeyboardInterrupt:
            print("\n👋 Exiting... Have a great day, Mr. Dhoundiyal!")
            break

if __name__ == "__main__":
    main()