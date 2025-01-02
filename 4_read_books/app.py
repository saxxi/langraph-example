import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import Graph
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from graphviz import Digraph

load_dotenv()

def embed_file(current_dir, persistent_directory, filename):
    # Define the directory containing the text file and the persistent directory
    file_path = os.path.join(current_dir, filename)

    # Check if the Chroma vector store already exists
    if not os.path.exists(persistent_directory):
        print("Persistent directory does not exist. Initializing vector store...")

        # Ensure the text file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"The file {file_path} does not exist. Please check the path."
            )

        # Read the text content from the file
        loader = TextLoader(file_path)
        documents = loader.load()

        # Split the document into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        # Display information about the split documents
        print("\n--- Document Chunks Information ---")
        print(f"Number of document chunks: {len(docs)}")
        print(f"Sample chunk:\n{docs[0].page_content}\n")

        # Create embeddings
        print("\n--- Creating embeddings ---")
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )  # Update to a valid embedding model if needed
        print("\n--- Finished creating embeddings ---")

        # Create the vector store and persist it automatically
        print("\n--- Creating vector store ---")
        db = Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_directory)
        print("\n--- Finished creating vector store ---")

    else:
        print("Vector store already exists. No need to initialize.")

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

def ask_question(_input):
    """Prompt for a question and handle quit conditions."""
    question = input(f"Ask a question about The Odyssey (type 'quit' to exit): ")
    if question.lower() in ['quit', 'exit', 'q', 'x']:
        return {"question": None}
    return {"question": question, "relevant_document_files": ["../data/books/odyssey.txt"]}

def embed_relevant_documents(input):
    if input.get("relevant_document_files") is None:
        return { "question": None }

    for file in input["relevant_document_files"]:
        embed_file(current_dir, persistent_directory, file)
    return input

def retrieve_relevant_documents(input):
    """Process the question and retrieve relevant documents."""
    if input.get("question") is None:
        return {"question": None, "relevant_docs": None}

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    relevant_docs = retriever.invoke(input["question"])
    return {
        "question": input["question"],
        "relevant_docs": relevant_docs
    }

def prepare_question(input):
    """Generate an answer based on the question and retrieved documents."""
    query = input["question"]
    if query is None:
        return { "prepared_question": None }

    if "relevant_docs" not in input:
        return { "prepared_question": None }

    prepared_question = (
        f"Question: {query}\n\n"
        f"Please provide a clear and concise answer to the above question based on the following documents:\n"
        f"{''.join([doc.page_content for doc in input['relevant_docs']])}\n\n"
        f"Please provide an answer based only on the provided documents."
    )
    return { "prepared_question": prepared_question }

def answer_question(input):
    """Generate an answer based on the question and retrieved documents."""
    if input.get("prepared_question") is None:
        return { "question": None }

    model = ChatOpenAI(temperature=0)
    response = model.invoke(input["prepared_question"])
    return response.content

# Create and configure the workflow graph
workflow = Graph()

# Add all nodes
workflow.add_node("ask_question", ask_question)
workflow.add_node("embed_relevant_documents", embed_relevant_documents)
workflow.add_node("retrieve_relevant_documents", retrieve_relevant_documents)
workflow.add_node("prepare_question", prepare_question)
workflow.add_node("answer_question", answer_question)

# Connect the nodes in sequence
workflow.add_edge("ask_question", "embed_relevant_documents")
workflow.add_edge("embed_relevant_documents", "retrieve_relevant_documents")
workflow.add_edge("retrieve_relevant_documents", "prepare_question")
workflow.add_edge("prepare_question", "answer_question")

# Set entry and finish points
workflow.set_entry_point("ask_question")
workflow.set_finish_point("answer_question")

# Visualize with Graphviz
dot = Digraph(comment='Workflow Visualization')
for node in workflow.nodes: dot.node(node, node)
for edge in workflow.edges: dot.edge(*edge)
dot.render('./4_read_books/workflow_visualization', format='png', cleanup=True)

app = workflow.compile()

while True:
    result = app.invoke("start")
    print("\nAnswer:", result)
    if result == { "question": None }:
        break
