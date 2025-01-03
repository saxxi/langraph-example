import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
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

chat_history = []  # Initialize chat history

def ask_query(_input):
    query = input("Ask a query (type 'quit' to exit): ")
    if query.lower() in ['quit', 'exit', 'q', 'x']:
        return { "step": "ask_query", "query": None}
    return { "step": "ask_query", "query": query}

def contextualize_query(input):
    if input.get("query") is None:
        return { "step": "contextualize_query", "query": None }

    print("""Contextualize the query using chat history.""")
    print(input)
    global chat_history
    chat_history.append({"role": "user", "content": input["query"]})
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user query "
        "which might reference context in the chat history, "
        "formulate a standalone query which can be understood "
        "without the chat history. Do NOT answer the query, just "
        "reformulate it if needed and otherwise return it as is."
    )
    model = ChatOpenAI(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]).format_prompt(chat_history=chat_history, input=input["query"])
    reformulated_query = model.invoke(prompt)
    print("---------")
    print(f"Query: {input['query']}")
    print("---------")
    return { "step": "contextualize_query", "query": reformulated_query, "relevant_document_files": input["relevant_document_files"] }

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

def ask_query(_input):
    """Prompt for a query and handle quit conditions."""

    query = input(f"Ask a query about The Odyssey (type 'quit' to exit): ")
    if query.lower() in ['quit', 'exit', 'q', 'x']:
        return { "step": "ask_query", "query": None}
    return { "step": "ask_query", "query": query, "relevant_document_files": ["../data/books/odyssey.txt"]}

def embed_relevant_documents(input):
    """Embed relevant documents."""

    if input.get("relevant_document_files") is None:
        return { "step": "embed_relevant_documents", "query": None }

    for file in input["relevant_document_files"]:
        embed_file(current_dir, persistent_directory, file)
    return input

def retrieve_relevant_documents(input):
    """Process the query and retrieve relevant documents."""

    if input.get("query") is None:
        return { "step": "retrieve_relevant_documents", "query": None, "relevant_docs": None}

    query = input.get("query").content
    if query is None:
        return { "step": "retrieve_relevant_documents", "query": None, "relevant_docs": None}

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    relevant_docs = retriever.invoke(query)
    return {
        "query": query,
        "relevant_docs": relevant_docs
    }

def prepare_query(input):
    """Generate an answer based on the query and retrieved documents."""

    query = input.get("query")
    if query is None or "relevant_docs" not in input:
        return { "step": "prepare_query", "prepared_query": None }

    prepared_query = (
        f"query: {query}\n\n"
        f"Please provide a clear and concise answer to the above query based on the following documents:\n"
        f"{''.join([doc.page_content for doc in input['relevant_docs']])}\n\n"
        f"Please provide an answer based only on the provided documents."
    )
    return { "step": "prepare_query", "prepared_query": prepared_query }

def answer_query(input):
    """Generate an answer and update chat history."""

    global chat_history
    if input.get("prepared_query") is None:
        return { "step": "answer_query", "query": None}
    model = ChatOpenAI(temperature=0)
    response = model.invoke(input["prepared_query"])
    chat_history.append({"role": "assistant", "content": response.content})
    return { "step": "answer_query", "answer": response.content }

# Create and configure the workflow graph
workflow = Graph()

# Add all nodes
workflow.add_node("ask_query", ask_query)
workflow.add_node("contextualize_query", contextualize_query)
workflow.add_node("embed_relevant_documents", embed_relevant_documents)
workflow.add_node("retrieve_relevant_documents", retrieve_relevant_documents)
workflow.add_node("prepare_query", prepare_query)
workflow.add_node("answer_query", answer_query)

# Connect the nodes in sequence
workflow.add_edge("ask_query", "contextualize_query")
workflow.add_edge("contextualize_query", "embed_relevant_documents")
workflow.add_edge("embed_relevant_documents", "retrieve_relevant_documents")
workflow.add_edge("retrieve_relevant_documents", "prepare_query")
workflow.add_edge("prepare_query", "answer_query")

# Set entry and finish points
workflow.set_entry_point("ask_query")
workflow.set_finish_point("answer_query")

# Visualize with Graphviz
dot = Digraph(comment='Workflow Visualization')
for node in workflow.nodes: dot.node(node, node)
for edge in workflow.edges: dot.edge(*edge)
dot.render('./4_read_books/workflow_visualization', format='png', cleanup=True)

app = workflow.compile()

while True:
    result = app.invoke("start")

    if result.get("answer"):
        print("Answer:", result["answer"])
        # Continue the loop
    elif result.get("query") is None:
        print("Goodbye!")
        break
