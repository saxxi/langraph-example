from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import Graph

load_dotenv()

model = ChatOpenAI(temperature=0)

def get_animal(_input):
    animal = input("What animal would you like to hear a joke about? ")
    return {"animal": animal}

def create_joke(animal):
    prompt = f"Create a short, funny joke about a {animal}. The joke should be appropriate for all ages."
    response = model.invoke(prompt)
    return response.content

workflow = Graph()
workflow.add_node("get_animal", get_animal)
workflow.add_node("create_joke", create_joke)
workflow.add_edge("get_animal", "create_joke")

workflow.set_entry_point("get_animal")
workflow.set_finish_point("create_joke")

app = workflow.compile()

result = app.invoke("some input")

print(result)
