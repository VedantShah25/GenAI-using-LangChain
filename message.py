from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

messages=[
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Explain what is langchain?")
]

result = model.invoke(messages) 

messages.append(AIMessage(content=result.content))

print(messages)