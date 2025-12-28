from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

template = ChatPromptTemplate([
    ('system', "You are a helpful {domain} expert."),
    ('human', "Explain the concept of {concept}.")
])    

prompt=template.invoke({
    "domain":"cricket",
    "concept":"duck-worth-lewis"
})

print(prompt)