from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

chat_template=ChatPromptTemplate([
    ('system', "You are a customer support assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human','{query}')
])

chat_history=[]

with open('chat_history.txt','r') as f:
    chat_history.extend(f.readlines())
    
##print(chat_history)

prompt=chat_template.invoke({
    "chat_history":chat_history, 'query':"How can I reset my password?"
})

print(prompt)