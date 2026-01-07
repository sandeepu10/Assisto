from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os

# LLM USAGE: OpenAI GPT is used as required in the assignment
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"

loader = TextLoader("policy.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

db = FAISS.from_documents(docs, OpenAIEmbeddings())
llm = ChatOpenAI(model="gpt-3.5-turbo")

qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

while True:
 query = input("Ask a question (type exit to quit): ")
 if query.lower() == "exit":
     break
 print(qa.run(query))
