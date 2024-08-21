import uvicorn
import os
from fastapi import FastAPI, BackgroundTasks, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
import asyncio
from typing import List
import tempfile
import io
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.prompts import PromptTemplate

from models import QueryRequest

# Set your OpenAI API key here
os.environ["OPENAI_API_KEY"] = "API-KEY_HERE"

app = FastAPI()

# Initialize language model
llm = OpenAI(model_name="gpt-4o",temperature=0.7)

# Initialize vector store for document storage
embeddings = OpenAIEmbeddings()

# Set the path for the persistent storage
persist_directory = 'db'

# Initialize or load the vector store
vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# Define tools for the ReAct agent
tools = [
    Tool(
        name="Search",
        func=lambda q: vector_store.similarity_search(q),
        description="useful for when you need to answer questions about the documents that have been uploaded. Input should be a fully formed question.",
    ),
]

# Define a custom prompt for the agent
custom_prompt = PromptTemplate(
    template="""You are an AI assistant that helps users find information about various topics in markdown format.
    You have access to a Search tool that can provide information from a knowledge base.
    If the Search tool doesn't return any results, use your general knowledge to answer the question.
    Always maintain a friendly and helpful tone.

    Human: {input}
    AI: To answer this question, I'll need to search for relevant information. Let me do that for you.

    {agent_scratchpad}

    Based on the information I've found (or using my general knowledge if no specific information was found), here's my answer:

    """,
    input_variables=["input", "agent_scratchpad"]
)

# Initialize the agent
react_agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True,
    agent_kwargs={
        "prefix": custom_prompt.template
    }
)

# Initialize conversation chain for chat functionality
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)

@app.post("/query")
async def query_knowledge_base(request: QueryRequest):
    async def generate():
        try:
            # Use the ReAct agent to process the query
            agent_response = await asyncio.to_thread(react_agent.run, request.query)
            
            # Use the conversation chain to maintain context and generate the final response
            response = conversation.predict(input=f"Query: {request.query}\nAgent response: {agent_response}\nPlease provide a concise and friendly response based on this information:")
            
            for word in response.split():
                yield f"{word} "
                await asyncio.sleep(0.05)  # Reduced delay for faster response
        except Exception as e:
            yield f"Error processing query: {str(e)}"

    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/train")
async def train_agent(
    background_tasks: BackgroundTasks,
    urls: str = Form(None),
    pdf_files: List[UploadFile] = File(None),
    text_files: List[UploadFile] = File(None)
):
    urls = urls.split(',') if urls else []
    
    if not urls and not pdf_files and not text_files:
        raise HTTPException(status_code=400, detail="No training data provided. Please provide URLs, PDF files, or text files.")
    
    try:
        # Read file contents immediately
        pdf_contents = []
        text_contents = []
        
        for pdf_file in pdf_files or []:
            content = await pdf_file.read()
            pdf_contents.append(content)
        
        for text_file in text_files or []:
            content = await text_file.read()
            text_contents.append(content)
        
        # Start background task with file contents instead of file objects
        background_tasks.add_task(process_training_data, urls, pdf_contents, text_contents)
        return {"message": "Training process started in the background"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting training process: {str(e)}")

async def process_training_data(urls, pdf_contents, text_contents):
    documents = []
    
    try:
        # Process URLs
        for url in urls:
            loader = WebBaseLoader(url)
            documents.extend(loader.load())
        
        # Process PDFs
        for pdf_content in pdf_contents:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(pdf_content)
                temp_file_path = temp_file.name
            
            try:
                loader = PyPDFLoader(temp_file_path)
                documents.extend(loader.load())
            finally:
                os.unlink(temp_file_path)
        
        # Process text files
        for text_content in text_contents:
            # Create a file-like object from the bytes
            text_file = io.StringIO(text_content.decode('utf-8'))
            loader = TextLoader(text_file)
            documents.extend(loader.load())
        
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        
        # Add documents to the vector store
        vector_store.add_documents(texts)
        
        # Persist the vector store
        vector_store.persist()
        
        print(f"Successfully processed and persisted {len(texts)} text chunks.")
    except Exception as e:
        print(f"Error in processing training data: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)