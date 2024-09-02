__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import asyncio
import random
from datetime import datetime

import streamlit as st




from langchain_google_genai import ChatGoogleGenerativeAI ,GoogleGenerativeAIEmbeddings

from langchain.prompts import PromptTemplate
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.output_parsers import JsonOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_community.document_loaders import PyPDFLoader
from typing import Any, Dict, List
from langchain_core.outputs import LLMResult





os.environ["GOOGLE_API_KEY"] = st.secrets["G_API_KEY"]

@st.cache_resource(show_spinner=True)
def load_vectorstore():
    loader = PyPDFLoader("Medical_Receptionist_QnA.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
    doc_splits = text_splitter.split_documents(docs)
    embd = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(documents=doc_splits, collection_name="rag-chroma", embedding=embd)
    return vectorstore.as_retriever()

retriever = load_vectorstore()



class MyCustomAsyncHandler(AsyncCallbackHandler):
    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        print("Starting LLM processing...")
        await asyncio.sleep(0.3)  # Simulate delay
        print("LLM is processing your request.")

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        print("LLM processing complete.")
        await asyncio.sleep(0.3)  # Simulate delay
        print("Here's your response.")


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.6,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    callbacks=[MyCustomAsyncHandler()],
)


async def check_if_emergency(message: str):
    check_emergency_prompt = PromptTemplate(
        template="""
        You are a medical assistant bot that determines if a user's situation is an emergency. 
        You will receive a message from a user, and your job is to check if it is an emergency. 
        If it's an emergency, return a JSON with `can_answer=true`. 
        Otherwise, return `can_answer=false`. 

        Return one of the following JSON:
        {{"reasoning": "User has difficulty breathing, which is an emergency.", "can_answer": true}}
        {{"reasoning": "User wants to leave a message.", "can_answer": false}}

        Message: {message} \n
       """,
        input_variables=["message"],
    )

    check_emergency_chain = check_emergency_prompt | llm | JsonOutputParser()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, check_emergency_chain.invoke, {"message": message})
    return result["can_answer"], result["reasoning"]

async def query_database(emergency: str):
    await asyncio.sleep(15)  # Artificial delay 
    
    
    docs = retriever.get_relevant_documents(emergency)
    context = docs[0].page_content if docs else "No information available."

    llm_prompt_template = PromptTemplate(template="""
    You are a helpful AI medical assistant. 
    You have the following context to answer the user's emergency: 

    {context}

    Now, generate a helpful response based on the emergency situation described:
    "{emergency}"

   """ , input_variables=["context" , "emergency"])

    llm_chain = llm_prompt_template | llm
    res = llm_chain.invoke({"context": context, "emergency" : emergency})
  

    return res.content


def generate_random_eta():
    return random.randint(5, 20)  # Random ETA between 5 to 20 min


async def handle_emergency_conversation(emergency: str, location: str):
    query_task = asyncio.create_task(query_database(emergency))
   
    eta = generate_random_eta()
    st.write(f"Dr. Adrin will be coming to {location} immediately. Estimated time of arrival: {eta} minutes.")
    show = False

    if not query_task.done():
        new_input = st.text_input("You:", key="key")
        if new_input.lower() == "exit":
           st.write("Ending conversation.")
           
            
        elif "late" in new_input.lower():
            if query_task.done():
                action = await query_task
                st.write(f"I understand that you are worried that Dr. Adrin will arrive too late, meanwhile we suggest you start CPR: {action}.")
                
            else:
                st.write("Please hold just a sec while I finish checking the immediate steps for you.")
                
        elif "location" in new_input.lower():
            st.write(f"Dr. Adrin is already on the way to {location}. ETA: {eta} minutes.")
            
        else:
            st.write("I'm still processing your emergency.")
    
    
    if st.session_state["action"] is None:
        with st.spinner("Fetching the necessary information..."):
            action = await query_task
            st.session_state["action"] = action
            show = True
    else:
        action = st.session_state["action"]
        show = True

    st.write(f"Immediate action: {action}")
    
    
    if show:

        follow_up = st.selectbox("Would you like to:", ["Ask a new question", "Clarify doubts about this advice"], key="follow_up_options")

        if follow_up == "Ask a new question":
            await handle_new_question()
        elif follow_up == "Clarify doubts about this advice":
            await handle_clarification(action)


async def handle_new_question():
    new_message = st.text_input("What's your question?", key="new_question")
    if new_message:
        st.write("Processing your new question...")
        st.session_state.clear()
        st.rerun()



async def handle_clarification(previous_action: str):
    clarification = st.text_input("Please specify your doubt:", key="clarification")
    if clarification:
        
        st.write("Let me clarify that for you...")
        response = await asyncio.run(clarify_doubt(previous_action, clarification))
        st.write(response)

async def clarify_doubt(previous_action: str, clarification: str):
    
    return f"Regarding the previous advice '{previous_action}', your clarification '{clarification}' means that you should continue to follow the given steps. If you're still unsure, contact the doctor immediately."



def main():
    st.title("AI Medical Receptionist")
    st.write("Welcome! How can I assist you today?")

    if "action" not in st.session_state:
        st.session_state["action"] = None

    user_input = st.text_input("You: ", key="input")

    if user_input:
        
        can_answer, reasoning = asyncio.run(check_if_emergency(user_input))
        
        if can_answer:
            st.write(reasoning)
          
            location = st.text_input("Meanwhile, can you tell me which area are you located right now?", key="location")
            if location:
                action = asyncio.run(handle_emergency_conversation(reasoning, location))
                st.write(f"Immediate action: {action}")
               
               
        else:
            
            user_message = st.text_input("Please leave your message.")
            if user_message:
                st.write("Thanks for the message, we will forward it to Dr. Adrin.")
    


if __name__ == "__main__":
    main()
