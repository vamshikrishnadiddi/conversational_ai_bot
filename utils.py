from sentence_transformers import SentenceTransformer
import pinecone
import openai
import streamlit as st
openai.api_key = "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"
model = SentenceTransformer('all-MiniLM-L6-v2')

pinecone.init(api_key='e145cf44-347e-4854-a802-525f5c8e3a05', environment='gcp-starter')
index = pinecone.Index('coto-index')
client = openai.OpenAI(api_key="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")
def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

def query_refiner(conversation):

    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    #messages=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    messages = [
        {
            "role": "system", 
            "content": "Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base."
        },
        {
            "role": "user",
            "content": {conversation}
        },
        ],
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    )
    return response['choices'][0]['text']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "user: "+st.session_state['requests'][i] + "\n"
        conversation_string += "assistant: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string