"""
    This file helps feeding the custom knowledge to Vector Databases 
    In simple words add knoeledge to chatbot
"""
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
import pinecone 
from langchain.vectorstores import Pinecone
from langchain.document_loaders import SeleniumURLLoader

def load_docs():
    urls = ["https://coto.world/privacypolicy","https://coto.world/spotlight-creator-program","https://coto.world/ar/about-us","https://coto.world/global/about-us","https://coto.world/privacypolicy/information-about-contacts-data","https://coto.world/cotonfts","https://coto.world/coto-uni","https://coto.world/community-roles-responsibilities",]
    urls.append('https://www.coto.world/')
    loader = SeleniumURLLoader(urls=urls)
    documents = loader.load()
    return documents

#documents = load_docs()

directory = './data'

def load_docs_dir(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs_dir(directory)
print(documents)
len(documents)



def split_docs(documents,chunk_size=500,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
print(len(docs))


embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


pinecone.init(
    api_key="e145cf44-347e-4854-a802-525f5c8e3a05",  # find at app.pinecone.io
    environment="gcp-starter"  # next to api key in console
)
index_name = "coto-index"
index = Pinecone.from_documents(docs, embeddings, index_name=index_name)

"""def get_similiar_docs(query,k=1,score=False):
  if score:
    similar_docs = index.similarity_search_with_score(query,k=k)
  else:
    similar_docs = index.similarity_search(query,k=k)
  return similar_docs"""
