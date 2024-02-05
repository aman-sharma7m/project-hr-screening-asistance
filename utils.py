import streamlit as st
import PyPDF2
from langchain.docstore.document import Document
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores.pinecone import Pinecone
from pinecone import Pinecone as pc, PodSpec
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter


def extract_text(pdf_file):
  pdf_obj=PyPDF2.PdfReader(pdf_file)
  text=''
  for page in pdf_obj.pages:
    data=page.extract_text()
    text+=data
  return text

def create_docs(pdf_list,unique_id):
  docs=[]
  for file in pdf_list:
    text=extract_text(file)
    docs.append(Document(page_content=text,metadata={'name':file.name,
                                                     'size':file.size,
                                                     'type':file.type,
                                                     'id':file.file_id,
                                                     'unique_id':unique_id}))
  return docs


def get_embeddings():
  return SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-V2')


def store_to_pinecone(docs,embeddings):
  pc_config=pc()
  index_name='hr-assist'
  for name in pc_config.list_indexes().names():
    if name!=index_name:
      try:
        pc_config.delete_index(name)
        print(f'delete index {index_name}')
      except Exception as e:
        print('no index is there')

  if pc_config.list_indexes().names()==[]:
    print('creating new index')
    pc_config.create_index(name=index_name,
                    dimension=embeddings.client.get_sentence_embedding_dimension(),
                    metric='dotproduct',
                    spec=PodSpec(environment='gcp-starter'))
  
  Pinecone.from_documents(docs,embeddings,index_name=index_name)


def get_docs_from_pinecone(query,num_count,embeddings,u_id):
  index_name='hr-assist'
  index=Pinecone.from_existing_index(index_name=index_name,embedding=embeddings)
  return index.similarity_search_with_score(query,int(num_count),{'unique_id':u_id})


def get_summary(rel_doc):
  ts=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=20)
  chunk_doc=ts.split_documents([rel_doc[0]])
  chain=load_summarize_chain(OpenAI(),chain_type='map_reduce',verbose=True)
  summary=chain.run(chunk_doc)
  return summary


