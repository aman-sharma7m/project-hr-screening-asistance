import streamlit as st 
from dotenv import load_dotenv
from uuid import uuid4
from utils import *
import time



#session_state
if 'unique_key' not in st.session_state:
  st.session_state['unique_key']=''


def main():
  st.set_page_config(page_title='hr-asist-tool')
  st.title('🙎‍♀️ HR-Resume Screening Assistance')
  st.subheader('I can help you in resume screening progress')

  #loading the keys 
  load_dotenv()

  job_desc=st.text_area("Please paste your 'JOB DESCRIPTION' here")
  num_count=st.text_input('No. of resume to return ')
  files=st.file_uploader('Upload resumes here, Only pdf files allowed',type=['pdf'],accept_multiple_files=True)

  button=st.button('Help me with the analysis')


  if button:
    with st.spinner('Wait for it........'):
      #getting unique id so that each session only uploaded file will be used to select the data
      st.session_state['unique_key']=uuid4().hex
      #sending the files to create docs
      docs=create_docs(files,st.session_state['unique_key'])
      st.write('creating the docs......')

      #get embeddings
      st.write('getting embeddings......')
      embeddings=get_embeddings()
      

      #store to pincone
      st.write('storing the docs......')
      store_to_pinecone(docs,embeddings)
      
      time.sleep(20)

      #retrieval of docs
      st.write('retrieve relevant docs......')
      relevant_docs=get_docs_from_pinecone(job_desc,num_count,embeddings,st.session_state['unique_key'])
      # st.write(relevant_docs)


    st.success('Process completed 😎😎')
    #summary
    for doc in relevant_docs:
      summary=get_summary(doc)
      st.divider()
      st.write(f"Name: {doc[0].metadata['name']}")
      st.write(f'Score:{doc[1]}')
      with st.expander('Show Summary'):
        st.write('Summary:')
        st.write(f'{summary}')




if __name__=='__main__':
  main()
