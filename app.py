import streamlit as st 
from dotenv import load_dotenv

#loading the keys 
load_dotenv()


def main():
  st.set_page_config(page_title='hr-asist-tool')
  st.title('🙎‍♀️HR-Resume Screening Assistance')
  st.subheader('I can help you in resume screening progress')

  job_desc=st.text_area("Please paste your 'JOB DESCRIPTION' here")
  num_count=st.text_input('No. of resume to return ')
  files=st.file_uploader('Upload resumes here, Only pdf files allowed',type=['pdf'],accept_multiple_files=True)

  button=st.button('Help me with the analysis')


  if button:
    pass


if __name__=='__main__':
  main()
