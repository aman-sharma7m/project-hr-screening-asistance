# Project HR-Resume Screening Assistance

## Introduction

we have a Streamlit application that assists in the screening of resumes for HR purposes. The application allows users to upload job descriptions and resumes in PDF format, and then performs an analysis to retrieve relevant resumes based on the job description.

## Installation & create environment

Clone the project

```bash
  git clone link_to_copy
```

Go to the project directory

```bash
  cd proj_dir
```

Create the enviroment

```bash
  conda create --prefix ./lang_env
  conda activate {path}/lang_env
  python -m ipykernel install --user --name=lang_env
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Insert Open-ai api key

```bash
  touch .env
  insert your key as following
  OPENAI_API_KEY=""
  HUGGINGFACEHUB_API_TOKEN=""
  PINECONE_API_KEY=""
```

Start the server

```bash
  streamlit run app.py
```

## Intro to libraries

Ticket Classification: Ticket classification is the process of categorizing user queries or tickets into different categories based on their content. This helps in efficiently assigning tickets to the appropriate department or team for resolution.

Natural Language Processing (NLP): NLP is a branch of artificial intelligence that focuses on the interaction between computers and human language. It involves techniques for understanding, analyzing, and generating human language.

Embeddings: Embeddings are numerical representations of words or sentences that capture their semantic meaning. They are used in NLP tasks such as text classification, information retrieval, and machine translation.

Pinecone: Pinecone is a vector database that allows fast and efficient similarity search on high-dimensional embeddings. It provides a scalable solution for storing and querying embeddings.

### Detailed Explanation:

This code is a Streamlit application that assists in resume screening for HR purposes.

First, the necessary libraries are imported: `streamlit`, `dotenv`, `uuid`, `utils`, and `time`.

The `st.session_state` is used to store a unique key for each session. If the key does not exist in the session state, it is initialized as an empty string.

The `main()` function is defined, which is the entry point of the application.

The page configuration is set using `st.set_page_config()`, specifying the page title as "hr-asist-tool".

The title and subheader of the application are displayed using `st.title()` and `st.subheader()`.

The `load_dotenv()` function is called to load any environment variables from a .env file.

A text area is displayed using `st.text_area()` to allow the user to paste the job description.

A text input is displayed using `st.text_input()` to allow the user to enter the number of resumes to return.

A file uploader is displayed using `st.file_uploader()` to allow the user to upload PDF files. Only PDF files are allowed.

A button is displayed using `st.button()` with the label "Help me with the analysis".

When the button is clicked, the code inside the `if button:` block is executed.

A spinner is displayed using `st.spinner()` with the message "Wait for it........".

A unique key is generated using `uuid4().hex` and stored in the session state.

The `create_docs()` function is called with the uploaded files and the unique key to create document objects.

A message is displayed using `st.write()` to indicate that the documents are being created.

The `get_embeddings()` function is called to get the embeddings of the documents.

A message is displayed to indicate that the embeddings are being retrieved.

The `store_to_pinecone()` function is called to store the documents and embeddings in Pinecone.

A message is displayed to indicate that the documents are being stored.

A delay of 20 seconds is added using `time.sleep()` to simulate a processing time.

The `get_docs_from_pinecone()` function is called to retrieve relevant documents based on the job description, number of resumes to return, embeddings, and unique key.

A message is displayed to indicate that the relevant documents are being retrieved.

The `get_summary()` function is called to get a summary of each relevant document.

A success message is displayed using `st.success()`.

A loop is used to iterate over each relevant document.

The name and score of the document are displayed using `st.write()`.

An expander is used to show the summary of the document when expanded.

The `if __name__=='__main__':` block is used to ensure that the `main()` function is only executed when the script is run directly, not when it is imported as a module.

extract_text(pdf_file): This function takes a PDF file as input and uses PyPDF2 to extract the text from the file. It returns the extracted text as a string.

create_docs(pdf_list, unique_id): This function takes a list of PDF files and a unique ID as input. It iterates over the PDF files, extracts the text using extract_text(), and creates a Document object for each file. The Document object includes the page content and metadata. The function returns a list of Document objects.

get_embeddings(): This function initializes and returns a SentenceTransformerEmbeddings object. The object is used to generate sentence embeddings for documents.

store_to_pinecone(docs, embeddings): This function takes a list of Document objects and a SentenceTransformerEmbeddings object as input. It creates a Pinecone index (if it doesn't exist) and stores the document embeddings in the index.

get_docs_from_pinecone(query, num_count, embeddings, u_id): This function takes a query string, the number of documents to retrieve, a SentenceTransformerEmbeddings object, and a unique ID as input. It retrieves the most similar documents to the query from the Pinecone index based on their embeddings. The function returns the retrieved documents along with their similarity scores.

get_summary(rel_doc): This function takes a relevant document (returned by get_docs_from_pinecone()) as input. It splits the document into smaller chunks using a text splitter, and then generates a summary of the document using a summarization chain based on OpenAI's language model.

## libraries

```
import streamlit as st
import PyPDF2
from langchain.docstore.document import Document
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores.pinecone import Pinecone
from pinecone import Pinecone as pc, PodSpec
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

```

## Conclusion

we have explained the code for an HR Assistant application. The code includes functions for extracting text from PDF files, creating document objects, storing documents in a vector store, retrieving documents based on a query, and generating summaries of documents. By understanding this code, you can build and customize your own HR Assistant application to streamline your HR processes.
