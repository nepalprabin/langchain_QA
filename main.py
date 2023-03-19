import os
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

import streamlit as st

# pydot env
from dotenv import load_dotenv

# custom import
from utils import load_data, convert_into_chunks, embeddings_, vector_database_setup, store_to_pinecone, docs_, create_index

# loading environment variables
load_dotenv()

# accessing environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV = os.getenv('PINECONE_API_ENV')
index_name = os.getenv('index_name')


# loading the data
# From here down is all the StreamLit UI.
st.header("LangChain Question Answering")


try:
    # uploaded_file = st.file_uploader('Upload a file')
    with st.form("upload-form", clear_on_submit=True):
        uploaded_file = st.file_uploader("Choose a file", accept_multiple_files=False,
                                         type=['pdf'],
                                         help="Upload a file")
        submitted = st.form_submit_button("Upload")

    data = load_data(uploaded_file)

    # Chunking data upto smaller components
    texts = convert_into_chunks(chunk_size=1000, data=data)

    # embeddings
    embeddings = embeddings_(openai_api_key=OPENAI_API_KEY)

    # initializing pinecone vectorstore
    vector_database_setup(PINECONE_API_KEY, PINECONE_API_ENV)

    # storing embeddings to pinecone
    docsearch = store_to_pinecone(texts, embeddings, index_name)

    # loading question answering chain
    def load_chain():
        llm = OpenAI(temperature=0)
        chain = load_qa_chain(llm, chain_type='stuff')
        return chain

    chain = load_chain()
    st.title("Ask a question")

    def get_text():
        input_text = st.text_area(
            'Please enter a question', "What is the text about")
        print(input_text)
        return input_text

    user_input = get_text()

    docs = docs_(docsearch=docsearch, query=user_input)
    output = chain.run(input_documents=docs, question=user_input)
    st.write(output)

except ValueError:
    pass
