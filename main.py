import os
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain

import streamlit as st

# Creating embeddings for semantic search
import pinecone

# loading the data
# From here down is all the StreamLit UI.
st.header("LangChain Question Answering")


try:
    # uploaded_file = st.file_uploader('Upload a file')
    with st.spinner('Wait for it...'):
        with st.form("upload-form", clear_on_submit=True):
            uploaded_file = st.file_uploader("Choose a file", accept_multiple_files=False,
                                             type=['pdf'],
                                             help="Upload a file")
            submitted = st.form_submit_button("Upload")

    # with st.spinner('Wait for it...'):
    #     time.sleep(5)

    loader = UnstructuredPDFLoader(uploaded_file)
    data = loader.load()

    # Chunking data upto smaller components
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])

    # initializing pinecone
    pinecone.init(
        api_key=os.environ['PINECONE_API_KEY'],
        environment=os.environ['PINECONE_API_ENV']
    )

    docsearch = Pinecone.from_texts(
        [t.page_content for t in texts], embeddings, index_name=index_name)

    # query = "What is the linkedin profile link?"
    # docs = docsearch.similarity_search(query, include_metadata=True)

    def load_chain():
        """Logic for loading the chain you want to use should go here."""
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
    print('-----', user_input)

    # if user_input:
    # docs = docs_(docsearch, user_input)
    docs = docsearch.similarity_search(user_input, include_metadata=True)
    output = chain.run(input_documents=docs, question=user_input)
    st.write(output)

    #     st.session_state.past.append(user_input)
    #     st.session_state.generated.append(output)

    # if st.session_state["generated"]:

    #     for i in range(len(st.session_state["generated"]) - 1, -1, -1):
    #         message(st.session_state["generated"][i], key=str(i))
    #         message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

except ValueError:
    pass