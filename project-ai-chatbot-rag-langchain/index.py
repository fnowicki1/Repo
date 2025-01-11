import os
from lib import utils
from lib.streaming import StreamHandler
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate



st.set_page_config(page_title="Chat", page_icon="📄")
st.header('AI chat - search for information in source documents with RAG')
st.write('This application has access to custom documents and can respond to user queries by referring to the content within those documents.')

class CustomDocChatbot:

    def __init__(self):
        utils.sync_st_session()
        self.llm = utils.configure_llm()
        self.embedding_model = utils.configure_embedding_model()

    @st.spinner('Analyzing documents..')
    def import_source_documents(self):
        # load documents
        docs = []
        files = []
        for file in os.listdir("data"):
            if file.endswith(".txt"):
                with open(os.path.join("data", file), encoding="utf-8") as f:
                    docs.append(os.path.join("data", f.read()))
                    files.append(file)

        # Split documents and store in vector db
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=30000,
            chunk_overlap=5000
        )

        splits = []
        for i, doc in enumerate(docs):
            for chunk in text_splitter.split_text(doc):
                splits.append(Document(page_content=chunk, metadata={"source": files[i]}))

        vectordb = DocArrayInMemorySearch.from_documents(splits, self.embedding_model)

        # Define retriever (similarity or mmr)
        retriever = vectordb.as_retriever(
            search_type='similarity',
            search_kwargs={'k':2, 'fetch_k':4}
        )

        # Setup memory for contextual conversation
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            output_key='answer',
            return_messages=True
        )

        system_message_prompt = SystemMessagePromptTemplate.from_template(
            """
            You are a chatbot tasked with answering questions based strictly on the information provided in the attached documents. These documents contain the book titled *Pathfinder Player Core*, which outlines the remastered rules for the second edition of Pathfinder.

            ### Key Guidelines:
            1. **Depend Exclusively on the Documents**: Base all answers solely on the content of the provided documents. Do not infer or include information that cannot be directly substantiated by the documents.
            2. **Cite Your Sources**: For every response, explicitly mention the name of the document and the specific section or page, if available, where the information was found.
            3. **Handle Uncertainty with Transparency**: If the documents do not provide an exact answer, explain this clearly and provide the closest possible answer using related information from the documents. Do not speculate beyond what the documents offer.
            4. **Example Creation**: If the user requests examples (e.g., a character sheet), create them using the character sheet layout described in the documents. Ensure every aspect of the example is consistent with the rules and guidelines in *Pathfinder Player Core*. Indicate which document sections were used to create the example.
            5. **Formatting**:
                - Use clear headings and bullet points where appropriate.
                - Include citations in brackets (e.g., *Pathfinder Player Core, Chapter 3, Page 45*).

            ### Task:
            Considering the above instructions, answer the following question. Depend strictly on the provided source documents and adhere to the specified guidelines for clarity and accuracy.

            {context}

            {question}
            
            """
        )

        prompt = ChatPromptTemplate.from_messages([system_message_prompt])

        # Setup LLM and QA chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False,
            combine_docs_chain_kwargs={"prompt": prompt}
        )

        return qa_chain

    @utils.enable_chat_history
    def main(self):
        user_query = st.chat_input(placeholder="Ask for information from documents")

        if user_query:
            qa_chain = self.import_source_documents()

            utils.display_msg(user_query, 'user')

            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())

                result = qa_chain.invoke(
                    {"question":user_query},
                    {"callbacks": [st_cb]}
                )
                response = result["answer"]
                st.session_state.messages.append({"role": "assistant", "content": response})
                utils.print_qa(CustomDocChatbot, user_query, response)

                

if __name__ == "__main__":
    obj = CustomDocChatbot()
    obj.main()