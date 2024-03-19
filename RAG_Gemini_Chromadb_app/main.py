
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import base64
# from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from youtube_transcript_api import YouTubeTranscriptApi
import streamlit as st
from langchain.prompts import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate
import speech_recognition as sr
from gtts import gTTS
from langchain_community.vectorstores import Qdrant
import shutil
from audio_recorder_streamlit import audio_recorder


# pip install youtube_transcript_api
# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain_community.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain_community.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        
        return None

    data = loader.load()
    return data





def webpage_loader(url):

    from langchain_community.document_loaders import WebBaseLoader
    loader = WebBaseLoader(f"{url}")
    data = loader.load()
    return data






def chunk_data(data, chunk_size=1100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    return chunks



# def create_embeddings_chroma(chunks,api_key,persist_directory='RAG_Gemini_Chromadb_app/chroma_db'):
   
#     embeddings = GooglePalmEmbeddings(google_api_key=api_key)
#     vector_store = Chroma.from_documents(chunks,embeddings,persist_directory=persist_directory)
#     vector_store = Chroma(persist_directory=persist_directory,embedding_function=embeddings)
#     vector_store.persist()
#     return vector_store




def create_embeddings_chroma(chunks, api_key):

    embeddings = GooglePalmEmbeddings(google_api_key=api_key)

    try:
        shutil.rmtree("RAG_Gemini_Chromadb_app/db")

    except:
        pass

   
    vector_store = Qdrant.from_documents(chunks, embeddings, path="RAG_Gemini_Chromadb_app/db", collection="store")

    return vector_store




# def delete_vector_store():
#     import subprocess
#     vector_store = Chroma(persist_directory='RAG_Gemini_Chromadb_app/chroma_db',embedding_function='RAG_Gemini_Chromadb_app/chroma_db')
#     command = ["zip", "-r", 'RAG_Gemini_Chromadb_app/chroma_db' , 'RAG_Gemini_Chromadb_app/chroma_db']
#     result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     vector_store.delete_collection()
#     vector_store.persist()
#     if result.returncode == 0:
#         print("Directory successfully zipped!")
#     else:
#         print("Error occurred while zipping the directory.")
#     os.system("rm -rf " + 'RAG_Gemini_Chromadb_app/chroma_db.zip')







def speech_to_text():
    recognizer = sr.Recognizer()
    # with sr.Microphone() as source:
        
    #     recognizer.adjust_for_ambient_noise(source)
    #     audio = recognizer.listen(source)
    try:
        
        # text = recognizer.recognize_google(audio)
        with sr.AudioFile("RAG_Gemini_Chromadb_app/record.mp3") as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.write("Sorry, I could not understand audio.")
        pass
    except sr.RequestError as e:
        st.write(f"Could not request results from Google Speech Recognition service,Try typing queries; {e}")
        pass






def generate_audio(text):
    try:
        print('done1')
        tts = gTTS(text=text, lang='en')
        tts.save("RAG_Gemini_Chromadb_app/output.mp3")
        audio_file = open("RAG_Gemini_Chromadb_app/output.mp3", "rb")
        audio_bytes = audio_file.read()
        
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        audio_tag = f'<audio src="data:audio/mp3;base64,{audio_base64}" controls autoplay>'
        st.markdown(audio_tag, unsafe_allow_html=True)
        os.remove("RAG_Gemini_Chromadb_app/output.mp3")
    except:
        print('failed')
        pass





def ask_question(q,chain):
    result = chain.invoke({'question':q})
    return result




def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']









if  __name__ == "__main__":

    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    key=os.environ.get('GEMINI_API_KEY')

    
    @st.cache_data
    def get_img_as_base64(file):
        with open(file,"rb") as f:
            data = f.read()   
        return base64.b64encode(data).decode()    

    img1 = get_img_as_base64('RAG_Gemini_Chromadb_app/ai_img9.jpeg')
    img2 = get_img_as_base64('RAG_Gemini_Chromadb_app/dark2.jpg')


    st.markdown(
    """
    <style>
    .main {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        opacity:1;
    }
    [data-testid=stSidebar] {
        background-image: url("data:image/png;base64,%s");
         background-position: right;
    }
    @media screen and (max-width: 768px) {
        .main {
            background-size: 800px; 
        }
  
    </style>
    """ % (img1,img2),
    unsafe_allow_html=True
    )

    
    st.write('<span style="font-size:8vw; font-weight:bolder; margin-left:-7vw; background: linear-gradient(45deg, #2c3e50, #4ca1af); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">aASKio', unsafe_allow_html=True)

    st.write('<span style="font-size:3.5vw; margin-left:-2vw; font-weight:bolder; background: linear-gradient(45deg, #3CA55C, #B5AC49); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Chat with Youtube Websites and Documents', unsafe_allow_html=True)



    with st.sidebar:
        st.write('<span style="color:green; font-weight:bolder;">Try for free!,without your api_key', unsafe_allow_html=True)
        st.write('<span style="color:green; font-weight:bolder;">In case of error get your api key here [Google Gemini](https://aistudio.google.com/app/apikey)', unsafe_allow_html=True)
        API_KEY = st.text_input('Your Gemini API key  ',type='password')
        if API_KEY:
            api_key = API_KEY
        else:
            api_key = key    

        uploaded_file = st.file_uploader('Upload a file',type=['pdf','docx','txt'])

        youtube_link = st.text_input("Paste YouTube or website Link Here ")
       
        add_data = st.button('Submit data',on_click=clear_history) 

        if (uploaded_file and add_data) or (youtube_link and add_data):
            
            if youtube_link:
                
                if "youtube.com" in youtube_link or "youtu.be" in youtube_link:
                    # Extracting video ID from the link
                    video_id = youtube_link.split("v=")[1] if "youtube.com" in youtube_link else youtube_link.split("/")[-1]
                    
                    # Constructing embedded YouTube video URL
                    embedded_url = f"https://www.youtube.com/embed/{video_id}"
                    

                    st.write(f"Embedded Video:")
                    st.write(f'<iframe width="280" height="170" src="{embedded_url}" frameborder="0" allowfullscreen></iframe>', unsafe_allow_html=True)


                    with st.spinner('Reading and embedding the file......'):
                        
                        try:
                            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                    

                            transcript_text = ""
                            for segment in transcript_list:
                                transcript_text += segment['text'] + "\n"

                            file_name = os.path.join('RAG_Gemini_Chromadb_app/','data.txt')
                            with open(file_name, 'w', encoding='utf-8') as f:
                                f.write(transcript_text)

                            data = load_document(file_name)
                            chunks = chunk_data(data)

                            
                            # delete_vector_store()
                            vector_store = create_embeddings_chroma(chunks,api_key)  
                            # vector_store = load_embeddings_chroma()


                            st.session_state.vs = vector_store
                            st.success('Process completed successfully!')

                        except:
                            st.warning("No trancript found for this video!")




                else:
                    url =youtube_link
                    try:
                        with st.spinner('Reading and embedding the file......'):
                            data = webpage_loader(url)
                            chunks = chunk_data(data)

                                
                            # delete_vector_store()
                            vector_store = create_embeddings_chroma(chunks,api_key)  
                            # vector_store = load_embeddings_chroma()


                            st.session_state.vs = vector_store
                            st.success('Process completed successfully!')

                    except:
                        st.error("Corrupted link!")

                    

                
            else:

                with st.spinner('Reading and embedding the file......'):
                    bytes_data = uploaded_file.read()
                    file_name = os.path.join('RAG_Gemini_Chromadb_app/',uploaded_file.name)
                    with open(file_name,'wb') as f:
                        f.write(bytes_data)
                    try:    

                        data = load_document(file_name)
                        chunks = chunk_data(data)
                        
                        # delete_vector_store()        
                        vector_store = create_embeddings_chroma(chunks,api_key)  
                        # vector_store = load_embeddings_chroma()

                        os.remove(f'{file_name}')
                        st.session_state.vs = vector_store
                        st.success('Process completed successfully!')
                    except:
                        os.remove(f'{file_name}')
                        st.error('Document format is not supported!')   
    
            try:


                with st.spinner('Generating abstract...........'):
                    if 'vs' in st.session_state: 

                            st.subheader("Abstract")
                            llm1 = ChatGoogleGenerativeAI(model='gemini-pro',temperature=0,google_api_key=api_key,convert_system_message_to_human=True)
                            print('done1')
                            vector_store1 = st.session_state.vs
                            print('done2')
                            retriever1 = vector_store1.as_retriever(search_type='similarity', search_kwargs={'k': 15})
                            print('done3')
                            memory1 = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
                            system_template1 = r'''

                                Use the following pieces of context to answer the user's question.

                                Context:```{context}```

                                ''' 

                            user_template1 = '''

                                Question: ```{question}```
                                Chat History: ```{chat_history}```
                                '''
                            messages1 = [
                            SystemMessagePromptTemplate.from_template(system_template1),
                            HumanMessagePromptTemplate.from_template(user_template1)
                            ]



                            qa_prompt1 = ChatPromptTemplate.from_messages(messages1)

                        

                            crc1 = ConversationalRetrievalChain.from_llm(
                                llm = llm1,
                                retriever = retriever1,
                                chain_type='stuff',
                                memory = memory1,  
                                combine_docs_chain_kwargs={'prompt':qa_prompt1},
                                verbose=True
                            
                                )
                            print('done4')
                            q = 'Generate a brief summary in points from this text'

                            abstract = ask_question(q,crc1)
                            print('done5')
                            st.write(abstract["answer"])
                            
                            
            except:
                st.warning("Invalid api key!")                
    
    try:
                     
                        llm = ChatGoogleGenerativeAI(model='gemini-pro',temperature=0,google_api_key=api_key,convert_system_message_to_human=True)
                        
                        vector_store = st.session_state.vs
                        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 5})
                        
                        memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
                    


                        system_template = r'''

                        Use the following pieces of context to answer the user's question.

                        Context:```{context}```

                        '''


                        user_template = '''

                        Question: ```{question}```
                        Chat History: ```{chat_history}```
                        '''

                        messages = [
                            SystemMessagePromptTemplate.from_template(system_template),
                            HumanMessagePromptTemplate.from_template(user_template)
                        ]

                    
                        qa_prompt = ChatPromptTemplate.from_messages(messages)

                    

                        crc = ConversationalRetrievalChain.from_llm(
                            llm = llm,
                            retriever = retriever,
                            memory = memory,
                            chain_type='stuff',  
                            combine_docs_chain_kwargs={'prompt':qa_prompt},
                            verbose=True
                            
                            )
                        
                        
                        audio = st.radio("AI voice assistant", ("ON", "OFF"))

                        question = st.text_input('Chat with the uploaded data:(refer as "this text" in case of confused answers)') 

                        col1, col2 =st.columns([2,12])

                        with col1:
                            ans = st.button("Answer")
                        with col2:
                            # audi = st.button("Tap and Ask")  
                            audi = audio_recorder()

                        # if ans or audi:

                        if ans:

                            with st.spinner('Generating answer........'):
                                    
                                answer = ask_question(question,crc)
                                flag = True
                                

                        elif audi:

                            recorded_file = "RAG_Gemini_Chromadb_app/record.mp3"
                            with open(recorded_file,"wb") as f:
                                f.write(audi)

                            #     with st.spinner('Listening..........'):

                            #         # st.write('Listining.......')
                            question = speech_to_text()

                            st.write(question)

                            with st.spinner('Generating answer.........'):

                                answer = ask_question(question,crc)
                                flag=True
                        if audio == "ON" and flag:

                            st.text_area('Answer : ',value=answer["answer"],height=200)
                            if flag:
                                generate_audio(answer["answer"])
                                flag=False

                        else:

                            st.text_area('Answer : ',value=answer["answer"],height=200) 

                        st.divider()
                        if 'history' not in st.session_state:
                            st.session_state.history = '' 
                        value = f'Q : {question} \nA : {answer["answer"]}'
                        st.session_state.history = f'{value} \n {"--"*64} \n {st.session_state.history}'
                        h = st.session_state.history  
                        st.text_area(label='Chat History',value=h,key='history',height=500)  


    except:
        pass

        
    