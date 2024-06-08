import streamlit as st
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma


def video_id_extractor(video_url):
    parsed_url = urlparse(video_url)
    query_params = parse_qs(parsed_url.query)
    video_ids = query_params.get('v')

    if video_ids:
        return video_ids[0]
    return None


def text_processing(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=600,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    knowledge_base = Chroma.from_texts(chunks, embeddings)
    return knowledge_base


def summarizer(database):
    query = "Summarize this transcripts of a podcast in 10-15 sentences"
    docs = database.similarity_search(query)
    llm = Ollama(model="llama3")
    query_template = f"{docs} \n\n\n {query}"
    response = llm.invoke(query_template)
    return response


if "test_text" not in st.session_state:
    st.session_state.test_text = ""

st.title("Youtube video summarizer")
st.markdown("""Welcome to **PodMaster**, the app that brings you *concise summaries* of YouTube podcasts. Stay 
    *informed* and *entertained* without the hassle of watching lengthy videos. Discover and enjoy your favorite content 
    on the go with **PodMaster** – where *every minute counts*! """)
url = st.text_input(label="Paste your URL here")

if url:
    video_id = video_id_extractor(url)
    with st.spinner("Fetching the video"):
        api_data = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        if api_data:
            extracted_subs = ""
            for chunk in api_data:
                extracted_subs += f"{chunk['text']} \n"

            expander = st.expander("Transcripts")
            expander.write(extracted_subs)

    with st.spinner("Processing text"):
        faiss_database = text_processing(extracted_subs)

    with st.spinner("Summarizing the video"):
        st.session_state.test_text = summarizer(faiss_database)

    st.subheader("Summary of the video")
    st.write(st.session_state.test_text)
