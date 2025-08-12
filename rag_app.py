from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# youtube video ID
video_id = "Gfr50f6ZBvo"

try:
    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id).find_transcript(["en"]).fetch()

    transcript = " ".join(chunk.text for chunk in transcript_list)

    # print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")

# Text Splitter

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

print(len(chunks))