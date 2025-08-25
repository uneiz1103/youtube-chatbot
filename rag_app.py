from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

#Step 1: Transcript Fetch
video_id = "Gfr50f6ZBvo"

try:
    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id).find_transcript(["en"]).fetch()
    transcript = " ".join(chunk.text for chunk in transcript_list)
except TranscriptsDisabled:
    print("No captions available for this video.")
    transcript = ""

#Step 2: Split text
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

#Step 3: Embedding + Vector Store
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':4})

#Step 4: LLM and Prompt
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know.

    {context}
    Question: {question}
    """,
    input_variables=['context', 'question']
)

#Step 5: Chain Building
def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

parallel_chain = RunnableParallel({
   'context': retriever | RunnableLambda(format_docs),
   'question': RunnablePassthrough()
})

parser = StrOutputParser()
main_chain = parallel_chain | prompt | llm | parser

#Step 6: Invoke Automated
result = main_chain.invoke("Can you summarize the video?")
print("Chain answer:\n", result)
