from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
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

# print(len(chunks))

# Embedding Generation and Storing in Vector Store
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
vector_store = FAISS.from_documents(chunks, embeddings)

# vector_store.index_to_docstore_id

# Retrieval
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':4})

retriever.invoke('What is deepmind')

# Augmentation

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables= ['context', 'question']
)

question = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs    = retriever.invoke(question)

# print(retrieved_docs)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

# print(context_text)

final_prompt = prompt.invoke({"context": context_text, "question": question})

# Generation
answer = llm.invoke(final_prompt)
print(answer.content)

# making chain
def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
   'context': retriever | RunnableLambda(format_docs),
   'question': RunnablePassthrough()
})

parallel_chain.invoke('who is demis')

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

main_chain.invoke('Can you summarize the video')