import re
import ollama
import os
import chromadb
import numpy as np
from sklearn.cluster import KMeans
from chromadb.utils import embedding_functions
from typing import List
import psycopg
from psycopg.rows import dict_row
import ast
from tqdm import tqdm
from colorama import Fore
import glob
from multiprocessing import Pool
import pickle
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain_core.prompts import PromptTemplate

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)


# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"]="text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}

persist_directory='/home/alberto/chroma_test_db'
source_directory='/home/alberto/source_documents'
vector_db_name = 'conversations'
doc_collection_name='documents'

DEFAULT_CHUNK_SIZE=500
DEFAULT_CHUNK_OVERLAP=50
DB_PARAMS= {
    'dbname': 'memory_agent',
    'user' : 'example_user',
    'password' : '123456',
    'host' :'localhost',
    'port' : '5432'
}

sentence_transformer_ef_nomic_embed_text = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text",
)


client = chromadb.PersistentClient(path=persist_directory)
ollama_emb = OllamaEmbeddings(model='nomic-embed-text')
db = Chroma(persist_directory=persist_directory, embedding_function=ollama_emb, collection_name=doc_collection_name)#sentence_transformer_ef_nomic_embed_text)#, client_settings=CHROMA_SETTINGS)
# client = chromadb.Client()

# client =Chroma(
#     collection_name=vector_db_name,
#     embedding_function=sentence_transformer_ef_nomic_embed_text,
#     persist_directory=persist_directory,
# )

system_prompt = ('You are an AI assistant that has memory of every conversation you have ever had with this user. '
                 'On every prompt from the user, the system has checked for any relevant messages you have had with the user. '
                 'If any embedded previous conversations are attached, use them for context to responding to the user, '
                 ' if the context  is relevant and useful to responding. If the recalled conversations are irrelevant, '
                 'disregard speaking about them and respond normally as an AI assistant. Do not talk about recalling conversations. '
                 'Just use any useful data from the prvious conversations and respond normally as an intelligent AI assistant. '
                 )

convo = [{'role': 'system', 'content': system_prompt}]


def connect_db():
    conn = psycopg.connect (**DB_PARAMS)
    return conn
##################################

def fetch_conversations():
    conn = connect_db()
    with conn.cursor (row_factory=dict_row) as cursor:
        cursor.execute('SELECT * from conversations')
        conversations = cursor.fetchall()
    conn.close()
    return conversations
##################################


def stream_response(prompt : str):
    response = ''
    stream = ollama.chat (model='llama3.2:latest' , messages=convo, stream=True)
    print (Fore.LIGHTGREEN_EX + "\nASSISTANT: ")

    for chunk in stream:
        content = chunk ['message']['content']
        response += content
        print (content, end='', flush=True)

    print ("\n")
    convo.append ({'role' : 'assistant' , 'content' : response})
    store_conversations (prompt, response)
##################################

def create_vector_db (conversations):
    try:
        client.delete_collection(name=vector_db_name)
    except (ValueError, IndexError):
        pass

    vector_db = client.create_collection(name=vector_db_name, embedding_function=sentence_transformer_ef_nomic_embed_text)

    for c in conversations:
        serialized_convo = f"prompt: {c['prompt']} response: {c['response']}"
        response = ollama.embeddings(model='nomic-embed-text', prompt=serialized_convo)
        embedding = response['embedding']
        vector_db.add(
            ids=[str(c['id'])],
            embeddings=[embedding],
            documents=[serialized_convo]
        )
##################


def classify_embedding(query, context):
    classify_msg = (
        'You are an embedding classification AI agent. Your input will be a prompt and one embedded chunk of text. '
        'You will not respond an an AI assistant. YOu only respond "yes" or "no". '
        'Determine wheter the context contains data that is directly related to the search query. '
        'If the context is seemingly exact what the search query needs, respond "yes" OR if it is anything but directly related'
        'respond "no". Do not respond "yes" unless the content is highly relevant to the search query.'
        )
    classify_convo = [
        {'role' : 'system', 'content' : classify_msg},
        {'role' : 'user' , 'content': f'SEARCH QUERY: What is the users name? \n\nEMBEDDED CONTEXT: You are Ai Austin. How Can I help today Austin?'},
        {'role' : 'assistant', 'content': 'yes'},
        {'role' : 'user' , 'content': f'SEARCH QUERY: Llama3 Python Voice Assistant \n\EMBEDDED CONTEXT: Siri is a voice assistant on Apple iOS and Mac OS. The voice assistant is designed to take voice prompts and help the user complete simple task on the device'},
        {'role' : 'assistant', 'content' : 'no'},
        {'role' : 'user', 'content': f'SEARCH QUERY: {query} \n\nEMBEDDED CONTEXT: {context}'},
    ]

    response = ollama.chat(model='llama3.2:latest', messages=classify_convo)

    return response['message']['content'].strip().lower()
#############################

def retrieve_embeddings(queries, results_per_query=2):
    embeddings = set()

    for query in tqdm(queries, desc='Processing queries to vector database'):
        response = ollama.embeddings(model='nomic-embed-text', prompt=query)
        query_embedding = response['embedding']
        vector_db = client.get_collection(name=vector_db_name, embedding_function=sentence_transformer_ef_nomic_embed_text)
        results = vector_db.query(query_embeddings=[query_embedding], n_results=results_per_query)
        best_embeddings = results['documents'][0]

        for best in best_embeddings:
            if best not in embeddings:
                if 'yes' in classify_embedding(query=query, context=best):
                    embeddings.add(best)

    return embeddings
#################

def store_conversations(prompt : str, response):
    conn = connect_db()
    with conn.cursor() as cursor:
        cursor.execute ('INSERT INTO conversations (timestamp, prompt, response) VALUES (CURRENT_TIMESTAMP, %s, %s)',
                        (prompt, response)
                        )
        conn.commit()
    conn.close()
########################

def create_queries (prompt : str):
    query_msg = (
        'You are a first principle reasoning search query AI agent. '
        'Your list of search queries will be ran on an embedding database of all your conversations '
        'you have evere had with the user. With first principles create a Python list of queries to '
        'search the embeddings database for any data that would be necessary to have acces to in '
        'order to correctly respond to the prompt. Your response mmust be a Python list with no syntaxx errors. '
        'Do not exaplain anything and do no ever generate anythin but a perfect syntax Python list'
        )
    query_convo = [
        {'role' : 'system', 'content': query_msg},
        {'role': 'user', 'content': 'Write an email to my car insurance company and create a pursiasive request for them to lower my monthly rate.'},
        {'role': 'assistant','content': '["What is the users name?", "What is the users current auto insurance provider?", "What is the monthly rate the user currently paus for auto insurance?"}'},
        {'role': 'user', 'content': 'How can I convert the speak function in my llama3 python voice assistant to use pyttsx3 instead of the OpenAI tts api'},
        {'role': 'assistant', 'content': '["Llama3 voice assitant", "Python voice assistant", "OpenAI TTS", "opeanai speak"]'},
        {'role': 'user', 'content': 'How Can I cook spaghetti carbonara?'},
        {'role': 'assistant', 'content': '["Spaghetti carbonara ingredients", "spaguetti carbonara recipe", "spaguett carbonara detailed steps"]'},
        {'role': 'user', 'content': prompt}]

    response = ollama.chat (model='llama3.2:latest', messages=query_convo)
    print (Fore.YELLOW + f'\nVector database queries: {response["message"]["content"]} \n')
    try:
        return ast.literal_eval(response['message']['content'])
    except:
        return [prompt]
#################################


def recall(prompt : str):
    queries = create_queries(prompt)
    embeddings = retrieve_embeddings(queries=queries)
    convo.append({'role': 'user', 'content' : f'MEMORIES: {embeddings} \n\n USER PROMPT: {prompt}'})
    print(f'\n{len(embeddings)} message: response embeddings added for context.')
###################################

def remove_last_conversation():
    conn = connect_db()
    with conn.cursor() as cursor:
        cursor.execute("DELETE FROM conversations WHERE id = (SELECT MAX(id) FROm conversations)")
        cursor.commit()
    conn.close()
##################################

def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(persist_directory):
        if os.path.exists(os.path.join(persist_directory, 'chroma.sqlite3')):
            return True
    return False
#######################################
def load_single_document(file_path: str) -> Document:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()[0]

    raise ValueError(f"Unsupported file extension '{ext}'")
#############################################################

def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    # try:
    #     pattern = re.compile(r'Tema (\d+)')
    #     sorted_filtered_files = sorted(filtered_files, key=lambda x: int(pattern.search(x).group(1)))
    # except AttributeError as e:
    sorted_filtered_files = filtered_files
    # print (f"INFORMATION: sorted_filtered_files => {sorted_filtered_files}")

    with Pool(processes=os.cpu_count()) as pool:
        print (f"INFORMATION: Using {os.cpu_count()} as processes")
        results = []
        with tqdm(total=len(sorted_filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, doc in enumerate(pool.imap_unordered(load_single_document, sorted_filtered_files)):
                results.append(doc)
                pbar.update()

    return results
#############################################################

def process_documents(ignored_files: List[str], summarization: bool, embeddings: HuggingFaceInstructEmbeddings) -> List[Document]:
    """
    Load documents and split in chunks, may raise value error when 0 texts processed
    """
    print(Fore.YELLOW + f"INFORMATION: Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print(Fore.YELLOW + "INFORMATION: No new documents to load and summarization not requested")
        exit(0)
    print(f"INFORMATION: Loaded {len(documents)} new documents from {source_directory}")
    chunk_size = DEFAULT_CHUNK_SIZE
    chunk_overlap = DEFAULT_CHUNK_OVERLAP
    print(Fore.YELLOW + f'INFORMATION: Chunk_Size => {chunk_size}  Chunk_Overlap => {chunk_overlap}')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"INFORMATION: Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    if len(texts) == 0:
        raise ValueError ('No text has been processed')
    if summarization and len(texts) > 0:
        most_important_chapter_indexes : List[int] = get_most_representative_chapters(texts, embeddings)
        # if we want a summary of a bunch of texts return most important chunks of text
        most_important_chapters = [texts[chapter_number] for chapter_number in most_important_chapter_indexes]
        if os.path.exists(f'./{persist_directory}'):
            with open(f'{persist_directory}/most_important_chapters.pickle', 'wb') as file:
                pickle.dump(most_important_chapters, file)
        else:
            print (f'WARNING: cannot save most important chunks inside {persist_directory}')
        return most_important_chapters
    # Return all chunks
    return texts
##############################################################################################################

def ingest_document (prompt : str) -> None:
    documents_dir = prompt.strip().split()[0]
    if not os.path.exists(documents_dir) or not os.path.isdir(os.path.basename(documents_dir)):
        print (Fore.RED + f'dir: {documents_dir} does not exists\n')
        return

    print(Fore.YELLOW + f"INFORMATION: Appending to existing vectorstore at {persist_directory}")
    #embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    if does_vectorstore_exist(persist_directory):
        # Update and store locally vectorstore
        print(Fore.YELLOW + f"INFORMATION: Appending to existing vectorstore at {persist_directory}")
        ##db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        #db = Chroma(persist_directory=persist_directory, embedding_function=sentence_transformer_ef_nomic_embed_text, client_settings=CHROMA_SETTINGS)
        collection = db.get()
        try:
            texts = process_documents(ignored_files=[metadata['source'] for metadata in collection['metadatas']], summarization=False, embeddings=embeddings)
        except ValueError as e:
            print (Fore.RED + e)
            exit (1)

        print(Fore.YELLOW + f"INFORMATION: Creating embeddings. May take some minutes...")
        db.add_documents(texts)
    else:
        # Create and store locally vectorstore
        print(Fore.YELLOW + "INFORMATION: Creating new vectorstore")
        try:
            texts = process_documents(ignored_files=[], summarization=False, embeddings=ollama_emb)
        except ValueError as e:
            print (Fore.RED + e)
            exit (1)

        print(Fore.YELLOW + f"INFORMATION: Creating embeddings. May take some minutes...")

        db = Chroma.from_documents(texts, ollama_emb, persist_directory=persist_directory, collection_name=doc_collection_name)#.persist()

    print(Fore.YELLOW + f"INFORMATION: Ingestion complete! You can now run privateGPT.py to query your documents")
#############################################################


system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

doc_query_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
def doc_search(prompt):
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retriever = db.as_retriever(search_kwargs={'k': 33,}) #search_type="similarity",
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [StreamingStdOutCallbackHandler()]
    model = OllamaLLM(model="llama3.2:latest")
    combine_docs_chain = create_stuff_documents_chain(model, doc_query_prompt)
    # retrieval_chain = (
    #     {"context": retriever | format_docs, "question": RunnablePassthrough()}
    #     | custom_rag_prompt
    #     | model
    #     | StrOutputParser()
    # )

    question_answer_chain = create_stuff_documents_chain(model, doc_query_prompt)
    retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)

    res = retrieval_chain.invoke({"input": prompt})
    answer, docs = res['answer'], res['context']
    # Print the result
    print(Fore.WHITE + "\n> DOC RETRIEVAL: Answer:")
    print(answer)
    # Print the relevant sources used for the answer
    for document in docs:
        print(Fore.BLUE + "> " + document.metadata["source"])#+ ":" + document.page_content)


    # Store query and reply
    convo.append({'role' : 'user', 'content':prompt})
    convo.append ({'role' : 'assistant' , 'content' : answer})
    store_conversations (prompt, answer)
########################################################



message_history=fetch_conversations()
create_vector_db (conversations=message_history)



while True:
    prompt = input (Fore.WHITE + "USER: \n")
    if prompt[0:7].lower() == '/recall':
        recall(prompt=prompt[8:])
        stream_response (prompt[8:])
    elif prompt[0:7].lower() == '/delete':
        remove_last_conversation()
        convo =convo[:-2]
        print('\n')
        continue
    elif prompt[0:9].lower() == '/memorize':
        prompt = prompt[10:]
        store_conversations(prompt=prompt, response='Memory Stored.')
        print('\n')
        continue
    elif prompt[0:5] == '/quit':
        break
    elif prompt[0:4] == '/doc':
        doc_search(prompt[4:])
        continue
    elif prompt[0:7] == '/ingest':
        ingest_document (prompt[7:])
        continue
    elif prompt[0:5] == '/help':
        print("COMMANDS: /recall /delete /memorize /quit /ingest /help")
        continue
    else:
        convo.append({'role' : 'user', 'content':prompt})
        stream_response (prompt)

db= None
