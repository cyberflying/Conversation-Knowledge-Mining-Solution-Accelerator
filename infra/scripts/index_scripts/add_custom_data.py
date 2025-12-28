import argparse
import json
import os
import regex as re
import struct
import time
from datetime import datetime, timedelta
from urllib.parse import urlparse

import pandas as pd
import pyodbc
from azure.ai.inference import ChatCompletionsClient, EmbeddingsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.identity import AzureCliCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    VectorSearch,
    VectorSearchProfile,
)
from azure.storage.filedatalake import DataLakeServiceClient

from content_understanding_client import AzureContentUnderstandingClient

# Constants and configuration
FILE_SYSTEM_CLIENT_NAME = "data"
DIRECTORY = 'custom_transcripts'
AUDIO_DIRECTORY = 'custom_audiodata'
INDEX_NAME = "call_transcripts_index"

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process custom data for knowledge mining')
parser.add_argument('--search_endpoint', required=True, help='Azure AI Search endpoint')
parser.add_argument('--openai_endpoint', required=True, help='Azure OpenAI endpoint')
parser.add_argument('--ai_project_endpoint', required=True, help='Azure AI Project endpoint')
parser.add_argument('--deployment_model', required=True, help='Azure OpenAI deployment model name')
parser.add_argument('--embedding_model', required=True, help='Azure OpenAI embedding model name')
parser.add_argument('--storage_account_name', required=True, help='Azure Storage Account name')
parser.add_argument('--sql_server', required=True, help='Azure SQL Server name')
parser.add_argument('--sql_database', required=True, help='Azure SQL Database name')
parser.add_argument('--cu_endpoint', required=True, help='Azure Content Understanding endpoint')
parser.add_argument('--cu_api_version', required=True, help='Azure Content Understanding API version')

args = parser.parse_args()

# Assign arguments to variables
SEARCH_ENDPOINT = args.search_endpoint
OPENAI_ENDPOINT = args.openai_endpoint
AI_PROJECT_ENDPOINT = args.ai_project_endpoint
DEPLOYMENT_MODEL = args.deployment_model
EMBEDDING_MODEL = args.embedding_model
STORAGE_ACCOUNT_NAME = args.storage_account_name
SQL_SERVER = args.sql_server
SQL_DATABASE = args.sql_database
CU_ENDPOINT = args.cu_endpoint
CU_API_VERSION = args.cu_api_version

# Azure DataLake setup
account_url = f"https://{STORAGE_ACCOUNT_NAME}.dfs.core.windows.net"
credential = AzureCliCredential(process_timeout=30)
service_client = DataLakeServiceClient(account_url, credential=credential, api_version='2023-01-03')
file_system_client = service_client.get_file_system_client(FILE_SYSTEM_CLIENT_NAME)
directory_name = DIRECTORY
paths = list(file_system_client.get_paths(path=directory_name))
print("Azure DataLake setup complete.")

# Azure Search setup
search_credential = AzureCliCredential(process_timeout=30)
search_client = SearchClient(SEARCH_ENDPOINT, INDEX_NAME, search_credential)
print("Azure Search setup complete.")

# Azure AI Foundry (Inference) clients (Managed Identity)
inference_endpoint = f"https://{urlparse(AI_PROJECT_ENDPOINT).netloc}/models"

chat_client = ChatCompletionsClient(
    endpoint=inference_endpoint,
    credential=credential,
    credential_scopes=["https://ai.azure.com/.default"],
)

embeddings_client = EmbeddingsClient(
    endpoint=inference_endpoint,
    credential=credential,
    credential_scopes=["https://ai.azure.com/.default"],
)

# SQL Server setup
try: 
    driver = "{ODBC Driver 18 for SQL Server}"
    token_bytes = credential.get_token("https://database.windows.net/.default").token.encode("utf-16-LE")
    token_struct = struct.pack(f"<I{len(token_bytes)}s", len(token_bytes), token_bytes)
    SQL_COPT_SS_ACCESS_TOKEN = 1256
    connection_string = f"DRIVER={driver};SERVER={SQL_SERVER};DATABASE={SQL_DATABASE};"
    conn = pyodbc.connect(connection_string, attrs_before={SQL_COPT_SS_ACCESS_TOKEN: token_struct})
    cursor = conn.cursor()
except: 
    driver = "{ODBC Driver 17 for SQL Server}"
    token_bytes = credential.get_token("https://database.windows.net/.default").token.encode("utf-16-LE")
    token_struct = struct.pack(f"<I{len(token_bytes)}s", len(token_bytes), token_bytes)
    SQL_COPT_SS_ACCESS_TOKEN = 1256
    connection_string = f"DRIVER={driver};SERVER={SQL_SERVER};DATABASE={SQL_DATABASE};"
    conn = pyodbc.connect(connection_string, attrs_before={SQL_COPT_SS_ACCESS_TOKEN: token_struct})
    cursor = conn.cursor()
print("SQL Server connection established.")

# Content Understanding client
cu_credential = AzureCliCredential(process_timeout=30)
cu_token_provider = get_bearer_token_provider(cu_credential, "https://cognitiveservices.azure.com/.default")
cu_client = AzureContentUnderstandingClient(
    endpoint=CU_ENDPOINT,
    api_version=CU_API_VERSION,
    token_provider=cu_token_provider
)
print("Content Understanding client initialized.")

# Utility functions
def get_embeddings(text: str):
    try:
        resp = embeddings_client.embed(model=EMBEDDING_MODEL, input=[text])
        return resp.data[0].embedding
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        raise
# --------------------------------------------------------------------------

def clean_spaces_with_regex(text):
    cleaned_text = re.sub(r'\s+', ' ', text)
    cleaned_text = re.sub(r'\.{2,}', '.', cleaned_text)
    return cleaned_text

def chunk_data(text, tokens_per_chunk=1024):
    text = clean_spaces_with_regex(text)
    sentences = text.split('. ')
    chunks, current_chunk, current_chunk_token_count = [], '', 0
    for sentence in sentences:
        tokens = sentence.split()
        if current_chunk_token_count + len(tokens) <= tokens_per_chunk:
            current_chunk += ('. ' if current_chunk else '') + sentence
            current_chunk_token_count += len(tokens)
        else:
            chunks.append(current_chunk)
            current_chunk, current_chunk_token_count = sentence, len(tokens)
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def prepare_search_doc(content, document_id, path_name):
    chunks = chunk_data(content)
    docs = []
    for idx, chunk in enumerate(chunks, 1):
        chunk_id = f"{document_id}_{str(idx).zfill(2)}"
        try:
            v_contentVector = get_embeddings(str(chunk))
        except Exception as e:
            print(f"Error getting embeddings on first try: {e}")
            time.sleep(30)
            try: 
                v_contentVector = get_embeddings(str(chunk))
            except Exception as e: 
                print(f"Error getting embeddings: {e}")
                v_contentVector = []
        docs.append({
            "id": chunk_id,
            "chunk_id": chunk_id,
            "content": chunk,
            "sourceurl": path_name.split('/')[-1],
            "contentVector": v_contentVector
        })
    return docs

def get_field_value(fields, field_name, default=""):
    field = fields.get(field_name, {})
    return field.get('valueString', default)

ANALYZER_ID = "ckm-json"
# Process files and insert into DB and Search - transcripts
conversationIds, docs, counter = [], [], 0
for path in paths:
    file_client = file_system_client.get_file_client(path.name)
    data_file = file_client.download_file()
    data = data_file.readall()
    try:
        response = cu_client.begin_analyze(ANALYZER_ID, file_location="", file_data=data)
        result = cu_client.poll_result(response)
        file_name = path.name.split('/')[-1].replace("%3A", "_")
        start_time = file_name.replace(".json", "")[-19:]
        timestamp_format = "%Y-%m-%d %H_%M_%S"
        start_timestamp = datetime.strptime(start_time, timestamp_format)
        conversation_id = file_name.split('convo_', 1)[1].split('_')[0]
        conversationIds.append(conversation_id)

        fields = result['result']['contents'][0]['fields']
        duration_str = get_field_value(fields, 'Duration', '0')
        try:
            duration = int(duration_str)
        except (ValueError, TypeError):
            duration = 0

        end_timestamp = str(start_timestamp + timedelta(seconds=duration)).split(".")[0]
        start_timestamp = str(start_timestamp).split(".")[0]
        
        summary = get_field_value(fields, 'summary')
        satisfied = get_field_value(fields, 'satisfied')
        sentiment = get_field_value(fields, 'sentiment')
        topic = get_field_value(fields, 'topic')
        key_phrases = get_field_value(fields, 'keyPhrases')
        complaint = get_field_value(fields, 'complaint')
        content = get_field_value(fields, 'content')

        cursor.execute(
            "INSERT INTO processed_data (ConversationId, EndTime, StartTime, Content, summary, satisfied, sentiment, topic, key_phrases, complaint) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (conversation_id, end_timestamp, start_timestamp, content, summary, satisfied, sentiment, topic, key_phrases, complaint)
        )
        conn.commit()
        docs.extend(prepare_search_doc(content, conversation_id, path.name))
        counter += 1
    except Exception as e:
        print(f"Error processing file {path.name}: {e}")
        pass
    if docs != [] and counter % 10 == 0:
        result = search_client.upload_documents(documents=docs)
        docs = []
if docs:
    search_client.upload_documents(documents=docs)

print(f"✓ Processed {counter} transcript files")

# Process files for audio data
ANALYZER_ID = "ckm-audio"

directory_name = AUDIO_DIRECTORY
paths = list(file_system_client.get_paths(path=directory_name))
docs = []
counter = 0
# process and upload audio files to search index - audio data
for path in paths:
    file_client = file_system_client.get_file_client(path.name)
    data_file = file_client.download_file()
    data = data_file.readall()
    try:
        # # Analyzer file
        response = cu_client.begin_analyze(ANALYZER_ID, file_location="", file_data=data)
        result = cu_client.poll_result(response)

        file_name = path.name.split('/')[-1]
        start_time = file_name.replace(".wav", "")[-19:]
        
        timestamp_format = "%Y-%m-%d %H_%M_%S"  # Adjust format if necessary
        start_timestamp = datetime.strptime(start_time, timestamp_format)

        conversation_id = file_name.split('convo_', 1)[1].split('_')[0]
        conversationIds.append(conversation_id)

        fields = result['result']['contents'][0]['fields']
        duration_str = get_field_value(fields, 'Duration', '0')
        try:
            duration = int(duration_str)
        except (ValueError, TypeError):
            duration = 0
        
        end_timestamp = str(start_timestamp + timedelta(seconds=duration))
        end_timestamp = end_timestamp.split(".")[0]
        start_timestamp = str(start_timestamp).split(".")[0]

        summary = get_field_value(fields, 'summary')
        satisfied = get_field_value(fields, 'satisfied')
        sentiment = get_field_value(fields, 'sentiment')
        topic = get_field_value(fields, 'topic')
        key_phrases = get_field_value(fields, 'keyPhrases')
        complaint = get_field_value(fields, 'complaint')
        content = get_field_value(fields, 'content')
        
        cursor.execute(f"INSERT INTO processed_data (ConversationId, EndTime, StartTime, Content, summary, satisfied, sentiment, topic, key_phrases, complaint) VALUES (?,?,?,?,?,?,?,?,?,?)", (conversation_id, end_timestamp, start_timestamp, content, summary, satisfied, sentiment, topic, key_phrases, complaint))    
        conn.commit()
    
        document_id = conversation_id

        docs.extend(prepare_search_doc(content, document_id, path.name))
        counter += 1
    except Exception as e:
        print(f"Error processing file {path.name}: {e}")
        pass

    if docs != [] and counter % 10 == 0:
        result = search_client.upload_documents(documents=docs)
        docs = []

# upload the last batch
if docs != []:
    search_client.upload_documents(documents=docs)

print(f"✓ Processed {counter} audio files")

# Topic mining and mapping
cursor.execute('SELECT distinct topic FROM processed_data')
rows = [tuple(row) for row in cursor.fetchall()]
column_names = [i[0] for i in cursor.description]
df = pd.DataFrame(rows, columns=column_names)
cursor.execute('DROP TABLE IF EXISTS km_mined_topics')
cursor.execute("""CREATE TABLE km_mined_topics (
    label varchar(255) NOT NULL PRIMARY KEY,
    description varchar(255)
);""")
conn.commit()
topics_str = ', '.join(df['topic'].tolist())

def call_gpt4(topics_str1, client):
    topic_prompt = f"""
        You are a data analysis assistant specialized in natural language processing and topic modeling. 
        Your task is to analyze the given text corpus and identify distinct topics present within the data.
        {topics_str1}
        1. Identify the key topics in the text using topic modeling techniques.
        2. Choose the right number of topics based on data. Try to keep it up to 8 topics.
        3. Assign a clear and concise label to each topic based on its content.
        4. Provide a brief description of each topic along with its label.
        5. Add parental controls, billing issues like topics to the list of topics if the data includes calls related to them.
        If the input data is insufficient for reliable topic modeling, indicate that more data is needed rather than making assumptions.
        Ensure that the topics and labels are accurate, relevant, and easy to understand.
        Return the topics and their labels in JSON format.Always add 'topics' node and 'label', 'description' attributes in json.
        Do not return anything else.
    """
    response = client.complete(
        model=DEPLOYMENT_MODEL,
        messages=[
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content=topic_prompt),
        ],
        temperature=0,
    )
    res = response.choices[0].message.content
    return json.loads(res.replace("```json", '').replace("```", ''))


max_tokens = 3096
res = call_gpt4(", ".join([]), chat_client)
for object1 in res['topics']:
    cursor.execute("INSERT INTO km_mined_topics (label, description) VALUES (?,?)", (object1['label'], object1['description']))
conn.commit()
print("Topics mined and inserted into km_mined_topics.")

cursor.execute('SELECT label FROM km_mined_topics')
rows = [tuple(row) for row in cursor.fetchall()]
column_names = [i[0] for i in cursor.description]
df_topics = pd.DataFrame(rows, columns=column_names)
mined_topics_list = df_topics['label'].tolist()
mined_topics = ", ".join(mined_topics_list)


def get_mined_topic_mapping(input_text, list_of_topics):
    prompt = f'''You are a data analysis assistant to help find the closest topic for a given text {input_text}
                from a list of topics - {list_of_topics}.
                ALWAYS only return a topic from list - {list_of_topics}. Do not add any other text.'''
    response = chat_client.complete(
        model=DEPLOYMENT_MODEL,
        messages=[
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content=prompt),
        ],
        temperature=0,
    )
    return response.choices[0].message.content


cursor.execute('SELECT * FROM processed_data')
rows = [tuple(row) for row in cursor.fetchall()]
column_names = [i[0] for i in cursor.description]
df_processed_data = pd.DataFrame(rows, columns=column_names)
df_processed_data = df_processed_data[df_processed_data['ConversationId'].isin(conversationIds)]
for _, row in df_processed_data.iterrows():
    mined_topic_str = get_mined_topic_mapping(row['topic'], str(mined_topics_list))
    cursor.execute("UPDATE processed_data SET mined_topic = ? WHERE ConversationId = ?", (mined_topic_str, row['ConversationId']))
conn.commit()
print("The field mined_topic of the table processed_data has been updated.")

# Update processed data for RAG
# Optimization: Insert unique records from processed_data into km_processed_data
cursor.execute("""
    INSERT INTO km_processed_data (
        ConversationId, StartTime, EndTime, Content, summary, 
        satisfied, sentiment, keyphrases, complaint, topic
    )
    SELECT 
        src.ConversationId, src.StartTime, src.EndTime, src.Content, src.summary, 
        src.satisfied, src.sentiment, src.key_phrases, src.complaint, src.mined_topic 
    FROM processed_data AS src
    WHERE NOT EXISTS (
        SELECT 1 
        FROM km_processed_data AS tgt 
        WHERE tgt.ConversationId = src.ConversationId
    )
""")
conn.commit()
print("km_processed_data table updated.")

# Update processed_data_key_phrases table
cursor.execute("""
    SELECT ConversationId, key_phrases, sentiment, mined_topic as topic, StartTime
    FROM processed_data d
    WHERE NOT EXISTS ( 
        SELECT 1
        FROM processed_data_key_phrases kp 
        WHERE kp.ConversationId = d.ConversationId
    )
""")
rows = [tuple(row) for row in cursor.fetchall()]
column_names = [i[0] for i in cursor.description]
df = pd.DataFrame(rows, columns=column_names)
df = df[df['ConversationId'].isin(conversationIds)]
for _, row in df.iterrows():
    key_phrases = row['key_phrases'].split(',')
    for key_phrase in key_phrases:
        key_phrase = key_phrase.strip()
        cursor.execute("INSERT INTO processed_data_key_phrases (ConversationId, key_phrase, sentiment, topic, StartTime) VALUES (?,?,?,?,?)",
                       (row['ConversationId'], key_phrase, row['sentiment'], row['topic'], row['StartTime']))
conn.commit()
print("processed_data_key_phrases table updated.")

# Adjust dates to current date
today = datetime.today()
cursor.execute("SELECT MAX(CAST(StartTime AS DATETIME)) FROM [dbo].[processed_data]")
max_start_time = cursor.fetchone()[0]
days_difference = (today - max_start_time).days - 1 if max_start_time else 0
cursor.execute("UPDATE [dbo].[processed_data] SET StartTime = FORMAT(DATEADD(DAY, ?, StartTime), 'yyyy-MM-dd HH:mm:ss'), EndTime = FORMAT(DATEADD(DAY, ?, EndTime), 'yyyy-MM-dd HH:mm:ss')", (days_difference, days_difference))
cursor.execute("UPDATE [dbo].[km_processed_data] SET StartTime = FORMAT(DATEADD(DAY, ?, StartTime), 'yyyy-MM-dd HH:mm:ss'), EndTime = FORMAT(DATEADD(DAY, ?, EndTime), 'yyyy-MM-dd HH:mm:ss')", (days_difference, days_difference))
cursor.execute("UPDATE [dbo].[processed_data_key_phrases] SET StartTime = FORMAT(DATEADD(DAY, ?, StartTime), 'yyyy-MM-dd HH:mm:ss')", (days_difference,))
conn.commit()

cursor.close()
conn.close()
print("✓ Data processing completed")
