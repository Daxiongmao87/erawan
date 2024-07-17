import os
import configparser
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
import json
import requests
from datetime import datetime
from werkzeug.exceptions import BadRequest
import logging
import time
from sqlalchemy import func, or_
import faiss
import numpy as np
import pickle

# Set up logging
logging.basicConfig(level=logging.DEBUG)

config = configparser.ConfigParser()
config.read('config.ini')

app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'bots.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['DEBUG'] = True  # Enable debug mode
db = SQLAlchemy(app)

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

class CRUDMixin:
    @classmethod
    def create(cls, **kwargs):
        instance = cls(**kwargs)
        return instance.save()

    def update(self, commit=True, **kwargs):
        for attr, value in kwargs.items():
            setattr(self, attr, value)
        return commit and self.save() or self

    def save(self, commit=True):
        db.session.add(self)
        if commit:
            db.session.commit()
        return self

    def delete(self, commit=True):
        db.session.delete(self)
        if commit:
            db.session.commit()

class Bot(db.Model, CRUDMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), unique=True, nullable=False)
    persona = db.Column(db.Text, nullable=False)
    endpoint_name = db.Column(db.String(120), nullable=False)
    inner_monologue = db.Column(db.Integer, default=0)

class Memory(db.Model, CRUDMixin):
    id = db.Column(db.Integer, primary_key=True)
    bot_id = db.Column(db.Integer, db.ForeignKey('bot.id'), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    content = db.Column(db.Text, nullable=False)
    bot = db.relationship('Bot', backref=db.backref('memories', lazy=True))

class Keyword(db.Model, CRUDMixin):
    id = db.Column(db.Integer, primary_key=True)
    memory_id = db.Column(db.Integer, db.ForeignKey('memory.id'), nullable=False)
    keyword = db.Column(db.String(120), nullable=False)
    memory = db.relationship('Memory', backref=db.backref('keywords', lazy=True))

class MainConversation(db.Model, CRUDMixin):
    id = db.Column(db.Integer, primary_key=True)
    role = db.Column(db.String(120), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    conversation = db.Column(db.Text, nullable=False)
    bot_name = db.Column(db.String(120), nullable=False)

class ApiEndpoint(db.Model, CRUDMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), unique=True, nullable=False)
    api_url = db.Column(db.String(255), nullable=False)
    token = db.Column(db.String(255), nullable=True)
    model = db.Column(db.String(120), nullable=False)
    context_length = db.Column(db.Integer, nullable=False)
    reserved_tokens_for_response = db.Column(db.Integer, nullable=False)

### RAG

def initialize_faiss(dimension):
    index = faiss.IndexFlatL2(dimension)
    return index

dimension = config.getint('embed_api_endpoint', 'dimension')
faiss_index = initialize_faiss(dimension)
faiss_mapping = {}

# Load FAISS mapping from disk if exists
if os.path.exists('faiss_mapping.pkl'):
    with open('faiss_mapping.pkl', 'rb') as f:
        faiss_mapping = pickle.load(f)

def save_faiss_mapping():
    with open('faiss_mapping.pkl', 'wb') as f:
        pickle.dump(faiss_mapping, f)

def get_embeddings(text):
    print("Get embeddings for: ", text)
    api_url = config.get('embed_api_endpoint', 'api_url')
    token = config.get('embed_api_endpoint', 'token')
    model = config.get('embed_api_endpoint', 'model')
    max_tokens = config.getint('embed_api_endpoint', 'context_length') - config.getint('embed_api_endpoint', 'reserved_tokens_for_response')  # Maximum tokens for the model

    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = f'Bearer {token}'

    # Truncate the input text to ensure it doesn't exceed the maximum tokens
    truncated_text = text[:max_tokens]

    json_data = {
        "input": truncated_text,
        "model": model
    }

    try:
        response = requests.post(api_url, headers=headers, json=json_data)
        response.raise_for_status() 
        response_json = response.json()
        embedding = response_json.get('data', [{}])[0].get('embedding')
        if embedding is None:
            raise ValueError("Embedding not found in response")
        #print(f"Generated Embedding: {embedding}")
        return np.array(embedding, dtype=np.float32)
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to get embeddings: {e} - Response text: {response.text}")
        return None
    except ValueError as e:
        logging.error(f"Error in API response: {e}")
        return None


def store_memory_faiss(bot_id, text, keywords):
    # Store embedding for the complete text
    text_embedding = get_embeddings(text)
    if text_embedding is not None:
        faiss_index.add(np.array([text_embedding]))
        faiss_index_id = faiss_index.ntotal - 1
        memory = Memory.create(bot_id=bot_id, timestamp=datetime.now(), content=text)
        faiss_mapping[memory.id] = faiss_index_id
        
        # Store embeddings for each keyword
        print("Keywords: ", keywords)
        for keyword in keywords:
            print("Saving embedding for: ", keyword)
            keyword_embedding = get_embeddings(keyword)
            if keyword_embedding is not None:
                faiss_index.add(np.array([keyword_embedding]))
                keyword_faiss_id = faiss_index.ntotal - 1
                faiss_mapping[f"{memory.id}_keyword_{keyword}"] = keyword_faiss_id
        
        save_faiss_mapping()
        return memory
    return None



def search_memory_faiss(query, k=10):
    query_embedding = get_embeddings(query)
    if query_embedding is not None:
        print(f"Query Embedding: {query_embedding}")  # Add this line
        distances, indices = faiss_index.search(np.array([query_embedding]), k)
        print(f"FAISS Search - distances: {distances}, indices: {indices}")  # Add this line
        return indices[0] if np.any(indices >= 0) else []
    else:
        print("Query came back empty")
    return []


def get_memory_by_index(index):
    for memory_id, faiss_id in faiss_mapping.items():
        if faiss_id == index:
            memory = Memory.query.get(memory_id)
            return memory.content
    return None

def combine_contexts(user_input, retrieved_texts):
    combined_context = user_input + "\n\n" + "\n\n".join(retrieved_texts)
    return combined_context

def delete_memory(memory_id):
    if memory_id in faiss_mapping:
        faiss_id = faiss_mapping.pop(memory_id)
        # Here we need to recreate the FAISS index without the deleted entry
        all_embeddings = [faiss_index.reconstruct(i) for i in range(faiss_index.ntotal) if i != faiss_id]
        faiss_index.reset()
        faiss_index.add(np.array(all_embeddings))
        Memory.query.filter_by(id=memory_id).delete()
        save_faiss_mapping()
        db.session.commit()


def embed_api_endpoint_call(prompt):
    api_url = config.get('embed_api_endpoint', 'api_url')
    token = config.get('embed_api_endpoint', 'token')
    model = config.get('embed_api_endpoint', 'model')

    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = f'Bearer {token}'

    json_data = {
        'model': model,
        'messages': [
            {
                "role": "system",
                "content": prompt
            }
        ],
        'format': 'json',  # Ensuring the word 'json' is included in the request
        'response_format': {
            'type': 'json_object'
        },
        'stream': False,
        'temperature': 1,
        'top_p': 1
    }

    try:
        response = requests.post(api_url, headers=headers, json=json_data)
        print(f"RESPONSE: {response.text}")
        response.raise_for_status()
        print("FINDING MESSAGE FIELD")
        print(response.json())
        response_text = find_message_content(response.json())
        print(f"RESPONSE TEXT: ", response_text)
        return response_text
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to call System API Endpoint: {e}")
        return None

###

def extract_keywords_with_model(conversation_text, latest_exchange):
    max_tokens = config.getint('system_api_endpoint', 'context_length') - config.getint('system_api_endpoint', 'reserved_tokens_for_response')
    max_tokens_conversation = int(max_tokens * 0.9)
    conversation_text=conversation_text[:max_tokens_conversation]
    prompt = f"""
    Extract relevant keywords from the following text.
    Focus on the latest exchange for keywords.
    
    Conversation History:
    {conversation_text}
    
    Latest Exchange:
    {latest_exchange}
    
    Respond with a JSON object in the format:
    {{
        "keywords": ["<keyword1>", "<keyword2>", "<keyword3>", ...]
    }}
    """
    
    api_url = config.get('system_api_endpoint', 'api_url')
    token = config.get('system_api_endpoint', 'token')
    model = config.get('system_api_endpoint', 'model')
    
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = f'Bearer {token}' 

    json_data = {
        "model": model,
        "messages": [{"role": "system", "content": prompt}],
        "max_tokens": config.getint('system_api_endpoint', 'reserved_tokens_for_response')
    }

    try:
        response = requests.post(api_url, headers=headers, json=json_data)
        response.raise_for_status()
        
        response_json = response.json()
        keywords = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
        print("Keywords: ", json.loads(keywords).get('keywords', []))
        return json.loads(keywords).get('keywords', [])
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to extract keywords: {e} - Response text: {response.text}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing the response JSON: {e}")
        return []


db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/bots', methods=['GET'])
def get_bots():
    bots = Bot.query.all()
    return jsonify([{'name': bot.name, 'persona': bot.persona} for bot in bots])

@app.route('/create_bot', methods=['POST'])
def create_bot():
    data = request.json
    bot_name = data.get('name')
    bot_persona = data.get('persona')
    endpoint_name = data.get('endpoint_name')
    inner_monologue = 1 if data.get('inner_monologue', False) else 0  # Use 1 or 0

    if not bot_name or not bot_persona or not endpoint_name:
        return jsonify({'error': 'Bot name, persona, and endpoint_name are required'}), 400

    existing_bot = Bot.query.filter_by(name=bot_name).first()
    if existing_bot:
        return jsonify({'error': 'Bot with this name already exists'}), 400

    bot = Bot.create(name=bot_name, persona=bot_persona, endpoint_name=endpoint_name, inner_monologue=inner_monologue)
    return jsonify({'name': bot.name, 'persona': bot.persona, 'endpoint_name': bot.endpoint_name, 'inner_monologue': bool(bot.inner_monologue)})

@app.route('/update_bot_api_endpoint', methods=['POST'])
def update_bot_api_endpoint():
    data = request.json
    bot_name = data.get('name')
    new_endpoint_name = data.get('endpoint_name')

    if not bot_name or not new_endpoint_name:
        return jsonify({'error': 'Bot name and new endpoint are required'}), 400

    bot = Bot.query.filter_by(name=bot_name).first()
    if not bot:
        return jsonify({'error': 'Bot not found'}), 404

    bot.update(endpoint_name=new_endpoint_name)
    return jsonify({'message': 'Bot API endpoint updated successfully'})

@app.route('/chat_history', methods=['GET'])
def chat_history():
    bot_name = request.args.get('bot_name')

    if not bot_name:
        return jsonify({'error': 'Bot name is required'}), 400

    conversations = MainConversation.query.filter_by(bot_name=bot_name).all()
    return jsonify([{'role': conv.role, 'conversation': conv.conversation, 'timestamp': conv.timestamp} for conv in conversations])

@app.route('/api_endpoints', methods=['GET'])
def get_api_endpoints():
    endpoints = ApiEndpoint.query.all()
    return jsonify([{'name': ep.name, 'api_url': ep.api_url, 'model': ep.model} for ep in endpoints])

@app.route('/add_api_endpoint', methods=['POST'])
def create_api_endpoint():
    data = request.json
    name = data.get('name')
    api_url = data.get('api_url')
    model = data.get('model')
    token = data.get('token', '')
    context_length = data.get('context_length')
    reserved_tokens_for_response = data.get('reserved_tokens_for_response')

    if not name or not api_url or not model or context_length is None or reserved_tokens_for_response is None:
        return jsonify({'error': 'Name, API URL, model, context_length, and reserved_tokens_for_response are required'}), 400

    existing_endpoint = ApiEndpoint.query.filter_by(name=name).first()
    if existing_endpoint:
        return jsonify({'error': 'Endpoint with this name already exists'}), 400

    endpoint = ApiEndpoint.create(name=name, api_url=api_url, model=model, token=token, context_length=context_length, reserved_tokens_for_response=reserved_tokens_for_response)
    return jsonify({'name': endpoint.name, 'api_url': endpoint.api_url, 'model': endpoint.model, 'context_length': endpoint.context_length, 'reserved_tokens_for_response': endpoint.reserved_tokens_for_response})

@app.route('/update_inner_monologue', methods=['POST'])
def update_inner_monologue():
    data = request.json
    bot_name = data.get('name')
    inner_monologue = data.get('inner_monologue')

    if not bot_name or inner_monologue is None:
        return jsonify({'error': 'Bot name and inner_monologue status are required'}), 400

    bot = Bot.query.filter_by(name=bot_name).first()
    if not bot:
        return jsonify({'error': 'Bot not found'}), 404

    bot.update(inner_monologue=1 if inner_monologue else 0)
    return jsonify({'message': 'Inner monologue status updated successfully'})


@app.route('/bot_info', methods=['GET'])
def bot_info():
    bot_name = request.args.get('bot_name')

    if not bot_name:
        return jsonify({'error': 'Bot name is required'}), 400

    bot = Bot.query.filter_by(name=bot_name).first()
    if not bot:
        return jsonify({'error': 'Bot not found'}), 404

    return jsonify({'name': bot.name, 'persona': bot.persona, 'endpoint_name': bot.endpoint_name})

@app.route('/update_api_endpoint', methods=['POST'])
def update_api_endpoint():
    data = request.json
    name = data.get('name')
    new_api_url = data.get('api_url')
    new_model = data.get('model')
    new_token = data.get('token', '')
    new_context_length = data.get('context_length')
    new_reserved_tokens_for_response = data.get('reserved_tokens_for_response')

    if not name or not new_api_url or not new_model or new_context_length is None or new_reserved_tokens_for_response is None:
        return jsonify({'error': 'Name, API URL, model, context_length, and reserved_tokens_for_response are required'}), 400

    endpoint = ApiEndpoint.query.filter_by(name=name).first()
    if not endpoint:
        return jsonify({'error': 'Endpoint not found'}), 404

    endpoint.update(api_url=new_api_url, model=new_model, token=new_token, context_length=new_context_length, reserved_tokens_for_response=new_reserved_tokens_for_response)
    return jsonify({'message': 'API endpoint updated successfully'})


@app.route('/delete_api_endpoint', methods=['DELETE'])
def delete_api_endpoint():
    name = request.args.get('name')

    if not name:
        return jsonify({'error': 'Name is required'}), 400

    endpoint = ApiEndpoint.query.filter_by(name=name).first()
    if not endpoint:
        return jsonify({'error': 'Endpoint not found'}), 404

    endpoint.delete()
    return jsonify({'message': 'API endpoint deleted successfully'})

def validate_and_sanitize_input(data):
    if not isinstance(data, dict):
        raise BadRequest("Invalid input format. Expected a JSON object.")
    required_fields = ['user_input', 'bot_name']
    for field in required_fields:
        if field not in data:
            raise BadRequest(f"Missing required field: {field}")
        if not isinstance(data[field], str) or not data[field].strip():
            raise BadRequest(f"Invalid value for field: {field}")
    return data

def is_valid_json(text):
    try:
        json.loads(text)
        return True
    except ValueError:
        return False

def make_api_call_with_retries(api_url, headers, json_data):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = requests.post(api_url, headers=headers, json=json_data)
            response.raise_for_status()

            # Check if the response content is valid JSON
            response_text = response.content.decode(errors='replace')
            if not is_valid_json(response_text):
                raise json.decoder.JSONDecodeError("Invalid JSON response", response_text, 0)
            else:
                return response.json()
        except (requests.exceptions.RequestException, ValueError, json.decoder.JSONDecodeError) as e:
            retries += 1
            if retries >= MAX_RETRIES:
                app.logger.error(f"Request failed after {MAX_RETRIES} attempts: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    app.logger.error(f"Response content: {e.response.content.decode(errors='replace')}")
                raise e
            app.logger.warning(f"Request failed. Retrying... {retries}/{MAX_RETRIES}")
            if hasattr(e, 'response') and e.response is not None:
                app.logger.warning(f"Response content: {e.response.content.decode(errors='replace')}")
            time.sleep(RETRY_DELAY)
        except Exception as e:  # Catch any other exception
            retries += 1
            if retries >= MAX_RETRIES:
                app.logger.error(f"Unexpected error after {MAX_RETRIES} attempts: {e}")
                raise e
            app.logger.warning(f"Unexpected error. Retrying... {retries}/{MAX_RETRIES}")
            time.sleep(RETRY_DELAY)

def get_bot_conversation(bot_name, limit = 0):
    if limit > 0:
        conversation = MainConversation.query.filter_by(bot_name=bot_name).order_by(MainConversation.timestamp.desc()).limit(limit).all()
        conversation.reverse()
    else:
        conversation = MainConversation.query.filter_by(bot_name=bot_name).order_by(MainConversation.timestamp.asc()).all()
    return conversation

def extract_keywords(text, latest_exchange):
    prompt = f"""
    Verbosely extract relevant keywords from the following text.  
    Be complete and thorough. 
    Utilize the conversation history mainly as a contextutal reference.
    Focus on the very latest exchange for keywords.

    Conversation History (ONLY FOR CONTEXTUAL REASONS):
    --- SEGMENT ---
    {text}
    --- SEGMENT ---

    Last Exchange (FOCUS HERE FOR KEYWORDS):
    --- SEGMENT ---
    {latest_exchange}
    --- SEGMENT ---

    NOTE: AVOID KEYWORDS THAT HAVE NO ASSOCIATION WITH THE TOPICS IN THE LAST EXCHANGE
    NOTE: MAKE SURE SEARCH TERMS ARE UNIQUE IDENTIFIERS.
    NOTE: INCLUDE AS MANY SYNONYMS OR RELATED WORDS FOR INCREASE SEARCH RELIABILITY.
    Respond with a JSON object in the format:
    {{
        "categories": ["<category1>", "<category2>", "<category3>", "etc ..."], # Contextually-complete categories IN ORDER OF SIGNIFICANCE
        "subjects": ["<subject1>", "<subject2>", "<subject3>", "etc ..."], # Contextually-complete subjects IN ORDER OF SIGNIFICANCE
        "keywords": ["<keyword1>", "<keyword2>", "<keyword3>", "<etc> ..."] # Contextually-complete keywords IN ORDER OF SIGNIFICANCE
    }}
    """
    print(f"Prompt: {prompt}")
    response = embed_api_endpoint_call(prompt)
    print("KEYWORDS RESPONSE: ", response)

    if response is None:
        return []

    try:
        response_json = json.loads(response)
        keywords = response_json.get("categories", []) + response_json.get("subjects", []) + response_json.get("keywords", [])
    except json.JSONDecodeError:
        print("Invalid JSON response:", response)
        return []

    return keywords
    
def store_memory(bot_id, timestamp, conversation_text, content):
    prompt = f"""
    Break the following group of texts into separate, yet coherent smaller groups of texts without losing any information.  
    Each group covering a specific subtopic, and only be around 4 sentences in length.

    Text:
    --- SEGMENT ---
    {content}
    --- SEGMENT ---
    Use the following JSON format:
    {{
        "topic": "<Overall topic of this text>",
        "number_of_subtopics": "<Number of subtopics to record from text>",
        "groups": [
            {{
                "text": "<USER/ASSISTANT>: <Text of subtopic 1>"
            }},
            {{
                "text": "<USER/ASSISTANT>: <Text of subtopic 2>"
            }},
            {{
                "text": "<USER/ASSISTANT>: <Text of subtopic 3>"
            }},
            {{
                "<etc>": "<etc>"
            }}                         
        ]
    }}
    """
    response = embed_api_endpoint_call(prompt)
    print(f"Store Memory Response: {response}")
    try:
        response_json = json.loads(response)
        groups = response_json.get("groups", [])
        
    except json.JSONDecodeError:
        print("Invalid JSON response:", response)
        return []
    keywords = [response_json.get("topic")]
    keywords.extend(extract_keywords(conversation_text, content))
    memory = Memory.create(bot_id=bot_id, timestamp=timestamp, content=content)
    for group in groups:
        text = group['text']
        new_keywords = extract_keywords(conversation_text, text)
        keywords.extend(new_keywords)
    # Flatten the keywords list and ensure unique keywords
    unique_keywords = set(keywords)
    print("TOTAL KEYWORDS: ", unique_keywords)
    for keyword in unique_keywords:
        Keyword.create(memory_id=memory.id, keyword=keyword)

def search_memories(bot_id, keywords):
    keyword_conditions = [Memory.content.ilike(f"%{keyword}%") for keyword in keywords]

    memories = (Memory.query
                .filter_by(bot_id=bot_id)
                .filter(or_(*keyword_conditions))
                .with_entities(Memory, func.count(Memory.content).label('keyword_count'))
                .group_by(Memory.id)
                .order_by(func.count(Memory.content).asc())
                .all())
    
    return [memory for memory, _ in memories]


def truncate_message_history(conversation, truncation_char_limit):
    print("truncation_char_limit: ", truncation_char_limit)
    total_chars = sum(len(msg.conversation) for msg in conversation)
    print("total_chars: ", total_chars)
    while total_chars > truncation_char_limit and len(conversation) > 1:
        total_chars -= len(conversation.pop(0).conversation)
    return conversation

def parse_summary_response(response):
    try:
        summary_json = json.loads(response)
        return summary_json.get("Summary", "")
    except json.JSONDecodeError:
        logging.error("Failed to decode JSON response")
        return ""

def summarize_memory(memories, keywords, system_memory_summary_char_limit, memory_summary_char_limit):
    memory_text = "\n".join([memory.content for memory in memories])
    api_url = config.get('embed_api_endpoint', 'api_url')
    context_length = config.getint('embed_api_endpoint', 'context_length')
    reserved_tokens_for_response = config.getint('embed_api_endpoint', 'reserved_tokens_for_response')
    max_prompt_length = system_memory_summary_char_limit

    # Split memory text into manageable chunks
    if len(memory_text) == 0:
        chunks = []
    else:
        chunks = [memory_text[i:i + max_prompt_length] for i in range(0, len(memory_text), max_prompt_length)]
        if not chunks:  # This handles the case where memory_text is smaller than max_prompt_length
            chunks = [memory_text]

    keyword_text = ", ".join(keywords)
    summaries = []
    print("CHUNKS LENGTH: ", len(chunks))
    for chunk in chunks:
        summary_prompt = f"""
        Summarize the following text as concisely as possible, maintaining ALL details and chronological order.
        Focus on the following keywords: {keyword_text}.

        DO NOT LIE, DO NOT GENERATE NEW CONTENT, DERIVE ONLY INFORMATION FROM THE TEXT.
        Text:
        --- SEGMENT ---
        {chunk}
        --- SEGMENT ---
        Use the following JSON format:
        {{
            "Summary": "<summary>"
        }}
        """
        summary_response = embed_api_endpoint_call(summary_prompt)
        summary = parse_summary_response(summary_response)
        if summary:
            summaries.append(summary)
    
    final_summary = ' '.join(summaries)
    print("Final Summary: ", final_summary)

    # Truncate the final summary if it exceeds the limit
    final_summary_text = final_summary if final_summary else ""
    if len(final_summary_text) > memory_summary_char_limit:
        final_summary_text = final_summary_text[:memory_summary_char_limit]

    return final_summary_text

def find_message_content(data):
    if isinstance(data, dict):
        if 'message' in data and isinstance(data['message'], dict) and 'content' in data['message']:
            return data['message']['content']
        for key, value in data.items():
            result = find_message_content(value)
            if result:
                return result
    elif isinstance(data, list):
        for item in data:
            result = find_message_content(item)
            if result:
                return result
    return None

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = validate_and_sanitize_input(request.json)
    except BadRequest as e:
        app.logger.error(f"Validation error: {e}")
        return jsonify({'error': str(e)}), 400

    user_input = data['user_input']
    bot_name = data['bot_name']

    bot = Bot.query.filter_by(name=bot_name).first()
    if not bot:
        return jsonify({'error': 'Bot not found'}), 404

    bot_persona = bot.persona
    endpoint_name = bot.endpoint_name

    endpoint = ApiEndpoint.query.filter_by(name=endpoint_name).first()
    if not endpoint:
        app.logger.error(f"Invalid endpoint name: {endpoint_name}")
        return jsonify({'error': 'Invalid endpoint name'}), 400

    api_url = endpoint.api_url
    token = endpoint.token
    model = endpoint.model
    context_length = endpoint.context_length
    reserved_tokens_for_response = endpoint.reserved_tokens_for_response

    if not api_url or not model:
        app.logger.error(f"Missing endpoint configuration: api_url={api_url}, model={model}")
        return jsonify({'error': 'API URL and model are required for the endpoint'}), 400

    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = f'Bearer {token}'

    prompt_token_limit = context_length - reserved_tokens_for_response
    prompt_char_limit = prompt_token_limit * 4

    latest_conversation = get_bot_conversation(bot_name, 2)

    latest_exchange = f"user: {user_input}"
    if len(latest_conversation) > 0:
        latest_exchange = f"{latest_conversation[-1].role}: {latest_conversation[-1].conversation}\n{latest_exchange}"
    print("LATEST EXCHANGE: ", latest_exchange)

    bot_conversation = get_bot_conversation(bot_name)
    bot_conversation = truncate_message_history(bot_conversation, int(prompt_char_limit * 1 / 5))
    conversation_text = "\n".join([f"{msg.role}: {msg.conversation}" for msg in bot_conversation])

    # Extract keywords using the model
    keywords = extract_keywords_with_model(conversation_text, latest_exchange)
    keyword_query = " ".join(keywords) if keywords else ""

    # Search related memories using FAISS
    related_memories_indices = search_memory_faiss(keyword_query)
    if related_memories_indices is None:
        related_memories_indices = []

    related_memories = [get_memory_by_index(index) for index in related_memories_indices if index != -1 and get_memory_by_index(index) is not None]

    # Combine context for response generation
    combined_context = combine_contexts(user_input, related_memories) if related_memories else user_input

    system_message = f"Your name is {bot_name}. Assume the role of {bot_name} and adhere to the following persona: {bot_persona}. For any formatting, use Markdown syntax. The following are related memories:\n{combined_context}"
    conversation_messages = [{
        'role': 'system',
        'content': system_message
    }]

    for message in bot_conversation:
        conversation_messages.append({
            'role': message.role,
            'content': message.conversation
        })

    conversation_messages.append({
        'role': 'user',
        'content': user_input
    })

    # If inner_monologue is enabled, add the inner monologue step
    if bot.inner_monologue:
        try:
            monologue_prompt = f"""
            Analyze the conversation and the user's latest response.  Provide your thoughts in the following sections: ANALYSIS, {bot_name}'S THOUGHTS, {bot_name}'S RESPONSE INTENT.
            
            CONVERSATION TEXT:
            --- SEGMENT ---
            {combined_context}
            --- SEGMENT ---

            USER'S RESPONSE:
            --- SEGMENT ---
            {user_input}
            --- SEGMENT ---
            """
            monologue_messages = [{
                'role': 'system',
                'content': system_message
            }]
            for message in bot_conversation:
                monologue_messages.append({
                    'role': message.role,
                    'content': message.conversation
                })
            monologue_messages.append({
                'role': 'user',
                'content': user_input
            })
            json_data = {
                'model': model,
                'stream': False,
                'temperature': 1,
                'top_p': 1,
                'messages': monologue_messages,
                'max_tokens': reserved_tokens_for_response
            }
            monologue_response = make_api_call_with_retries(api_url, headers, json_data)
            inner_monologue_response = find_message_content(monologue_response)
            conversation_messages.append({
                'role': 'assistant',
                'content': f"**Inner Monologue:** {inner_monologue_response}"
            })
            json_data['messages'] = conversation_messages  # Update json_data with the inner monologue
        except (requests.exceptions.RequestException, ValueError, json.decoder.JSONDecodeError) as e:
            app.logger.error(f"Error during inner monologue API call: {e}")
            return jsonify({'error': str(e)}), 500

    json_data = {
        'model': model,
        'stream': False,
        'temperature': 1,
        'top_p': 1,
        'messages': conversation_messages,
        'max_tokens': reserved_tokens_for_response
    }
    try:
        response = make_api_call_with_retries(api_url, headers, json_data)
        chatbot_response = response
        response_text = find_message_content(chatbot_response)
    except (requests.exceptions.RequestException, ValueError, json.decoder.JSONDecodeError) as e:
        app.logger.error(f"Error during API call: {e}")
        return jsonify({'error': str(e)}), 500

    MainConversation.create(role='user', conversation=user_input, timestamp=datetime.now(), bot_name=bot_name)
    MainConversation.create(role='assistant', conversation=response_text, timestamp=datetime.now(), bot_name=bot_name)

    latest_exchange = f"ASSISTANT:\n{response_text}\nUSER:\n{user_input}"
    keywords_second = extract_keywords_with_model(conversation_text, latest_exchange)
    
    print ("KEYWORDS: ", keywords + keywords_second)
    exit
    store_memory_faiss(bot.id, latest_exchange, keywords + keywords_second)

    return jsonify({'response': response_text})


@app.route('/delete_bot', methods=['DELETE'])
def delete_bot():
    bot_name = request.args.get('bot_name')

    if not bot_name:
        return jsonify({'error': 'Bot name is required'}), 400

    bot = Bot.query.filter_by(name=bot_name).first()
    if not bot:
        return jsonify({'error': 'Bot not found'}), 404

    # Manually delete related records
    memories = Memory.query.filter_by(bot_id=bot.id).all()
    for memory in memories:
        Keyword.query.filter_by(memory_id=memory.id).delete()
        memory.delete()

    bot.delete()
    db.session.commit()

    return jsonify({'message': 'Bot deleted successfully'})

if __name__ == '__main__':
    app.run(debug=True)
