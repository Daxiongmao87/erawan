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
from sqlalchemy import func, or_, text
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
        print("Called create")
        instance = cls(**kwargs)
        return instance.save()

    def update(self, commit=True, **kwargs):
        print("Called update")
        for attr, value in kwargs.items():
            setattr(self, attr, value)
        return commit and self.save() or self

    def save(self, commit=True):
        print("Called save")
        db.session.add(self)
        if commit:
            db.session.commit()
        return self

    def delete(self, commit=True):
        print("Called delete")
        db.session.delete(self)
        if commit:
            db.session.commit()

class Bot(db.Model, CRUDMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), unique=True, nullable=False)
    persona = db.Column(db.Text, nullable=False)
    endpoint_name = db.Column(db.String(120), nullable=False)
    inner_monologue = db.Column(db.Integer, default=0)

class MainConversation(db.Model, CRUDMixin):
    id = db.Column(db.Integer, primary_key=True)
    role = db.Column(db.String(120), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    conversation = db.Column(db.Text, nullable=False)
    bot_name = db.Column(db.String(120), nullable=False)

class ApiEndpoint(db.Model, CRUDMixin):
    id = db.Column(db.Integer, primary_key=True)
    api_id = db.Column(db.Integer, nullable=True)  # Allow nullable initially for the first entry
    variable_name = db.Column(db.String(120), nullable=False)
    variable_value = db.Column(db.Text, nullable=False)


### RAG

dimension = config.getint('embed_api_endpoint', 'dimension')
faiss_indices = {}
faiss_mappings = {}

def initialize_faiss_for_bot(bot_id, dimension):
    print("Called initialize_faiss_for_bot")
    if bot_id not in faiss_indices:
        print(f"Initializing FAISS for bot {bot_id}")
        faiss_indices[bot_id] = faiss.IndexFlatL2(dimension)
        faiss_mappings[bot_id] = {}

# Initialize FAISS indices for existing bots
def initialize_bots():
    print("Called initialize_bots")
    with app.app_context():
        bots = Bot.query.all()
        for bot in bots:
            initialize_faiss_for_bot(bot.id, dimension)

def save_faiss_mapping():
    print("Called save_faiss_mapping")
    with open('faiss_mapping.pkl', 'wb') as f:
        pickle.dump(faiss_mappings, f)

def get_embeddings(text):
    print("Called get_embeddings")
    api_url = config.get('embed_api_endpoint', 'api_url')
    headers = {'Content-Type': 'application/json'}
    api_key = config.get('embed_api_endpoint', 'api_key', fallback=None)
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}' 

    # Prepare payload with dynamic fields
    json_data = get_config_payload('embed_api_endpoint')
    json_data['input'] = text[:json_data.get('max_tokens', len(text))]

    try:
        response = requests.post(api_url, headers=headers, json=json_data)
        response.raise_for_status()
        response_json = response.json()
        embedding = response_json.get('data', [{}])[0].get('embedding')
        if embedding is None:
            raise ValueError("Embedding not found in response")
        return np.array(embedding, dtype=np.float32)
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to get embeddings: {e} - Response text: {response.text}")
        return None
    except ValueError as e:
        logging.error(f"Error in API response: {e}")
        return None


def store_memory_faiss(bot_id, text, keywords):
    print("Called store_memory_faiss")
    initialize_faiss_for_bot(bot_id, dimension)  # Ensures an index is there
    text_embedding = get_embeddings(text)
    if text_embedding is not None:
        index = faiss_indices[bot_id]
        index.add(np.array([text_embedding]))
        faiss_index_id = index.ntotal - 1
        memory_id = f"memory_{faiss_index_id}"
        faiss_mappings[bot_id][memory_id] = faiss_index_id
        print(f"Memory {memory_id} added to FAISS for bot {bot_id}")

        # Store embeddings for each keyword
        print("Keywords: ", keywords)
        for keyword in keywords:
            print("Saving embedding for: ", keyword)
            keyword_embedding = get_embeddings(keyword)
            if keyword_embedding is not None:
                faiss_indices[bot_id].add(np.array([keyword_embedding]))
                keyword_faiss_id = faiss_indices[bot_id].ntotal - 1
                faiss_mappings[bot_id][f"{memory_id}_keyword_{keyword}"] = keyword_faiss_id

        save_faiss_mapping()
        return memory_id
    return None

def search_memory_faiss(bot_id, query, k=10):
    print("Called search_memory_faiss")
    query_embedding = get_embeddings(query)
    if query_embedding is not None:
        distances, indices = faiss_indices[bot_id].search(np.array([query_embedding]), k)
        return indices[0] if np.any(indices >= 0) else []
    return []

def delete_memory(memory_id):
    print("Called delete_memory")
    if memory_id in faiss_mapping:
        faiss_id = faiss_mapping.pop(memory_id)
        # Here we need to recreate the FAISS index without the deleted entry
        all_embeddings = [faiss_index.reconstruct(i) for i in range(faiss_index.ntotal) if i != faiss_id]
        faiss_index.reset()
        faiss_index.add(np.array(all_embeddings))
        Memory.query.filter_by(id=memory_id).delete()
        save_faiss_mapping()
        db.session.commit()

def get_config_payload(section):
    print("Called get_config_payload")
    config_items = {key: config.get(section, key) for key in config[section]}
    # Convert integer-like strings to integers
    for key, value in config_items.items():
        if value.isdigit():
            config_items[key] = int(value)
    return config_items

def embed_api_endpoint_call(api_id, prompt):
    print("Called embed_api_endpoint_call")
    
    # Retrieve the API configurations based on api_id
    configurations = get_api_configurations(api_id)
    
    # Extract the API URL and API key
    api_url = configurations.get('api_url')
    headers = {'Content-Type': 'application/json'}
    if 'api_key' in configurations:
        headers['Authorization'] = f'Bearer {configurations["api_key"]}'
    
    # Use the new build_json_payload function
    json_data = build_json_payload(configurations)
    json_data_replaced = replace_json_payload_placeholders(json_data, '{"embed_input": {prompt}}')
    # Make the API call
    try:
        response = requests.post(api_url, headers=headers, json=json_data_replaced)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to call System API Endpoint: {e}")
        return None

###

def build_json_payload(configurations):
    print("Called build_json_payload")
    # Build the dynamic JSON payload
    print(configurations)
    json_data = {key: value for key, value in configurations.items() if key not in ['api_url', 'name', 'api_key']}
    
    return json_data


def replace_json_payload_placeholders(json_data, replacements):
    print("Called replace_json_payload_placeholders")
    # If the input is already a dictionary, we skip json.loads
    if isinstance(json_data, str):
        json_dict = json.loads(json_data)
    else:
        json_dict = json_data  # It is already a dict
    
    # Replace the placeholders with the actual values
    for key, value in json_dict.items():
        if isinstance(value, dict):  # Handle nested dictionaries
            json_dict[key] = replace_json_payload_placeholders(value, replacements)
        elif isinstance(value, str):  # Check if the value is a string
            # Check if the string matches a placeholder format and replace
            for placeholder, replacement in replacements.items():
                if value == f"{{{placeholder}}}":
                    json_dict[key] = replacement
    
    # Convert the dictionary back to a json string
    return json.dumps(json_dict)




def extract_keywords_with_model(conversation_text, latest_exchange):
    print("Called extract_keywords_with_model")
    
    try:
        max_tokens = config.getint('system_api_endpoint', 'max_tokens')
    except configparser.NoOptionError:
        max_tokens = int(len(conversation_text))
    max_tokens_conversation = int(max_tokens * 0.9)
    conversation_text = conversation_text[:max_tokens_conversation]
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

    # Prepare payload with dynamic fields
    json_data = get_config_payload('embed_api_endpoint')
    headers = {'Content-Type': 'application/json'}
    print("test 1")
    if "api_key" in json_data:
        headers['Authorization'] = f'Bearer {json_data["api_key"]}'
    print("Test 2")
    api_url = json_data["api_url"]
    json_data = build_json_payload(json_data)
    json_data_replaced = replace_json_payload_placeholders(json_data, {"embed_input": prompt})
    print("Loading json")
    json_obj = json.loads(json_data_replaced)
    print(json.dumps(json_obj, indent=4))
    try:
        response = requests.post(api_url, headers=headers, json=json_obj)
        response.raise_for_status()

        response_json = response.json()
        keywords = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
        return json.loads(keywords).get('keywords', [])
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to extract keywords: {e} - Response text: {response.text}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing the response JSON: {e}")
        return []

@app.route('/')
def index():
    print("Called index")
    return render_template('index.html')

@app.route('/bots', methods=['GET'])
def get_bots():
    print("Called get_bots")
    bots = Bot.query.all()
    return jsonify([{'name': bot.name, 'persona': bot.persona} for bot in bots])

@app.route('/create_bot', methods=['POST'])
def create_bot():
    print("Called create_bot")
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
    initialize_faiss_for_bot(bot.id, dimension)
    return jsonify({'name': bot.name, 'persona': bot.persona, 'endpoint_name': bot.endpoint_name, 'inner_monologue': bool(bot.inner_monologue)})

@app.route('/update_bot_api_endpoint', methods=['POST'])
def update_bot_api_endpoint():
    print("Called update_bot_api_endpoint")
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
    print("Called chat_history")
    bot_name = request.args.get('bot_name')

    if not bot_name:
        return jsonify({'error': 'Bot name is required'}), 400

    conversations = MainConversation.query.filter_by(bot_name=bot_name).all()
    return jsonify([{'role': conv.role, 'conversation': conv.conversation, 'timestamp': conv.timestamp} for conv in conversations])

@app.route('/api_endpoints', methods=['GET'])
def get_api_endpoints():
    print("Called get_api_endpoints")
    endpoints = ApiEndpoint.query.group_by(ApiEndpoint.api_id).all()

    response = []
    for ep in endpoints:
        endpoint_data = {
            'api_id': ep.api_id,
            'configurations': {}
        }

        # Retrieve all configurations associated with this api_id
        configurations = ApiEndpoint.query.filter_by(api_id=ep.api_id).all()
        for config in configurations:
            endpoint_data['configurations'][config.variable_name] = config.variable_value

        response.append(endpoint_data)

    return jsonify(response)

@app.route('/add_api_endpoint', methods=['POST'])
def add_api_endpoint():
    print("Called add_api_endpoint")
    data = request.json
    name = data.get('name')
    api_url = data.get('api_url')
    config = data.get('config')

    if not name or not api_url:
        return jsonify({'error': 'API Name and API URL are required'}), 400

    # Step 1: Retrieve the current highest api_id and increment it
    result = db.session.execute(text("SELECT IFNULL(MAX(api_id), 0) + 1 AS new_api_id FROM api_endpoint"))
    new_api_id = result.scalar()

    # Step 2: Insert the 'name' entry with the generated new_api_id
    name_entry = ApiEndpoint(api_id=new_api_id, variable_name='name', variable_value=name)
    db.session.add(name_entry)

    # Step 3: Insert the 'api_url' entry with the correct api_id
    api_url_entry = ApiEndpoint(api_id=new_api_id, variable_name='api_url', variable_value=api_url)
    db.session.add(api_url_entry)

    # Step 4: Add the rest of the configuration entries with the associated api_id
    for entry in config:
        variable_name = entry.get('variable_name')
        variable_value = entry.get('variable_value')

        if not variable_name or not variable_value:
            return jsonify({'error': f'Invalid configuration: {entry}'}), 400
        
        config_entry = ApiEndpoint(api_id=new_api_id, variable_name=variable_name, variable_value=variable_value)
        db.session.add(config_entry)

    db.session.commit()

    return jsonify({'message': 'API endpoint created successfully', 'api_id': new_api_id})

@app.route('/bot_info', methods=['GET'])
def bot_info():
    print("Called bot_info")
    bot_name = request.args.get('bot_name')

    if not bot_name:
        return jsonify({'error': 'Bot name is required'}), 400

    bot = Bot.query.filter_by(name=bot_name).first()
    if not bot:
        return jsonify({'error': 'Bot not found'}), 404

    return jsonify({'name': bot.name, 'persona': bot.persona, 'endpoint_name': bot.endpoint_name})

def get_api_configurations(api_id):
    print("Called get_api_configurations")
    api_endpoint = ApiEndpoint.query.filter_by(api_id=api_id).all()
    # Convert query into JSON format
    configurations = {entry.variable_name: entry.variable_value for entry in api_endpoint}
    return configurations

@app.route('/update_api_endpoint', methods=['POST'])
def update_api_endpoint():
    print("Called update_api_endpoint")
    data = request.json
    api_id = data.get('api_id')
    new_api_url = data.get('api_url')
    config = data.get('config')

    if not api_id or not new_api_url:
        return jsonify({'error': 'API ID and new API URL are required'}), 400

    # Update the API URL (this could be in a separate table if needed)
    # Update the configurations
    for entry in config:
        variable_name = entry.get('variable_name')
        variable_value = entry.get('variable_value')
        
        if not variable_name or not variable_value:
            return jsonify({'error': f'Invalid entry in config: {entry}'}), 400
        
        config_entry = ApiEndpoint.query.filter_by(api_id=api_id, variable_name=variable_name).first()
        if config_entry:
            config_entry.update(variable_value=variable_value)
        else:
            ApiEndpoint.create(api_id=api_id, variable_name=variable_name, variable_value=variable_value)

    return jsonify({'message': 'Endpoint updated successfully'})

@app.route('/delete_api_endpoint', methods=['DELETE'])
def delete_api_endpoint():
    print("Called delete_api_endpoint")
    name = request.args.get('name')

    if not name:
        return jsonify({'error': 'Name is required'}), 400

    # Find the `api_id` associated with the provided name
    endpoint_entry = ApiEndpoint.query.filter_by(variable_name='name', variable_value=name).first()
    if not endpoint_entry:
        return jsonify({'error': 'Endpoint not found'}), 404

    # Retrieve the `api_id` from the found entry
    api_id = endpoint_entry.api_id

    # Delete all entries with the found `api_id`
    ApiEndpoint.query.filter_by(api_id=api_id).delete()
    db.session.commit()

    return jsonify({'message': 'API endpoint deleted successfully'})


def validate_and_sanitize_input(data):
    print("Called validate_and_sanitize_input")
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
    print("Called is_valid_json")
    try:
        json.loads(text)
        return True
    except ValueError:
        return False

def make_api_call_with_retries(configurations, prompt, system_prompt=None):
    print("Called make_api_call_with_retries")
    print(configurations)
    api_key = configurations["api_key"] if "api_key" in configurations else None
    api_url = configurations["api_url"]
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    print(configurations)
    json_data = build_json_payload(configurations)
    
    # Replace placeholders in json_data
    replacements = {
        'system_prompt': system_prompt,
        'messages': prompt
    }
    json_data_replaced = replace_json_payload_placeholders(json.dumps(json_data), replacements)
    
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = requests.post(api_url, headers=headers, json=json.loads(json_data_replaced))
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
                logging.error(f"Request failed after {MAX_RETRIES} attempts: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    logging.error(f"Response content: {e.response.content.decode(errors='replace')}")
                raise e
            logging.warning(f"Request failed. Retrying... {retries}/{MAX_RETRIES}")
            if hasattr(e, 'response') and e.response is not None:
                logging.warning(f"Response content: {e.response.content.decode(errors='replace')}")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            retries += 1
            if retries >= MAX_RETRIES:
                logging.error(f"Unexpected error after {MAX_RETRIES} attempts: {e}")
                raise e
            logging.warning(f"Unexpected error. Retrying... {retries}/{MAX_RETRIES}")
            time.sleep(RETRY_DELAY)

def get_bot_conversation(bot_name, limit = 0):
    print("Called get_bot_conversation")
    if limit > 0:
        conversation = MainConversation.query.filter_by(bot_name=bot_name).order_by(MainConversation.timestamp.desc()).limit(limit).all()
        conversation.reverse()
    else:
        conversation = MainConversation.query.filter_by(bot_name=bot_name).order_by(MainConversation.timestamp.asc()).all()
    return conversation

def extract_keywords(text, latest_exchange):
    print("Called extract_keywords")
    
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
    

def find_message_content(data):
    print("Called find_message_content")
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

@app.route('/delete_bot', methods=['DELETE'])
def delete_bot():
    print("Called delete_bot")
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


@app.route('/chat', methods=['POST'])
def chat():
    print("Called chat")
    
    try:
        data = validate_and_sanitize_input(request.json)
    except BadRequest as e:
        app.logger.error(f"Validation error: {e}")
        return jsonify({'error': str(e)}), 400

    user_input = data['user_input']
    bot_name = data['bot_name']
    
    try:
        bot, configurations = retrieve_bot_and_configurations(bot_name)
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    
    try:
        conversation_messages = build_conversation_context(bot, user_input, configurations)
    except Exception as e:
        app.logger.error(f"Error building conversation context: {e}")
        return jsonify({'error': str(e)}), 500

    try:
        response_text = finalize_conversation_and_make_api_call(bot, configurations, conversation_messages)
    except Exception as e:
        app.logger.error(f"Error during API call: {e}")
        return jsonify({'error': str(e)}), 500
    
    store_conversation_and_keywords(bot, user_input, response_text)
    
    return jsonify({'response': response_text})

def retrieve_bot_and_configurations(bot_name):
    print("Called retrieve_bot_and_configurations")
    bot = Bot.query.filter_by(name=bot_name).first()
    if not bot:
        raise ValueError('Bot not found')

    endpoint_entry = ApiEndpoint.query.filter_by(variable_name='name', variable_value=bot.endpoint_name).first()
    if not endpoint_entry:
        raise ValueError('Invalid endpoint name')

    configurations = get_api_configurations(endpoint_entry.api_id)
    return bot, configurations

def build_conversation_context(bot, user_input, configurations):
    print("Called build_conversation_context")
    latest_conversation = get_bot_conversation(bot.name, 2)
    conversation_text = "\n".join([f"{msg.role}: {msg.conversation}" for msg in latest_conversation])
    keywords = extract_keywords_with_model(conversation_text, user_input)
    keyword_query = " ".join(keywords) if keywords else ""

    related_memories_indices = search_memory_faiss(bot.id, keyword_query)
    if related_memories_indices is None:
        related_memories_indices = []

    #get related memories from faiss
    related_memories = []
    for memory_id in related_memories_indices:
        related_memories.append(find_endpoint_variable_value(configurations, memory_id))

    combined_context = create_combined_context(user_input, related_memories)

    system_prompt = f"Your name is {bot.name}. Assume the role of {bot.name} and adhere to the following persona: {bot.persona}. For any formatting, use Markdown syntax. The following are related memories:\n{combined_context}"
    conversation_messages = [{'role': 'system', 'content': system_prompt}]

    for message in latest_conversation:
        conversation_messages.append({'role': message.role, 'content': message.conversation})

    conversation_messages.append({'role': 'user', 'content': user_input})
    return conversation_messages

def find_endpoint_variable_value(endpoints, target_variable):
    print("Called find_endpoint_variable_value")
    print(endpoints)
    for endpoint in endpoints:
        # print object's properties
        print(endpoint)
        if endpoint.variable_name == target_variable:
            return endpoint.variable_value
    return None

def finalize_conversation_and_make_api_call(bot, configurations, conversation_messages):
    print("Called finalize_conversation_and_make_api_call")
    response = make_api_call_with_retries(configurations, conversation_messages, conversation_messages[-1]['content'])
    return find_message_content(response)

def store_conversation_and_keywords(bot, user_input, response_text):
    print("Called store_conversation_and_keywords")
    # Store the user input and response text in the database
    MainConversation.create(role='user', conversation=user_input, timestamp=datetime.now(), bot_name=bot.name)
    MainConversation.create(role='assistant', conversation=response_text, timestamp=datetime.now(), bot_name=bot.name)

    # Extract keywords from the user input and response text
    latest_exchange = f"ASSISTANT:\n{response_text}\nUSER:\n{user_input}"
    keywords_first = extract_keywords_with_model(user_input, latest_exchange)
    keywords_second = extract_keywords_with_model(response_text, latest_exchange)
    # Store the user input and response text with the extracted keywords
    store_memory_faiss(bot.id, latest_exchange, keywords_first + keywords_second)

def create_combined_context(user_input, related_memories):
    print("Called create_combined_context")
    # Combine the user input with related memories
    if related_memories:
        return f"{user_input}\n\nRelated Memories:\n" + "\n".join(related_memories)
    return user_input

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        initialize_bots()
    app.run(debug=True)
