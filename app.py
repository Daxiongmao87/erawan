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

# Set up logging
logging.basicConfig(level=logging.DEBUG)

config = configparser.ConfigParser()
config.read('config.ini')

app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'memory.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['DEBUG'] = True  # Enable debug mode
db = SQLAlchemy(app)

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Configurable parameters
TOTAL_TOKEN_LIMIT = 128000
RESERVED_TOKENS_FOR_RESPONSE = 4096
PROMPT_TOKEN_LIMIT = TOTAL_TOKEN_LIMIT - RESERVED_TOKENS_FOR_RESPONSE
TRUNCATION_LIMIT = int(PROMPT_TOKEN_LIMIT * 2 / 5)
MEMORY_SUMMARY_LIMIT = int(PROMPT_TOKEN_LIMIT * 1 / 5)
LATEST_MESSAGES_LIMIT = int(PROMPT_TOKEN_LIMIT * 1 / 5)

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


db.create_all()

def system_api_endpoint_call(prompt):
    api_url = config.get('system_api_endpoint', 'api_url')
    token = config.get('system_api_endpoint', 'token')
    model = config.get('system_api_endpoint', 'model')

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
        'temperature': 1,
        'top_p': 1
    }

    try:
        response = requests.post(api_url, headers=headers, json=json_data)
        print(f"RESPONSE: {response.text}")
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to call System API Endpoint: {e}")
        return None

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

    if not bot_name or not bot_persona or not endpoint_name:
        return jsonify({'error': 'Bot name, persona, and endpoint_name are required'}), 400

    existing_bot = Bot.query.filter_by(name=bot_name).first()
    if existing_bot:
        return jsonify({'error': 'Bot with this name already exists'}), 400

    bot = Bot.create(name=bot_name, persona=bot_persona, endpoint_name=endpoint_name)
    return jsonify({'name': bot.name, 'persona': bot.persona, 'endpoint_name': bot.endpoint_name})

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

def get_bot_conversation(bot_name):
    conversation = MainConversation.query.filter_by(bot_name=bot_name).order_by(MainConversation.timestamp.desc()).limit(2).all()
    return conversation

def extract_keywords(text, latest_exchange):
    prompt = f"""
    Verbosely extract relevant keywords from the following text.  
    Be complete and thorough. 
    Utilize the previous exchanges in the conversation history mainly as a contextutal reference.
    Focus on the very latest exchange for keywords.

    Last Exchange:
    {latest_exchange}

    Respond with a JSON object in the format:
    {{
        "keywords": ["<keyword1>", "<keyword2>", "<keyword3>", "<etc> ..."]
    }}:
    {text}
    """
    print(f"Prompt: {prompt}")
    response = system_api_endpoint_call(prompt)
    print("KEYWORDS RESPONSE: ", response)

    if response is None:
        return []

    try:
        response_json = json.loads(response)
        keywords = response_json.get("keywords", [])
    except json.JSONDecodeError:
        print("Invalid JSON response:", response)
        return []

    return keywords

    
def store_memory(bot_id, timestamp, content):
    memory = Memory.create(bot_id=bot_id, timestamp=timestamp, content=content)
    keywords = extract_keywords(content, content)

    for keyword in keywords:
        Keyword.create(memory_id=memory.id, keyword=keyword)

def search_memories(bot_id, keywords):
    memories = Memory.query.filter_by(bot_id=bot_id).join(Keyword).filter(Keyword.keyword.in_(keywords)).all()
    return memories

def truncate_message_history(conversation):
    total_tokens = sum(len(msg.conversation) // 4 for msg in conversation)
    while total_tokens > TRUNCATION_LIMIT and len(conversation) > 1:
        total_tokens -= len(conversation.pop(0).conversation) // 4
    return conversation

def parse_summary_response(response):
    try:
        summary_json = json.loads(response)
        return summary_json.get("Summary", "")
    except json.JSONDecodeError:
        logging.error("Failed to decode JSON response")
        return ""

def summarize_memory(memories, keywords):
    memory_text = "\n".join([memory.content for memory in memories])
    api_url = config.get('system_api_endpoint', 'api_url')
    context_length = config.getint('system_api_endpoint', 'context_length')
    reserved_tokens_for_response = config.getint('system_api_endpoint', 'reserved_tokens_for_response')
    max_prompt_length = context_length - reserved_tokens_for_response

    # Split memory text into manageable chunks
    chunks = [memory_text[i:i + max_prompt_length * 4] for i in range(0, len(memory_text), max_prompt_length * 4)]

    keyword_text = ", ".join(keywords)
    summaries = []
    for chunk in chunks:
        summary_prompt = f"""
        Summarize the following text as concisely as possible, maintaining ALL details and chronological order.
        Focus on the following keywords: {keyword_text}.

        Text:
        ---
        {chunk}
        ---
        Use the following JSON format:
        {{
            "Summary": "<summary>"
        }}
        """
        summary_response = system_api_endpoint_call(summary_prompt)
        summary = parse_summary_response(summary_response)
        if summary:
            summaries.append(summary)
    
    final_summary_prompt = f"""
    Combine the following summaries into a single, concise summary that maintains ALL details.
    Focus on the following keywords: {keyword_text}.

    Text:
    ---
    {' '.join(summaries)}
    ---
    Use the following JSON format:
    {{
        "Summary": "<summary>"
    }}
    """
    final_summary_response = system_api_endpoint_call(final_summary_prompt)
    final_summary = parse_summary_response(final_summary_response)

    # Truncate the final summary if it exceeds the limit
    final_summary_text = final_summary if final_summary else ""
    if len(final_summary_text) > max_prompt_length * 4:
        final_summary_text = final_summary_text[:max_prompt_length * 4]

    return final_summary_text


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

    # Calculate prompt token limit
    prompt_token_limit = context_length - reserved_tokens_for_response
    truncation_limit = int(prompt_token_limit * 2 / 5)
    memory_summary_limit = int(prompt_token_limit * 1 / 5)
    latest_messages_limit = int(prompt_token_limit * 1 / 5)

    # Step 1: Get the latest conversation exchange
    bot_conversation = get_bot_conversation(bot_name)
    bot_conversation = truncate_message_history(bot_conversation)
    conversation_text = "\n".join([f"{msg.role}: {msg.conversation}" for msg in bot_conversation])

    latest_exchange = f"user: {user_input}"
    if len(bot_conversation) > 0:
        latest_exchange = f"{bot_conversation[-1].role}: {bot_conversation[-1].conversation}\n{latest_exchange}"

    # Step 2: Extract keywords from the latest conversation
    keywords = extract_keywords(conversation_text, latest_exchange)
    if not keywords:
        return jsonify({'error': 'Failed to extract keywords'}), 500

    # Step 3: Search for related memories
    related_memories = search_memories(bot.id, keywords)
    summarized_memory = summarize_memory(related_memories, keywords)


    system_message = f"Your name is {bot_name}. Assume the role of {bot_name} and adhere to the following persona: {bot_persona}. For any formatting, use Markdown syntax. The following are related memories:\n{summarized_memory}"
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
    print(f"CONVERSATION MESSAGES: {conversation_messages}")
    json_data = {
        'model': model,
        'stream': False,
        'temperature': 1,
        'top_p': 1,
        'messages': conversation_messages,
        'max_tokens': reserved_tokens_for_response
    }
    try:
        app.logger.debug(f"API request content: {json_data}")
        response = make_api_call_with_retries(api_url, headers, json_data)
        
        app.logger.debug(f"API response content: {response}")

        chatbot_response = response
        response_text = chatbot_response.get('choices', [{}])[0].get('message', {}).get('content', '')

    except (requests.exceptions.RequestException, ValueError, json.decoder.JSONDecodeError) as e:
        app.logger.error(f"Error during API call: {e}")
        return jsonify({'error': str(e)}), 500

    MainConversation.create(role='user', conversation=user_input, timestamp=datetime.now(), bot_name=bot_name)
    MainConversation.create(role='assistant', conversation=response_text, timestamp=datetime.now(), bot_name=bot_name)

    # Step 4: Store the latest conversation exchange in memory
    latest_exchange = f"USER:\n{user_input}\nASSISTANT:\n{response_text}"
    store_memory(bot.id, datetime.now(), latest_exchange)

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
