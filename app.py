# Import necessary libraries
from flask import Flask, request, jsonify
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from datetime import datetime
import uuid
import json

# Add these imports to the existing imports
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from flask import session, redirect, url_for

# Load environment variables from .env file
load_dotenv()

# Initialize Flask application
app = Flask(__name__)

# Constants
SECRET_KEY = os.urandom(24)  # Generate a random secret key for session

# Add this: Initialize chat history storage with session support
chat_histories = {}
MAX_HISTORY_LENGTH = 10
HISTORY_DIR = "chat_histories"
USERS_FILE = os.path.join(HISTORY_DIR, "users.json")
HISTORY_FILE = os.path.join(HISTORY_DIR, "chat_histories.json")

PERSONAS = {
    "default": "You are a helpful assistant providing concise and accurate answers. Maintain context from the conversation history.",
    "expert": "You are an expert AI assistant with deep knowledge across many fields. Provide detailed, technical, and accurate information. Use academic language and cite sources when possible.",
    "friendly": "You are a friendly and approachable assistant. Use casual language, be conversational, and add occasional humor. Keep explanations simple and relatable.",
    "concise": "You are a concise assistant that values brevity. Provide short, direct answers with minimal elaboration. Use bullet points when appropriate."
}

FEW_SHOT_EXAMPLES = {
    "factual": [
        {"question": "What is the capital of France?", "answer": "The capital of France is Paris."},
        {"question": "Who wrote 'Romeo and Juliet'?", "answer": "William Shakespeare wrote 'Romeo and Juliet'."}
    ],
    "opinion": [
        {"question": "What's the best programming language for beginners?", "answer": "For beginners, Python is often recommended due to its readable syntax and gentle learning curve. However, the 'best' language depends on your goals. JavaScript is great for web development, Swift for iOS apps, etc."},
        {"question": "Is artificial intelligence dangerous?", "answer": "AI presents both opportunities and risks. While it can solve complex problems, concerns exist about job displacement, privacy, and autonomous weapons. Responsible development with ethical guidelines is important."}
    ],
    "instruction": [
        {"question": "How do I make pancakes?", "answer": "Basic pancake recipe: Mix 1 cup flour, 2 tbsp sugar, 2 tsp baking powder, and a pinch of salt. Add 1 cup milk, 2 tbsp melted butter, and 1 egg. Stir until just combined (lumps are okay). Cook on a hot greased pan until bubbles form, then flip and cook until golden."},
        {"question": "How can I improve my public speaking?", "answer": "To improve public speaking: 1) Practice regularly, 2) Record yourself and review, 3) Join a group like Toastmasters, 4) Know your material thoroughly, 5) Start with small audiences, and 6) Focus on breathing and posture."}
    ]
}

def classify_question_type(question):
    """
    Classify the type of question to select appropriate few-shot examples.
    Returns one of: 'factual', 'opinion', 'instruction', or None.
    """
    # Simple rule-based classification
    question = question.lower()

    # Check for factual questions (who, what, when, where)
    if any(q in question for q in ["what is", "who is", "when did", "where is", "how many"]):
        return "factual"

    # Check for opinion questions
    if any(q in question for q in ["what do you think", "opinion", "best", "worst", "should i", "better"]):
        return "opinion"

    # Check for instruction questions (how to)
    if any(q in question for q in ["how do i", "how to", "steps", "guide", "tutorial", "instructions"]):
        return "instruction"

    # Default to None if no clear classification
    return None

def create_prompt_template(persona="default", question_type=None):
    """
    Create a dynamic prompt template based on the persona and question type.
    """
    system_message = PERSONAS.get(persona, PERSONAS["default"])

    messages = [
        ("system", system_message),
    ]
    # Add few-shot examples if we have a question type
    if question_type and question_type in FEW_SHOT_EXAMPLES:
        examples = FEW_SHOT_EXAMPLES[question_type]

        # Add examples to the system message
        examples_text = "\n\nHere are some examples of how to answer this type of question:\n"
        for example in examples:
            examples_text += f"\nQuestion: {example['question']}\nAnswer: {example['answer']}\n"

        messages = [
            ("system", system_message + examples_text),
        ]
    # Add the actual question placeholder
    messages.append(("human", "{question}"))

    return ChatPromptTemplate.from_messages(messages)

def save_chat_histories():
    """
    Save all chat histories to a JSON file.
    """
    try:
        # Convert datetime objects to strings for JSON serialization
        serializable_histories = {}
        for session_id, history in chat_histories.items():
            serializable_histories[session_id] = history

        with open(HISTORY_FILE, 'w') as f:
            json.dump(serializable_histories, f, indent=2)
        print(f"Chat histories saved to {HISTORY_FILE}")
        return True
    except Exception as e:
        print(f"Error saving chat histories: {str(e)}")
        return False

def load_chat_histories():
    """
    Load all chat histories from a JSON file.
    Returns a dictionary of chat histories.
    """
    global chat_histories
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                chat_histories = json.load(f)
            print(f"Chat histories loaded from {HISTORY_FILE}")
            return True
        else:
            print(f"No chat history file found at {HISTORY_FILE}")
            return False
    except Exception as e:
        print(f"Error loading chat histories: {str(e)}")
        return False

def save_users(users):
    """
    Save users dictionary to a JSON file.
    """
    global USERS_FILE
    try:
        with open(USERS_FILE, 'w') as f:
            # Don't save the actual password, only the hash
            serializable_users = {}
            for username, user_data in users.items():
                serializable_users[username] = {
                    'password_hash': user_data['password_hash'],
                    'sessions': user_data['sessions']
                }
            json.dump(serializable_users, f, indent=2)
        print(f"Users saved to {USERS_FILE}")
        return True
    except Exception as e:
        print(f"Error saving users: {str(e)}")
        return False

def load_users():
    """
    Load users from a JSON file.
    Returns a dictionary of users.
    """
    global USERS_FILE
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r') as f:
                users = json.load(f)
            print(f"Users loaded from {USERS_FILE}")
            return users
        else:
            print(f"No users file found at {USERS_FILE}, creating default users")
            # Create a default admin user
            default_users = {
                'admin': {
                    'password_hash': generate_password_hash('admin'),
                    'sessions': []
                }
            }
            save_users(default_users)
            return default_users
    except Exception as e:
        print(f"Error loading users: {str(e)}")
        return {'admin': {'password_hash': generate_password_hash('admin'), 'sessions': []}}

# Initialize users dictionary
users = load_users()

def login_required(f):
    """
    Decorator to require login for specific routes.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)
    return decorated_function

load_chat_histories()

# Initialize session management
app.config['SECRET_KEY'] = SECRET_KEY

# Section 1: Configure and Validate Azure OpenAI Environment Variables
required_vars = {
    "AZURE_OPENAI_API_KEY": "API key",
    "AZURE_OPENAI_API_ENDPOINT": "endpoint",
    "AZURE_OPENAI_API_VERSION": "API version",
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": "deployment name"
}
for var, desc in required_vars.items():
    if not os.getenv(var):
        raise ValueError(f"Missing {desc} in environment variables. Check your .env file.")

# Section 2: Initialize the Azure OpenAI Model with LangChain
try:
    llm = AzureChatOpenAI(
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        temperature=0.7,
        max_tokens=500
    )
except Exception as e:
    raise RuntimeError(f"Failed to initialize AzureChatOpenAI: {str(e)}")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant providing concise and accurate answers. Maintain context from the conversation history."),
    ("human", "{question}")
])


# Section 4: Create a Chain
chain = prompt_template | llm

# Section 5: Define the REST API Endpoint
@app.route('/register', methods=['POST'])
def register():
    """
    REST API endpoint to register a new user.
    Expects a JSON payload with 'username' and 'password'.
    Returns a confirmation message.
    """
    global users

    try:
        data = request.get_json()
        if not data or 'username' not in data or 'password' not in data:
            return jsonify({"error": "Missing username or password in request body"}), 400

        username = data['username']
        password = data['password']

        if username in users:
            return jsonify({"error": "Username already exists"}), 400

        users[username] = {
            'password_hash': generate_password_hash(password),
            'sessions': []
        }

        save_users(users)

        return jsonify({
            "message": f"User {username} registered successfully",
            "status": "success"
        }), 201

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {type(e).__name__}: {str(e)}"}), 500

@app.route('/login', methods=['POST'])
def login():
    """
    REST API endpoint to log in a user.
    Expects a JSON payload with 'username' and 'password'.
    Returns a confirmation message and session ID.
    """
    global users

    try:
        data = request.get_json()
        if not data or 'username' not in data or 'password' not in data:
            return jsonify({"error": "Missing username or password in request body"}), 400

        username = data['username']
        password = data['password']

        if username not in users:
            return jsonify({"error": "Invalid username or password"}), 401

        if not check_password_hash(users[username]['password_hash'], password):
            return jsonify({"error": "Invalid username or password"}), 401

        # Generate a new session ID for this user
        new_session_id = str(uuid.uuid4())
        users[username]['sessions'].append(new_session_id)

        # Store username in session
        session['username'] = username
        session['session_id'] = new_session_id

        save_users(users)

        return jsonify({
            "message": f"User {username} logged in successfully",
            "status": "success",
            "session_id": new_session_id
        }), 200

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {type(e).__name__}: {str(e)}"}), 500

@app.route('/logout', methods=['POST'])
@login_required
def logout():
    """
    REST API endpoint to log out a user.
    Expects a JSON payload with 'session_id'.
    Returns a confirmation message.
    """
    global users

    try:
        data = request.get_json() or {}
        session_id = data.get('session_id', session.get('session_id'))
        username = session.get('username')

        if username in users and session_id in users[username]['sessions']:
            users[username]['sessions'].remove(session_id)
            save_users(users)

        session.pop('username', None)
        session.pop('session_id', None)

        return jsonify({
            "message": "Logged out successfully",
            "status": "success"
        }), 200

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {type(e).__name__}: {str(e)}"}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """
    REST API endpoint to ask a question to Azure OpenAI.
    Expects a JSON payload with 'question' field, optional 'session_id', and optional 'persona'.
    Returns the model's response as JSON along with the chat history for that session.
    """
    global chat_histories, users

    try:
        # Get JSON data from the request
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "Missing 'question' in request body"}), 400

        # Extract question, session_id, and persona from the request
        question = data['question']
        session_id = data.get('session_id', session.get('session_id', 'default_session'))
        persona = data.get('persona', 'default')

        # Check if this is an authenticated session
        is_authenticated = False
        session_username = None

        # Check if the session is in the flask session
        if 'username' in session and session.get('session_id') == session_id:
            is_authenticated = True
            session_username = session.get('username')
        else:
            # Check if the session is associated with any user
            for username, user_data in users.items():
                if session_id in user_data['sessions']:
                    is_authenticated = True
                    session_username = username
                    break

        print(f"Question received from {'authenticated ' + session_username if is_authenticated else 'unauthenticated'} session {session_id}: {question}")

        # Initialize session history if it doesn't exist
        if session_id not in chat_histories:
            chat_histories[session_id] = []

        # Get the chat history for this session
        session_history = chat_histories[session_id]

        # Create context from chat history
        context = ""
        if session_history:
            context = "Previous conversation:\n"
            for entry in session_history:
                context += f"Human: {entry['question']}\nAI: {entry['answer']}\n"

        # Prepare the question with context if there's history
        contextualized_question = question
        if context:
            contextualized_question = f"{context}\nHuman: {question}"

# Classify the question type for few-shot examples
        question_type = classify_question_type(question)

        # Create a dynamic prompt template based on persona and question type
        prompt_template = create_prompt_template(persona, question_type)

        # Create a new chain with the dynamic prompt template
        dynamic_chain = prompt_template | llm

        # Invoke the chain with the user's question
        response = dynamic_chain.invoke({
            "question": contextualized_question
        })
        print(f"Response for session {session_id}: {response.content}")

                # Update chat history for this session
        chat_histories[session_id].append({
            "question": question,
            "answer": response.content,
            "timestamp": str(datetime.now()),
            "user": session_username if is_authenticated else "anonymous",
            "persona": persona
        })

        # Limit chat history to MAX_HISTORY_LENGTH entries
        if len(chat_histories[session_id]) > MAX_HISTORY_LENGTH:
            chat_histories[session_id] = chat_histories[session_id][-MAX_HISTORY_LENGTH:]

        # Save chat histories
        save_chat_histories()

        # Return the response with chat history for this session
        return jsonify({
            "answer": response.content,
            "status": "success",
            "session_id": session_id,
            "authenticated": is_authenticated,
            "username": session_username if is_authenticated else None,
            "persona": persona,
            "question_type": question_type,
            "history": chat_histories[session_id]
        }), 200

    except KeyError as e:
        return jsonify({"error": f"KeyError: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {type(e).__name__}: {str(e)}"}), 500

@app.route('/personas', methods=['GET'])
def get_personas():
    """
    REST API endpoint to retrieve all available personas.
    Returns a list of all personas as JSON.
    """
    return jsonify({
        "personas": list(PERSONAS.keys()),
        "descriptions": PERSONAS
    }), 200

# Add new endpoints for chat history management
@app.route('/history', methods=['GET'])
def get_history():
    """
    REST API endpoint to retrieve the chat history for a specific session.
    Expects a query parameter 'session_id'.
    Returns the chat history for that session as JSON.
    """
    session_id = request.args.get('session_id', 'default_session')

    if session_id not in chat_histories:
        return jsonify({
            "history": [],
            "count": 0,
            "session_id": session_id
        }), 200

    return jsonify({
        "history": chat_histories[session_id],
        "count": len(chat_histories[session_id]),
        "session_id": session_id
    }), 200

@app.route('/sessions', methods=['GET'])
def get_sessions():
    """
    REST API endpoint to retrieve all active session IDs.
    Returns a list of all session IDs as JSON.
    """
    return jsonify({
        "sessions": list(chat_histories.keys()),
        "count": len(chat_histories)
    }), 200

@app.route('/clear-history', methods=['POST'])
def clear_history():
    """
    REST API endpoint to clear the chat history for a specific session.
    Expects a JSON payload with 'session_id'.
    Returns a confirmation message.
    """
    data = request.get_json() or {}
    session_id = data.get('session_id', 'default_session')

    if session_id in chat_histories:
        chat_histories[session_id] = []
        message = f"Chat history for session {session_id} cleared successfully"
    else:
        message = f"No history found for session {session_id}"

    # Save chat history
    save_chat_histories()

    return jsonify({
        "message": message,
        "status": "success",
        "session_id": session_id
    }), 200

@app.route('/clear-all-history', methods=['POST'])
def clear_all_history():
    """
    REST API endpoint to clear all chat histories for all sessions.
    Returns a confirmation message.
    """
    global chat_histories
    session_count = len(chat_histories)
    chat_histories = {}

    # Save chat history
    save_chat_histories()

    return jsonify({
        "message": f"Chat history cleared for all {session_count} sessions",
        "status": "success"
    }), 200

@app.route('/save-histories', methods=['POST'])
def save_histories_endpoint():
    """
    REST API endpoint to manually save all chat histories.
    Returns a confirmation message.
    """
    success = save_chat_histories()
    if success:
        return jsonify({
            "message": "Chat histories saved successfully",
            "status": "success"
        }), 200
    else:
        return jsonify({
            "message": "Failed to save chat histories",
            "status": "error"
        }), 500


@app.route('/generate-session', methods=['GET'])
def generate_session():
    """
    REST API endpoint to generate a new unique session ID.
    Returns the generated session ID as JSON.
    """
    new_session_id = str(uuid.uuid4())
    return jsonify({
        "session_id": new_session_id,
        "status": "success"
    }), 200

# Section 7: Run the Flask Application
if __name__ == '__main__':
    # Start the Flask server on port 3000
    app.run(host='0.0.0.0', port=3000, debug=True)