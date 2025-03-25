# Import necessary libraries
from flask import Flask, request, jsonify
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import datetime
import uuid


# Load environment variables from .env file
load_dotenv()

# Initialize Flask application
app = Flask(__name__)

# Add this: Initialize chat history storage with session support
chat_histories = {}
MAX_HISTORY_LENGTH = 10

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
@app.route('/ask', methods=['POST'])
def ask_question():
    """
    REST API endpoint to ask a question to GPT-4o.
    Expects a JSON payload with 'question' field and optional 'session_id'.
    Returns the model's response as JSON along with the chat history for that session.
    """
    global chat_histories

    try:
        # Get JSON data from the request
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "Missing 'question' in request body"}), 400

        # Extract question and session_id from the request
        question = data['question']
        session_id = data.get('session_id', 'default_session')
        print(f"Question received from session {session_id}: {question}")

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

        # Invoke the chain with the user's question
        response = chain.invoke({
            "question": contextualized_question
        })
        print(f"Response for session {session_id}: {response.content}")

        # Update chat history for this session
        chat_histories[session_id].append({
            "question": question,
            "answer": response.content,
            "timestamp": str(datetime.datetime.now())
        })

        # Limit chat history to MAX_HISTORY_LENGTH entries
        if len(chat_histories[session_id]) > MAX_HISTORY_LENGTH:
            chat_histories[session_id] = chat_histories[session_id][-MAX_HISTORY_LENGTH:]

        # Return the response with chat history for this session
        return jsonify({
            "answer": response.content,
            "status": "success",
            "session_id": session_id,
            "history": chat_histories[session_id]
        }), 200

    except KeyError as e:
        return jsonify({"error": f"KeyError: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {type(e).__name__}: {str(e)}"}), 500

# Add this new endpoint after the ask_question function
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

# Add this new endpoint after the get_history function
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
    return jsonify({
        "message": f"Chat history cleared for all {session_count} sessions",
        "status": "success"
    }), 200

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

# Section 7: Run the Flask Application
if __name__ == '__main__':
    # Start the Flask server on port 3000
    app.run(host='0.0.0.0', port=3000, debug=True)