# ChatBot API Tutorial - Part 2

## Current Application Overview

Our ChatBot API is a Flask-based REST API that integrates with Azure OpenAI to provide AI-powered responses to user questions. Currently, the application offers a single endpoint (`/ask`) that accepts a question from the user, sends it to Azure OpenAI's language model via LangChain, and returns the AI-generated response. Each request is independent, with no memory of previous interactions.

## Section 1: Adding Chat History Functionality

In this section, we'll modify our application to maintain a chat history of the 10 most recent interactions. This will allow our chatbot to have context-aware conversations by remembering previous questions and answers.

### Step 1: Modify the app.py file to store chat history

First, we need to add a data structure to store our chat history. Open your `app.py` file and add the following code after the imports section:

```python
# Import necessary libraries
from flask import Flask, request, jsonify
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize Flask application
app = Flask(__name__)

# Add this: Initialize chat history storage
chat_history = []
MAX_HISTORY_LENGTH = 10
```

### Step 2: Update the prompt template to include chat history

Now, we need to modify our prompt template to include the chat history. Replace the existing prompt template with the following:

```python
# Replace the existing prompt template with this:
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant providing concise and accurate answers. Maintain context from the conversation history."),
    ("human", "{question}")
])
```

### Step 3: Update the ask_question function to use and update chat history

Next, we need to modify our `/ask` endpoint to use and update the chat history. Replace the existing `ask_question` function with this updated version:

```python
@app.route('/ask', methods=['POST'])
def ask_question():
    """
    REST API endpoint to ask a question to GPT-4o.
    Expects a JSON payload with 'question' field.
    Returns the model's response as JSON along with the chat history.
    """
    global chat_history

    try:
        # Get JSON data from the request
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "Missing 'question' in request body"}), 400

        # Extract question from the request
        question = data['question']
        print(f"Question received: {question}")

        # Create context from chat history
        context = ""
        if chat_history:
            context = "Previous conversation:\n"
            for entry in chat_history:
                context += f"Human: {entry['question']}\nAI: {entry['answer']}\n"

        # Prepare the question with context if there's history
        contextualized_question = question
        if context:
            contextualized_question = f"{context}\nHuman: {question}"

        # Invoke the chain with the user's question
        response = chain.invoke({
            "question": contextualized_question
        })
        print(f"Response: {response.content}")

        # Update chat history
        chat_history.append({
            "question": question,
            "answer": response.content
        })

        # Limit chat history to MAX_HISTORY_LENGTH entries
        if len(chat_history) > MAX_HISTORY_LENGTH:
            chat_history = chat_history[-MAX_HISTORY_LENGTH:]

        # Return the response with chat history
        return jsonify({
            "answer": response.content,
            "status": "success",
            "history": chat_history
        }), 200

    except KeyError as e:
        return jsonify({"error": f"KeyError: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {type(e).__name__}: {str(e)}"}), 500
```

### Step 4: Add a new endpoint to view the chat history

Let's add a new endpoint to view the current chat history:

```python
# Add this new endpoint after the ask_question function
@app.route('/history', methods=['GET'])
def get_history():
    """
    REST API endpoint to retrieve the current chat history.
    Returns the current chat history as JSON.
    """
    return jsonify({
        "history": chat_history,
        "count": len(chat_history)
    }), 200
```

### Step 5: Add an endpoint to clear the chat history

Finally, let's add an endpoint to clear the chat history when needed:

```python
# Add this new endpoint after the get_history function
@app.route('/clear-history', methods=['POST'])
def clear_history():
    """
    REST API endpoint to clear the chat history.
    Returns a confirmation message.
    """
    global chat_history
    chat_history = []
    return jsonify({
        "message": "Chat history cleared successfully",
        "status": "success"
    }), 200
```

## Testing the Chat History Feature

Now that we've implemented chat history functionality, let's test it using curl commands. Make sure your Flask application is running:

```bash
python app.py
```

### Test 1: Ask a question

```bash
curl -X POST http://localhost:3000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the capital of France?"}'
```

### Test 2: Ask a follow-up question that references the previous question

```bash
curl -X POST http://localhost:3000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is its population?"}'
```

### Test 3: View the chat history

```bash
curl -X GET http://localhost:3000/history
```

### Test 4: Ask more questions to fill up the history

```bash
curl -X POST http://localhost:3000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What are some famous landmarks there?"}'
```

```bash
curl -X POST http://localhost:3000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Tell me about the cuisine in this city."}'
```

### Test 5: Clear the chat history

```bash
curl -X POST http://localhost:3000/clear-history
```

### Test 6: Verify the history is cleared

```bash
curl -X GET http://localhost:3000/history
```

## Section 2: Adding Multi-User Support with Session IDs

Now that we have chat history functionality, let's enhance our application to support multiple users simultaneously. We'll implement a session-based system where each user has their own unique session ID and chat history.

### Step 1: Update imports and data structure

First, we need to add the necessary imports and modify our data structure to store chat histories for multiple sessions. Update the imports and chat history storage in your `app.py` file:

```python
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
```

### Step 2: Update the ask_question function to support session IDs

Next, we need to modify our `/ask` endpoint to accept a session ID and maintain separate chat histories for each session. Replace the existing `ask_question` function with this updated version:

```python
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
```

### Step 3: Update the history endpoint to support session IDs

Now, let's modify the `/history` endpoint to retrieve the chat history for a specific session:

```python
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
```

### Step 4: Add an endpoint to list all active sessions

Let's add a new endpoint to list all active session IDs:

```python
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
```

### Step 5: Update the clear-history endpoint to support session IDs

Next, let's modify the `/clear-history` endpoint to clear the chat history for a specific session:

```python
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
```

### Step 6: Add an endpoint to clear all chat histories

Let's add a new endpoint to clear all chat histories for all sessions:

```python
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
```

### Step 7: Add an endpoint to generate a new session ID

Finally, let's add an endpoint to generate a new unique session ID:

```python
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
```

## Testing the Multi-User Session Feature

Now that we've implemented multi-user support with session IDs, let's test it using curl commands. Make sure your Flask application is running:

```bash
python app.py
```

### Test 1: Generate new session IDs for two different users

```bash
# Generate a session ID for User 1
curl -X GET http://localhost:3000/generate-session
```

Save the returned session ID as USER1_SESSION.

```bash
# Generate a session ID for User 2
curl -X GET http://localhost:3000/generate-session
```

Save the returned session ID as USER2_SESSION.

### Test 2: Ask questions with different session IDs

```bash
# User 1 asks a question
curl -X POST http://localhost:3000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the capital of France?", "session_id":"'$USER1_SESSION'"}'
```

```bash
# User 2 asks a different question
curl -X POST http://localhost:3000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the largest planet in our solar system?", "session_id":"'$USER2_SESSION'"}'
```

### Test 3: Ask follow-up questions with different session IDs

```bash
# User 1 asks a follow-up question
curl -X POST http://localhost:3000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is its population?", "session_id":"'$USER1_SESSION'"}'
```

```bash
# User 2 asks a follow-up question
curl -X POST http://localhost:3000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"How many moons does it have?", "session_id":"'$USER2_SESSION'"}'
```

### Test 4: View chat histories for different sessions

```bash
# View User 1's chat history
curl -X GET "http://localhost:3000/history?session_id=$USER1_SESSION"
```

```bash
# View User 2's chat history
curl -X GET "http://localhost:3000/history?session_id=$USER2_SESSION"
```

### Test 5: List all active sessions

```bash
curl -X GET http://localhost:3000/sessions
```

### Test 6: Clear chat history for a specific session

```bash
# Clear User 1's chat history
curl -X POST http://localhost:3000/clear-history \
  -H "Content-Type: application/json" \
  -d '{"session_id":"'$USER1_SESSION'"}'
```

### Test 7: Clear all chat histories

```bash
curl -X POST http://localhost:3000/clear-all-history
```

## What's Next?

In the next section, we'll explore how to enhance our chatbot with additional features such as conversation persistence between server restarts, user identification, and more advanced prompt engineering techniques.
