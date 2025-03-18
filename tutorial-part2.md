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

## What's Next?

In the next section, we'll explore how to enhance our chatbot with additional features such as conversation persistence between server restarts, user identification, and more advanced prompt engineering techniques.
