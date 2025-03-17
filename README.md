# ChatBot API

A Flask-based REST API that integrates with Azure OpenAI to provide AI-powered responses to user questions.

## Overview

This project creates a simple REST API endpoint that allows users to ask questions and receive responses from Azure OpenAI's language models using LangChain.

## Features

- REST API endpoint for submitting questions
- Integration with Azure OpenAI services
- Environment-based configuration
- Error handling and validation

## Prerequisites

- Python 3.11
- Azure OpenAI API access
- Required environment variables (see below)

## Setup Instructions

### 1. Create a Python 3.11 Virtual Environment

```bash
# Create a virtual environment
python3.11 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root with the following variables:

```
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_API_ENDPOINT=your_endpoint
AZURE_OPENAI_API_VERSION=your_api_version
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=your_deployment_name
```

## Running the Application

```bash
python app.py
```

The server will start on port 3000 and can be accessed at http://localhost:3000.

## API Usage

### Ask a Question

**Endpoint:** `POST /ask`

**Request Body:**

```json
{
  "question": "Your question here"
}
```

**Response:**

```json
{
  "answer": "AI-generated response",
  "status": "success"
}
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- 400: Missing or invalid request parameters
- 500: Server-side errors
