# Copyright (C) 2025 AIDC-AI
# Licensed under the MIT License.

import json
import os
import asyncio
import time
from typing import Optional, Dict, Any, TypedDict, List, Union
import logging

import server
from aiohttp import web
import aiohttp
import base64
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory dictionary to store session messages
session_messages = {}

# Add at the beginning of the file
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "public")

# Define types using TypedDict
class Node(TypedDict):
    name: str  # Node name
    description: str  # Node description
    image: str  # Node image URL, can be empty
    github_url: str  # Node GitHub address
    from_index: int  # Node position in the list
    to_index: int  # Node position in the list

class NodeInfo(TypedDict):
    existing_nodes: List[Node]  # Installed nodes
    missing_nodes: List[Node]  # Missing nodes

class Workflow(TypedDict, total=False):
    id: Optional[int]  # Workflow ID
    name: Optional[str]  # Workflow name
    description: Optional[str]  # Workflow description
    image: Optional[str]  # Workflow image
    workflow: Optional[str]  # Workflow JSON

class ExtItem(TypedDict):
    type: str  # Extension type
    data: Union[dict, list]  # Extension data

class ChatResponse(TypedDict):
    session_id: str  # Session ID
    text: Optional[str]  # Response text
    finished: bool  # Whether the conversation is finished
    type: str  # Response type
    format: str  # Response format
    ext: Optional[List[ExtItem]]  # Extension information

def get_workflow_templates():
    templates = []
    workflows_dir = os.path.join(STATIC_DIR, "workflows")
    
    for filename in os.listdir(workflows_dir):
        if filename.endswith('.json'):
            with open(os.path.join(workflows_dir, filename), 'r') as f:
                template = json.load(f)
                templates.append(template)
    
    return templates

@server.PromptServer.instance.routes.get("/workspace/fetch_messages_by_id")
async def fetch_messages(request):
    session_id = request.query.get('session_id')
    data = await asyncio.to_thread(fetch_messages_sync, session_id)
    return web.json_response(data)

def fetch_messages_sync(session_id):
    logger.info(f"fetch_messages: {session_id}")
    return session_messages.get(session_id, [])

@server.PromptServer.instance.routes.post("/workspace/workflow_gen")
async def workflow_gen(request):
    logger.info("Received workflow_gen request")
    req_json = await request.json()
    logger.info(f"Request JSON: {req_json}")
    
    response = web.StreamResponse(
        status=200,
        reason='OK',
        headers={
            'Content-Type': 'application/json',
            'X-Content-Type-Options': 'nosniff'
        }
    )
    await response.prepare(request)
    
    session_id = req_json.get('session_id')
    user_message = req_json.get('message')
    
    # Create user message
    user_msg = {
        "id": str(len(session_messages.get(session_id, []))),
        "content": user_message,
        "role": "user"
    }
    
    if "workflow" in user_message.lower():
        workflow = {
            "name": "basic_image_gen",
            "description": "Create a basic image generation workflow",
            "image": "https://placehold.co/600x400",
            "workflow": "{ ... }"  # Your workflow JSON here
        }
        
        chat_response = ChatResponse(
            session_id=session_id,
            text="",
            finished=False,
            type="workflow_option",
            format="text",
            ext=[{"type": "workflows", "data": [workflow]}]
        )
        
        await response.write(json.dumps(chat_response).encode() + b"\n")
        
        message = "Let me help you choose a workflow. Here are some options available:"
        accumulated = ""
        for char in message:
            accumulated += char
            chat_response["text"] = accumulated
            await response.write(json.dumps(chat_response).encode() + b"\n")
            await asyncio.sleep(0.01)
        
        chat_response["finished"] = True
        chat_response["text"] = message
        await response.write(json.dumps(chat_response).encode() + b"\n")
        
    elif "recommend" in user_message.lower():
        existing_nodes = [
            {
                "name": "LoraLoader",
                "description": "Load LoRA weights for conditioning.",
                "image": "",
                "github_url": "https://github.com/CompVis/taming-transformers",
                "from_index": 0,
                "to_index": 0
            },
            {
                "name": "KSampler",
                "description": "Generate images using K-diffusion sampling.",
                "image": "",
                "github_url": "https://github.com/CompVis/taming-transformers",
                "from_index": 0,
                "to_index": 0
            }
        ]
        
        missing_nodes = [
            {
                "name": "CLIPTextEncode",
                "description": "Encode text prompts for conditioning.",
                "image": "",
                "github_url": "https://github.com/CompVis/clip-interrogator",
                "from_index": 0,
                "to_index": 0
            }
        ]
        
        node_info = {
            "existing_nodes": existing_nodes,
            "missing_nodes": missing_nodes
        }
        
        chat_response = ChatResponse(
            session_id=session_id,
            text="",
            finished=False,
            type="downstream_node_recommend",
            format="text",
            ext=[{"type": "node_info", "data": node_info}]
        )
        
        await response.write(json.dumps(chat_response).encode() + b"\n")
        
        message = "Here are some recommended nodes:"
        accumulated = ""
        for char in message:
            accumulated += char
            chat_response["text"] = accumulated
            await response.write(json.dumps(chat_response).encode() + b"\n")
            await asyncio.sleep(0.01)
        
        chat_response["finished"] = True
        chat_response["text"] = message
        await response.write(json.dumps(chat_response).encode() + b"\n")
        
    else:
        chat_response = ChatResponse(
            session_id=session_id,
            text="",
            finished=False,
            type="message",
            format="text",
            ext=[{"type": "guides", "data": ["Create a workflow", "Search for nodes", "Get node recommendations"]}]
        )
        
        await response.write(json.dumps(chat_response).encode() + b"\n")
        
        message = "I can help you with workflows, nodes, and more. Try asking about:"
        accumulated = ""
        for char in message:
            accumulated += char
            chat_response["text"] = accumulated
            await response.write(json.dumps(chat_response).encode() + b"\n")
            await asyncio.sleep(0.01)
        
        chat_response["finished"] = True
        chat_response["text"] = message
        await response.write(json.dumps(chat_response).encode() + b"\n")
    
    if session_id not in session_messages:
        session_messages[session_id] = []
    
    session_messages[session_id].extend([user_msg])
    
    await response.write_eof()
    return response

async def upload_to_oss(file_data: bytes, filename: str) -> str:
    # Implement your OSS upload logic here
    # Return the URL of the uploaded file
    pass

@server.PromptServer.instance.routes.post("/api/chat/invoke")
async def invoke_chat(request):
    data = await request.json()
    session_id = data['session_id']
    prompt = data['prompt']
    images = data.get('images', [])

    # Get LLM provider configuration from headers
    ollama_base_url = request.headers.get('Ollama-Base-Url')
    ollama_model = request.headers.get('Ollama-Model')
    encrypted_openai_api_key = request.headers.get('Encrypted-Openai-Api-Key')
    openai_base_url = request.headers.get('Openai-Base-Url')

    # Prepare the response stream
    response_stream = web.StreamResponse(
        status=200,
        reason='OK',
        headers={
            'Content-Type': 'application/json',
            'X-Content-Type-Options': 'nosniff'
        }
    )
    await response_stream.prepare(request)

    try:
        if ollama_base_url and ollama_model:
            logger.info(f"Using Ollama: {ollama_base_url}, Model: {ollama_model}")
            await ollama_chat_stream(
                session_id,
                prompt,
                ollama_base_url,
                ollama_model,
                response_stream
            )
        elif encrypted_openai_api_key and openai_base_url:
            logger.info(f"Using OpenAI: {openai_base_url}")
            # Decrypt API key (assuming a private key is available on the server)
            # For this example, we'll use a placeholder for decryption
            # In a real application, you'd use your private key to decrypt
            decrypted_api_key = decrypt_api_key(encrypted_openai_api_key)
            await openai_chat_stream(
                session_id,
                prompt,
                decrypted_api_key,
                openai_base_url,
                response_stream
            )
        else:
            # Fallback to existing hardcoded logic or a default LLM
            logger.info("No specific LLM provider configured, using default logic.")
            await handle_default_chat_logic(
                session_id,
                prompt,
                response_stream
            )

    except Exception as e:
        logger.error(f"Error in invoke_chat: {e}")
        error_response = ChatResponse(
            session_id=session_id,
            text=f"An error occurred: {str(e)}",
            finished=True,
            type="error",
            format="text"
        )
        await response_stream.write(json.dumps(error_response).encode() + b"\n")
    finally:
        await response_stream.write_eof()
    
    return response_stream

async def ollama_chat_stream(
    session_id: str,
    prompt: str,
    base_url: str,
    model: str,
    response_stream: web.StreamResponse
):
    try:
        async with aiohttp.ClientSession() as session:
            ollama_api_url = f"{base_url}/api/chat"
            messages = [{"role": "user", "content": prompt}] # Simplified for example

            async with session.post(ollama_api_url, json={"model": model, "messages": messages, "stream": True}) as resp:
                async for chunk in resp.content.iter_any():
                    try:
                        # Ollama sends newline-delimited JSON objects
                        decoded_chunk = chunk.decode('utf-8')
                        for line in decoded_chunk.splitlines():
                            if line.strip():
                                ollama_response = json.loads(line)
                                if "content" in ollama_response["message"]:
                                    text_content = ollama_response["message"]["content"]
                                    chat_response = ChatResponse(
                                        session_id=session_id,
                                        text=text_content,
                                        finished=False,
                                        type="message",
                                        format="text"
                                    )
                                    await response_stream.write(json.dumps(chat_response).encode() + b"\n")
                                    await response_stream.flush()
                                if ollama_response.get("done"):
                                    chat_response = ChatResponse(
                                        session_id=session_id,
                                        text="",
                                        finished=True,
                                        type="message",
                                        format="text"
                                    )
                                    await response_stream.write(json.dumps(chat_response).encode() + b"\n")
                                    await response_stream.flush()
                                    break # Exit loop if done
                    except json.JSONDecodeError:
                        logger.warning(f"Could not decode JSON from Ollama chunk: {line}")
                    except Exception as e:
                        logger.error(f"Error processing Ollama stream chunk: {e}")
                        raise

    except aiohttp.ClientError as e:
        logger.error(f"Ollama connection error: {e}")
        raise Exception(f"Could not connect to Ollama server: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in ollama_chat_stream: {e}")
        raise

async def openai_chat_stream(
    session_id: str,
    prompt: str,
    api_key: str,
    base_url: str,
    response_stream: web.StreamResponse
):
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        messages = [{"role": "user", "content": prompt}] # Simplified for example

        stream = client.chat.completions.create(
            model="gpt-3.5-turbo", # Or dynamically select model
            messages=messages,
            stream=True,
        )
        
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                chat_response = ChatResponse(
                    session_id=session_id,
                    text=content,
                    finished=False,
                    type="message",
                    format="text"
                )
                await response_stream.write(json.dumps(chat_response).encode() + b"\n")
                await response_stream.flush()
        
        chat_response = ChatResponse(
            session_id=session_id,
            text="",
            finished=True,
            type="message",
            format="text"
        )
        await response_stream.write(json.dumps(chat_response).encode() + b"\n")
        await response_stream.flush()

    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise Exception(f"OpenAI API error: {e}")

async def handle_default_chat_logic(
    session_id: str,
    user_message: str,
    response_stream: web.StreamResponse
):
    # This is the existing hardcoded logic from workflow_gen, adapted for invoke_chat
    # In a real scenario, you might integrate a default LLM here or return a helpful message
    chat_response = ChatResponse(
        session_id=session_id,
        text="",
        finished=False,
        type="message",
        format="text",
        ext=[{"type": "guides", "data": ["Create a workflow", "Search for nodes", "Get node recommendations"]}]
    )
    
    await response_stream.write(json.dumps(chat_response).encode() + b"\n")
    
    message = "I can help you with workflows, nodes, and more. Try asking about:"
    accumulated = ""
    for char in message:
        accumulated += char
        chat_response["text"] = accumulated
        await response_stream.write(json.dumps(chat_response).encode() + b"\n")
        await asyncio.sleep(0.01)
    
    chat_response["finished"] = True
    chat_response["text"] = message
    await response_stream.write(json.dumps(chat_response).encode() + b"\n")

# Placeholder for decryption - IMPORTANT: Implement secure decryption in production
def decrypt_api_key(encrypted_key: str) -> str:
    # In a real application, you would use your RSA private key to decrypt the API key.
    # This is a placeholder and should NOT be used in production as is.
    logger.warning("Using placeholder decryption for API key. Implement proper RSA decryption in production!")
    
    # Example: Load your private key (replace with your actual key loading)
    # with open("private_key.pem", "rb") as key_file:
    #     private_key = serialization.load_pem_private_key(
    #         key_file.read(),
    #         password=None, # Replace with your key password if applicable
    #         backend=default_backend()
    #     )
    # decrypted_bytes = private_key.decrypt(
    #     base64.b64decode(encrypted_key),
    #     padding.OAEP(
    #         mgf=padding.MGF1(algorithm=hashes.SHA256()),
    #         algorithm=hashes.SHA256(),
    #         label=None
    #     )
    # )
    # return decrypted_bytes.decode('utf-8')
    
    # For demonstration, returning a dummy key or the encrypted key itself (unsafe)
    # You MUST replace this with actual decryption logic.
    return "sk-YOUR_DECRYPTED_OPENAI_API_KEY" # Replace with actual decrypted key

@server.PromptServer.instance.routes.post("/api/chat/verify_openai_key")
async def verify_openai_key(request):
    encrypted_api_key = request.headers.get('Encrypted-Openai-Api-Key')
    openai_base_url = request.headers.get('Openai-Base-Url', 'https://api.openai.com/v1')

    if not encrypted_api_key:
        return web.json_response({"success": False, "message": "Encrypted API key is missing."}, status=400)

    try:
        decrypted_api_key = decrypt_api_key(encrypted_api_key)
        
        # Attempt to list models to verify the key
        client = OpenAI(api_key=decrypted_api_key, base_url=openai_base_url)
        await client.models.list()
        
        return web.json_response({"success": True, "message": "API key is valid."})
    except Exception as e:
        logger.error(f"OpenAI API key verification failed: {e}")
        return web.json_response({"success": False, "message": f"API key verification failed: {str(e)}"}, status=401)