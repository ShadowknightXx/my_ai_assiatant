from flask import Flask, request, jsonify, send_from_directory
from groq import Groq
import os
import requests

app = Flask(__name__)

# Initialize Groq client safely
api_key = os.environ.get('GROQ_API_KEY')
if not api_key:
    print("WARNING: GROQ_API_KEY environment variable is not set!")
client = Groq(api_key=api_key)

conversation_history = []

def search_web(query):
    try:
        url = f"https://api.duckduckgo.com/?q={requests.utils.quote(query)}&format=json&no_html=1"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, timeout=5, headers=headers)
        data = res.json()
        result = data.get('AbstractText', '')
        if not result:
            topics = data.get('RelatedTopics', [])
            if topics and isinstance(topics[0], dict):
                result = topics[0].get('Text', '')
        return result or "No web results found."
    except Exception as e:
        return f"Web search failed: {str(e)}"

@app.route('/')
def home():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        web_context = search_web(user_message)

        messages = [
            {
                'role': 'system',
                'content': f'''You are a highly capable AI assistant that can answer anything.
You have access to the following web search result to help answer the user's question:

WEB RESULT: {web_context}

Use this information if relevant, otherwise use your own knowledge.
Always give detailed, helpful answers.'''
            },
            *conversation_history,
            {'role': 'user', 'content': user_message}
        ]

        response = client.chat.completions.create(
            model='mixtral-8x7b-32768',
            messages=messages,
            max_tokens=2048
        )

        ai_reply = response.choices[0].message.content
        conversation_history.append({'role': 'user', 'content': user_message})
        conversation_history.append({'role': 'assistant', 'content': ai_reply})
        return jsonify({'response': ai_reply})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file'}), 400
        file = request.files['file']
        text = file.read().decode('utf-8', errors='ignore')
        conversation_history.append({
            'role': 'user',
            'content': f'I am uploading a document for you to read and answer questions about:\n\n{text[:4000]}'
        })
        response = client.chat.completions.create(
            model='mixtral-8x7b-32768',
            messages=[
                {'role': 'system', 'content': 'You are a helpful AI. Acknowledge the document and summarize it briefly.'},
                *conversation_history
            ]
        )
        ai_reply = response.choices[0].message.content
        conversation_history.append({'role': 'assistant', 'content': ai_reply})
        return jsonify({'response': ai_reply})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    conversation_history.clear()
    return jsonify({'status': 'cleared'})

if __name__ == '__main__':
    app.run(debug=True)
