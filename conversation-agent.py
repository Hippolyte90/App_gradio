#pip install Flask==3.0.3 gunicorn==23.0.0 Werkzeug==3.0.3 python-dotenv numpy pandas scikit-learn matplotlib gensim openai
#pip install gradio tiktoken faiss-cpu datasets sentencepiece google-generativeai unstructured plotly jupyter-dash pydub
#pip install accelerate sentence_transformers feedparser speedtest-cli


import os
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai
import gradio as gr
from flask import Flask

# Charger les variables d'environnement
load_dotenv(override=True)

# Initialiser Flask
app = Flask(__name__)

@app.route("/")
def home():
    return "Conversation Agent is running!"

# Récupérer les clés API
openai_api_key = os.getenv('OPENAI_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

if groq_api_key:
    print(f"GROQ API Key exists and begins {groq_api_key[:8]}")
else:
    print("GROQ API Key not set")

if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:8]}")
else:
    print("Google API Key not set")

# Initialiser les clients API
openai = OpenAI()
google.generativeai.configure(api_key=google_api_key)

system_message = "You are a helpful assistant"

# We will use GPT-4o-mini for wrapping 

def message_gpt(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
      ]
    completion = openai.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
    )
    return completion.choices[0].message.content

# Define this variable and then pass js=force_dark_mode when creating the Interface

force_dark_mode = """
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

##  GPT model
def stream_gpt(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
      ]
    stream = openai.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
        stream=True
    )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result
        
# Gemini model
def stream_gemini(prompt):
    gemini = google.generativeai.GenerativeModel(
        model_name='gemini-2.0-flash',
        system_instruction=system_message
    )
    result = gemini.generate_content(prompt, stream=True)
    response = ""
    for chuck in result:
        text = chuck.text
        response += text or ""
        yield response

##  Groq model
def stream_groq(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
      ]
    groq = OpenAI(api_key = groq_api_key, base_url="https://api.groq.com/openai/v1")
    stream = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        stream=True
    )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result
    
def stream_model(prompt, model):
    if model=="GPT":
        result = stream_gpt(prompt)
    elif model=="Gemini":
        result = stream_gemini(prompt)
    elif model == "Groq":
        result = stream_groq(prompt)
    else:
        raise ValueError("Unknown model")
    yield from result
    
# Interface Gradio
view = gr.Interface(
    fn=stream_model,
    inputs=[gr.Textbox(label="Your message:", lines = 8), gr.Dropdown(["GPT", "Gemini", "Groq"], label="Select model", value="GPT")],
    outputs=[gr.Markdown(label="Response:")],
    flagging_mode="never",
    js=force_dark_mode
)

if __name__ == "__main__":
    # Lancer Gradio
    view.launch()
    # Note: Flask est configuré mais pas lancé pour éviter les conflits
    # Si vous voulez utiliser Flask à la place, commentez la ligne view.launch() et décommentez la ligne suivante:
    # app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))