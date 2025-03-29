from flask import Flask, render_template, request, jsonify
import os
import logging
import json
import re
import random
import requests
from datetime import datetime
import uuid
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("sharma.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# DeepSeek API setup
DEEPSEEK_API_KEY = "sk-732ea226899242339e5d25944abbafd7"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

# Load sharma-health-model
MODEL_PATH = "sharma-health-model"
try:
    # Check if we're in production (Render) environment
    if os.environ.get('RENDER'):
        logger.info("Running in Render environment, using DeepSeek API instead of local model")
        model, tokenizer = None, None
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("sharma-health-model loaded")
except Exception as e:
    logger.error(f"Error loading sharma-health-model: {e}")
    model, tokenizer = None, None

# Load health_data.json
DATA_PATH = "health_data.json"
try:
    # Create an empty DataFrame if the file doesn't exist (for Render deployment)
    if not os.path.exists(DATA_PATH):
        logger.warning(f"{DATA_PATH} not found, creating empty DataFrame")
        health_data = pd.DataFrame(columns=["input", "output"])
    else:
        health_data = pd.read_json(DATA_PATH)
        logger.info(f"Loaded {len(health_data)} entries from health_data.json")
except Exception as e:
    logger.error(f"Error loading health_data.json: {e}")
    health_data = pd.DataFrame(columns=["input", "output"])

# Preprocess health_data
if not health_data.empty and "input" in health_data.columns and len(health_data["input"]) > 0:
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf_vectorizer.fit_transform(health_data["input"])
    logger.info("TF-IDF matrix initialized")
else:
    # Create empty vectorizer and matrix if no data
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = None
    logger.warning("Empty health_data, TF-IDF matrix not initialized")

# Conversation history
conversation_history = []

# Memory class
class Memory:
    def __init__(self):
        self.memories = {}
        self.memory_file = "health_memories.json"
        self.load_memories()
    
    def load_memories(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                self.memories = json.load(f)
    
    def save_memories(self):
        with open(self.memory_file, 'w') as f:
            json.dump(self.memories, f)
    
    def add_memory(self, key, value, context=None):
        memory_id = str(uuid.uuid4())
        self.memories[memory_id] = {"key": key, "value": value, "context": context}
        self.save_memories()
        return memory_id
    
    def get_memory(self, query):
        return [m for m in self.memories.values() if query.lower() in m["key"].lower() or query.lower() in m["value"].lower()]

# Tool system
class HealthToolSystem:
    def __init__(self):
        self.tools = {
            "obgyn": self.obgyn_info,
            "pregnancy": self.pregnancy_info,
            "neonatal": self.neonatal_care,
            "remember": self.remember_health_info,
            "recall": self.recall_health_info
        }
    
    def obgyn_info(self, topic=""): 
        return f"Got an OB-GYN question about {topic or 'general stuff'}? I can tell you about checkups, fertility, or more—whatcha curious about?"
    
    def pregnancy_info(self, stage="", concern=""): 
        info = stage or concern or "general pregnancy stuff"
        return f"Pregnancy chat time! For '{info}', I can cover stages like the first trimester (weeks 1-12) or concerns like nutrition. What do you want to dive into?"
    
    def neonatal_care(self, topic=""): 
        return f"Newborn stuff is my jam! {topic or 'What’s up with your little one?'} I can talk feeding, jaundice, or anything neonatal—what’s the scoop?"
    
    def remember_health_info(self, info=""): 
        return memory_system.add_memory("health_info", info, "health") if info else "What should I jot down for you?"
    
    def recall_health_info(self, query=""): 
        memories = memory_system.get_memory(query)
        return "\n".join([m["value"] for m in memories]) if memories else "Hmm, nothing in my notes yet!"
    
    def execute_tool(self, tool_name, **kwargs):
        return self.tools.get(tool_name, lambda **x: "That’s not my thing—let’s stick to neonatal, pregnancy, or OB-GYN!")(**kwargs)

# Initialize systems
memory_system = Memory()
health_tool_system = HealthToolSystem()

class HealthContext:
    def __init__(self):
        self.current_context = {"topic": None, "last_query": None}
    
    def update_context(self, message):
        health_topics = ["neonatal", "pregnancy", "obgyn", "baby", "birth", "fetal", "labor", "ectopic"]
        for topic in health_topics:
            if topic in message.lower():
                self.current_context["topic"] = topic
                break
        self.current_context["last_query"] = message
        return self.current_context

health_context = HealthContext()

@app.route('/')
def index():
    logger.info("Index accessed")
    return render_template('index.html', current_time=datetime.now().strftime('%H:%M:%S'))

@app.route('/api/chat', methods=['POST'])
def chat():
    logger.info("Chat API accessed")
    data = request.get_json(silent=True)
    if not data or 'message' not in data:
        logger.warning("Invalid or missing message")
        return jsonify({'error': 'No message provided'}), 400
    
    user_message = data['message'].strip()
    logger.info(f"Processing message: {user_message}")
    
    conversation_history.append({
        'sender': 'user',
        'message': user_message,
        'timestamp': datetime.now().strftime('%H:%M:%S')
    })
    
    current_context = health_context.update_context(user_message)
    logger.info(f"Context: {current_context}")
    
    tool_match = re.search(r"(?:can you|please)?\s*(?:check|track|tell me about)?\s*(?:my)?\s*(\w+)\s*(?:for|about|of)?\s*(.+)?", user_message.lower())
    tool_response = None
    if tool_match:
        tool = tool_match.group(1)
        args = tool_match.group(2) or ""
        logger.info(f"Tool detected: {tool}, args: {args}")
        health_tool_mapping = {"obgyn": "obgyn", "pregnancy": "pregnancy", "neonatal": "neonatal", "remember": "remember", "recall": "recall"}
        if tool in health_tool_mapping:
            try:
                tool_response = health_tool_system.execute_tool(health_tool_mapping[tool], topic=args)
                logger.info(f"Tool response: {tool_response}")
            except Exception as e:
                logger.error(f"Tool error: {e}")
                tool_response = "Oops, my tools got tangled! Let’s try something else—what’s up?"
    
    bot_response = generate_health_response(user_message, tool_response, current_context)
    logger.info(f"Bot response: {bot_response}")
    
    conversation_history.append({
        'sender': 'Sharma',
        'message': bot_response,
        'timestamp': datetime.now().strftime('%H:%M:%S')
    })
    
    return jsonify({
        'response': bot_response,
        'history': conversation_history[-2:]
    })

@app.route('/api/history', methods=['GET'])
def history():
    logger.info("History accessed")
    return jsonify({'history': conversation_history})

def generate_health_response(message, tool_response=None, context=None):
    if tool_response:
        logger.info(f"Returning tool response: {tool_response}")
        return tool_response

    # Greeting
    if re.match(r"^(hi|hello|hey|greetings)(\s+sharma)?$", message.lower()):
        logger.info("Greeting detected")
        return "Hey there! I’m Sharma, your go-to pal for neonatal, pregnancy, and OB-GYN chats. What’s on your mind—something about babies, bumps, or women’s health?\n\nWanna save this chat? Just say 'remember this'!"

    # Step 1: Use sharma-health-model to analyze relevance and intent
    relevant_keywords = ["neonatal", "pregnancy", "obgyn", "baby", "birth", "fetal", "labor", "newborn", "ectopic"]
    has_relevant_keyword = any(keyword in message.lower() for keyword in relevant_keywords)
    
    intent_prompt = f"""Classify the intent of this query as 'relevant' (neonatal, pregnancy, OB-GYN) or 'irrelevant'. Respond with 'relevant' or 'irrelevant'.
Query: {message}"""
    intent = "irrelevant"
    if model and tokenizer:
        try:
            inputs = tokenizer(intent_prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = model.generate(**inputs, max_length=50, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
            intent = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
            logger.info(f"sharma-health-model intent: {intent}")
        except Exception as e:
            logger.error(f"sharma-health-model intent error: {e}")
            intent = "irrelevant" if not has_relevant_keyword else "relevant"

    if intent == "irrelevant" and not has_relevant_keyword:
        logger.info("Non-relevant query detected")
        if "baby" in message.lower() or "babies" in message.lower():
            return "Hey, I caught 'baby' in there! I’m all about neonatal stuff—wanna chat about newborn health instead? Maybe something like jaundice or feeding?\n\nWanna save this chat? Just say 'remember this'!"
        return "Hey, I’m all about neonatal, pregnancy, and OB-GYN goodies! That’s a bit outside my lane—how about we talk babies or pregnancy instead? What’s your next move?\n\nWanna save this chat? Just say 'remember this'!"

    # Step 2: Search health_data.json if TF-IDF matrix is available
    if tfidf_matrix is not None:
        logger.info(f"Searching health_data.json for: {message}")
        try:
            query_vec = tfidf_vectorizer.transform([message.lower()])
            similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
            best_match_idx = similarities.argmax()
            similarity_score = similarities[best_match_idx]
            
            if similarity_score > 0.6:
                matched_input = health_data.iloc[best_match_idx]["input"]
                matched_output = health_data.iloc[best_match_idx]["output"]
                logger.info(f"Match found: {matched_input} (score: {similarity_score})")
                if isinstance(matched_output, dict):
                    options = "\n".join([f"{k}: {v}" for k, v in matched_output.items()])
                    bot_response = f"Alright, '{message}' sounds like a puzzle! Here's what I dug up:\n{options}\nWhat do you think fits best—or should we dig deeper into something neonatal or pregnancy-related?\n\nWanna save this chat? Just say 'remember this'!"
                    return bot_response
                else:
                    bot_response = f"Here's the scoop on '{message}': {matched_output}. Pretty neat, huh? Want me to expand on that or switch gears?\n\nWanna save this chat? Just say 'remember this'!"
                    return bot_response
        except Exception as e:
            logger.error(f"Error in TF-IDF search: {e}")
            # Continue to next step if search fails
    else:
        logger.warning("TF-IDF matrix not available, skipping health_data search")

    # Step 3: Use sharma-health-model to refine the prompt
    refine_prompt = f"""You are an agentic AI assistant. Refine this query into a clear, specific question about neonatal, pregnancy, or OB-GYN topics. Keep it concise and relevant.
Query: {message}
Refined:"""
    refined_query = message
    if model and tokenizer:
        try:
            inputs = tokenizer(refine_prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = model.generate(**inputs, max_length=100, temperature=0.8, pad_token_id=tokenizer.eos_token_id)
            refined_query = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Refined:")[1].strip()
            logger.info(f"sharma-health-model refined query: {refined_query}")
        except Exception as e:
            logger.error(f"sharma-health-model refine error: {e}")
            refined_query = message  # Fallback to original

    # Step 4: Call DeepSeek with refined query
    logger.info(f"Calling DeepSeek API with refined query: {refined_query}")
    prompt = f"""You are Sharma, a super friendly, proactive medical expert focused only on neonatal, pregnancy, and OB-GYN topics. Respond in a warm, conversational tone with flair. Be agentic—anticipate needs, suggest next steps, and keep it strictly relevant to neonatal, pregnancy, or OB-GYN. If vague, ask a fun follow-up. Do NOT misinterpret terms like 'ectopic' or invent terms like '1st-grade pregnancy'.
Question: {refined_query}
Answer:"""
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": refined_query}
        ],
        "max_tokens": 300,
        "temperature": 0.9
    }
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    
    try:
        response = requests.post(DEEPSEEK_URL, json=payload, headers=headers)
        response.raise_for_status()
        response_data = response.json()
        logger.info(f"DeepSeek raw response: {response_data}")
        if "choices" in response_data and response_data["choices"]:
            bot_response = response_data["choices"][0]["message"]["content"].strip()
            logger.info(f"DeepSeek response: {bot_response}")
        else:
            bot_response = "Hmm, DeepSeek's being shy! Let's pivot—what's your next neonatal or pregnancy question?\n\nWanna save this chat? Just say 'remember this'!"
            logger.error(f"Invalid DeepSeek response: {response_data}")
    except requests.RequestException as e:
        logger.error(f"DeepSeek API error: {e}")
        if model and tokenizer:
            try:
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
                outputs = model.generate(**inputs, max_length=200, temperature=0.9, pad_token_id=tokenizer.eos_token_id)
                bot_response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[1].strip()
                logger.info(f"Fallback sharma-health-model response: {bot_response}")
            except Exception as e:
                logger.error(f"Fallback model error: {e}")
                bot_response = "Whoops, both my brains are hiccupping! Got any baby or pregnancy questions I can tackle with my notes?\n\nWanna save this chat? Just say 'remember this'!"
        else:
            bot_response = "I'm having trouble connecting to my knowledge source right now. Let's try a different question about neonatal care, pregnancy, or OB-GYN topics!\n\nWanna save this chat? Just say 'remember this'!"

    if context and context.get("topic"):
        bot_response += f" (Oh, we're vibing on {context['topic']}—love that!)"
    
    memories = memory_system.get_memory(message)
    if memories:
        bot_response += "\n\nBy the way, I've got some notes:\n" + "\n".join([f"• {m['value']}" for m in memories])
    else:
        bot_response += "\n\nWanna save this chat in my memory bank? Just say 'remember this'!"
    return bot_response

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)