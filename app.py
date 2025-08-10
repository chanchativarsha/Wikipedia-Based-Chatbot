from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os

from langchain.agents import initialize_agent, Tool
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Get Gemini API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in .env")

# Use the correct Gemini model
llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",  # ✅ Correct model name
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

# Wikipedia search tool
wiki_api_wrapper = WikipediaAPIWrapper()
wiki_tool = Tool(
    name="Wikipedia",
    func=WikipediaQueryRun(api_wrapper=wiki_api_wrapper).run,
    description="Search Wikipedia for information"
)

# Create the agent
agent = initialize_agent(
    tools=[wiki_tool],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    try:
        response = agent.run(user_message)
    except Exception as e:
        response = f"Error: {e}"
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
