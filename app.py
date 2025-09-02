from flask import Flask, render_template
from flask import request, jsonify, abort

# Added per Cohere docs
from langchain_cohere import ChatCohere
from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain


app = Flask(__name__)
# Initialize cohere once per app instance
chat = ChatCohere()

def answer_from_knowledgebase(message):
    # TODO: Write your code here
    return ""

def search_knowledgebase(message):
    # TODO: Write your code here
    sources = ""
    return sources

def answer_as_chatbot(message):

    # Initialize the LLM only once (stored on the function object)
    if not hasattr(answer_as_chatbot, "_chat"):
        # One-time creation of the Cohere chat model
        answer_as_chatbot._chat = ChatCohere()
        # And an empty list that will hold the conversation history
        answer_as_chatbot._memory = []

    # Append the new user turn to the history
    answer_as_chatbot._memory.append(HumanMessage(content=message))

    # Ask Cohere for a reply using the *entire* history
    ai_msg: AIMessage = answer_as_chatbot._chat.invoke(answer_as_chatbot._memory)

    # Store the bot’s reply so future calls can see it
    answer_as_chatbot._memory.append(ai_msg)

    # Return just the text to the caller
    return ai_msg.content

@app.route('/kbanswer', methods=['POST'])
def kbanswer():
    # TODO: Write your code here
    
    # call answer_from_knowledebase(message)
        
    # Return the response as JSON
    return 

@app.route('/search', methods=['POST'])
def search():    
    # Search the knowledgebase and generate a response
    # (call search_knowledgebase())
    
    # Return the response as JSON
    return

@app.route('/answer', methods=['POST'])
def answer():
    message = request.json['message']
    
    # Generate a response
    response_message = answer_as_chatbot(message)
    
    # Return the response as JSON
    return jsonify({'message': response_message}), 200

@app.route("/")
def index():
    return render_template("index.html", title="")

if __name__ == "__main__":
    app.run()