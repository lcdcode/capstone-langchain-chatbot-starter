from flask import Flask, render_template
from flask import request, jsonify, abort

# Added per Cohere docs
# from langchain_cohere import ChatCohere
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain.memory import ConversationBufferWindowMemory
# from langchain.chains import ConversationChain
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

app = Flask(__name__)

def answer_from_knowledgebase(message):
    # TODO: Write your code here
    return ""

def search_knowledgebase(message):
    # TODO: Write your code here
    sources = ""
    return sources

def answer_as_chatbot(message):

    # Create LLM/prompt only once
    if not hasattr(answer_as_chatbot, "_chain"):
        # a simple prompt that appends the conversation history
        template_str = (
            "{history}\n"
            "Human: {human_input}\n"
            "Assistant:"
        )
        prompt = PromptTemplate(
            input_variables=["history", "human_input"],
            template=template_str,
        )
        # Cohere LLM (text‑generation endpoint, not chat)
        llm = Cohere(
            model="command-xlarge",
            temperature=0.7,
        )
        # Build a chain that plugs the prompt into the LLM
        answer_as_chatbot._chain = LLMChain(llm=llm, prompt=prompt)

        # Keep the conversation history as a plain string
        answer_as_chatbot._history = ""

    # Run the chain with the new user turn
    assistant_reply = answer_as_chatbot._chain.run(
        {
            "history": answer_as_chatbot._history,
            "human_input": message,
        }
    ).strip()

    # Update the history for the next call -- keep the history in the same format that the prompt expects
    new_entry = f"Human: {message}\nAssistant: {assistant_reply}\n"
    answer_as_chatbot._history += new_entry

    # Return the reply
    return assistant_reply

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