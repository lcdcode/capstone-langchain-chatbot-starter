from flask import Flask, render_template
from flask import request, jsonify, abort
import os
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Chroma

app = Flask(__name__)

def load_db():
    try:
        embeddings = CohereEmbeddings(cohere_api_key=os.environ["COHERE_API_KEY"], user_agent="my-app")
        vectordb = Chroma(persist_directory='db', embedding_function=embeddings)
        qa = RetrievalQA.from_chain_type(
            llm=Cohere(),
            chain_type="refine",
            retriever=vectordb.as_retriever(),
            return_source_documents=True
        )
        return qa
    except Exception as e:
        print("Error:", e)
        raise RuntimeError("Failed to load the knowledgebase.") from e

qa = load_db()

def answer_from_knowledgebase(message):
    # Just return error if we can't get to the kb.
    if qa is None:
        return "Knowledgebase is not available."
    
    # in case we get no results
    try:
        res = qa({"query": message})
        return res['result']
    except IndexError:
        return "No relevant information found in the knowledge base."
    except Exception as e:
        return f"An error occurred: {str(e)}"

def search_knowledgebase(message):
    # Just return error if we can't get to the kb.
    if qa is None:
        return "Knowledgebase is not available."
    
    # in case we get no results
    try:
        res = qa({"query": message})
        sources = ""
        for count, source in enumerate(res['source_documents'],1):
            sources += "Source " + str(count) + "\n"
            sources += source.page_content + "\n"
        return sources
    except IndexError:
        return "No relevant information found in the knowledge base."
    except Exception as e:
        return f"An error occurred: {str(e)}"

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
    # Get the user message and search database
    message = request.json['message']
    response_message = answer_from_knowledgebase(message)
    # Return JSON
    return jsonify({'message': response_message}), 200


@app.route('/search', methods=['POST'])
def search():    
    # Search the knowledgebase and generate a response
    # (call search_knowledgebase())
    message = request.json['message']
    response_message = search_knowledgebase(message)
    # Return JSON
    return jsonify({'message': response_message}), 200

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