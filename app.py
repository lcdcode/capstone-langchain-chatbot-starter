from flask import Flask, render_template
from flask import request, jsonify, abort
import os
from langchain.llms import Cohere
from langchain_cohere.chat_models import ChatCohere
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain, RetrievalQA
from langchain.embeddings import CohereEmbeddings
from langchain_community.vectorstores import Chroma

app = Flask(__name__)

def load_db():
    try:
        embeddings = CohereEmbeddings(cohere_api_key=os.environ["COHERE_API_KEY"], user_agent="my-app")
        vectordb = Chroma(persist_directory='db', embedding_function=embeddings)
        # Use ChatCohere for new Chat API versus old Generate API
        chat_llm = ChatCohere()
        qa = RetrievalQA.from_chain_type(
            llm=chat_llm,
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

    # Use chat-compatible input for new Cohere Chat API
    try:
        res = qa.invoke({"query": message})
        # The result may be in 'text' or 'result' depending on chain config
        return res.get('text') or res.get('result') or str(res)
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
        res = qa.invoke({"query": message})
        sources = ""
        for count, source in enumerate(res.get('source_documents', []), 1):
            sources += f"Source {count}\n"
            sources += getattr(source, 'page_content', str(source)) + "\n"
        return sources if sources else (res.get('text') or res.get('result') or str(res))
    except IndexError:
        return "No relevant information found in the knowledge base."
    except Exception as e:
        return f"An error occurred: {str(e)}"

def answer_as_chatbot(message):

    # Create LLM/prompt only once
    if not hasattr(answer_as_chatbot, "_chain"):
        # Use ChatPromptTemplate for chat models
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        # Use ChatCohere for Cohere's chat API
        llm = ChatCohere()
        answer_as_chatbot._chain = LLMChain(llm=llm, prompt=prompt)
        answer_as_chatbot._history = []  # list of messages

    # Prepare the history for the chat API
    history = answer_as_chatbot._history.copy()
    # Add the new user message
    history.append({"role": "user", "content": message})

    # Run the chain with the new user turn
    response = answer_as_chatbot._chain.invoke({
        "history": history,
        "input": message
    })
    assistant_reply = response["text"] if "text" in response else str(response)

    # Update the history for the next call
    answer_as_chatbot._history.append({"role": "assistant", "content": assistant_reply})

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