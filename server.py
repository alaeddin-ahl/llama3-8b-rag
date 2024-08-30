from flask import Flask, request, jsonify
from local_rag import getFilesFromFolder, embedPDFDocuments, retrieveTextChunks, augmentPrompt, getResponse

app = Flask(__name__)

def start(): 
    print ("starting..")
    files = getFilesFromFolder("pdfs/")
    app.embeddedTextChunksIndex = embedPDFDocuments(files)
    print("started")

@app.route('/start', methods=['GET'])
def handle_start():
    start()
    return jsonify({"status": "success"})

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json()
    message = data.get('message')
    if not message:
        return jsonify({"status": "error", "message": "No message provided"})
    
    print(f"Received query: {message}")

    mostSimilarTextChunks = retrieveTextChunks(app.embeddedTextChunksIndex, message)

    # Augment prompt with additional context
    fullPrompt = augmentPrompt(message, mostSimilarTextChunks)

    r = getResponse(fullPrompt)

    print(f"Response: {r}")

    return jsonify({"status": "success", "data": {"response": r}})


if __name__ == '__main__':
    start()
    app.run(port=8080)
    
