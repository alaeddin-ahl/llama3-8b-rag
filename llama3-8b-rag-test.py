from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
import os, ollama

def getFilesFromFolder(folder):
    return [folder + f for f in os.listdir(folder)]

def embedPDFDocuments(filePaths):
    # Load document and split into chunks
    pdfTextChunks = []
    print("Loading PDFs and splitting them into chunks..")

    # Configure split PDF in text chunks of 800 characters with an overlap of 80 characters
    textSplitConfiguration = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
        )

    # Load and split all PDF files into text chunks
    for filePath in filePaths:
        pdfLoader = PyPDFLoader(filePath)
        newPDFTextChunks = pdfLoader.load_and_split(textSplitConfiguration)
        pdfTextChunks += newPDFTextChunks
        # print(f"Loaded {len(pdfTextChunks)} chunks.")
        # print(newPDFTextChunks[10].metadata)
        # print(newPDFTextChunks[10].page_content)
        # print("")

    # Create embeddings using llama3 model and index them so that we can easily do similarity search later on
    # Facebook/Meta AI Similarity Search
    print(f"Embedding and indexing {len(pdfTextChunks)} text chunks..")
    # pdfTextChunks = pdfTextChunks[:1]
    embeddedTextChunksIndex = FAISS.from_documents(pdfTextChunks, OllamaEmbeddings(model="llama3"))
    return embeddedTextChunksIndex

def retrieveTextChunks(embeddedTextChunksIndex, prompt):
    print("Searching for best fits for query..")
    return embeddedTextChunksIndex.similarity_search_with_score(prompt, k=2)

def augmentPrompt(originalPrompt, mostSimilarTextChunks):
    print("Augment prompt with additional data..")

    augmentedPrompt  = f"Respond to the query at the end with the help of the following context:\n"
    
    for (documentChunk, score) in mostSimilarTextChunks:
        augmentedPrompt += "\n---\n"
        augmentedPrompt += f"SIMILARITY SCORE: {score}"
        augmentedPrompt += f"METADATA: {str(documentChunk.metadata)}"
        augmentedPrompt += f"CONTENT: {documentChunk.page_content}"
    augmentedPrompt += f"\n------\n"

    augmentedPrompt += f"Respond to the query '{originalPrompt}'with the help the context above. Answer short and precisely. Reference the source and page in your answer."
    return augmentedPrompt

def generateResponse(prompt):
    print("Generating response..")

    # Create the response
    stream = ollama.chat(
        model='llama3',
        messages=[{'role': 'user', 'content': prompt}],
        stream=True,
    )

    print("\nResponse:\n")

    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

# Index PDF vector embeddings
files = getFilesFromFolder("pdfs/")
embeddedTextChunksIndex = embedPDFDocuments(files)

# Retrieve top similar chunks
prompt = "What is the dress code for the workshop?"
mostSimilarTextChunks = retrieveTextChunks(embeddedTextChunksIndex, prompt)

# Augment prompt with additional context
fullPrompt = augmentPrompt(prompt, mostSimilarTextChunks)

# Generate response
generateResponse(fullPrompt)

