from docx import Document
from textwrap import wrap
from sentence_transformers import SentenceTransformer
import chromadb
import requests

# Step 1: Read the Word document
doc = Document("Customer_Service_QA_Knowledge_Base.docx")

text = ""

for para in doc.paragraphs:
    if para.text.strip():
        text += para.text.strip() + "\n"

# This reads the Q&A tables inside the Word document
for table in doc.tables:
    for row in table.rows:
        row_text = []
        for cell in row.cells:
            if cell.text.strip():
                row_text.append(cell.text.strip())
        if row_text:
            text += " | ".join(row_text) + "\n"

# Step 2: Split the text into chunks
chunks = wrap(text, 500)

print("Number of chunks created:", len(chunks))

# Step 3: Convert chunks into embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(chunks).tolist()

# Step 4: Store embeddings in ChromaDB
client = chromadb.Client()

try:
    client.delete_collection(name="customer_service_qa")
except:
    pass

collection = client.create_collection(name="customer_service_qa")

collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

print("ChromaDB vector database created successfully.")
print("Number of chunks stored:", collection.count())

# Step 5: Function to ask the chatbot
def ask_chatbot(question):
    question_embedding = embedding_model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=3
    )

    retrieved_chunks = results["documents"][0]
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a helpful customer service assistant.

Use only the information from the context below to answer the user's question.
If the answer is not available in the context, say that you do not have enough information.

Context:
{context}

User Question:
{question}

Answer:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:1b",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]

# Step 6: Simple terminal chat
print("\nCustomer Service RAG Chatbot is ready.")
print("Type 'exit' to stop.\n")

while True:
    user_question = input("Ask a question: ")

    if user_question.lower() == "exit":
        print("Chatbot stopped.")
        break

    answer = ask_chatbot(user_question)

    print("\nAnswer:")
    print(answer)
    print("-" * 80)