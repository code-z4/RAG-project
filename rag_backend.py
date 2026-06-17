from sentence_transformers import SentenceTransformer
import chromadb
import requests
import csv
import os
import json

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "customer_service_qa"
MODEL_NAME = "llama3.2:1b"
CSV_FILE = "lead_queries.csv"
MAX_HISTORY = 6

conversation_history = []

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load existing persistent ChromaDB
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_NAME)


def call_ollama(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"].strip()


def retrieve_context(question, top_k=3):
    question_embedding = embedding_model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k
    )

    retrieved_chunks = results["documents"][0]
    return "\n\n".join(retrieved_chunks)


def classify_lead_with_llm(user_query):
    prompt = f"""
You are a strict lead classification assistant.

Classify the user query into exactly one category.

Important priority rule:
If the query is about password, login, refund, return, order tracking, delivery, damaged item, payment issue, app issue, website issue, or technical help, classify it as Support Query.

Categories:
Hot Lead: Strong buying intent, asks for purchase, quote, demo, pricing, subscription, sales contact, or business requirement.
Warm Lead: Interested or exploring options but not ready to buy.
Cold Lead: Casual browsing or vague interest.
Support Query: Existing customer support issue or help request.

User Query:
{user_query}

Return only one label:
Hot Lead
Warm Lead
Cold Lead
Support Query
"""

    result = call_ollama(prompt)

    valid_labels = ["Support Query", "Hot Lead", "Warm Lead", "Cold Lead"]

    for label in valid_labels:
        if label.lower() in result.lower():
            return label

    return "Support Query"


def extract_details_with_llm(user_query):
    prompt = f"""
You are a strict information extraction assistant.

Extract details ONLY if they are explicitly mentioned in the user query.

Fields:
- Product Interest
- Team Size
- Budget
- Timeline
- Contact Request

Rules:
1. Do not guess.
2. Do not infer.
3. If not explicitly mentioned, return "Not mentioned".
4. Contact Request should be "Yes" only if the user explicitly asks to be contacted, called, emailed, or requests a demo.
5. Return ONLY valid JSON.
6. No explanations.

User Query:
{user_query}

Return JSON exactly like this:

{{
  "Product Interest": "Not mentioned",
  "Team Size": "Not mentioned",
  "Budget": "Not mentioned",
  "Timeline": "Not mentioned",
  "Contact Request": "No"
}}
"""

    result = call_ollama(prompt)

    try:
        start = result.find("{")
        end = result.rfind("}") + 1

        json_text = result[start:end]

        details = json.loads(json_text)

        required_keys = [
            "Product Interest",
            "Team Size",
            "Budget",
            "Timeline",
            "Contact Request"
        ]

        for key in required_keys:
            if key not in details:
                details[key] = "Not mentioned"

        return details

    except:
        return {
            "Product Interest": "Not mentioned",
            "Team Size": "Not mentioned",
            "Budget": "Not mentioned",
            "Timeline": "Not mentioned",
            "Contact Request": "No"
        }


def save_to_csv(user_query, lead_type, extracted_details, bot_response):
    file_exists = os.path.isfile(CSV_FILE)

    with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "User Query",
                "Lead Type",
                "Extracted Details",
                "Bot Response"
            ])

        writer.writerow([
            user_query,
            lead_type,
            json.dumps(extracted_details),
            bot_response
        ])


def get_conversation_memory():
    return "\n".join(conversation_history[-MAX_HISTORY:])


def update_memory(user_query, bot_response):
    conversation_history.append(f"User: {user_query}")
    conversation_history.append(f"Assistant: {bot_response}")

    while len(conversation_history) > MAX_HISTORY:
        conversation_history.pop(0)


def ask_chatbot(user_query):
    context = retrieve_context(user_query)
    memory = get_conversation_memory()

    prompt = f"""
You are a helpful customer service and sales assistant.

Use the retrieved context to answer the user's question.
Also consider the recent conversation history if it is useful.
If the answer is not available in the context, say you do not have enough information.

Conversation History:
{memory}

Retrieved Context:
{context}

User Question:
{user_query}

Answer:
"""

    bot_response = call_ollama(prompt)

    lead_type = classify_lead_with_llm(user_query)
    extracted_details = extract_details_with_llm(user_query)

    save_to_csv(
        user_query,
        lead_type,
        extracted_details,
        bot_response
    )

    update_memory(user_query, bot_response)

    return lead_type, extracted_details, bot_response


print("\nCustomer Service RAG Chatbot with Persistent VectorDB, LLM Lead Qualification, and Memory is ready.")
print("Type 'exit' to stop.\n")

while True:
    user_question = input("Ask a question: ")

    if user_question.lower() == "exit":
        print("Chatbot stopped.")
        break

    lead_type, extracted_details, answer = ask_chatbot(user_question)

    print("\nLead Type:")
    print(lead_type)

    print("\nExtracted Details:")
    print(extracted_details)

    print("\nAnswer:")
    print(answer)

    print("\nSaved to lead_queries.csv")
    print("-" * 80)

