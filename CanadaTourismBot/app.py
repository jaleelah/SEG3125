import os
import gradio as gr
from google.generativeai import GenerativeModel, configure, types
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from typing import List, Tuple

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
configure(api_key=GOOGLE_API_KEY)

class MyApp:
    def __init__(self) -> None:
        self.documents = []
        self.embeddings = None
        self.index = None
        self.load_pdf("Canada.pdf")
        self.build_vector_db()

    def load_pdf(self, file_path: str) -> None:
        """Extracts text from a PDF file and stores it in the app's documents."""
        doc = fitz.open(file_path)
        self.documents = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            self.documents.append({"page": page_num + 1, "content": text})
        print("PDF processed successfully!")

    def build_vector_db(self) -> None:
        """Builds a vector database using the content of the PDF."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = model.encode([doc["content"] for doc in self.documents], show_progress_bar=True)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))
        print("Vector database built successfully!")

    def search_documents(self, query: str, k: int = 3) -> List[str]:
        """Searches for relevant documents using vector similarity."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query], show_progress_bar=False)
        D, I = self.index.search(np.array(query_embedding), k)
        results = [self.documents[i]["content"] for i in I[0]]
        return results if results else ["No relevant documents found."]

app = MyApp()

def respond(message: str, history: List[Tuple[str, str]]):
    system_message = (
        "You are a friendly, fun Canadian travel agent. "
        "You help clients plan vacations in Canada by learning about their personalities and recommending appropriate destinations and activities. "
        "You must say one thing at a time and ask follow-up questions to continue the chat."
        "Assurez-vous que votre sortie est dans la même langue que l'entrée."
    )
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    if any(
        keyword in message.lower()
        for keyword in ["province", "city", "activity", "where to", "accommodation"]
    ):
        retrieved_docs = app.search_documents(message)
        context = "\n".join(retrieved_docs)
        if context.strip():
            messages.append({"role": "system", "content": "Relevant documents: " + context})

    model = GenerativeModel("gemini-1.5-pro-latest")
    generation_config = types.GenerationConfig(
        temperature=0.7,
        max_output_tokens=1024,
    )
    
    try:
        response = model.generate_content([message], generation_config=generation_config)
        response_content = response.text if hasattr(response, "text") else "No response generated."
    except Exception as e:
        response_content = f"An error occurred while generating the response: {str(e)}"

    history.append((message, response_content))
    return history, ""

with gr.Blocks(theme=gr.themes.Glass(primary_hue = "violet")) as demo:
    gr.Markdown("**Your Canadian Travel Agent**")

    chatbot = gr.Chatbot()

    with gr.Row():
        txt_input = gr.Textbox(
            show_label=False,
            placeholder="",
            lines=1
        )
        submit_btn = gr.Button("Submit", scale=1)
        refresh_btn = gr.Button("Refresh Chat", scale=1, variant="secondary")

    example_questions = [
        ["Do you have any recommendations on things to do in Ottawa?"],
        ["I'm not sure which city to visit. Can you help me decide?"],
        ["Donnez-moi une liste d’options de transport entre Calgary et Edmonton."],
        ["Combien dois-je m’attendre à dépenser pour un voyage d’une semaine à Toronto ?"],
        ["I really like the outdoors. What kind of trip would you recommend for that?"],
        ["Dans quel hôtel devrais-je séjourner à Québec?"],
        ["What's the best province to visit in the winter?"],
        ["What kind of food options are available in Saskatchewan?"],
        ["Quel est le moyen le moins cher pour se rendre à Halifax depuis la Floride ?"],
        ["Give me a list of 8-day long vacation plans including at least 2 cities each."],
        ["What's the best city to go to for a history-lover?"]
    ]

    gr.Examples(examples=example_questions, inputs=[txt_input])

    submit_btn.click(fn=respond, inputs=[txt_input, chatbot], outputs=[chatbot, txt_input])
    refresh_btn.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch()