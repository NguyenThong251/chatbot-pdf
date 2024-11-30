import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class PDFChatbot:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Model embedding
        self.index = None
        self.text_chunks = []

    def load_pdf(self):
        # Đọc nội dung PDF và chia thành các đoạn văn
        reader = PdfReader(self.pdf_path)
        self.text_chunks = [
            page.extract_text() for page in reader.pages if page.extract_text()
        ]

    def build_index(self):
        # Tạo chỉ mục FAISS từ các đoạn văn
        embeddings = self.model.encode(self.text_chunks)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings))

    def answer_question(self, question):
        # Trả lời câu hỏi dựa trên dữ liệu
        question_embedding = self.model.encode([question])
        distances, indices = self.index.search(question_embedding, k=1)
        best_match = indices[0][0]
        return self.text_chunks[best_match] if best_match < len(self.text_chunks) else "No relevant information found."
