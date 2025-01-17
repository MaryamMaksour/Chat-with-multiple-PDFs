# Chat-with-multiple-PDFs

Project Title: Interactive PDF Chatbot with Streamlit
About the Project:

This project is an interactive web application that allows users to chat with content extracted from multiple PDF files. By leveraging advanced AI models and embedding-based search, it provides accurate answers to user questions based on uploaded documents. Built using Streamlit, it offers a user-friendly interface for document processing and question-answering tasks.

Key Features:

PDF Upload and Processing: Users can upload multiple PDFs, and the content is extracted for further analysis.
Chunked Text Analysis: Splits document content into manageable text chunks for efficient embedding and retrieval.
Embedding-Based Search: Utilizes Google Generative AI embeddings for semantic search and similarity matching.
Contextual QA: Provides detailed answers to user queries based on document content, ensuring accuracy and transparency.
Web Interface: Built with Streamlit, offering an intuitive and interactive user experience.
Technologies Used:

Streamlit: For creating the web interface and handling user interactions.
PyPDF2: To extract text from PDF files.
LangChain: For integrating embeddings, text splitting, and QA chains.
Google Generative AI: Used for generating embeddings and conversational responses.
FAISS: A local vector database for fast similarity search.
dotenv: For secure API key management.
How It Works:

PDF Processing:
The uploaded PDFs are processed to extract raw text using PyPDF2.
The text is split into smaller chunks for better embedding and retrieval performance.
Vector Store Creation:
Text chunks are embedded using Google Generative AI embeddings.
The embeddings are stored in a local FAISS vector database for semantic search.
Question Answering:
Users input questions via the web interface.
Relevant text chunks are retrieved using similarity search from FAISS.
Google Generative AI provides detailed and contextual answers based on the retrieved chunks.
User-Friendly Interface:
The Streamlit interface guides users through uploading, processing, and querying their documents.
Use Cases:

Knowledge Management: Transform static PDF documents into interactive resources for employees or customers.
Educational Tools: Make course materials or academic papers more accessible through conversational queries.
Legal Document Review: Quickly search and query legal contracts or agreements.
Why It Stands Out: This project integrates multiple cutting-edge technologies to create an accessible and powerful tool for document interaction. It showcases expertise in building end-to-end AI solutions, combining NLP, vector databases, and web application development into a cohesive product.
