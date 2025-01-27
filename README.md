# Medical Query Assistant

This is a medical query assistant built using Streamlit, which allows users to interact with medical documents (PDF or web URL) and ask related questions. The assistant processes the documents and provides answers based on the content using advanced AI models powered by Groq and Pinecone.

## Features

- **PDF Upload**: Upload medical PDFs for analysis and querying.
- **Web URL**: Process medical-related content from any web URL (including PDFs).
- **Question-Answer Interface**: Ask medical questions, and the assistant will provide answers based on the content in the processed document.
- **Query History**: View a history of the questions asked and answers provided.
- **Powered by Groq and Pinecone**: Uses state-of-the-art AI for text vectorization and search.

## Installation

### Prerequisites
- Python 3.8 or higher
- Install dependencies using `pip`


medical_assistant.py           # Main Streamlit app
requirements.txt               # List of required Python packages
.env                           # Environment variables for API keys
## Installation

### Prerequisites

- Python 3.8 or higher
- Required API keys for **Groq** and **Pinecone**

### Install Dependencies

Clone the repository and install the dependencies using:


pip install -r requirements.txt

### API_KEY
- GROQ_API_KEY=<your_groq_api_key>
- PINECONE_API_KEY=<your_pinecone_api_key>

### Run Application by Typing:
streamlit run app.py


### Explanation:

- **DocumentProcessor**: Responsible for extracting and processing text from PDFs and web URLs. The `process_pdf` function handles PDFs, while `process_url` fetches and processes content from URLs.
- **AIAssistant**: Manages the AI-related tasks, such as embedding creation, vector storage, and querying the Pinecone vector store for answers.
- **MedicalUI**: Handles the user interface for interacting with the assistant, uploading documents, entering queries, and displaying results.



