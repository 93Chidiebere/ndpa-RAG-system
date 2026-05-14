# NDPA RAG System

This is a Retrieval-Augmented Generation (RAG) system built with FastAPI to answer questions about the Nigerian Data Protection Act (NDPA) 2023.

It uses:
- **FAISS** for vector similarity search
- **BM25** for keyword search
- **SentenceTransformers** for generating text embeddings (`all-MiniLM-L6-v2`)
- **TinyLlama** (`TinyLlama-1.1B-Chat-v1.0`) for generating answers

## How to Deploy to Hugging Face Spaces (Free)

Hugging Face Spaces provides enough free RAM (16GB) to host this application smoothly. Follow these steps:

1. Create an account on [Hugging Face](https://huggingface.co/) if you don't have one.
2. Go to your Hugging Face profile and click **"New Space"**.
3. Name your space (e.g., `NDPA-Legal-Assistant`).
4. Select **"Docker"** as the Space SDK. (Choose the blank Docker template).
5. Choose the **Free (Basic CPU)** hardware.
6. Click **Create Space**.
7. Connect your GitHub repository to Hugging Face or simply upload all your project files directly to the Space's Files section.
   
Make sure the following files are included in the repository/Space:
- `app.py`
- `requirements.txt`
- `Dockerfile`
- `chunks.json`
- `ndpa_faiss.index`

Once uploaded, Hugging Face will automatically build the Docker container using the `Dockerfile` and start your app. You'll get a public URL that you can share with anyone!

## Running Locally

If you want to run this locally:

1. Install dependencies: `pip install -r requirements.txt`
2. Run the preparation script (only needed once if `chunks.json` doesn't exist): `python save_data.py`
3. Start the FastAPI server: `uvicorn app:app --reload`
4. Visit `http://localhost:8000` in your browser.
