# NDIS AI Assistant (2026 Stable)

A Conversational RAG (Retrieval-Augmented Generation) assistant designed to help users navigate the complex National Disability Insurance Scheme (NDIS) guidelines.

## Key Features
* **Context-Aware Retrieval:** Uses FAISS and Google Gemini Embeddings to retrieve relevant NDIS documentation.
* **Conversational Memory:** Implements Streamlit session state to maintain context across multi-turn dialogues.
* **Source Attribution:** Provides transparent "Information Sources" for every answer to ensure accuracy and reduce hallucinations.
* **LCEL Orchestration:** Built using LangChain Expression Language for a modular, production-ready data pipeline.

## Tech Stack
* **Language:** Python 3.11
* **LLM:** Google Gemini 2.5 Flash
* **Embeddings:** Google Gemini Embedding 001
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Frameworks:** LangChain, Streamlit

## Technical Problem Solving: The "404 NOT_FOUND" Fix
During development, a critical issue was identified where the Google API returned a `404 NOT_FOUND` for standard embedding models. 

**The Solution:** 1.  Identified a mismatch between the `v1beta` API path and model availability in early 2026.
2.  Refactored the embedding configuration to use the `models/gemini-embedding-001` alias, ensuring compatibility across stable and beta endpoints.
3.  Resolved Pydantic validation errors by implementing `itemgetter` to isolate string inputs for the retriever within the LCEL dictionary structure.

## Installation & Setup
1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`.
3. Install dependencies: `pip install -r requirements.txt`.
4. Create a `.env` file based on `.env.example` and add your `GOOGLE_API_KEY`.
5. Run the app: `streamlit run app.py`.