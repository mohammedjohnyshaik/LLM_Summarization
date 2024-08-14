from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import openai
import faiss
import numpy as np
from dotenv import load_dotenv
import os
import fitz

load_dotenv()
app = FastAPI()
openai.api_key = os.getenv("api_key")
class QueryRequest(BaseModel):
    query: str

def pagelevel_pdf_process(pdf_path):
    document = fitz.open(pdf_path)
    page_chunks_ = []
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text = page.get_text()
        page_chunks_.append(text) 
    return page_chunks_

def paraleve_pdf_process(pdf_path):
    paragraph_chunks_ = []
    page_chunks_ = pagelevel_pdf_process(pdf_path)  
    for page_text in page_chunks_:
        page_paragraphs = page_text.split('\n\n')  
        for para in page_paragraphs:
            if para.strip():
                paragraph_chunks_.append(para.strip())
    return paragraph_chunks_

def get_embeddings(chunks, model="text-embedding-3-small"):
    embeddings = []
    for chunk in chunks:
        chunk = chunk.replace("\n", " ")
        response = openai.embeddings.create(
            input=[chunk],
            model=model
        )
        embeddings.append(response.data[0].embedding)
    return embeddings

def store_embeddings(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    embedding_array = np.array(embeddings).astype('float32')
    index.add(embedding_array)
    return index

def truncate_text(text, max_tokens):
    tokens = text.split()[:max_tokens]
    return ' '.join(tokens)

def generate_response(query, index, chunks, temperature=0.5, top_p=1.0):
    query_vector = get_embeddings([query])[0]
    _, top_indices = index.search(np.array([query_vector]), k=5)
    relevant_chunks = [chunks[i] for i in top_indices[0]]
    context = " ".join(relevant_chunks)
    response =  openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Summarize the pdf"},
            {"role": "user", "content": context + "\n\nQ: " + query}
        ],
        max_tokens=150,
        temperature=temperature,
        top_p=top_p
    )
    return response.choices[0].message.content.strip()

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global pdf_chunks, index
    try:
        # Save the uploaded file
        pdf_path = f"/tmp/{file.filename}"
        with open(pdf_path, "wb") as buffer:
            buffer.write(await file.read())

        # Process the PDF and create embeddings
        pdf_chunks = paraleve_pdf_process(pdf_path)
        embeddings = get_embeddings(pdf_chunks)
        index = store_embeddings(embeddings)
        return {"message": "PDF processed and embeddings created successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/generate-response/")
async def generate_response_endpoint(request: QueryRequest):
    try:
        if pdf_chunks is None or index is None:
            raise HTTPException(status_code=400, detail="PDF not uploaded or processed.")

        response = generate_response(
            query=request.query,
            index=index,
            chunks=pdf_chunks,
            temperature=0.5, 
            top_p=0.9        
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
