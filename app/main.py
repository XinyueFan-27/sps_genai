from typing import Union
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from pydantic import BaseModel, Field
from app.bigram_model import BigramModel
from app.embeddings import calculate_embedding
from app.classifier import predict_image
from app.gan_model import generate_digits_png
from app.diffusion_model import generate_diffusion_png
from app.energy_model import generate_ebm_png

app = FastAPI()

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

class EmbeddingRequest(BaseModel):
    word: str

class DigitGenerationRequest(BaseModel):
    num_images: int = Field(16, ge=1, le=64)

class DiffusionRequest(BaseModel):
    num_images: int = Field(16, ge=1, le=64)
    num_steps: int = Field(50, ge=1, le=1000)  # DDIM sampling steps

class EBMRequest(BaseModel):
    num_images: int = Field(16, ge=1, le=64)
    steps: int = Field(256, ge=1, le=2000)      # Langevin steps

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    vector = calculate_embedding(request.word)
    return {
        "word": request.word,
        "dimensions": len(vector),
        "embedding": vector.tolist(),
    }

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")
    image_bytes = await file.read()
    try:
        result = predict_image(image_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the image.")

    return {"filename": file.filename, **result}


@app.post("/generate_digits", response_class=Response)
def generate_digits(request: DigitGenerationRequest):
    png_bytes = generate_digits_png(request.num_images)
    return Response(content=png_bytes, media_type="image/png")


@app.post("/generate_diffusion", response_class=Response)
def generate_diffusion(request: DiffusionRequest):
    try:
        png_bytes = generate_diffusion_png(request.num_images, request.num_steps)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return Response(content=png_bytes, media_type="image/png")


@app.post("/generate_ebm", response_class=Response)
def generate_ebm(request: EBMRequest):
    try:
        png_bytes = generate_ebm_png(request.num_images, request.steps)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return Response(content=png_bytes, media_type="image/png")
