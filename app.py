from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle


def load_model():
    with open("./classifiers/linear_svc_pipeline.pkl", "rb") as file:
        pipeline = pickle.load(file)
        
    return pipeline

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class request_body(BaseModel):
    comment: str



@app.get("/")
def main():
    return {"message": "Hello World"}


@app.post("/predict")
def predict(data: request_body):
    pipeline = load_model()
    res = pipeline.predict([data.comment])

    response = True if res[0] else False

    return {"is_toxic": response}