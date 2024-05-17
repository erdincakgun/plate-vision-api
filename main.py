from typing import Annotated
import magic
from fastapi import FastAPI, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from utils import convert, detect, read, encode

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/")
async def create_file(file: Annotated[bytes, File()]):
    mime = magic.Magic(mime=True)
    file_type = mime.from_buffer(file)

    if not file_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only images are allowed."
        )

    try:
        file_size = len(file)
        if file_size > 1048576:
            raise HTTPException(
                status_code=400,
                detail="Image file size exceeds 1 MB."
            )
        img = convert(file)
        plate_images = detect(img)
        plate_numbers = read(plate_images)
        encoded_plates = encode(plate_images)
        return {"encoded_plates": encoded_plates, "plate_numbers": plate_numbers}

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
