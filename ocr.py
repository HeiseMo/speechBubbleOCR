# server/ocr.py
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import os
from uuid import uuid4
from PIL import Image
from manga_ocr import MangaOcr
from sse_starlette.sse import EventSourceResponse
import json

# server/ocr.py
app = FastAPI()
mocr = MangaOcr()
UPLOAD_DIR = "uploads"

# Define CORS settings
origins = [
    "http://localhost",      # Allow requests from localhost
    "http://localhost:3000", # Allow requests from a specific port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# server/ocr.py
# find all speech bubbles in the given comic page and return a list of cropped speech bubbles (with possible false positives)
def findSpeechBubbles(imagePath, method):

    # read image
    image = cv2.imread(imagePath)
    # gray scale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # filter noise
    imageGrayBlur = cv2.GaussianBlur(imageGray, (3, 3), 0)
    if method != 'simple':
        # recognizes more complex bubble shapes
        imageGrayBlurCanny = cv2.Canny(imageGrayBlur, 50, 500)
        binary = cv2.threshold(imageGrayBlurCanny, 235,
                               255, cv2.THRESH_BINARY)[1]
    else:
        # recognizes only rectangular bubbles
        binary = cv2.threshold(imageGrayBlur, 235, 255, cv2.THRESH_BINARY)[1]

    # find contours
    contours = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    # get the list of cropped speech bubbles

    croppedImageList = []
    i = 0
    for contour in contours:

        contour = contour.astype(np.int32)
        rect = cv2.boundingRect(contour)
        [x, y, w, h] = rect

        # filter out speech bubble candidates with unreasonable size
        if w < 500 and w > 40 and h < 500 and h > 40:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            croppedImage = image[y:y+h, x:x+w]
            croppedImageList.append(croppedImage)
            cv2.imwrite(UPLOAD_DIR+"/"+'cropped/'+ str(i)+".jpg", croppedImage)
            i += 1

    return croppedImageList

# server/ocr.py
async def text_generator(request):
    i = 1
    image_list = os.listdir(os.path.join(UPLOAD_DIR,'cropped'))
    for img in image_list:
        if await request.is_disconnected():
            print("client disconnected!!!")
            break
        if i <= len(image_list):
            text = mocr(os.path.join(UPLOAD_DIR,'cropped',img))
            print(text)
            i+=1
            yield json.dumps({"id": i,"source":text})
        else:
            print("OCR complete!")
            break
# server/ocr.py

@app.post("/api/ocr")
async def upload_file(request: Request,file: UploadFile = File(...)):
    try:
        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(os.path.join(UPLOAD_DIR,"cropped"))

        file_extension = file.filename.split(".")[-1]
        new_filename = f"{uuid4()}.{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, new_filename)

        with open(file_path, "wb") as f:
            f.write(file.file.read())

        try:
            findSpeechBubbles(file_path,"simple")
        except Exception as e:
            print(e)
            return JSONResponse(content={"error": "An error occurred" }, status_code=500)

        try:
            event_generator =  text_generator(request)
            return EventSourceResponse(event_generator)
        except Exception as e:
            print(e)
            return JSONResponse(content={"error": "An error occurred"}, status_code=500)

    except Exception as e:
        print(e)

    return JSONResponse(content={"error": "An error occurred while uploading the file"}, status_code=500)