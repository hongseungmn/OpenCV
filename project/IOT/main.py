from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from stream import get_stream_video

app = FastAPI()
#uvicorn main:app --reload --host=0.0.0.0 --port=8000  
def video_streaming():
    return get_stream_video()

@app.get("/video")
def main():
    return StreamingResponse(video_streaming(), media_type="multipart/x-mixed-replace; boundary=frame")

