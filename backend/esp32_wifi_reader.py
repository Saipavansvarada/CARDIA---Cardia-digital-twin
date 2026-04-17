# backend/esp32_wifi_reader.py
from fastapi import APIRouter, Request

router = APIRouter()

@router.post("/sensor-data")
async def receive_sensor_data(request: Request):
    data = await request.json()
    """
    Example data:
    {
        "hr": 72,
        "hrv": 45,
        "load": 20,
        "sleep": 7
    }
    """
    # Save to DB or forward to processing pipeline
    return {"status": "received", "data": data}