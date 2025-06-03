from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
import json
import asyncio
import time
import uuid
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from .models.wav2lip_model import Wav2LipProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
wav2lip_processor: Optional[Wav2LipProcessor] = None
security = HTTPBearer(auto_error=False)

# Processing statistics
processing_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "average_processing_time": 0.0,
    "last_request_time": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    global wav2lip_processor
    try:
        logger.info("Starting Real-Time Lip-Syncing WebSocket API...")
        logger.info("Initializing Wav2Lip processor...")
        wav2lip_processor = Wav2LipProcessor()
        logger.info(f"Wav2Lip processor initialized successfully")
        logger.info(f"Device: {wav2lip_processor.device}")
        logger.info(f"Model: {wav2lip_processor.checkpoint_path}")
        logger.info("API ready to accept connections!")
    except Exception as e:
        logger.error(f"Failed to initialize Wav2Lip processor: {e}")

    yield

    # Shutdown
    logger.info("Shutting down API...")
    # Notify all connected clients
    await manager.broadcast({
        "type": "server_shutdown",
        "message": "Server is shutting down"
    })

app = FastAPI(
    title="Real-Time Lip-Syncing WebSocket API",
    description="A real-time lip-syncing API powered by Wav2Lip with WebSocket support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_info: Dict[str, Dict] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str = None):
        """Connect a new WebSocket client"""
        await websocket.accept()
        
        if not client_id:
            client_id = str(uuid.uuid4())
        
        self.active_connections[client_id] = websocket
        self.connection_info[client_id] = {
            "connected_at": datetime.now(),
            "requests_count": 0,
            "last_activity": datetime.now()
        }
        
        logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")
        
        await self.send_personal_message({
            "type": "connected",
            "client_id": client_id,
            "server_info": {
                "api_version": "1.0.0",
                "wav2lip_ready": wav2lip_processor is not None,
                "supported_formats": ["wav", "mp3", "jpg", "png"],
                "max_fps": 30,
                "server_time": datetime.now().isoformat()
            }
        }, client_id)
        
        return client_id
    
    def disconnect(self, client_id: str):
        """Disconnect a WebSocket client"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            if client_id in self.connection_info:
                connection_duration = datetime.now() - self.connection_info[client_id]["connected_at"]
                logger.info(f"Client {client_id} disconnected after {connection_duration}. Total connections: {len(self.active_connections)}")
                del self.connection_info[client_id]
    
    async def broadcast(self, message: Dict):
        """Broadcast message to all connected clients"""
        for client_id in list(self.active_connections.keys()):
            await self.send_personal_message(message, client_id)
    
    def _serialize_datetime(self, obj):
        """Custom JSON serializer for datetime objects"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    
    async def send_personal_message(self, message: Dict, client_id: str):
        """Send message to specific client"""
        if client_id in self.active_connections:
            try:
                json_message = json.dumps(message, default=self._serialize_datetime)
                await self.active_connections[client_id].send_text(json_message)
                if client_id in self.connection_info:
                    self.connection_info[client_id]["last_activity"] = datetime.now()
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
    
    def get_connection_stats(self):
        """Get connection statistics"""
        return {
            "total_connections": len(self.active_connections),
            "connections": {
                client_id: {
                    "connected_at": info["connected_at"].isoformat(),
                    "requests_count": info["requests_count"],
                    "last_activity": info["last_activity"].isoformat()
                }
                for client_id, info in self.connection_info.items()
            }
        }

manager = ConnectionManager()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Basic authentication placeholder"""
    return credentials.credentials if credentials else "anonymous"

@app.get("/")
async def serve_web_interface():
    """Serve the main web interface"""
    return FileResponse('static/index.html')

@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "message": "Real-Time Lip-Syncing WebSocket API",
        "version": "1.0.0",
        "status": "running",
        "wav2lip_ready": wav2lip_processor is not None,
        "device": wav2lip_processor.device if wav2lip_processor else "unknown",
        "endpoints": {
            "websocket": "/ws",
            "health": "/health",
            "process": "/process",
            "stats": "/stats",
            "docs": "/docs",
            "web_interface": "/",
            "api_info": "/api"
        },
        "supported_formats": {
            "audio": ["wav", "mp3"],
            "image": ["jpg", "jpeg", "png"],
            "video": ["mp4"]
        }
    }

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "wav2lip_processor": wav2lip_processor is not None,
        "connections": len(manager.active_connections),
        "stats": processing_stats
    }
    
    if wav2lip_processor:
        health_status.update({
            "device": wav2lip_processor.device,
            "checkpoint": wav2lip_processor.checkpoint_path,
            "model_loaded": wav2lip_processor.model_loaded
        })
    
    return health_status

@app.get("/stats")
async def get_stats(user: str = Depends(get_current_user)):
    """Get API statistics"""
    return {
        "processing_stats": processing_stats,
        "connection_stats": manager.get_connection_stats(),
        "system_info": {
            "wav2lip_ready": wav2lip_processor is not None,
            "device": wav2lip_processor.device if wav2lip_processor else "unknown"
        }
    }

@app.post("/process")
async def process_lip_sync(request: Dict[str, Any], user: str = Depends(get_current_user)):
    """
    Enhanced HTTP endpoint for lip-sync processing
    
    Expected request format:
    {
        "audio": "base64_encoded_audio_data",
        "image": "base64_encoded_image_data", 
        "fps": 25.0 (optional),
        "quality": "normal" (optional: "low", "normal", "high")
    }
    """
    global processing_stats
    
    if not wav2lip_processor:
        raise HTTPException(status_code=503, detail="Wav2Lip processor not available")
    
    start_time = time.time()
    processing_stats["total_requests"] += 1
    processing_stats["last_request_time"] = datetime.now().isoformat()
    
    try:
        # Validate input
        audio_base64 = request.get("audio")
        image_base64 = request.get("image")
        fps = request.get("fps", 25.0)
        quality = request.get("quality", "normal")
        
        if not audio_base64 or not image_base64:
            processing_stats["failed_requests"] += 1
            raise HTTPException(status_code=400, detail="Missing audio or image data")
        
        if fps > 30 or fps < 1:
            processing_stats["failed_requests"] += 1
            raise HTTPException(status_code=400, detail="FPS must be between 1 and 30")
        
        if audio_base64.startswith('data:'):
            audio_base64 = audio_base64.split(',')[1]
        if image_base64.startswith('data:'):
            image_base64 = image_base64.split(',')[1]
        
        # Process lip-sync
        logger.info(f"Processing lip-sync request for user {user} - FPS: {fps}, Quality: {quality}")
        result_video_base64 = await asyncio.to_thread(
            wav2lip_processor.process_base64,
            audio_base64,
            image_base64,
            fps
        )
        
        processing_time = time.time() - start_time
        processing_stats["successful_requests"] += 1
        
        # Update average processing time
        if processing_stats["average_processing_time"] == 0.0:
            processing_stats["average_processing_time"] = processing_time
        else:
            processing_stats["average_processing_time"] = (
                processing_stats["average_processing_time"] * 0.8 + processing_time * 0.2
            )
        
        logger.info(f"Lip-sync processing completed in {processing_time:.2f}s")
        
        return {
            "status": "success",
            "video": f"data:video/mp4;base64,{result_video_base64}",
            "processing_time": round(processing_time, 2),
            "fps": fps,
            "quality": quality,
            "metadata": {
                "video_size": len(result_video_base64),
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        processing_stats["failed_requests"] += 1
        logger.error(f"Error processing lip-sync: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Enhanced WebSocket endpoint for real-time lip-sync processing
    
    Message Types:
    
    1. Connection:
       Client -> Server: (automatic on connect)
       Server -> Client: {"type": "connected", "client_id": "...", "server_info": {...}}
    
    2. Ping/Pong:
       Client -> Server: {"type": "ping", "timestamp": 123456789}
       Server -> Client: {"type": "pong", "timestamp": 123456789}
    
    3. Processing:
       Client -> Server: {
           "type": "process",
           "data": {
               "audio": "base64_encoded_audio_data",
               "image": "base64_encoded_image_data",
               "fps": 25.0,
               "quality": "normal"
           }
       }
       
       Server -> Client: {"type": "status", "message": "Processing..."}
       Server -> Client: {
           "type": "result",
           "status": "success|error",
           "data": {"video": "base64_encoded_video_data"},
           "metadata": {...}
       }
    """
    client_id = None
    global processing_stats
    
    try:
        client_id = await manager.connect(websocket)
        
        while True:
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                message_type = message.get("type")
                
                # Update client activity
                if client_id in manager.connection_info:
                    manager.connection_info[client_id]["last_activity"] = datetime.now()
                
                if message_type == "ping":
                    await manager.send_personal_message({
                        "type": "pong", 
                        "timestamp": message.get("timestamp"),
                        "server_time": time.time()
                    }, client_id)
                    continue
                
                elif message_type == "process":
                    if not wav2lip_processor:
                        await manager.send_personal_message({
                            "type": "error",
                            "message": "Wav2Lip processor not available"
                        }, client_id)
                        continue
                    
                    # Update request count
                    if client_id in manager.connection_info:
                        manager.connection_info[client_id]["requests_count"] += 1
                    
                    start_time = time.time()
                    processing_stats["total_requests"] += 1
                    processing_stats["last_request_time"] = datetime.now().isoformat()
                    
                    # Process lip-sync request
                    request_data = message.get("data", {})
                    audio_base64 = request_data.get("audio")
                    image_base64 = request_data.get("image")
                    fps = request_data.get("fps", 25.0)
                    quality = request_data.get("quality", "normal")
                    
                    # Validate input
                    if not audio_base64 or not image_base64:
                        processing_stats["failed_requests"] += 1
                        await manager.send_personal_message({
                            "type": "error",
                            "message": "Missing audio or image data"
                        }, client_id)
                        continue
                    
                    if fps > 30 or fps < 1:
                        processing_stats["failed_requests"] += 1
                        await manager.send_personal_message({
                            "type": "error",
                            "message": "FPS must be between 1 and 30"
                        }, client_id)
                        continue
                    
                    try:
                        # Clean base64 data
                        if audio_base64.startswith('data:'):
                            audio_base64 = audio_base64.split(',')[1]
                        if image_base64.startswith('data:'):
                            image_base64 = image_base64.split(',')[1]
                        
                        await manager.send_personal_message({
                            "type": "status",
                            "message": f"Processing lip-sync (FPS: {fps}, Quality: {quality})...",
                            "progress": 0
                        }, client_id)
                        
                        result_video_base64 = await asyncio.to_thread(
                            wav2lip_processor.process_base64,
                            audio_base64,
                            image_base64,
                            fps
                        )
                        
                        processing_time = time.time() - start_time
                        processing_stats["successful_requests"] += 1
                        
                        # Update average processing time
                        if processing_stats["average_processing_time"] == 0.0:
                            processing_stats["average_processing_time"] = processing_time
                        else:
                            processing_stats["average_processing_time"] = (
                                processing_stats["average_processing_time"] * 0.8 + processing_time * 0.2
                            )
                        
                        await manager.send_personal_message({
                            "type": "result",
                            "status": "success",
                            "data": {
                                "video": f"data:video/mp4;base64,{result_video_base64}",
                                "fps": fps,
                                "quality": quality
                            },
                            "metadata": {
                                "processing_time": round(processing_time, 2),
                                "video_size": len(result_video_base64),
                                "timestamp": datetime.now().isoformat(),
                                "client_id": client_id
                            }
                        }, client_id)
                        
                        logger.info(f"WebSocket lip-sync completed for {client_id} in {processing_time:.2f}s")
                        
                    except Exception as e:
                        processing_stats["failed_requests"] += 1
                        logger.error(f"Error processing lip-sync for {client_id}: {e}")
                        await manager.send_personal_message({
                            "type": "error",
                            "message": f"Processing error: {str(e)}",
                            "error_code": "PROCESSING_FAILED"
                        }, client_id)
                
                elif message_type == "get_stats":
                    await manager.send_personal_message({
                        "type": "stats",
                        "data": {
                            "processing_stats": processing_stats,
                            "your_stats": manager.connection_info.get(client_id, {}),
                            "server_connections": len(manager.active_connections)
                        }
                    }, client_id)
                
                else:
                    await manager.send_personal_message({
                        "type": "error",
                        "message": f"Unknown message type: {message_type}",
                        "error_code": "UNKNOWN_MESSAGE_TYPE"
                    }, client_id)
                    
            except json.JSONDecodeError:
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "error_code": "INVALID_JSON"
                }, client_id)
                
    except WebSocketDisconnect:
        if client_id:
            manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        if client_id:
            manager.disconnect(client_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=True
    ) 