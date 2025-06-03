"""
Enhanced WebSocket Client for Real-Time Lip-Syncing API Testing
Demonstrates full WebSocket communication capabilities with the enhanced API
"""

import asyncio
import websockets
import json
import base64
import time
import os
from pathlib import Path
from typing import Optional

class LipSyncWebSocketClient:
    """Enhanced WebSocket client for lip-sync API testing"""
    
    def __init__(self, uri: str = "ws://localhost:8000/ws"):
        self.uri = uri
        self.websocket = None
        self.client_id = None
        self.connected = False
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "processing_requests": 0,
            "successful_responses": 0,
            "errors": 0
        }
    
    async def connect(self):
        """Connect to WebSocket server"""
        try:
            print(f"Connecting to {self.uri}...")
            self.websocket = await websockets.connect(self.uri)
            self.connected = True
            print("Connected to WebSocket server")
            
            # Wait for welcome message
            welcome_msg = await self.websocket.recv()
            welcome_data = json.loads(welcome_msg)
            
            if welcome_data.get("type") == "connected":
                self.client_id = welcome_data.get("client_id")
                server_info = welcome_data.get("server_info", {})
                print(f"Welcome! Client ID: {self.client_id}")
                print(f"Server API Version: {server_info.get('api_version')}")
                print(f"Wav2Lip Ready: {server_info.get('wav2lip_ready')}")
                print(f"Supported Formats: {server_info.get('supported_formats')}")
                self.stats["messages_received"] += 1
            
            return True
            
        except Exception as e:
            print(f"Connection failed: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from WebSocket server"""
        if self.websocket and self.connected:
            await self.websocket.close()
            self.connected = False
            print("Disconnected from server")
    
    async def send_message(self, message: dict):
        """Send message to server"""
        if not self.connected:
            print("Not connected to server")
            return False
        
        try:
            await self.websocket.send(json.dumps(message))
            self.stats["messages_sent"] += 1
            return True
        except Exception as e:
            print(f"Failed to send message: {e}")
            return False
    
    async def receive_message(self) -> Optional[dict]:
        """Receive message from server"""
        if not self.connected:
            return None
        
        try:
            message = await self.websocket.recv()
            data = json.loads(message)
            self.stats["messages_received"] += 1
            return data
        except Exception as e:
            print(f"Failed to receive message: {e}")
            return None
    
    async def ping(self):
        """Send ping to server"""
        print("Sending ping...")
        timestamp = time.time()
        
        await self.send_message({
            "type": "ping",
            "timestamp": timestamp
        })
        
        response = await self.receive_message()
        if response and response.get("type") == "pong":
            latency = time.time() - timestamp
            print(f"Pong received! Latency: {latency*1000:.2f}ms")
            return latency
        else:
            print("No pong response received")
            return None
    
    async def get_server_stats(self):
        """Request server statistics"""
        print("Requesting server statistics...")
        
        await self.send_message({"type": "get_stats"})
        
        response = await self.receive_message()
        if response and response.get("type") == "stats":
            stats_data = response.get("data", {})
            print("Server Statistics:")
            print(f"Processing Stats: {stats_data.get('processing_stats', {})}")
            print(f"Your Stats: {stats_data.get('your_stats', {})}")
            print(f"Active Connections: {stats_data.get('server_connections', 0)}")
            return stats_data
        else:
            print("Failed to get server statistics")
            return None
    
    def load_file_as_base64(self, file_path: str) -> Optional[str]:
        """Load file and encode as base64"""
        try:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return None
            
            with open(file_path, "rb") as f:
                file_data = f.read()
                b64_data = base64.b64encode(file_data).decode('utf-8')
                print(f"Loaded {file_path}: {len(file_data)} bytes -> {len(b64_data)} chars")
                return b64_data
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            return None
    
    async def process_lip_sync(self, audio_file: str, image_file: str, fps: float = 25.0, quality: str = "normal"):
        """Process lip-sync with audio and image files"""
        print(f"Starting lip-sync processing...")
        print(f"Audio: {audio_file}")
        print(f"Image: {image_file}")
        print(f"FPS: {fps}, Quality: {quality}")
        
        audio_b64 = self.load_file_as_base64(audio_file)
        image_b64 = self.load_file_as_base64(image_file)
        
        if not audio_b64 or not image_b64:
            print("Failed to load input files")
            return None
        
        # Send processing request
        await self.send_message({
            "type": "process",
            "data": {
                "audio": audio_b64,
                "image": image_b64,
                "fps": fps,
                "quality": quality
            }
        })
        
        self.stats["processing_requests"] += 1
        print("Processing request sent, waiting for response...")
        
        # Wait for response
        while True:
            response = await self.receive_message()
            if not response:
                break
            
            response_type = response.get("type")
            
            if response_type == "status":
                message = response.get("message", "Processing...")
                progress = response.get("progress", 0)
                print(f"Status: {message} ({progress}%)")
            
            elif response_type == "result":
                status = response.get("status")
                if status == "success":
                    print("Processing completed successfully!")
                    
                    data = response.get("data", {})
                    metadata = response.get("metadata", {})
                    
                    video_b64 = data.get("video", "")
                    processing_time = metadata.get("processing_time", 0)
                    video_size = metadata.get("video_size", 0)
                    
                    print(f"Video generated:")
                    print(f"Processing time: {processing_time}s")
                    print(f"Video size: {video_size} characters")
                    print(f"FPS: {data.get('fps', fps)}")
                    print(f"Quality: {data.get('quality', quality)}")
                    
                    # Save result
                    if video_b64.startswith('data:video/mp4;base64,'):
                        video_data = video_b64.split(',')[1]
                    else:
                        video_data = video_b64
                    
                    try:
                        video_bytes = base64.b64decode(video_data)
                        output_file = f"websocket_output_{int(time.time())}.mp4"
                        with open(output_file, "wb") as f:
                            f.write(video_bytes)
                        print(f"Video saved as: {output_file}")
                        
                        self.stats["successful_responses"] += 1
                        return {
                            "success": True,
                            "output_file": output_file,
                            "processing_time": processing_time,
                            "video_size": len(video_bytes)
                        }
                    except Exception as e:
                        print(f"Failed to save video: {e}")
                        self.stats["errors"] += 1
                        return None
                else:
                    print(f"Processing failed: {response.get('message', 'Unknown error')}")
                    self.stats["errors"] += 1
                    return None
            
            elif response_type == "error":
                error_msg = response.get("message", "Unknown error")
                error_code = response.get("error_code", "UNKNOWN")
                print(f"Error [{error_code}]: {error_msg}")
                self.stats["errors"] += 1
                return None
    
    def print_client_stats(self):
        """Print client statistics"""
        print(f"\nClient Statistics:")
        print(f"Messages sent: {self.stats['messages_sent']}")
        print(f"Messages received: {self.stats['messages_received']}")
        print(f"Processing requests: {self.stats['processing_requests']}")
        print(f"Successful responses: {self.stats['successful_responses']}")
        print(f"Errors: {self.stats['errors']}")
        if self.stats['processing_requests'] > 0:
            success_rate = (self.stats['successful_responses'] / self.stats['processing_requests']) * 100
            print(f"   Success rate: {success_rate:.1f}%")

async def run_comprehensive_test():
    """Run comprehensive WebSocket API test"""
    print("Starting Comprehensive WebSocket API Test")
    print("=" * 60)
    
    client = LipSyncWebSocketClient()
    
    try:
        if not await client.connect():
            return
        
        # Test 1: Ping test
        print("\nPing/Pong")
        print("-" * 30)
        latency = await client.ping()
        
        # Test 2: Get server statistics
        print("\nServer Statistics")
        print("-" * 30)
        stats = await client.get_server_stats()
        
        # Test 3: Process lip-sync
        print("\nLip-Sync Processing")
        print("-" * 30)
        
        # Check for test files
        audio_file = "ttsMP3.com_VoiceText_2025-6-3_12-18-2.mp3"
        image_file = "Donald_Trump_official_portrait.jpg"
        
        if os.path.exists(audio_file) and os.path.exists(image_file):
            result = await client.process_lip_sync(
                audio_file=audio_file,
                image_file=image_file,
                fps=25.0,
                quality="normal"
            )
            
            if result and result.get("success"):
                print(f"Test completed successfully!")
                print(f"Output: {result['output_file']}")
                print(f"Processing time: {result['processing_time']}s")
        else:
            print(f"Test files not found:")
            print(f"Audio: {audio_file} {'YES' if os.path.exists(audio_file) else 'NO'}")
            print(f"Image: {image_file} {'YES' if os.path.exists(image_file) else 'NO'}")
        
        # Test 4: Error handling
        print("\nError Handling")
        print("-" * 30)
        
        # Send invalid message
        await client.send_message({"type": "invalid_type"})
        error_response = await client.receive_message()
        if error_response and error_response.get("type") == "error":
            print("Error handling works correctly")
        
        print("\nFinal Statistics")
        print("-" * 30)
        await client.get_server_stats()
        client.print_client_stats()
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed: {e}")
    finally:
        await client.disconnect()

async def run_interactive_mode():
    """Run interactive WebSocket client"""
    print("Interactive WebSocket Client Mode")
    print("Commands: ping, stats, process <audio> <image>, quit")
    print("=" * 60)
    
    client = LipSyncWebSocketClient()
    
    if not await client.connect():
        return
    
    try:
        while True:
            command = input("\n> ").strip().split()
            
            if not command:
                continue
            
            if command[0] == "quit":
                break
            elif command[0] == "ping":
                await client.ping()
            elif command[0] == "stats":
                await client.get_server_stats()
            elif command[0] == "process" and len(command) >= 3:
                audio_file = command[1]
                image_file = command[2]
                fps = float(command[3]) if len(command) > 3 else 25.0
                quality = command[4] if len(command) > 4 else "normal"
                
                await client.process_lip_sync(audio_file, image_file, fps, quality)
            else:
                print("Unknown command. Available: ping, stats, process <audio> <image> [fps] [quality], quit")
    
    except KeyboardInterrupt:
        print("\nInteractive mode interrupted")
    finally:
        await client.disconnect()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        asyncio.run(run_interactive_mode())
    else:
        asyncio.run(run_comprehensive_test()) 