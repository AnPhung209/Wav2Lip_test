"""
Docker Test Script for Real-Time Lip-Syncing WebSocket API
Tests if the application runs entirely inside Docker
"""

import asyncio
import websockets
import json
import time
import requests

def test_docker_health():
    """Test if the Docker container is healthy"""
    print("🔍 Testing Docker container health...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"Health check passed!")
            print(f"Wav2Lip ready: {data.get('wav2lip_processor', False)}")
            print(f"Device: {data.get('device', 'unknown')}")
            return True
        else:
            print(f"Health check failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_docker_web_interface():
    """Test if the web interface is accessible"""
    print("\nTesting web interface...")
    
    try:
        response = requests.get("http://localhost:8000/", timeout=10)
        if response.status_code == 200 and "Lip Sync Magic" in response.text:
            print("Web interface is accessible!")
            return True
        else:
            print(f"Web interface failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"Web interface failed: {e}")
        return False

async def test_docker_websocket():
    """Test WebSocket connection to Docker container"""
    print("\nTesting WebSocket connection...")
    
    try:
        uri = "ws://localhost:8000/ws"
        async with websockets.connect(uri) as websocket:
            print("WebSocket connected!")
            
            # Wait for welcome message
            welcome = await websocket.recv()
            welcome_data = json.loads(welcome)
            
            if welcome_data.get("type") == "connected":
                print(f"Welcome message received!")
                print(f"Client ID: {welcome_data.get('client_id')}")
                print(f"Server API: {welcome_data.get('server_info', {}).get('api_version')}")
                
                # Test ping
                ping_message = {
                    "type": "ping",
                    "timestamp": time.time()
                }
                await websocket.send(json.dumps(ping_message))
                
                pong = await websocket.recv()
                pong_data = json.loads(pong)
                
                if pong_data.get("type") == "pong":
                    print("Ping/Pong test passed!")
                    return True
                else:
                    print("Ping/Pong test failed")
                    return False
            else:
                print("Invalid welcome message")
                return False
                
    except Exception as e:
        print(f"WebSocket test failed: {e}")
        return False

def test_docker_api_endpoints():
    """Test various API endpoints"""
    print("\nTesting API endpoints...")
    
    endpoints = [
        ("/api", "API info"),
        ("/stats", "Statistics"),
        ("/docs", "Documentation")
    ]
    
    success = True
    for endpoint, name in endpoints:
        try:
            response = requests.get(f"http://localhost:8000{endpoint}", timeout=10)
            if response.status_code == 200:
                print(f"{name} endpoint working")
            else:
                print(f"{name} endpoint failed: HTTP {response.status_code}")
                success = False
        except Exception as e:
            print(f"{name} endpoint failed: {e}")
            success = False
    
    return success

async def main():
    """Run all Docker tests"""
    print("Docker Test Suite for Real-Time Lip-Syncing API")
    print("=" * 60)
    
    tests = [
        ("Health Check", test_docker_health),
        ("Web Interface", test_docker_web_interface),
        ("API Endpoints", test_docker_api_endpoints),
        ("WebSocket", test_docker_websocket)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        if asyncio.iscoroutinefunction(test_func):
            result = await test_func()
        else:
            result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    
    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"   {test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\nOverall Result:")
    if all_passed:
        print("ALL TESTS PASSED - Docker deployment is working perfectly!")
        print("The application runs entirely inside Docker!")
    else:
        print("Some tests failed - Docker deployment needs attention")
    
    return all_passed

if __name__ == "__main__":
    asyncio.run(main()) 