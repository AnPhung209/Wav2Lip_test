"""
Test script to verify Wav2Lip model can actually process data
"""

import sys
import os
from pathlib import Path
import base64
from app.models.wav2lip_model import Wav2LipProcessor

# Add wav2lip to path
wav2lip_path = Path("wav2lip")
sys.path.append(str(wav2lip_path))

def test_model_processing():
    """Test if the Wav2Lip model can actually process data"""
    print("🧪 Testing Wav2Lip Model Processing with Real Data")
    print("=" * 50)
    
    try:
        # Initialize processor
        print("Initializing Wav2Lip processor...")
        processor = Wav2LipProcessor()
        print(f"Processor initialized with model: {processor.checkpoint_path}")
        print(f"Using device: {processor.device}")
        
        # Check if provided files exist
        image_file = "Donald_Trump_official_portrait.jpg"
        audio_file = "ttsMP3.com_VoiceText_2025-6-3_12-18-2.mp3"
        
        if not os.path.exists(image_file):
            print(f"Image file not found: {image_file}")
            return False
            
        if not os.path.exists(audio_file):
            print(f"Audio file not found: {audio_file}")
            return False
        
        print(f"Using image: {image_file}")
        print(f"Using audio: {audio_file}")
        
        # Load the actual files
        print("Loading provided files...")
        with open(image_file, "rb") as f:
            image_bytes = f.read()
        
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
        
        print(f"Image size: {len(image_bytes)} bytes ({len(image_bytes)/1024:.1f} KB)")
        print(f"Audio size: {len(audio_bytes)} bytes ({len(audio_bytes)/1024:.1f} KB)")
        
        # Test processing with real data
        print("Testing actual processing with real data...")
        print("This may take a moment as the model loads and processes...")
        print("The model will analyze the face and sync lips to the audio...")
        
        try:
            result_video = processor.process_audio_and_image(
                audio_bytes, 
                image_bytes, 
                fps=25.0
            )
            
            print(f"Processing successful!")
            print(f"Output video size: {len(result_video)} bytes ({len(result_video)/1024:.1f} KB)")
            
            # Save result
            output_filename = "trump_lipsync_output.mp4"
            with open(output_filename, "wb") as f:
                f.write(result_video)
            
            print(f"Result saved as {output_filename}")
            print("You can now play the video to see the lip-syncing result!")
            return True
            
        except Exception as e:
            print(f"Processing failed: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print(f"Full traceback:\n{traceback.format_exc()}")
            return False
        
    except Exception as e:
        print(f"Test failed: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        return False

def test_base64_encoding():
    """Test base64 encoding/decoding of the files"""
    print("\nTesting Base64 Encoding (for WebSocket API)")
    print("=" * 50)
    
    try:
        image_file = "Donald_Trump_official_portrait.jpg"
        audio_file = "ttsMP3.com_VoiceText_2025-6-3_12-18-2.mp3"
        
        # Test base64 encoding
        with open(image_file, "rb") as f:
            image_bytes = f.read()
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        print(f"Image base64 length: {len(image_b64)} characters")
        print(f"Audio base64 length: {len(audio_b64)} characters")
        
        # Test decoding
        decoded_image = base64.b64decode(image_b64)
        decoded_audio = base64.b64decode(audio_b64)
        
        print(f"Decoded image size: {len(decoded_image)} bytes")
        print(f"Decoded audio size: {len(decoded_audio)} bytes")
        
        assert len(decoded_image) == len(image_bytes), "Image decode mismatch"
        assert len(decoded_audio) == len(audio_bytes), "Audio decode mismatch"
        
        print("Base64 encoding/decoding working perfectly")
        return True
        
    except Exception as e:
        print(f"Base64 test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_model_processing()
    
    if success:
        print("\nModel processing test PASSED!")
        print("Wav2Lip is ready for real-time use with actual data")
        
        #Test base64 encoding
        base64_success = test_base64_encoding()
        
        if base64_success:
            print("\nBase64 encoding test PASSED!")
            print("Ready for WebSocket API with base64 data transfer")
        
    else:
        print("\nModel processing test FAILED!")
        print("Check model files and dependencies") 