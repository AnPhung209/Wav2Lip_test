import sys
import cv2
import numpy as np
import torch
from pathlib import Path
import tempfile
import base64
import subprocess

wav2lip_path = Path(__file__).parent.parent.parent / "wav2lip"
sys.path.append(str(wav2lip_path))

import audio as wav2lip_audio
import face_detection
from models import Wav2Lip

class Wav2LipProcessor:
    def __init__(self, checkpoint_path: str = None, device: str = "auto"):
        """
        Initialize Wav2Lip processor
        
        Args:
            checkpoint_path: Path to the Wav2Lip model checkpoint. If None, will auto-detect.
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        self.device = self._get_device(device)
        self.checkpoint_path = checkpoint_path or self._find_model_checkpoint()
        self.model = None
        self.img_size = 96
        self.mel_step_size = 16
        
        # Default parameters
        self.face_det_batch_size = 16
        self.wav2lip_batch_size = 128
        self.pads = [0, 10, 0, 0]
        self.resize_factor = 1
        self.nosmooth = False
        self.model_loaded = False
        
    def _find_model_checkpoint(self) -> str:
        """Auto-detect available Wav2Lip model checkpoint"""
        possible_paths = [
            "checkpoints/Wav2Lip-SD-NOGAN.pt",
            "checkpoints/Wav2Lip-SD-GAN.pt", 
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                print(f"Found Wav2Lip model: {path}")
                return path
        
        raise FileNotFoundError("No Wav2Lip model checkpoint found. Please ensure you have downloaded a model to checkpoints/ or models/ directory.")
    
    def _get_device(self, device: str) -> str:
        """Determine the device to use"""
        if device == "auto":
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _load_model(self):
        """Load the Wav2Lip model"""
        if not self.model_loaded:
            print(f"Loading Wav2Lip model from {self.checkpoint_path}")
            
            try:
                # Try loading as a regular PyTorch checkpoint first
                if self.device == 'cuda':
                    checkpoint = torch.load(self.checkpoint_path, weights_only=False)
                else:
                    checkpoint = torch.load(self.checkpoint_path, 
                                          map_location=self.device,
                                          weights_only=False)
                
                # If it's a regular checkpoint, load it the usual way
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    self.model = Wav2Lip()
                    s = checkpoint["state_dict"]
                    new_s = {}
                    for k, v in s.items():
                        new_s[k.replace('module.', '')] = v
                    self.model.load_state_dict(new_s)
                    self.model = self.model.to(self.device)
                    self.model.eval()
                else:
                    raise ValueError("Not a regular checkpoint, trying TorchScript loading")
                    
            except (RuntimeError, ValueError) as e:
                # Try loading as TorchScript model
                print("Checkpoint appears to be TorchScript, loading with torch.jit.load")
                try:
                    if self.device == 'cuda':
                        self.model = torch.jit.load(self.checkpoint_path)
                    else:
                        self.model = torch.jit.load(self.checkpoint_path, map_location=self.device)
                    
                    self.model = self.model.to(self.device)
                    self.model.eval()
                except Exception as jit_error:
                    raise RuntimeError(f"Failed to load model as both regular checkpoint and TorchScript: {e}, {jit_error}")
            
            self.model_loaded = True
            print("Wav2Lip model loaded successfully")
    
    def _get_smoothened_boxes(self, boxes, T=5):
        """Smooth face detection boxes over time"""
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i : i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes
    
    def _face_detect(self, images):
        """Detect faces in images"""
        detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                               flip_input=False, device=self.device)
        
        batch_size = self.face_det_batch_size
        
        while True:
            predictions = []
            try:
                for i in range(0, len(images), batch_size):
                    batch = np.array(images[i:i + batch_size])
                    predictions.extend(detector.get_detections_for_batch(batch))
                break
            except RuntimeError:
                if batch_size == 1:
                    raise RuntimeError('Image too big to run face detection on GPU')
                batch_size //= 2
                print(f'Recovering from OOM error; New batch size: {batch_size}')
                continue
        
        results = []
        pady1, pady2, padx1, padx2 = self.pads
        
        for rect, image in zip(predictions, images):
            if rect is None:
                raise ValueError('Face not detected! Ensure the image contains a face.')
            
            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)
            
            results.append([x1, y1, x2, y2])
        
        boxes = np.array(results)
        if not self.nosmooth:
            boxes = self._get_smoothened_boxes(boxes, T=5)
        
        results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] 
                  for image, (x1, y1, x2, y2) in zip(images, boxes)]
        
        del detector
        return results
    
    def _datagen(self, frames, mels):
        """Generate data batches for Wav2Lip inference"""
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
        
        # Face detection
        face_det_results = self._face_detect(frames)
        
        for i, m in enumerate(mels):
            idx = i % len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx]
            
            face = cv2.resize(face, (self.img_size, self.img_size))
            
            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)
            
            if len(img_batch) >= self.wav2lip_batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
                
                img_masked = img_batch.copy()
                img_masked[:, self.img_size//2:] = 0
                
                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
                
                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
        
        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
            
            img_masked = img_batch.copy()
            img_masked[:, self.img_size//2:] = 0
            
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            
            yield img_batch, mel_batch, frame_batch, coords_batch
    
    def _find_ffmpeg(self):
        """Try to find FFmpeg executable"""
        possible_paths = [
            "ffmpeg",
            "C:\\ffmpeg\\bin\\ffmpeg.exe",
            "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe",
            "ffmpeg.exe"
        ]
        
        for ffmpeg_path in possible_paths:
            try:
                result = subprocess.run([ffmpeg_path, '-version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return ffmpeg_path
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        raise FileNotFoundError(
            "FFmpeg not found. Please install FFmpeg and ensure it's in your PATH, "
            "or install it to C:\\ffmpeg\\bin\\"
        )
    
    def process_audio_and_image(self, audio_data: bytes, image_data: bytes, fps: float = 25.0) -> bytes:
        """
        Process audio and image to generate lip-synced video
        Args:
            audio_data: Audio data in bytes
            image_data: Image data in bytes
            fps: Frames per second for output video
        Returns:
            Video data in bytes (MP4 format)
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            
            # Save input files
            audio_path = temp_dir / "input_audio.wav"
            image_path = temp_dir / "input_image.jpg"
            output_video_path = temp_dir / "output_video.mp4"
            temp_video_path = temp_dir / "temp_video.avi"
            
            with open(audio_path, 'wb') as f:
                f.write(audio_data)
            with open(image_path, 'wb') as f:
                f.write(image_data)
            
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError("Could not load image")
            
            frames = [image]
            
            # Process audio
            wav = wav2lip_audio.load_wav(str(audio_path), 16000)
            mel = wav2lip_audio.melspectrogram(wav)
            
            if np.isnan(mel.reshape(-1)).sum() > 0:
                raise ValueError('Mel contains nan! Try adding small epsilon noise to the audio.')
            
            # Generate mel chunks
            mel_chunks = []
            mel_idx_multiplier = 80. / fps
            i = 0
            while True:
                start_idx = int(i * mel_idx_multiplier)
                if start_idx + self.mel_step_size > len(mel[0]):
                    mel_chunks.append(mel[:, len(mel[0]) - self.mel_step_size:])
                    break
                mel_chunks.append(mel[:, start_idx : start_idx + self.mel_step_size])
                i += 1
            
            self._load_model()
            
            # Generate video
            frame_h, frame_w = frames[0].shape[:-1]
            out = None
            
            try:
                out = cv2.VideoWriter(str(temp_video_path), 
                                    cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
                
                gen = self._datagen(frames.copy(), mel_chunks)
                
                for img_batch, mel_batch, frame_batch, coords_batch in gen:
                    img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
                    mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)
                    
                    with torch.no_grad():
                        pred = self.model(mel_batch, img_batch)
                    
                    pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
                    
                    for p, f, c in zip(pred, frame_batch, coords_batch):
                        y1, y2, x1, x2 = c
                        p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                        f[y1:y2, x1:x2] = p
                        out.write(f)
                
            finally:
                if out is not None:
                    out.release()
                    out = None
            
            # Convert to MP4 with audio using FFmpeg
            try:
                ffmpeg_path = self._find_ffmpeg()
                command = [
                    ffmpeg_path, '-y', '-i', str(audio_path), '-i', str(temp_video_path),
                    '-strict', '-2', '-q:v', '1', str(output_video_path)
                ]
                
                subprocess.run(command, check=True, capture_output=True)
                
                # Read output video
                with open(output_video_path, 'rb') as f:
                    return f.read()
                    
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                print("Returning video without audio sync")
                with open(temp_video_path, 'rb') as f:
                    return f.read()
    
    def process_base64(self, audio_base64: str, image_base64: str, fps: float = 25.0) -> str:
        """
        Process base64 encoded audio and image
        Args:
            audio_base64: Base64 encoded audio data
            image_base64: Base64 encoded image data
            fps: Frames per second
        Returns:
            Base64 encoded video data
        """
        audio_data = base64.b64decode(audio_base64)
        image_data = base64.b64decode(image_base64)
        
        video_data = self.process_audio_and_image(audio_data, image_data, fps)
        
        return base64.b64encode(video_data).decode('utf-8') 