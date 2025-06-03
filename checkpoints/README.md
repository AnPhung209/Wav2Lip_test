# Model Checkpoints Directory

This directory contains the AI model files required for lip-syncing. The model files are not included in the Git repository due to their large size (139MB each).

## 📥 **Download Required Models**

You need to download **at least one** of these Wav2Lip models:

### **Option 1: Wav2Lip-SD-NOGAN.pt (Recommended)**
- **Size**: 139MB
- **Quality**: Better accuracy, more stable
- **Use case**: General purpose, better lip-sync accuracy

### **Option 2: Wav2Lip-SD-GAN.pt**  
- **Size**: 139MB
- **Quality**: Better visual quality, more realistic
- **Use case**: When visual quality is more important than accuracy

## 🔗 **Download Links**

Download from the official Wav2Lip google drive:
- [Wav2Lip-SD-NOGAN.pt](https://drive.google.com/drive/folders/153HLrqlBNxzZcHi17PEvP09kkAfzRshM)
- [Wav2Lip-SD-GAN.pt](https://drive.google.com/drive/folders/153HLrqlBNxzZcHi17PEvP09kkAfzRshM)

## 📂 **Installation Instructions**

1. Download one or both model files
2. Place them in this `checkpoints/` directory
3. Verify the files:
   ```bash
   ls -la checkpoints/
   # Should show:
   # Wav2Lip-SD-NOGAN.pt (139MB)
   # Wav2Lip-SD-GAN.pt (139MB) [optional]
   ```

## ✅ **Verification**

After downloading, test the setup:
```bash
python test_model_processing.py
```

The application will automatically detect and load the available model.

---

**Note**: The face detection model (s3fd.pth) downloads automatically when needed. 