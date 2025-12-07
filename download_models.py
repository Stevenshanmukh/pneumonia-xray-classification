#!/usr/bin/env python3
"""
Download pretrained models from GitHub Releases
"""
import os
import requests
from tqdm import tqdm

GITHUB_REPO = "yourusername/pneumonia-xray-classification"  # UPDATE THIS!
RELEASE_TAG = "v1.0"

MODELS = {
    "baseline_cnn_best.pth": {
        "size": "298.50 MB",
        "accuracy": "84.94%"
    },
    "densenet121_best.pth": {
        "size": "80.47 MB",
        "accuracy": "84.13%"
    },
    "efficientnet_best.pth": {
        "size": "46.35 MB",
        "accuracy": "90.87%"
    },
    "vit_best.pth": {
        "size": "977.57 MB",
        "accuracy": "92.31%"
    },
    "pneumonia_classifier_efficientnet.onnx": {
        "size": "15.29 MB",
        "type": "ONNX"
    },
    "pneumonia_classifier_vit.onnx": {
        "size": "327.75 MB",
        "type": "ONNX"
    }
}

def download_file(url, destination):
    """Download file with progress bar"""
    print(f"\nDownloading {os.path.basename(destination)}...")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    with open(destination, 'wb') as file, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    
    print(f"✅ Downloaded: {destination}")

def main():
    print("="*70)
    print("PNEUMONIA CLASSIFICATION - MODEL DOWNLOADER")
    print("="*70)
    
    print("\nAvailable models:")
    print("\nPyTorch Models (.pth):")
    for name, info in MODELS.items():
        if name.endswith('.pth'):
            print(f"  • {name:30s} - {info['size']:10s} (Acc: {info['accuracy']})")
    
    print("\nONNX Models (for deployment):")
    for name, info in MODELS.items():
        if name.endswith('.onnx'):
            print(f"  • {name:50s} - {info['size']}")
    
    print("\n" + "="*70)
    print("DOWNLOAD OPTIONS")
    print("="*70)
    print("\n1. EfficientNet only (recommended for Streamlit)")
    print("2. All PyTorch models")
    print("3. All ONNX models")
    print("4. Everything")
    print("5. Custom selection")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    to_download = []
    
    if choice == "1":
        to_download = ["efficientnet_best.pth", "pneumonia_classifier_efficientnet.onnx"]
    elif choice == "2":
        to_download = [name for name in MODELS.keys() if name.endswith('.pth')]
    elif choice == "3":
        to_download = [name for name in MODELS.keys() if name.endswith('.onnx')]
    elif choice == "4":
        to_download = list(MODELS.keys())
    elif choice == "5":
        print("\nEnter model names (comma-separated):")
        for i, name in enumerate(MODELS.keys(), 1):
            print(f"  {i}. {name}")
        selection = input("\nYour selection: ").strip()
        to_download = [name.strip() for name in selection.split(',') if name.strip() in MODELS]
    else:
        print("Invalid choice!")
        return
    
    print(f"\nWill download {len(to_download)} file(s):")
    for name in to_download:
        print(f"  • {name} ({MODELS[name]['size']})")
    
    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    # Download files
    base_url = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}"
    
    for filename in to_download:
        url = f"{base_url}/{filename}"
        
        if filename.endswith('.pth'):
            destination = f"models/{filename}"
        elif filename.endswith('.onnx'):
            destination = f"deployment/{filename}"
        else:
            destination = filename
        
        try:
            download_file(url, destination)
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
    
    print("\n" + "="*70)
    print("✅ DOWNLOAD COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. For Streamlit: cd deployment && streamlit run streamlit_app.py")
    print("  2. For training: Open Jupyter notebooks")

if __name__ == "__main__":
    main()
