#!/usr/bin/env python3
"""
Download SMPLest-X model weights from HuggingFace Hub
"""

import os
from huggingface_hub import hf_hub_download

def download_smplest_x_weights():
    """Download SMPLest-X model weights and config files"""
    
    repo_id = "waanqii/SMPLest-X"
    print(f"📥 Downloading SMPLest-X weights from {repo_id}")
    print("=" * 60)
    
    # Define download targets
    downloads = [
        {
            "filename": "smplest_x_h.pth.tar",
            "local_dir": "pretrained_models/smplest_x_h/",
            "description": "SMPLest-X-Huge model weights (8.2GB)"
        },
        {
            "filename": "config_base.py", 
            "local_dir": "pretrained_models/smplest_x_h/",
            "description": "Base configuration file"
        },
        {
            "filename": "hd_sample_humandata.zip",
            "local_dir": "data/",
            "description": "Sample human data"
        }
    ]
    
    success_count = 0
    
    for download in downloads:
        try:
            print(f"\n📦 Downloading {download['description']}...")
            print(f"   File: {download['filename']}")
            print(f"   Destination: {download['local_dir']}")
            
            # Ensure local directory exists
            os.makedirs(download['local_dir'], exist_ok=True)
            
            # Download file to local directory
            print(f"   🔄 Starting download...")
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=download['filename'],
                local_dir=download['local_dir']
            )
            print(f"   🔄 Download completed")
            
            print(f"   ✅ Downloaded to: {downloaded_path}")
            
            # Check file size
            file_size = os.path.getsize(downloaded_path)
            size_mb = file_size / (1024 * 1024)
            print(f"   📊 File size: {size_mb:.1f} MB")
            
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ Failed to download {download['filename']}: {e}")
    
    print(f"\n{'='*60}")
    print(f"📋 Download Summary")
    print(f"{'='*60}")
    print(f"✅ Successfully downloaded: {success_count}/{len(downloads)} files")
    
    if success_count == len(downloads):
        print("🎉 All files downloaded successfully!")
        return True
    else:
        print(f"⚠️  {len(downloads) - success_count} files failed to download")
        return False

def verify_file_structure():
    """Verify that all files are in the correct locations"""
    
    print(f"\n🔍 Verifying file structure...")
    print("=" * 40)
    
    expected_files = [
        "pretrained_models/smplest_x_h/smplest_x_h.pth.tar",
        "pretrained_models/smplest_x_h/config_base.py", 
        "data/hd_sample_humandata.zip"
    ]
    
    all_present = True
    
    for file_path in expected_files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"✅ {file_path} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {file_path} - Missing!")
            all_present = False
    
    print("\n📁 Directory structure:")
    print("pretrained_models/")
    if os.path.exists("pretrained_models"):
        for root, _, files in os.walk("pretrained_models"):
            level = root.replace("pretrained_models", "").count(os.sep)
            indent = "  " * (level + 1)
            print(f"{indent}{os.path.basename(root)}/")
            sub_indent = "  " * (level + 2)
            for file in files:
                file_path = os.path.join(root, file)
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"{sub_indent}{file} ({size_mb:.1f} MB)")
    
    return all_present

if __name__ == "__main__":
    print("🤖 SMPLest-X Weight Downloader")
    print("Powered by HuggingFace Hub")
    print()
    
    # Download weights
    download_success = download_smplest_x_weights()
    
    # Verify structure
    structure_ok = verify_file_structure()
    
    if download_success and structure_ok:
        print("\n🎯 Setup Complete!")
        print("SMPLest-X model weights are ready for use.")
        print()
        print("Next steps:")
        print("1. Download SMPL/SMPL-X body models to human_models/human_model_files/")
        print("2. For training: Download ViTPose weights to pretrained_models/")
        print("3. Run inference: sh scripts/inference.sh smplest_x_h your_video.mp4 30")
    else:
        print("\n⚠️  Setup incomplete. Please check the errors above.")
        exit(1)