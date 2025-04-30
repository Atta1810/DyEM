import os
import zipfile
import argparse
from google.colab import files
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='Zip and download processed Kolmogorov Flow dataset from Colab')
    parser.add_argument('--data-dir', type=str, default='./data/kf/',
                        help='Directory containing the data files')
    parser.add_argument('--zip-name', type=str, default='kf_dataset.zip',
                        help='Name of the output zip file')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Ensure the data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found.")
        return
    
    # Check if there are files to zip
    files_to_zip = [f for f in os.listdir(args.data_dir) if f.endswith('.h5')]
    if not files_to_zip:
        print(f"No .h5 files found in {args.data_dir}")
        return
    
    # Print file sizes before zipping
    total_size = 0
    print(f"Files to be included in {args.zip_name}:")
    for file in files_to_zip:
        file_path = os.path.join(args.data_dir, file)
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        total_size += size_mb
        print(f" - {file}: {size_mb:.2f} MB")
    print(f"Total uncompressed size: {total_size:.2f} MB")
    
    # Create temporary directory for zipping if needed
    temp_dir = os.path.join(os.getcwd(), 'temp_zip_dir')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    # Copy files to temp directory with a simpler structure
    for file in files_to_zip:
        src_path = os.path.join(args.data_dir, file)
        dst_path = os.path.join(temp_dir, file)
        shutil.copy2(src_path, dst_path)
    
    # Zip the files
    zip_path = os.path.join(os.getcwd(), args.zip_name)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files_to_zip:
            file_path = os.path.join(temp_dir, file)
            arcname = file  # Store with just the filename, not the full path
            print(f"Adding {file} to zip...")
            zipf.write(file_path, arcname=arcname)
    
    # Calculate compressed size
    compressed_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Zip file created: {args.zip_name} ({compressed_size_mb:.2f} MB)")
    print(f"Compression ratio: {compressed_size_mb/total_size:.2f}x")
    
    # Download the zip file
    print("Starting download... (this may take a while depending on the file size)")
    files.download(zip_path)
    print("Download initiated. If it doesn't start automatically, check your browser's download settings.")
    
    # Clean up
    shutil.rmtree(temp_dir)
    print("Temporary directory cleaned up.")

if __name__ == "__main__":
    main()
