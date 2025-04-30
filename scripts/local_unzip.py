import os
import zipfile
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Unzip Kolmogorov Flow dataset on local machine')
    parser.add_argument('--zip-file', type=str, required=True,
                        help='Path to the downloaded zip file')
    parser.add_argument('--output-dir', type=str, default='./data/kf/',
                        help='Directory to extract files to')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Ensure the zip file exists
    if not os.path.exists(args.zip_file):
        print(f"Error: Zip file '{args.zip_file}' not found.")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    
    # Print zip file info
    zip_size_mb = os.path.getsize(args.zip_file) / (1024 * 1024)
    print(f"Processing zip file: {args.zip_file} ({zip_size_mb:.2f} MB)")
    
    # List contents before extracting
    with zipfile.ZipFile(args.zip_file, 'r') as zipf:
        file_list = zipf.namelist()
        print(f"Found {len(file_list)} files in the archive:")
        for file in file_list:
            info = zipf.getinfo(file)
            size_mb = info.file_size / (1024 * 1024)
            print(f" - {file}: {size_mb:.2f} MB")
    
    # Extract files
    print(f"\nExtracting files to {args.output_dir}...")
    with zipfile.ZipFile(args.zip_file, 'r') as zipf:
        zipf.extractall(args.output_dir)
    
    # Verify extraction
    extracted_files = [f for f in os.listdir(args.output_dir) if os.path.isfile(os.path.join(args.output_dir, f))]
    print(f"Extraction complete. {len(extracted_files)} files extracted to {args.output_dir}")
    
    # Optional: Display info about extracted H5 files
    h5_files = [f for f in extracted_files if f.endswith('.h5')]
    if h5_files:
        print("\nExtracted H5 files:")
        for file in h5_files:
            file_path = os.path.join(args.output_dir, file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f" - {file}: {size_mb:.2f} MB")
            
            # If h5py is available, show basic info about the file
            try:
                import h5py
                with h5py.File(file_path, 'r') as f:
                    print(f"   • Keys: {list(f.keys())}")
                    if 'times' in f:
                        times_obj = f['times']
                        obj_type = type(times_obj).__name__
                        
                        # Just report the type without trying to access attributes
                        print(f"   • Times dataset present (type: {obj_type})")
                        
                        # If it's a group, we can safely list its keys
                        if isinstance(times_obj, h5py.Group):
                            try:
                                group_keys = list(times_obj.keys())
                                print(f"     - Contains {len(group_keys)} entries")
                            except Exception as e:
                                print(f"     - Error accessing group: {e}")                    
                                if 'attrs' in dir(f):
                                    print(f"   • Attributes: {list(f.attrs.keys())}")
            except ImportError:
                print("   • Install h5py to see detailed information about H5 files")
            except Exception as e:
                print(f"   • Error reading H5 file: {e}")

if __name__ == "__main__":
    main()
