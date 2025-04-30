import numpy as np
import h5py
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess Kolmogorov Flow dataset')
    parser.add_argument('--data-path', type=str, default='./data/kf/results.h5',
                        help='Path to the input H5 file')
    parser.add_argument('--output-path', type=str, default='./data/kf/processed.h5',
                        help='Path to save the processed H5 file')
    parser.add_argument('--truncate-before', type=float, default=10000,
                        help='Time before which to truncate data')
    parser.add_argument('--downsample-factor', type=int, default=1,
                        help='Factor by which to downsample spatially')
    return parser.parse_args()

def calculate_vorticity(u_field):
    """Calculate vorticity from velocity field.
    Assumes u_field has shape (2, nx, ny) where u_field[0] is u and u_field[1] is v."""
    nx, ny = u_field[0].shape
    
    # Calculate spatial derivatives
    du_dy = np.zeros((nx, ny))
    dv_dx = np.zeros((nx, ny))
    
    # Calculate du/dy
    du_dy[:, 1:-1] = (u_field[0][:, 2:] - u_field[0][:, :-2]) / 2
    du_dy[:, 0] = u_field[0][:, 1] - u_field[0][:, 0]
    du_dy[:, -1] = u_field[0][:, -1] - u_field[0][:, -2]
    
    # Calculate dv/dx
    dv_dx[1:-1, :] = (u_field[1][2:, :] - u_field[1][:-2, :]) / 2
    dv_dx[0, :] = u_field[1][1, :] - u_field[1][0, :]
    dv_dx[-1, :] = u_field[1][-1, :] - u_field[1][-2, :]
    
    # Vorticity = dv/dx - du/dy
    vorticity = dv_dx - du_dy
    
    return vorticity

def downsample(field, factor):
    """Downsample field by averaging over blocks."""
    if factor == 1:
        return field
    
    if len(field.shape) == 2:
        # Single field
        nx, ny = field.shape
        new_nx, new_ny = nx // factor, ny // factor
        result = np.zeros((new_nx, new_ny))
        
        for i in range(new_nx):
            for j in range(new_ny):
                result[i, j] = np.mean(field[i*factor:(i+1)*factor, j*factor:(j+1)*factor])
        
        return result
    elif len(field.shape) == 3:
        # Multiple fields (e.g., velocity components)
        nf, nx, ny = field.shape
        new_nx, new_ny = nx // factor, ny // factor
        result = np.zeros((nf, new_nx, new_ny))
        
        for f in range(nf):
            for i in range(new_nx):
                for j in range(new_ny):
                    result[f, i, j] = np.mean(field[f, i*factor:(i+1)*factor, j*factor:(j+1)*factor])
        
        return result
    else:
        raise ValueError(f"Unexpected field shape: {field.shape}")

def main():
    args = parse_args()
    
    print(f"Reading data from {args.data_path}")
    with h5py.File(args.data_path, 'r') as f:
        fields = f['fields'][:]
        times = f['times'][:]
        
        # Copy metadata
        metadata = {}
        for key in f.attrs.keys():
            metadata[key] = f.attrs[key]
    
    # Find time index to truncate before
    truncate_idx = 0
    if args.truncate_before > 0:
        for i, t in enumerate(times):
            if t >= args.truncate_before:
                truncate_idx = i
                break
    
    print(f"Truncating: keeping {len(times) - truncate_idx} out of {len(times)} snapshots")
    fields = fields[truncate_idx:]
    times = times[truncate_idx:]
    
    # Calculate vorticity
    print("Calculating vorticity...")
    vorticity = np.zeros((len(fields), fields.shape[2], fields.shape[3]))
    for i, field in enumerate(fields):
        vorticity[i] = calculate_vorticity(field)
        if i % 100 == 0:
            print(f"Processed {i}/{len(fields)} snapshots")
    
    # Downsample if needed
    if args.downsample_factor > 1:
        print(f"Downsampling by factor {args.downsample_factor}...")
        downsampled_fields = np.zeros((len(fields), 2, 
                                       fields.shape[2] // args.downsample_factor, 
                                       fields.shape[3] // args.downsample_factor))
        downsampled_vorticity = np.zeros((len(vorticity), 
                                         vorticity.shape[1] // args.downsample_factor, 
                                         vorticity.shape[2] // args.downsample_factor))
        
        for i in range(len(fields)):
            downsampled_fields[i] = downsample(fields[i], args.downsample_factor)
            downsampled_vorticity[i] = downsample(vorticity[i], args.downsample_factor)
            
        fields = downsampled_fields
        vorticity = downsampled_vorticity
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Save processed data
    print(f"Saving processed data to {args.output_path}")
    with h5py.File(args.output_path, 'w') as f:
        f.create_dataset('velocity', data=fields)
        f.create_dataset('vorticity', data=vorticity)
        f.create_dataset('times', data=times)
        
        # Save metadata
        for key, value in metadata.items():
            f.attrs[key] = value
        f.attrs['truncate_before'] = args.truncate_before
        f.attrs['downsample_factor'] = args.downsample_factor
    
    print("Preprocessing complete!")

if __name__ == "__main__":
    main()
