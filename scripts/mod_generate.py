import numpy as np
import h5py
import argparse
import os
from kolsol.numpy.solver import KolSol

def parse_args():
    parser = argparse.ArgumentParser(description='Generate Kolmogorov Flow dataset')
    parser.add_argument('--data-path', type=str, default='./data/kf/results.h5', 
                        help='Path to save the H5 file')
    parser.add_argument('--resolution', type=int, default=64,
                        help='Resolution of the simulation')
    parser.add_argument('--re', type=float, default=14.4,
                        help='Reynolds number')
    parser.add_argument('--time-simulation', type=float, default=10000,
                        help='Total simulation time')
    parser.add_argument('--nf', type=int, default=2,
                        help='Number of forced modes')
    parser.add_argument('--dt', type=float, default=0.01,
                        help='Time step')
    parser.add_argument('--nk', type=int, default=32,
                        help='Number of modes in each dimension')
    parser.add_argument('--ndim', type=int, default=2,
                        help='Number of dimensions')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
    
    print(f"Starting simulation with parameters: RE={args.re}, NF={args.nf}, Resolution={args.resolution}")
    
    # Adjust nk based on resolution if needed
    nk = args.resolution // 2  # Use half the resolution for the number of modes
    
    # Instantiate solver with proper parameters
    ks = KolSol(nk=nk, nf=args.nf, re=args.re, ndim=args.ndim)
    dt = args.dt
    
    # Define initial conditions
    u_hat = ks.random_field(magnitude=1.0, sigma=1.0)
    print("Initial conditions set")
    
    # Calculate number of steps needed
    transient_steps = 1000  # Skip initial transients
    total_steps = int(args.time_simulation / dt)
    
    # Simulate :: run over transients
    print("Running transient simulation...")
    for step in range(transient_steps):
        u_hat += dt * ks.dynamics(u_hat)
        if step % 100 == 0:
            print(f"Transient step {step}/{transient_steps}")
    
    # Prepare storage for results
    times = []
    fields = []
    
    # Simulate :: generate results
    print("Generating results...")
    save_interval = max(1, int(total_steps / 1000))  # Save approximately 1000 frames
    
    for step in range(total_steps):
        u_hat += dt * ks.dynamics(u_hat)
        
        if step % save_interval == 0:
            # Generate physical field
            u_field = ks.fourier_to_phys(u_hat, nref=args.resolution)
            
            # Store the field and time
            fields.append(u_field)
            times.append(step * dt)
            
        if step % (total_steps // 10) == 0:
            print(f"Progress: {step}/{total_steps} steps ({step/total_steps*100:.1f}%)")
    
    # Convert to arrays
    fields = np.array(fields)
    times = np.array(times)
    
    # Save results to H5 file
    print(f"Saving results to {args.data_path}...")
    with h5py.File(args.data_path, 'w') as f:
        f.create_dataset('fields', data=fields)
        f.create_dataset('times', data=times)
        
        # Save metadata
        f.attrs['re'] = args.re
        f.attrs['nf'] = args.nf
        f.attrs['resolution'] = args.resolution
        f.attrs['time_simulation'] = args.time_simulation
    
    print(f"Successfully saved {len(fields)} snapshots to {args.data_path}")

if __name__ == "__main__":
    main()
