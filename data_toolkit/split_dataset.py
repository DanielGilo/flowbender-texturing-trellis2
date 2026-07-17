import os
import argparse
import pandas as pd
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='Directory containing metadata.csv')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Ratio of test set (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    metadata_path = os.path.join(args.root, 'metadata.csv')
    if not os.path.exists(metadata_path):
        raise ValueError(f'metadata.csv not found in {args.root}. Please run build_metadata.py first.')

    print(f'Loading metadata from {metadata_path}...')
    metadata = pd.read_csv(metadata_path)
    
    # Ensure sha256 is index for combine_first
    if 'sha256' in metadata.columns:
        metadata.set_index('sha256', inplace=True)

    # Helper to merge sub-metadata
    def merge_sub_metadata(root_dir, sub_path):
        full_path = os.path.join(root_dir, sub_path, 'metadata.csv')
        if os.path.exists(full_path):
            print(f'Merging {full_path}...')
            sub_meta = pd.read_csv(full_path)
            if 'sha256' in sub_meta.columns:
                sub_meta.set_index('sha256', inplace=True)
            return sub_meta
        return None

    # Merge raw (downloads)
    sub = merge_sub_metadata(args.root, 'raw')
    if sub is not None: metadata = metadata.combine_first(sub)

    # Merge latents
    for latent_type in ['pbr_latents', 'shape_latents', 'ss_latents']:
        latent_root = os.path.join(args.root, latent_type)
        if os.path.exists(latent_root):
            for subdir in os.listdir(latent_root):
                if not os.path.isdir(os.path.join(latent_root, subdir)):
                    continue
                sub = merge_sub_metadata(latent_root, subdir)
                if sub is not None: metadata = metadata.combine_first(sub)

    metadata.reset_index(inplace=True)

    if 'split' in metadata.columns:
        print('Metadata already contains "split" column. Overwriting...')
    
    # Determine valid assets based on availability
    valid_mask = None
    if 'pbr_latent_encoded' in metadata.columns:
        valid_mask = metadata['pbr_latent_encoded'].fillna(False).astype(bool)
        count = valid_mask.sum()
        if count > 0:
            print(f"Using assets with PBR latents as base. Found {count} assets.")
        else:
            print("Warning: 'pbr_latent_encoded' column found but no assets are marked as True.")
            valid_mask = None

    if valid_mask is None:
        if 'local_path' in metadata.columns:
            valid_mask = metadata['local_path'].notna()
            print(f"Using downloaded assets as base. Found {valid_mask.sum()} assets.")
        else:
            valid_mask = np.ones(len(metadata), dtype=bool)
            print(f"Using all assets as base. Found {valid_mask.sum()} assets.")

    print(f'Splitting dataset with test ratio {args.test_ratio}...')
    np.random.seed(args.seed)
    
    # Initialize split column (reset to ensure clean state)
    metadata['split'] = None
    
    # Get indices of valid assets
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) > 0:
        # Create a random permutation of valid indices
        perm = np.random.permutation(len(valid_indices))
        test_size = int(len(valid_indices) * args.test_ratio)
        
        # Map back to original dataframe indices
        test_indices = valid_indices[perm[:test_size]]
        train_indices = valid_indices[perm[test_size:]]
        
        # Assign splits
        metadata.loc[train_indices, 'split'] = 'train'
        metadata.loc[test_indices, 'split'] = 'test'
    
    print('Saving updated metadata...')
    metadata.to_csv(metadata_path, index=False)
    
    train_count = (metadata['split'] == 'train').sum()
    test_count = (metadata['split'] == 'test').sum()
    print(f'Done. Train: {train_count}, Test: {test_count}')
