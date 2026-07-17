import os
import json
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from utils import sphere_hammersley_sequence


BLENDER_LINK = 'https://ftp.halifax.rwth-aachen.de/blender/release/Blender4.5/blender-4.5.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-4.5.1-linux-x64/blender'

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6 libxfixes3 libgl1')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-4.5.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')


def _render_cond(file_path, metadatum, root, num_cond_views, cond_resolution=1024):
    # Compatibility Patch
    if isinstance(metadatum, str):
        sha256 = metadatum
    else:
        sha256 = metadatum['sha256']
    # Build conditional view camera
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_cond_views):
        y, p = sphere_hammersley_sequence(i, num_cond_views, offset)
        yaws.append(y)
        pitchs.append(p)
    fov_min, fov_max = 10, 70
    radius_min = np.sqrt(3) / 2 / np.sin(fov_max / 360 * np.pi)
    radius_max = np.sqrt(3) / 2 / np.sin(fov_min / 360 * np.pi)
    k_min = 1 / radius_max**2
    k_max = 1 / radius_min**2
    ks = np.random.uniform(k_min, k_max, (num_cond_views,))
    radius = [1 / np.sqrt(k) for k in ks]
    fov = [2 * np.arcsin(np.sqrt(3) / 2 / r) for r in radius]
    cond_views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]

    output_folder = os.path.join(root, 'renders_cond', sha256)
    os.makedirs(output_folder, exist_ok=True)
    
    args = [
        BLENDER_PATH, '-b', '-t', '1', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'render_cond.py'),
        '--',
        '--object', os.path.expanduser(file_path),
        '--cond_views', json.dumps(cond_views),
        '--cond_resolution', str(cond_resolution),
        '--cond_output_folder', output_folder,
        '--engine', 'BLENDER_EEVEE_NEXT',
    ]
    if file_path.endswith('.blend'):
        args.insert(1, file_path)
    
    ret = subprocess.run(args, capture_output=True, text=True)
    
    # Check for both process failure and missing output file (silent failure)
    output_file = os.path.join(output_folder, 'transforms.json')
    if ret.returncode != 0 or not os.path.exists(output_file):
        print(f"Blender failed for {sha256} (RC={ret.returncode}):\nSTDOUT:\n{ret.stdout}\nSTDERR:\n{ret.stderr}", file=sys.stderr)
    
    if os.path.exists(output_file):
        return {'sha256': sha256, 'cond_rendered': True}


if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--download_root', type=str, default=None,
                        help='Directory to save the downloaded files')
    parser.add_argument('--render_cond_root', type=str, default=None,
                        help='Directory to save the mesh dumps')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--num_cond_views', type=int, default=16,
                        help='Number of conditional views to render')
    parser.add_argument('--cond_resolution', type=int, default=1024,
                        help='Resolution of the conditional views')
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=8)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))
    opt.download_root = opt.download_root or opt.root
    opt.render_cond_root = opt.render_cond_root or opt.root

    os.makedirs(os.path.join(opt.render_cond_root, 'renders_cond', 'new_records'), exist_ok=True)
    
    # install blender
    print('Checking blender...', flush=True)
    _install_blender()

    # get file list
    if not os.path.exists(os.path.join(opt.root, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv')).set_index('sha256')
    if os.path.exists(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.root, 'aesthetic_scores','metadata.csv')).set_index('sha256'))
    if os.path.exists(os.path.join(opt.download_root, 'raw', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.download_root, 'raw', 'metadata.csv')).set_index('sha256'))
    if os.path.exists(os.path.join(opt.render_cond_root, 'renders_cond', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.render_cond_root, 'renders_cond', 'metadata.csv')).set_index('sha256'))
    metadata = metadata.reset_index()
    if opt.instances is None:
        metadata = metadata[metadata['local_path'].notna()]
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if 'cond_rendered' in metadata.columns:
            metadata = metadata[(metadata['cond_rendered'] != True)]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    # filter out objects that are already processed
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor, \
        tqdm(total=len(metadata), desc="Filtering existing objects") as pbar:
        def check_sha256(sha256):
            if os.path.exists(os.path.join(opt.render_cond_root, 'renders_cond', sha256, 'transforms.json')):
                records.append({'sha256': sha256, 'cond_rendered': True})
            pbar.update()
        executor.map(check_sha256, metadata['sha256'].values)
        executor.shutdown(wait=True)
    existing_sha256 = set(r['sha256'] for r in records)
    metadata = metadata[~metadata['sha256'].isin(existing_sha256)]

    print(f'Processing {len(metadata)} objects...')

    # process objects
    func = partial(_render_cond, root=opt.render_cond_root, num_cond_views=opt.num_cond_views, cond_resolution=opt.cond_resolution)
    
    # Set workers to user input or default to CPU count - 2 (leave room for OS)
    num_workers = opt.max_workers if opt.max_workers else (os.cpu_count() - 2)
    print(f"Starting execution with {num_workers} workers...", flush=True)

    new_results = []
    
    # Use ThreadPoolExecutor (ProcessPool is not needed since subprocess.call releases GIL)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Dictionary to map futures to sha256 for better error reporting
        future_to_sha = {}
        
        # Use itertuples for faster iteration than iterrows
        for row in metadata.itertuples(index=False):
            # Convert namedtuple back to dict-like if needed, or access via attributes
            # Assuming 'local_path' and 'sha256' are columns in metadata
            file_path = os.path.join(opt.download_root, row.local_path)
            
            # Reconstruct the dict for the function (or modify function to accept namedtuple)
            # The simplest way here is passing the row as a dict to match your existing logic
            row_dict = row._asdict() 
            
            future = executor.submit(func, file_path, row_dict)
            future_to_sha[future] = row.sha256

        # Process results as they complete
        for future in tqdm(as_completed(future_to_sha), total=len(future_to_sha), desc='Rendering objects'):
            try:
                res = future.result()
                if res is not None:
                    new_results.append(res)
            except Exception as e:
                sha = future_to_sha[future]
                print(f"Task failed for {sha}: {e}", file=sys.stderr)

    
    # Merge results
    cond_rendered = pd.DataFrame(new_results)
    if not records:
        final_df = cond_rendered
    else:
        # If cond_rendered is empty, just use records
        if cond_rendered.empty:
            final_df = pd.DataFrame.from_records(records)
        else:
            final_df = pd.concat([cond_rendered, pd.DataFrame.from_records(records)])
        
    final_df.to_csv(os.path.join(opt.render_cond_root, 'renders_cond', 'new_records', f'part_{opt.rank}.csv'), index=False)