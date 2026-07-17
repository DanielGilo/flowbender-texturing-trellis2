from typing import *
import json
from abc import abstractmethod
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2


class StandardDatasetBase(Dataset):
    """
    Base class for standard datasets.

    Args:
        roots (str): paths to the dataset
    """

    def __init__(self,
        roots: str,
        split: str = 'train',
        **kwargs,
    ):
        super().__init__()
        self.split = split
        try:
            self.roots = json.loads(roots)
            root_type = 'obj'
        except:
            self.roots = roots.split(',')
            root_type = 'list'
        self.instances = []
        self.metadata = pd.DataFrame()
        
        self._stats = {}
        if root_type == 'obj':
            for key, root in self.roots.items():
                self._stats[key] = {}
                metadata = pd.DataFrame(columns=['sha256']).set_index('sha256')
                for _, r in root.items():
                    metadata = metadata.combine_first(pd.read_csv(os.path.join(r, 'metadata.csv')).set_index('sha256'))
                if 'split' in metadata.columns:
                    metadata = metadata[metadata['split'] == self.split]
                self._stats[key]['Total'] = len(metadata)
                metadata, stats = self.filter_metadata(metadata)
                self._stats[key].update(stats)
                self.instances.extend([(root, sha256) for sha256 in metadata.index.values])
                self.metadata = pd.concat([self.metadata, metadata])
        else:
            for root in self.roots:
                key = os.path.basename(root)
                self._stats[key] = {}
                metadata = pd.read_csv(os.path.join(root, 'metadata.csv'))
                if 'split' in metadata.columns:
                    metadata = metadata[metadata['split'] == self.split]
                self._stats[key]['Total'] = len(metadata)
                metadata, stats = self.filter_metadata(metadata)
                self._stats[key].update(stats)
                self.instances.extend([(root, sha256) for sha256 in metadata['sha256'].values])
                metadata.set_index('sha256', inplace=True)
                self.metadata = pd.concat([self.metadata, metadata])
            
    @abstractmethod
    def filter_metadata(self, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        pass
    
    @abstractmethod
    def get_instance(self, root, instance: str) -> Dict[str, Any]:
        pass
        
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index) -> Dict[str, Any]:
        try:
            root, instance = self.instances[index]
            return self.get_instance(root, instance)
        except Exception as e:
            print(f'Error loading {instance}: {e}')
            return self.__getitem__(np.random.randint(0, len(self)))
        
    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Split: {self.split}')
        lines.append(f'  - Total instances: {len(self)}')
        lines.append(f'  - Sources:')
        for key, stats in self._stats.items():
            lines.append(f'    - {key}:')
            for k, v in stats.items():
                lines.append(f'      - {k}: {v}')
        return '\n'.join(lines)


class ImageConditionedMixin:
    def __init__(self, roots, *, image_size=518, **kwargs):
        self.image_size = image_size
        super().__init__(roots, **kwargs)
    
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata['cond_rendered'].notna()]
        stats['Cond rendered'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
       
        image_root = os.path.join(root['render_cond'], instance)
        with open(os.path.join(image_root, 'transforms.json')) as f:
            metadata = json.load(f)
        n_views = len(metadata['frames'])
        view = np.random.randint(n_views)
        frame_metadata = metadata['frames'][view]

        image_path = os.path.join(image_root, frame_metadata['file_path'])
        image = Image.open(image_path)
        W, H = image.size

        if 'envmap_path' in frame_metadata:
            envmap_path = os.path.join(image_root, frame_metadata['envmap_path'])
            if os.path.exists(envmap_path):
                envmap_image = cv2.imread(envmap_path, cv2.IMREAD_UNCHANGED)
                envmap_image = cv2.cvtColor(envmap_image, cv2.COLOR_BGR2RGB)
                pack['cond_envmap'] = torch.tensor(envmap_image, dtype=torch.float32)


        # No cropping, so camera parameters align with the original image.
        center = [W / 2, H / 2]
        hsize = max(W, H) / 2
        aug_hsize = hsize
        aug_center = center

        # Calculate camera matrices
        camera_angle_x = frame_metadata['camera_angle_x']
        f_norm = (W / 2) / (aug_hsize * np.tan(camera_angle_x / 2)) / 2
        cx_norm = (W / 2 - aug_center[0]) / aug_hsize / 2 + 0.5
        cy_norm = -(aug_center[1] - H / 2) / aug_hsize / 2 + 0.5
        
        pack['cond_intrinsics'] = torch.tensor([
            [f_norm, 0, cx_norm],
            [0, f_norm, cy_norm],
            [0, 0, 1]
        ]).float()
        pack['cond_extrinsics'] = torch.linalg.inv(torch.tensor(frame_metadata['transform_matrix']).float())
        pack['cond_extrinsics'][3, :] = torch.tensor([0, 0, 0, 1]).float()
        
         # Coordinate system conversion: Flip Y and Z axes
        # This converts from OpenCV-style (Right, Down, Forward) to OpenGL-style (Right, Up, Back)
        # which is expected by the renderer.
        pack['cond_extrinsics'][1:3, :] *= -1

        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        image = image * alpha.unsqueeze(0)
        pack['cond'] = image
       
        return pack


class MultiImageConditionedMixin:
    def __init__(self, roots, *, image_size=518, max_image_cond_view = 4, **kwargs):
        self.image_size = image_size
        self.max_image_cond_view = max_image_cond_view
        super().__init__(roots, **kwargs)

    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata['cond_rendered'].notna()]
        stats['Cond rendered'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
       
        image_root = os.path.join(root['render_cond'], instance)
        with open(os.path.join(image_root, 'transforms.json')) as f:
            metadata = json.load(f)

        n_views = len(metadata['frames'])
        n_sample_views = np.random.randint(1, self.max_image_cond_view+1)

        assert n_views >= n_sample_views, f'Not enough views to sample {n_sample_views} unique images.'

        sampled_views = np.random.choice(n_views, size=n_sample_views, replace=False)

        cond_images = []
        for v in sampled_views:
            frame_info = metadata['frames'][v]
            image_path = os.path.join(image_root, frame_info['file_path'])
            image = Image.open(image_path)

            alpha = np.array(image.getchannel(3))
            bbox = np.array(alpha).nonzero()
            bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
            center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
            aug_hsize = hsize
            aug_center = center
            aug_bbox = [
                int(aug_center[0] - aug_hsize),
                int(aug_center[1] - aug_hsize),
                int(aug_center[0] + aug_hsize),
                int(aug_center[1] + aug_hsize),
            ]

            img = image.crop(aug_bbox)
            img = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            alpha = img.getchannel(3)
            img = img.convert('RGB')
            img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
            alpha = torch.tensor(np.array(alpha)).float() / 255.0
            img = img * alpha.unsqueeze(0)

            cond_images.append(img)

        pack['cond'] = [torch.stack(cond_images, dim=0)]  # (V,3,H,W)
        return pack
