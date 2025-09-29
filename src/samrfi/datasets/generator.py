"""
High-Level Synthetic Measurement Set Generator

Complete pipeline for generating synthetic measurement sets with realistic RFI
for training SAM-RFI models and benchmarking against other flaggers.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
import json
from datetime import datetime

from .synthetic_ms_legacy import ObservationConfig, RFIConfig, SyntheticVisibilityGenerator
from .ms_writer import MSWriter

logger = logging.getLogger(__name__)


class SyntheticDatasetGenerator:
    """
    Generate complete synthetic datasets for SAM-RFI training
    """
    
    def __init__(self, output_dir: str = "synthetic_datasets"):
        """
        Initialize dataset generator
        
        Args:
            output_dir: Directory for output datasets
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        (self.output_dir / 'measurement_sets').mkdir(exist_ok=True)
        (self.output_dir / 'ground_truth').mkdir(exist_ok=True)
        (self.output_dir / 'configs').mkdir(exist_ok=True)
        (self.output_dir / 'training_data').mkdir(exist_ok=True)
        
        logger.info(f"Synthetic dataset generator initialized: {self.output_dir}")
    
    def generate_training_dataset(self, 
                                dataset_name: str,
                                num_observations: int = 10,
                                obs_config: Optional[ObservationConfig] = None,
                                rfi_configs: Optional[List[RFIConfig]] = None) -> Dict:
        """
        Generate complete training dataset with multiple observations
        
        Args:
            dataset_name: Name for the dataset
            num_observations: Number of synthetic observations
            obs_config: Observation configuration (uses default if None)
            rfi_configs: List of RFI configurations for variety
            
        Returns:
            Dictionary with dataset metadata
        """
        logger.info(f"Generating training dataset '{dataset_name}' with {num_observations} observations")
        
        # Use default configurations if not provided
        if obs_config is None:
            obs_config = self._get_default_obs_config()
        
        if rfi_configs is None:
            rfi_configs = self._get_default_rfi_configs(num_observations)
        
        # Ensure we have enough RFI configs
        while len(rfi_configs) < num_observations:
            rfi_configs.extend(rfi_configs)
        
        dataset_metadata = {
            'dataset_name': dataset_name,
            'creation_time': datetime.now().isoformat(),
            'num_observations': num_observations,
            'observation_config': obs_config.__dict__,
            'observations': []
        }
        
        # Generate each observation
        for i in range(num_observations):
            obs_name = f"{dataset_name}_obs_{i:03d}"
            rfi_config = rfi_configs[i % len(rfi_configs)]
            
            logger.info(f"Generating observation {i+1}/{num_observations}: {obs_name}")
            
            obs_metadata = self.generate_single_observation(
                obs_name, obs_config, rfi_config
            )
            dataset_metadata['observations'].append(obs_metadata)
        
        # Save dataset metadata
        metadata_file = self.output_dir / 'configs' / f"{dataset_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(dataset_metadata, f, indent=2)
        
        logger.info(f"Dataset '{dataset_name}' generated successfully")
        logger.info(f"Metadata saved to: {metadata_file}")
        
        return dataset_metadata
    
    def generate_single_observation(self, 
                                   obs_name: str,
                                   obs_config: ObservationConfig,
                                   rfi_config: RFIConfig) -> Dict:
        """
        Generate a single synthetic observation
        
        Args:
            obs_name: Name for this observation
            obs_config: Observation configuration
            rfi_config: RFI configuration
            
        Returns:
            Observation metadata dictionary
        """
        logger.info(f"Generating observation: {obs_name}")
        
        # Generate synthetic visibilities
        vis_gen = SyntheticVisibilityGenerator(obs_config, rfi_config)
        
        # Create clean visibilities
        clean_vis = vis_gen.generate_clean_visibilities()
        
        # Inject RFI
        corrupted_vis, rfi_mask = vis_gen.inject_rfi(clean_vis)
        
        # Create measurement sets - fresh MSWriter instance for each MS
        
        # # Clean MS (no RFI)
        # clean_ms_path = self.output_dir / 'measurement_sets' / f"{obs_name}_clean.ms"
        # MSWriter(obs_config).create_measurement_set(
        #     str(clean_ms_path), clean_vis, include_rfi_flags=False
        # )
        
        # # Corrupted MS (with RFI, no flags)
        # corrupted_ms_path = self.output_dir / 'measurement_sets' / f"{obs_name}_corrupted.ms"
        # MSWriter(obs_config).create_measurement_set(
        #     str(corrupted_ms_path), corrupted_vis, include_rfi_flags=False
        # )
        
        # Ground truth MS (with RFI flags)
        truth_ms_path = self.output_dir / 'measurement_sets' / f"{obs_name}_truth.ms"
        MSWriter(obs_config).create_measurement_set(
            str(truth_ms_path), corrupted_vis, rfi_mask, include_rfi_flags=True
        )
        
        # Save ground truth flags as numpy arrays
        gt_dir = self.output_dir / 'ground_truth' / obs_name
        gt_dir.mkdir(exist_ok=True)
        
        np.save(gt_dir / 'clean_visibilities.npy', clean_vis)
        np.save(gt_dir / 'corrupted_visibilities.npy', corrupted_vis) 
        np.save(gt_dir / 'rfi_mask.npy', rfi_mask)
        
        # Calculate RFI statistics
        total_points = rfi_mask.size
        flagged_points = np.sum(rfi_mask)
        rfi_fraction = flagged_points / total_points
        
        obs_metadata = {
            'observation_name': obs_name,
            'clean_ms': str(clean_ms_path),
            'corrupted_ms': str(corrupted_ms_path),
            'truth_ms': str(truth_ms_path),
            'ground_truth_dir': str(gt_dir),
            'rfi_statistics': {
                'total_points': int(total_points),
                'flagged_points': int(flagged_points),
                'rfi_fraction': float(rfi_fraction)
            },
            'observation_config': obs_config.__dict__,
            'rfi_config': rfi_config.__dict__
        }
        
        logger.info(f"Observation generated: {rfi_fraction:.3f} RFI fraction")
        return obs_metadata
    
    def create_flagger_comparison_script(self, dataset_name: str) -> str:
        """
        Create script to run other flaggers (aoflagger, CASA) on the dataset
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Path to the generated script
        """
        import inspect
        import subprocess
        import time
        import shutil
        
        # Define the functions that will go in the script
        def run_aoflagger(ms_path, strategy_file=None):
            """Run AOFlagger on measurement set"""
            import subprocess
            import time
            
            cmd = ['aoflagger']
            if strategy_file:
                cmd.extend(['-strategy', strategy_file])
            cmd.append(str(ms_path))
            
            print(f"Running AOFlagger on {ms_path}")
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            duration = time.time() - start_time
            
            return {
                'success': result.returncode == 0,
                'duration': duration,
                'stdout': result.stdout,
                'stderr': result.stderr
            }

        def run_casa_flagger(ms_path, algorithm='tfcrop', **kwargs):
            """Run CASA flagger (tfcrop or rflag)"""
            from pathlib import Path
            
            # Generate CASA script
            casa_script = f'''
vis = "{ms_path}"
flagdata(vis=vis, mode="{algorithm}", **{kwargs})
'''
            
            script_path = f"casa_{algorithm}_{Path(ms_path).stem}.py"
            with open(script_path, 'w') as f:
                f.write(casa_script)
            
            print(f"CASA {algorithm} script generated: {script_path}")
            print(f"Run with: casa --nologger -c {script_path}")
            
            return {'script_path': script_path}

        def compare_flaggers():
            """Compare different flagging algorithms"""
            import json
            import shutil
            from pathlib import Path
            
            # Load dataset metadata
            metadata_file = Path(__file__).parent / 'configs' / f'{dataset_name}_metadata.json'
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            results = {}
            
            for obs_meta in metadata['observations']:
                obs_name = obs_meta['observation_name']
                corrupted_ms = obs_meta['corrupted_ms']
                truth_ms = obs_meta['truth_ms']
                
                print(f"\\nProcessing {obs_name}...")
                
                # Copy MS for each flagger test
                # AOFlagger test
                aof_ms = corrupted_ms.replace('.ms', '_aoflagger.ms')
                shutil.copytree(corrupted_ms, aof_ms, dirs_exist_ok=True)
                aof_result = run_aoflagger(aof_ms)
                
                # CASA tfcrop test  
                tfcrop_ms = corrupted_ms.replace('.ms', '_tfcrop.ms')
                shutil.copytree(corrupted_ms, tfcrop_ms, dirs_exist_ok=True)
                tfcrop_result = run_casa_flagger(tfcrop_ms, 'tfcrop')
                
                # CASA rflag test
                rflag_ms = corrupted_ms.replace('.ms', '_rflag.ms')
                shutil.copytree(corrupted_ms, rflag_ms, dirs_exist_ok=True)
                rflag_result = run_casa_flagger(rflag_ms, 'rflag')
                
                results[obs_name] = {
                    'aoflagger': aof_result,
                    'tfcrop': tfcrop_result,
                    'rflag': rflag_result,
                    'ground_truth_ms': truth_ms
                }
            
            # Save comparison results
            results_file = Path(__file__).parent / 'flagger_comparison_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\\nComparison results saved to: {results_file}")
            print("\\nNext steps:")
            print("1. Run the generated CASA scripts")
            print("2. Use SAM-RFI to analyze all flagged datasets")
            print("3. Compare performance metrics")

        # Write the script file
        script_path = self.output_dir / f"run_flaggers_{dataset_name}.py"
        
        with open(script_path, 'w') as f:
            f.write("#!/usr/bin/env python3\n")
            f.write('"""\n')
            f.write(f'Flagger Comparison Script for {dataset_name}\n')
            f.write('\n')
            f.write('Runs aoflagger, CASA tfcrop, and CASA rflag on synthetic measurement sets\n')
            f.write('for benchmarking against SAM-RFI results.\n')
            f.write('"""\n\n')
            
            # Write the functions using inspect
            f.write(inspect.getsource(run_aoflagger))
            f.write('\n\n')
            f.write(inspect.getsource(run_casa_flagger))
            f.write('\n\n')
            f.write(inspect.getsource(compare_flaggers))
            f.write('\n\nif __name__ == "__main__":\n    compare_flaggers()\n')
        
        # Make executable
        script_path.chmod(0o755)
        
        logger.info(f"Flagger comparison script created: {script_path}")
        return str(script_path)
    
    def export_training_patches(self, 
                               dataset_name: str, 
                               patch_size: int = 256,
                               overlap: int = 64) -> str:
        """
        Export dataset as training patches for SAM-RFI
        
        Args:
            dataset_name: Name of dataset to export
            patch_size: Size of patches to extract
            overlap: Overlap between patches
            
        Returns:
            Path to exported training data
        """
        logger.info(f"Exporting training patches for '{dataset_name}'")
        
        # Load dataset metadata
        metadata_file = self.output_dir / 'configs' / f"{dataset_name}_metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        export_dir = self.output_dir / 'training_data' / dataset_name
        export_dir.mkdir(exist_ok=True)
        
        all_patches = []
        all_masks = []
        
        for obs_meta in metadata['observations']:
            obs_name = obs_meta['observation_name']
            corrupted_ms = obs_meta['corrupted_ms']
            ground_truth_dir = Path(obs_meta['ground_truth_dir'])
            
            logger.info(f"Processing {obs_name} for patch extraction...")
            
            # Load ground truth data
            corrupted_vis = np.load(ground_truth_dir / 'corrupted_visibilities.npy')
            rfi_mask = np.load(ground_truth_dir / 'rfi_mask.npy')
            
            # Extract patches from each baseline/polarization
            for baseline_idx in range(corrupted_vis.shape[0]):
                for pol_idx in range(corrupted_vis.shape[3]):
                    # Get waterfall data [channels, time]
                    vis_data = visibilities[baseline_idx, :, :, time_idx]  # [pol, chan]
                    vis_data = np.array(vis_data, dtype=np.complex64, copy=True)
                    waterfall = np.abs(vis_data).T
                    mask = rfi_mask[baseline_idx, :, :, pol_idx].T
                    
                    # Extract patches
                    patches, positions = self._extract_patches(
                        waterfall, patch_size, overlap
                    )
                    mask_patches, _ = self._extract_patches(
                        mask.astype(np.float32), patch_size, overlap
                    )
                    
                    all_patches.extend(patches)
                    all_masks.extend(mask_patches > 0.5)  # Convert back to boolean
        
        # Save training data
        patches_array = np.stack(all_patches)
        masks_array = np.stack(all_masks)
        
        np.save(export_dir / 'training_patches.npy', patches_array)
        np.save(export_dir / 'training_masks.npy', masks_array)
        
        # Create training metadata
        training_meta = {
            'dataset_name': dataset_name,
            'patch_size': patch_size,
            'overlap': overlap,
            'num_patches': len(all_patches),
            'rfi_fraction': float(np.mean(masks_array)),
            'export_time': datetime.now().isoformat()
        }
        
        with open(export_dir / 'training_metadata.json', 'w') as f:
            json.dump(training_meta, f, indent=2)
        
        logger.info(f"Training patches exported: {len(all_patches)} patches")
        logger.info(f"Export directory: {export_dir}")
        
        return str(export_dir)
    
    def _extract_patches(self, data: np.ndarray, patch_size: int, overlap: int) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """Extract overlapping patches from 2D data"""
        patches = []
        positions = []
        
        step_size = patch_size - overlap
        
        for y in range(0, data.shape[0] - patch_size + 1, step_size):
            for x in range(0, data.shape[1] - patch_size + 1, step_size):
                patch = data[y:y+patch_size, x:x+patch_size]
                patches.append(patch)
                positions.append((y, x))
        
        return patches, positions
    
    def _get_default_obs_config(self) -> ObservationConfig:
        """Get default observation configuration"""
        return ObservationConfig(
            num_antennas=8,
            num_spw=4,
            channels_per_spw=256,
            start_frequency=1.0e9,  # L-band
            channel_width=1e6,
            total_duration=1800.0,  # 30 minutes
            integration_time=10.0
        )
    
    def _get_default_rfi_configs(self, num_configs: int) -> List[RFIConfig]:
        """Get variety of RFI configurations for training diversity"""
        configs = []
        
        # Light RFI scenario
        configs.append(RFIConfig(
            broadband_probability=0.01,
            narrowband_lines=2,
            transient_events=3,
            periodic_signals=1,
            satellite_passes=1
        ))
        
        # Medium RFI scenario
        configs.append(RFIConfig(
            broadband_probability=0.03,
            narrowband_lines=5,
            transient_events=8,
            periodic_signals=3,
            satellite_passes=2
        ))
        
        # Heavy RFI scenario
        configs.append(RFIConfig(
            broadband_probability=0.05,
            narrowband_lines=8,
            transient_events=15,
            periodic_signals=5,
            satellite_passes=3
        ))
        
        # Extend list to match requested number
        while len(configs) < num_configs:
            configs.extend(configs)
        
        return configs[:num_configs]


if __name__ == "__main__":
    # Simple test of the generator
    generator = SyntheticDatasetGenerator("test_output")
    print("SyntheticDatasetGenerator created successfully")