"""
Fast rainfall data loader for ANUGA simulations.

Pre-loads all rainfall CSV files for a given date at startup,
avoiding slow line-by-line reading during the evolve loop.
"""

import os
import numpy as np
from glob import glob


class RainfallLoader:
    """Pre-loads and caches rainfall data for fast access during simulation."""

    def __init__(self, date_str, rainfall_factor, base_dir="rainfall_data", verbose=True):
        """
        Initialize rainfall loader.

        Args:
            date_str: Date string in format "DD_MM_YYYY"
            rainfall_factor: Conversion factor for rainfall units
            base_dir: Base directory containing rainfall subdirectories
            verbose: Print loading progress
        """
        self.date_str = date_str
        self.rainfall_factor = rainfall_factor
        self.base_dir = base_dir
        self.verbose = verbose

        # Cache: timestep -> numpy array of rainfall values
        self._cache = {}
        self._source_type = None  # 'gpm', 'gfs', 'imd_wrf', etc.

        # Determine which rainfall source is available and pre-load
        self._detect_and_preload()

    def _detect_and_preload(self):
        """Detect available rainfall source and pre-load all files."""

        # Priority order of rainfall sources (matching original script)
        sources = [
            ('imd_pt_bhub', f"imd/daily/pt_data_bhubaneshwar/{self.date_str}"),
            ('imd_rain25', f"imd/daily/rgdata_rain_25/{self.date_str}"),
            ('imd_wrf', f"imd/wrf/{self.date_str}"),
            ('imd_gfs', f"imd/gfs/{self.date_str}"),
            ('gpm', f"gpm/{self.date_str}"),
            ('gfs', f"gfs/{self.date_str}"),
        ]

        for source_name, subdir in sources:
            source_dir = os.path.join(self.base_dir, subdir)
            if os.path.isdir(source_dir):
                csv_files = glob(os.path.join(source_dir, "*.csv"))
                if csv_files:
                    self._source_type = source_name
                    if self.verbose:
                        print(f"RainfallLoader: Found {len(csv_files)} {source_name} files in {source_dir}")
                    self._preload_files(csv_files, source_name)
                    return

        if self.verbose:
            print("RainfallLoader: No rainfall files found, will use zero rainfall")

    def _preload_files(self, csv_files, source_name):
        """Pre-load all CSV files into memory."""

        for filepath in csv_files:
            # Extract timestep from filename
            # Format: gpm_DD_MM_YYYY_TIMESTEP_0.csv or imd_DD_MM_YYYY_TIMESTEP.csv
            basename = os.path.basename(filepath)
            parts = basename.replace('.csv', '').split('_')

            # Find the timestep - it's after the date parts
            # e.g., gpm_11_09_2021_10800_0 -> timestep = 10800.0
            try:
                # Handle both "10800_0" (10800.0) and "10800" formats
                if len(parts) >= 5:
                    # Try to find timestep after date (DD_MM_YYYY)
                    # gpm_DD_MM_YYYY_TIMESTEP_DECIMAL or imd_DD_MM_YYYY_TIMESTEP
                    timestep_idx = 4  # After prefix and 3 date parts
                    if parts[timestep_idx].isdigit():
                        timestep = float(parts[timestep_idx])
                        if len(parts) > timestep_idx + 1 and parts[timestep_idx + 1].isdigit():
                            # Handle decimal part (10800_0 -> 10800.0)
                            timestep = float(f"{parts[timestep_idx]}.{parts[timestep_idx + 1]}")
                    else:
                        continue
                else:
                    continue
            except (ValueError, IndexError):
                if self.verbose:
                    print(f"RainfallLoader: Could not parse timestep from {basename}")
                continue

            # Load the CSV - just single column of values
            try:
                # Fast numpy load - assumes single column CSV
                data = np.loadtxt(filepath, delimiter=',', usecols=0 if ',' in open(filepath).readline() else None)
                # Apply rainfall factor
                data = data * self.rainfall_factor
                self._cache[timestep] = data

                if self.verbose and len(self._cache) <= 3:
                    print(f"  Loaded t={timestep}: {len(data)} values, "
                          f"min={data.min():.2e}, max={data.max():.2e}")
            except Exception as e:
                if self.verbose:
                    print(f"RainfallLoader: Error loading {filepath}: {e}")

        if self.verbose:
            print(f"RainfallLoader: Pre-loaded {len(self._cache)} timesteps into memory")

    def get_rainfall_at_time(self, t):
        """
        Get rainfall array for a given simulation time.

        Returns:
            numpy array of rainfall values (per vertex), or None if no data
        """
        # Round to nearest available timestep
        if not self._cache:
            return None

        # Find exact or closest timestep
        if t in self._cache:
            return self._cache[t]

        # Find closest timestep that's <= t
        available = sorted(self._cache.keys())
        closest = None
        for ts in available:
            if ts <= t:
                closest = ts
            else:
                break

        if closest is not None:
            return self._cache[closest]

        return None

    def has_data(self):
        """Check if any rainfall data was loaded."""
        return len(self._cache) > 0

    def get_timesteps(self):
        """Get list of available timesteps."""
        return sorted(self._cache.keys())

    @property
    def source_type(self):
        """Get the type of rainfall source being used."""
        return self._source_type


def preload_gpm_rainfall(date_str, rainfall_factor, base_dir="rainfall_data", verbose=True):
    """
    Convenience function to pre-load GPM rainfall data.

    Args:
        date_str: Date in "DD_MM_YYYY" format
        rainfall_factor: Conversion factor (e.g., gpm_rainfall_factor)
        base_dir: Base directory for rainfall data
        verbose: Print progress

    Returns:
        RainfallLoader instance
    """
    return RainfallLoader(date_str, rainfall_factor, base_dir, verbose)
