import joblib
import numpy as np
from pathlib import Path

# the model raises some warnings that are not relevant to the user
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message=".*does not have valid feature names.*")


class CoveragePrediction:
    # Grid resolution in meters. Smaller = more accurate but slower to precompute.
    GRID_RESOLUTION = 20  # meters

    def __init__(self,
                 cp_path: str = "coverage_prediction/random_forest_model_esch_belval_100.pkl",
                 qos_path: str = "coverage_prediction/modello_qos_v3.pkl",
                 map_bounds: tuple = None):
        """
        Loads models and precomputes a lookup grid for the entire map.
        At training time, each predict() is O(1) array lookup instead of
        traversing 200 Random Forest trees.

        :param map_bounds: (x_min, y_min, x_max, y_max) in SUMO coordinates.
                           If None, defaults to the Esch-Belval bounding box.
        """
        print("   [CP] Caricamento modelli SINR e QoS...")
        cp_model = joblib.load(str(Path(cp_path)))
        qos_model = joblib.load(str(Path(qos_path)))

        # Use all CPUs only for the one-time grid precomputation
        if hasattr(cp_model, 'n_jobs'):
            cp_model.n_jobs = -1
        if hasattr(qos_model, 'n_jobs'):
            qos_model.n_jobs = -1

        # Default bounding box for Esch-Belval SUMO network (SUMO coordinates).
        # Adjust if the network is larger or positioned differently.
        if map_bounds is None:
            map_bounds = (3000.0, 1000.0, 10000.0, 8000.0)

        x_min, y_min, x_max, y_max = map_bounds
        res = self.GRID_RESOLUTION

        xs = np.arange(x_min, x_max + res, res)
        ys = np.arange(y_min, y_max + res, res)

        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.xs = xs
        self.ys = ys

        # Build flat array of all (x, y) grid points for batch prediction
        xx, yy = np.meshgrid(xs, ys)
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])

        n_pts = len(grid_points)
        print(f"   [CP] Precomputing coverage grid ({len(xs)}x{len(ys)} = {n_pts:,} points at {res}m resolution)...")

        sinr_values = cp_model.predict(grid_points)
        qos_values  = qos_model.predict(grid_points)

        # Reshape into 2D lookup tables: index as [y_idx, x_idx]
        self.sinr_grid = sinr_values.reshape(len(ys), len(xs))
        self.qos_grid  = qos_values.reshape(len(ys), len(xs))

        print(f"   [CP] Grid precomputed! Lookups ora sono O(1) per step.")

        # Free the large models from memory — they are no longer needed
        del cp_model, qos_model

    def _get_indices(self, position):
        """Convert SUMO (x, y) coordinates to grid indices."""
        x, y = position
        x = float(np.clip(x, self.x_min, self.x_max))
        y = float(np.clip(y, self.y_min, self.y_max))
        xi = int((x - self.x_min) / self.GRID_RESOLUTION)
        yi = int((y - self.y_min) / self.GRID_RESOLUTION)
        xi = min(xi, len(self.xs) - 1)
        yi = min(yi, len(self.ys) - 1)
        return yi, xi

    def predict(self, position):
        """Returns SINR at position via O(1) grid lookup."""
        yi, xi = self._get_indices(position)
        return float(self.sinr_grid[yi, xi])

    def predict_qos(self, position):
        """Returns QoS at position via O(1) grid lookup."""
        yi, xi = self._get_indices(position)
        return float(self.qos_grid[yi, xi])