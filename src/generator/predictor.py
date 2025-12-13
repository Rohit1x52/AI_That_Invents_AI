import json
import math
import pickle
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

class ParamPredictor:
    def __init__(self, model_path: str = "param_predictor.pkl"):
        self.scale = 1.0
        self.bias = 0.0
        self.is_fitted = False
        self.model_path = model_path
        self.load()

    def _calculate_theoretical_params(self, bp: Dict[str, Any]) -> int:
        total = 0
        in_ch = bp.get("input_shape", [3, 32, 32])[0]
        
        # Stem
        stem_f = bp.get("stem", {}).get("filters", 32)
        total += 3 * 3 * in_ch * stem_f
        in_ch = stem_f
        
        # Stages
        for s in bp.get("stages", []):
            filters = int(s.get("filters", 32))
            depth = int(s.get("depth", 1))
            k = int(s.get("kernel", 3))
            s_type = s.get("type", "conv_block")
            
            # Expansion logic for blocks
            expansion = 4 if s_type == "bottleneck_block" else (6 if s_type == "inverted_residual" else 1)
            
            # Param count for one block
            block_params = 0
            if s_type == "depthwise_conv":
                block_params = (k*k*in_ch) + (in_ch*filters)
            elif s_type == "bottleneck_block":
                mid = filters
                out = filters * expansion
                block_params = (in_ch*mid) + (k*k*mid) + (mid*out) # Simplified 1x1 + 3x3 + 1x1
                if in_ch != out: block_params += in_ch * out # Shortcut projection
            else:
                # Standard Conv
                block_params = k*k*in_ch*filters

            total += block_params
            if depth > 1:
                # Subsequent blocks in stage (in_ch usually equals out_ch now, unless bottleneck)
                # For simplicity in this robust estimator, we assume in_ch updates
                next_in = filters * expansion
                next_block = 0
                if s_type == "bottleneck_block":
                    next_block = (next_in*filters) + (k*k*filters) + (filters*next_in)
                else:
                    next_block = k*k*next_in*filters
                
                total += next_block * (depth - 1)

            in_ch = filters * expansion

        # Head
        num_classes = bp.get("num_classes", 10)
        total += in_ch * num_classes
        
        return int(total)

    def estimate_params(self, bp: Dict[str, Any]) -> int:
        theoretical = self._calculate_theoretical_params(bp)
        
        if not self.is_fitted:
            return theoretical
            
        # Apply Log-Space Correction: log(y) = scale * log(x) + bias
        # y = exp(bias) * x^scale
        try:
            log_t = math.log(max(1, theoretical))
            log_pred = (log_t * self.scale) + self.bias
            return int(math.exp(log_pred))
        except OverflowError:
            return theoretical

    def fit(self, pairs: List[Tuple[Dict, int]]):
        if not pairs: return
        
        # Extract X (theoretical) and Y (actual)
        X = []
        Y = []
        
        for bp, actual in pairs:
            th = self._calculate_theoretical_params(bp)
            if th > 0 and actual > 0:
                X.append(math.log(th))
                Y.append(math.log(actual))
        
        if len(X) < 2: return

        # Simple Linear Regression: Y = aX + b
        n = len(X)
        sum_x = sum(X)
        sum_y = sum(Y)
        sum_xy = sum(x*y for x,y in zip(X,Y))
        sum_xx = sum(x*x for x in X)
        
        denom = (n * sum_xx - sum_x * sum_x)
        if abs(denom) < 1e-9: return

        self.scale = (n * sum_xy - sum_x * sum_y) / denom
        self.bias = (sum_y - self.scale * sum_x) / n
        self.is_fitted = True
        
        self.save()

    def fit_from_dkb(self, dkb_client):
        rows = dkb_client.query_architectures()
        pairs = []
        for r in rows:
            try:
                if r["params"] and r["blueprint_json"]:
                    bp = json.loads(r["blueprint_json"])
                    pairs.append((bp, r["params"]))
            except: 
                continue
        
        if len(pairs) > 5:
            self.fit(pairs)

    def save(self):
        with open(self.model_path, 'wb') as f:
            pickle.dump({"scale": self.scale, "bias": self.bias, "fitted": self.is_fitted}, f)

    def load(self):
        if Path(self.model_path).exists():
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.scale = data["scale"]
                    self.bias = data["bias"]
                    self.is_fitted = data["fitted"]
            except:
                pass