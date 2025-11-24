from typing import Dict, Optional, List, Tuple
import math

class ParamPredictor:
    def __init__(self):
        # simple learned bias/scale (trained via fit); fallback to heuristic
        self._scale = 1.0
        self._bias = 0.0
        self._trained = False

    @staticmethod
    def heuristic(bp: Dict) -> int:
        # same crude heuristic used earlier (total_filters * depth * 10)
        stages = bp.get("stages", [])
        depth = sum(int(s.get("depth", 1)) for s in stages)
        total_filters = sum(int(s.get("filters", 0) or 0) for s in stages)
        est = int(total_filters * max(1, depth) * 10)
        return max(0, est)

    def estimate_params(self, bp: Dict) -> int:
        h = self.heuristic(bp)
        if self._trained:
            return max(0, int(h * self._scale + self._bias))
        return h

    def fit_from_pairs(self, pairs: List[Tuple[int,int]]):
        if not pairs:
            return
        xs = [float(x) for x, y in pairs]
        ys = [float(y) for x, y in pairs]
        n = len(xs)
        mean_x = sum(xs)/n
        mean_y = sum(ys)/n
        num = sum((xs[i]-mean_x)*(ys[i]-mean_y) for i in range(n))
        den = sum((xs[i]-mean_x)**2 for i in range(n)) or 1.0
        self._scale = num/den
        self._bias = mean_y - self._scale*mean_x
        self._trained = True

    def fit_from_dkb(self, dkb_client):
        # collect pairs (heuristic_est, true_params) for rows with params not null
        pairs = []
        rows = dkb_client.query_architectures()
        for r in rows:
            try:
                true = r.get("params")
                if true is None:
                    continue
                bp = r.get("blueprint_json")
                if isinstance(bp, str):
                    import json
                    bp = json.loads(bp)
                est = self.heuristic(bp)
                pairs.append((est, int(true)))
            except Exception:
                continue
        if pairs:
            self.fit_from_pairs(pairs)
