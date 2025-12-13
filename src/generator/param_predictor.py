import math
from typing import Dict, Any, List, Tuple

class ParamPredictor:
    def __init__(self):
        self.correction_factor = 1.0 

    def extract_features(self, bp: Dict[str, Any]) -> Dict[str, float]:
        stages = bp.get("stages", [])
        
        depths = [int(s.get("depth", 1)) for s in stages]
        filters = [int(s.get("filters", 32)) for s in stages]
        kernels = [int(s.get("kernel", 3)) for s in stages]
        
        total_depth = sum(depths)
        max_width = max(filters) if filters else 0
        avg_kernel = sum(kernels) / len(kernels) if kernels else 3.0
        
        bottleneck_count = sum(1 for s in stages if s.get("type") == "bottleneck_block")
        dw_count = sum(1 for s in stages if s.get("type") == "depthwise_conv")
        
        return {
            "depth": float(total_depth),
            "total_filters": float(sum(filters)),
            "max_width": float(max_width),
            "avg_kernel": float(avg_kernel),
            "bottleneck_ratio": bottleneck_count / max(1, len(stages)),
            "depthwise_ratio": dw_count / max(1, len(stages)),
            "input_ch": float(bp.get("input_shape", [3, 32, 32])[0])
        }

    def _calculate_block_params(self, type_str: str, cin: int, cout: int, k: int, expansion: int = 1) -> int:
        if type_str == "depthwise_conv":
            dw = k * k * cin 
            pw = cin * cout
            return dw + pw
            
        elif type_str == "bottleneck_block":
            mid = cout 
            final = cout * 4 
            
            pw1 = cin * mid
            spatial = k * k * mid 
            pw2 = mid * final
            
            shortcut = 0
            if cin != final:
                shortcut = cin * final
                
            return pw1 + spatial + pw2 + shortcut
            
        elif type_str == "inverted_residual":
            hidden_dim = int(round(cin * expansion))
            
            pw1 = cin * hidden_dim
            dw = k * k * hidden_dim 
            pw2 = hidden_dim * cout
            
            return pw1 + dw + pw2

        else:
            return k * k * cin * cout

    def estimate_params(self, bp: Dict[str, Any]) -> int:
        in_ch = bp.get("input_shape", [3, 32, 32])[0]
        total_params = 0
        
        stem_filters = bp.get("stem", {}).get("filters", 32)
        total_params += 3 * 3 * in_ch * stem_filters
        in_ch = stem_filters
        
        for s in bp.get("stages", []):
            s_type = s.get("type", "conv_block")
            filters = int(s.get("filters", 32))
            depth = int(s.get("depth", 1))
            kernel = int(s.get("kernel", 3))
            
            expansion = 1
            if s_type == "inverted_residual":
                expansion = s.get("expansion_ratio", 6)
            
            first_block_params = self._calculate_block_params(s_type, in_ch, filters, kernel, expansion)
            total_params += first_block_params
            
            out_ch = filters
            if s_type == "bottleneck_block":
                out_ch = filters * 4

            if depth > 1:
                subsequent_block_params = self._calculate_block_params(s_type, out_ch, filters, kernel, expansion)
                total_params += subsequent_block_params * (depth - 1)
            
            in_ch = out_ch

        num_classes = bp.get("num_classes", 10)
        total_params += in_ch * num_classes
        
        return int(total_params * self.correction_factor)

    def calibrate(self, history: List[Tuple[Dict, int]]):
        if not history: return
        
        ratios = []
        for bp, actual in history:
            est = self.estimate_params(bp)
            if est > 0:
                ratios.append(actual / est)
        
        if ratios:
            import statistics
            self.correction_factor = statistics.median(ratios)

rule_based_estimator = ParamPredictor()

def rule_based_param_estimate(bp: Dict[str, Any]) -> int:
    return rule_based_estimator.estimate_params(bp)