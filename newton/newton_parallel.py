import torch
from typing import List, Tuple, Optional
import numpy as np

def newton_parallel(shares: np.ndarray | List[Tuple[float, float]], x_target: float = 0.0) -> float:
    """
    Calcula o segredo usando a forma de Newton na GPU.
    
    Estratégia de Paralelização:
    Embora o cálculo das ordens de diferenças divididas seja sequencial (ordem 1 -> ordem 2 -> ...),
    o cálculo de todos os elementos DENTRO da mesma ordem pode ser feito em paralelo na GPU.
    
    Ex: Para calcular todas as diferenças de 1ª ordem:
    [(y1-y0)/(x1-x0), (y2-y1)/(x2-x1), (y3-y2)/(x3-x2), ...]
    A GPU executa todas essas divisões simultaneamente.
    """

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    
    # Garantir que a entrada seja um tensor
    if isinstance(shares, np.ndarray):
        shares_tensor = torch.from_numpy(shares).to(dtype=torch.float32, device=device)
    else:
        shares_tensor = torch.tensor(shares, dtype=torch.float32, device=device)
        
    n = shares_tensor.shape[0]
    if n == 0: return 0.0
    X = shares_tensor[:, 0] 

    current_diffs = shares_tensor[:, 1].clone()
    coeffs = torch.empty(n, dtype=torch.float32, device=device)
    coeffs[0] = current_diffs[0]

    for j in range(1, n):
        num_elements = n - j
        numerator = current_diffs[1:num_elements+1] - current_diffs[0:num_elements]

        denominator = X[j:n] - X[0:num_elements]

        new_diffs = numerator / denominator
        current_diffs[:num_elements] = new_diffs
        coeffs[j] = current_diffs[0]
    
    result = coeffs[n-1]
    for i in range(n - 2, -1, -1):
        result = result * (x_target - X[i]) + coeffs[i]

    return result.item()
