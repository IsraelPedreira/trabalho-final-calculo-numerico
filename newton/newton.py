import numpy as np
from typing import Tuple

class NewtonIncremental:
    """Newton com suporte incremental para streaming."""
    
    def __init__(self):
        self.x = []
        self.y = []
        self.coeffs = []
    
    def adicionar_share(self, share: Tuple[float, float]) -> None:
        """Adiciona um share e atualiza incrementalmente."""
        xi, yi = float(share[0]), float(share[1])
        self.x.append(xi)
        self.y.append(yi)
        
        n = len(self.x)
        if n == 1:
            self.coeffs = [yi]
        else:
            new_coeff = yi
            for j in range(n - 1):
                numerador = new_coeff - self.coeffs[j]
                denominador = xi - self.x[j]
                new_coeff = numerador / denominador
            self.coeffs.append(new_coeff)
    
    def obter_segredo(self, x_alvo: float = 0.0) -> float:
        """Avalia o polinômio em x_alvo."""
        if not self.coeffs:
            return 0.0
        
        n = len(self.coeffs)
        segredo = 0.0
        
        for i in range(n):
            termo = self.coeffs[i]
            for j in range(i):
                termo *= x_alvo - self.x[j]
            segredo += termo
        
        return segredo
    
    def reset(self):
        """Limpa o estado interno."""
        self.x = []
        self.y = []
        self.coeffs = []


def newton(shares: np.ndarray) -> float:
    """Wrapper para Newton no modo batch (compatível com InterpBenchmark)."""
    reconstrutor = NewtonIncremental()
    for i in range(len(shares)):
        reconstrutor.adicionar_share((shares[i][0], shares[i][1]))
    return reconstrutor.obter_segredo()