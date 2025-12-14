import numpy as np
from typing import Tuple, List, Union
from fractions import Fraction
from benchmark import StreamingInterpolator

class NewtonStreaming(StreamingInterpolator):
    """Newton com suporte incremental para streaming."""
    
    def __init__(self):
        self.x = []
        self.y = []
        self.coeffs = []
    
    def add_share(self, share: Tuple[float, float]) -> None:
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
    
    def get_secret(self, x_alvo: float = 0.0) -> float:
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
    reconstrutor = NewtonStreaming()
    for i in range(len(shares)):
        reconstrutor.add_share((shares[i][0], shares[i][1]))
    return reconstrutor.get_secret()

class NewtonStreamingFraction(StreamingInterpolator):
    """
    Versão do Newton Incremental usando Aritmética Racional Exata (Fraction).
    Garante erro zero de arredondamento, mas é significativamente mais lento.
    """
    
    def __init__(self):
        self.x = []
        self.y = []
        self.coeffs = [] 
    
    def add_share(self, share: Tuple[float, float]) -> None:
        """Adiciona um share e atualiza incrementalmente usando Fraction."""
        xi = Fraction(share[0])
        yi = Fraction(share[1])
        
        self.x.append(xi)
        self.y.append(yi)
        
        n = len(self.x)
        
        if n == 1:
            self.coeffs = [yi]
        else:
            new_coeff = yi
            for j in range(n - 1):
                numerator = new_coeff - self.coeffs[j]
                denominator = xi - self.x[j]
                new_coeff = numerator / denominator
                
            self.coeffs.append(new_coeff)
    
    def get_secret(self, x_alvo: float = 0.0) -> float:
        """Avalia o polinômio em x_alvo usando aritmética exata."""
        if not self.coeffs:
            return 0.0
    
        target_x = Fraction(x_alvo)
        n = len(self.coeffs)

        segredo = Fraction(0, 1)
        
        for i in range(n):
            term = self.coeffs[i]
            for j in range(i):
                term *= (target_x - self.x[j])
            segredo += term
        
        return float(segredo)
    
    def reset(self):
        self.x = []
        self.y = []
        self.coeffs = []

def newton_fraction(shares: Union[np.ndarray, List[Tuple[float, float]]]) -> float:
    """Wrapper para Newton Fraction no modo batch."""
    reconstrutor = NewtonStreamingFraction()
    for i in range(len(shares)):
        reconstrutor.add_share((shares[i][0], shares[i][1]))
    return reconstrutor.get_secret()