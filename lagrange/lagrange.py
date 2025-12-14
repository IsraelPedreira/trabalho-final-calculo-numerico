from typing import List
from fractions import Fraction
from benchmark import StreamingInterpolator

def lagrange(shares: list[tuple[float, float]], x: float = 0) -> float:
    """
    
    Reconstrói o segredo a partir das partes usando interpolação de Lagrange.

    :param shares: Uma lista de tuplas (x_i, y_i) representando as partes.
    :param x: O valor de x no qual avaliar o polinômio. 0 para o segredo.
    :return: O segredo reconstruído.
    """
    if shares is None:
        raise ValueError("A lista de partes não pode estar vazia")

    if len(shares) < 2:
        raise ValueError(
            "São necessárias pelo menos 2 partes para reconstruir o segredo"
        )

    secret = 0.0 
    n = len(shares)

    for i in range(n):
        x_i = float(shares[i][0])  
        term = float(shares[i][1]) 

        for j in range(n):
            if j != i:
                x_j = float(shares[j][0])
                
                if x_i == x_j:
                    raise ValueError(f"Valores de x duplicados encontrados: {x_i}")

                term *= (x - x_j) / (x_i - x_j)

        secret += term

    return secret

def lagrange_fraction(shares: List[tuple[float, float]], x: float = 0) -> float:
    """
    Reconstrói o segredo usando Frações para precisão infinita.
    
    NOTA: Muito mais lento que a versão float, mas elimina o erro de arredondamento.
    """
    if shares is None:
        raise ValueError("A lista de partes não pode estar vazia")

    if len(shares) < 2:
        raise ValueError("São necessárias pelo menos 2 partes")

    secret = Fraction(0, 1)
    
    target_x = Fraction(x)
    
    n = len(shares)

    for i in range(n):
        x_i = Fraction(shares[i][0])
        y_i = Fraction(shares[i][1])
        
        term = y_i

        for j in range(n):
            if j != i:
                x_j = Fraction(shares[j][0])
                
                if x_i == x_j:
                    raise ValueError(f"Valores de x duplicados encontrados: {x_i}")

                # Cálculo exato sem perda de precisão
                term *= (target_x - x_j) / (x_i - x_j)

        secret += term

    return float(secret)

class LagrangeStreaming(StreamingInterpolator):
    """
    Wrapper para simular comportamento de streaming com Lagrange.
    
    NOTA: Como Lagrange não é incremental, esta classe armazena todo o histórico
    e recicla todo o cálculo a cada chamada de get_secret(). Isso é intencionalmente
    ineficiente para fins de comparação com Newton.
    """
    def __init__(self):
        self.shares_history = []

    def add_share(self, share: tuple[float, float]) -> None:
        """Apenas armazena o share na memória."""
        self.shares_history.append(share)

    def get_secret(self) -> float:
        """
        Recalcula o segredo do zero usando todos os shares acumulados.
        """
        if len(self.shares_history) < 2:
            return 0.0
        return lagrange(self.shares_history)

    def reset(self) -> None:
        self.shares_history = []

class LagrangeStreamingFraction(StreamingInterpolator):
    """
    Versão de Lagrange Streaming usando Fractions para precisão infinita.
    
    NOTA: Muito mais lento que a versão float, mas elimina o erro de arredondamento.
    """
    def __init__(self):
        self.shares_history = []

    def add_share(self, share: tuple[float, float]) -> None:
        """Apenas armazena o share na memória."""
        self.shares_history.append(share)

    def get_secret(self) -> float:
        """
        Recalcula o segredo do zero usando todos os shares acumulados com Fractions.
        """
        if len(self.shares_history) < 2:
            return 0.0
        return lagrange_fraction(self.shares_history)

    def reset(self) -> None:
        self.shares_history = []

if __name__ == "__main__":
    partes = [(1, 1234), (2, 38), (3, 91011)]
    segredo = lagrange(partes)
    print(f"segredo: {segredo}")
