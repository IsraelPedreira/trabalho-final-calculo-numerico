import torch
from typing import List, Tuple


def lagrange_parallel(shares: List[Tuple[float, float]], x: float = 0.0) -> float:
    """
    Reconstrói o segredo a partir das partes usando interpolação de Lagrange. VETORIZADO NA GPU.

    :param shares: Uma lista de tuplas (x_i, y_i) representando as partes.
    :param x: O valor de x no qual avaliar o polinômio. 0 para o segredo.
    :return: O segredo reconstruído.
    """

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # print("Device:", device)

    x_vals: List[float] = [s[0] for s in shares]
    y_vals: List[float] = [s[1] for s in shares]

    X: torch.Tensor = torch.tensor(x_vals, device=device, dtype=torch.float32)
    Y: torch.Tensor = torch.tensor(y_vals, device=device, dtype=torch.float32)
    k: int = len(X)

    diff_matrix: torch.Tensor = X.view(-1, 1) - X
    eye: torch.Tensor = torch.eye(k, device=device)
    diff_matrix = diff_matrix + eye  # evita divisão por zero

    numerators: torch.Tensor = x - X
    terms_matrix: torch.Tensor = numerators / diff_matrix

    mask: torch.Tensor = 1 - eye
    terms_matrix = terms_matrix * mask + eye  # substitui diagonal por 1

    L_i: torch.Tensor = torch.prod(terms_matrix, dim=1)
    return torch.sum(Y * L_i).item()


if __name__ == "__main__":

    partes = [(1, 1234), (2, 38), (3, 91011)]
    segredo = lagrange_parallel(partes)
    print(f"segredo: {segredo}")
