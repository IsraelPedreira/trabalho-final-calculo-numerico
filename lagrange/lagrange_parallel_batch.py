import torch
from typing import Tuple

def lagrange_parallel_batch(shares_batch: torch.Tensor, x_alvo: float = 0.0) -> torch.Tensor:
    """
    Reconstrói o segredo a partir das partes usando interpolação de Lagrange. VETORIZADO NA GPU (BATCHES).

    :param shares: Uma lista de tuplas (x_i, y_i) representando as partes.
    :param x: O valor de x no qual avaliar o polinômio. 0 para o segredo.
    :return: O segredo reconstruído.
    """
    device = shares_batch.device
    _, k, _ = shares_batch.shape

    X = shares_batch[:, :, 0]
    Y = shares_batch[:, :, 1]

    diff_matrix = X.unsqueeze(2) - X.unsqueeze(1)
    eye = torch.eye(k, device=device).unsqueeze(0)
    diff_matrix = diff_matrix + eye

    numerators = (x_alvo - X).unsqueeze(2)
    terms_matrix = numerators / diff_matrix

    mask = 1 - eye
    terms_matrix = terms_matrix * mask + eye

    L_i = torch.prod(terms_matrix, dim=2)
    segredos = torch.sum(Y * L_i, dim=1)

    return segredos

if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = 'cuda'
        print("Device:", DEVICE)
        
        BATCH_SIZE = 10000
        K = 50
        
        dados_x = torch.randn(BATCH_SIZE, K, 1, device=DEVICE)
        dados_y = torch.randn(BATCH_SIZE, K, 1, device=DEVICE)
        batch_input = torch.cat([dados_x, dados_y], dim=2)

        _ = lagrange_parallel_batch(batch_input)
        torch.cuda.synchronize()

        segredos_recuperados = lagrange_parallel_batch(batch_input)
        print(f"Shape da saída: {segredos_recuperados.shape}")
        
    else:
        print("Sem GPU para teste.")
