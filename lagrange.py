def lagrange(shares: list[tuple[int, int]], x: int = 0) -> int:
    """
    Reconstrói o segredo a partir das partes usando interpolação de Lagrange.

    :param shares: Uma lista de tuplas (x_i, y_i) representando as partes.
    :param x: O valor de x no qual avaliar o polinômio. 0 para o segredo.
    :return: O segredo reconstruído: int.
    """
    if not shares:
        raise ValueError("A lista de partes não pode estar vazia")
    
    if len(shares) < 2:
        raise ValueError("São necessárias pelo menos 2 partes para reconstruir o segredo")
    
    secret: int = 0
    n = len(shares)

    for i in range(n):
        x_i: int = shares[i][0]
        term = shares[i][1]

        for j in range(n):
            if j != i:
                x_j: int = shares[j][0]
                if x_i == x_j:
                    raise ValueError(f"Valores de x duplicados encontrados: {x_i}")
                
                term *= (x - x_j) / (x_i - x_j)
        
        secret += term

    return round(secret)

if __name__ == "__main__":
    # Exemplo de uso
    partes = [(1, 1234), (2, 38), (3, 91011)]
    segredo = lagrange(partes)
    print(f"O segredo reconstruído é: {segredo}")
    