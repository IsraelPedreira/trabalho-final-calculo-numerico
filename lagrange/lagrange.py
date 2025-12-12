def lagrange(shares: list[tuple[float, float]], x: float = 0) -> float:
    """
    Versão OTIMIZADA: usa float ao invés de Fraction.
    
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

    secret = 0.0  # float ao invés de Fraction(0)
    n = len(shares)

    for i in range(n):
        x_i = float(shares[i][0])  # float ao invés de Fraction
        term = float(shares[i][1])  # float ao invés de Fraction

        for j in range(n):
            if j != i:
                x_j = float(shares[j][0])
                
                if x_i == x_j:
                    raise ValueError(f"Valores de x duplicados encontrados: {x_i}")

                # Operação de float pura - ~1000x mais rápida
                term *= (x - x_j) / (x_i - x_j)

        secret += term

    return secret


if __name__ == "__main__":
    partes = [(1, 1234), (2, 38), (3, 91011)]
    segredo = lagrange(partes)
    print(f"segredo: {segredo}")
