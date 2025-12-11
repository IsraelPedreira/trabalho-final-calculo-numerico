import random
import time
from fractions import Fraction

# --- Configuração do Cenário ---
# O polinômio tem grau k-1
SEGREDO_REAL = 1456


def gerar_shares(k, n_total):
    """Gera n_total pedaços (x, y) de um polinômio onde P(0) = SEGREDO"""
    coefs = [SEGREDO_REAL] + [random.randint(1, 100) for _ in range(k - 1)]
    shares = []
    for x in range(1, n_total + 1):
        y = sum([c * (x**i) for i, c in enumerate(coefs)])
        shares.append((x, y))
    return shares


# --- 1. Reconstrução via LAGRANGE ---
def recuperar_lagrange(shares):
    # No SSS, queremos saber P(0)
    x = [s[0] for s in shares]
    y = [s[1] for s in shares]

    segredo = Fraction(0, 1)
    k = len(shares)

    # Alvo é x=0
    target = 0

    for i in range(k):
        term = Fraction(y[i], 1)
        for j in range(k):
            if i != j:
                term *= Fraction(target - x[j], x[i] - x[j])
        segredo += term

    print(f"Segredo: {segredo}")
    return segredo


# --- 3. EXECUÇÃO DO BENCHMARK ---
tamanhos_k = [5, 10, 20, 50, 100]
tempos_lagrange = []
tempos_newton = []

print(f"{'K (Shares)':<10} | {'Lagrange (ms)':<15} | {'Newton (ms)':<15}")
print("-" * 45)

for k in tamanhos_k:
    # Gerar k shares (o mínimo necessário para recuperar)
    # shares = gerar_shares(k, k)
    shares = gerar_shares(k, k)
    x_vals = [s[0] for s in shares]
    y_vals = [s[1] for s in shares]

    # Teste Lagrange
    start = time.time()

    recuperar_lagrange(shares)
    tempos_lagrange.append((time.time() - start) * 1000)


    print(f"{k:<10} | {tempos_lagrange[-1]:.4f}          |")
