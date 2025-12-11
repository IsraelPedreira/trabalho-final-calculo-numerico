import numpy as np
import math
import time
import matplotlib.pyplot as plt


def f(x):
    """Função a ser interpolada."""
    return np.sin(x)


def pontos_uniformes(a, b, n):
    """Gera n pontos igualmente espaçados."""
    return np.linspace(a, b, n)


def pontos_chebyshev(a, b, n):
    """Gera n pontos de Chebyshev no intervalo [a, b]."""
    k = np.arange(n)
    x_cheb = np.cos((2*k + 1) * math.pi / (2*n)) 
    return 0.5*(a+b) + 0.5*(b-a)*x_cheb


def lagrange_interpolar(xs, ys, x0):
    """Avalia interpolação de Lagrange diretamente (O(n^2))."""
    total = 0.0
    n = len(xs)
    for i in range(n):
        li = 1.0
        for j in range(n):
            if i != j:
                li *= np.divide((x0 - xs[j]), (xs[i] - xs[j]))
        total += ys[i] * li
    return total


def newton_coef(xs, ys):
    """Retorna os coeficientes das diferenças divididas (O(n^2))."""
    n = len(xs)
    coef = ys.copy()
    
    for j in range(1, n):
        for i in range(n-1, j-1, -1):
            coef[i] = (coef[i] - coef[i-1])/(xs[i] - xs[i-j])
            
    return coef


def newton_avaliar(xs, coef, x0):
    """Avalia o polinômio de Newton em x0 usando o método de Horner (O(n))."""
    n = len(coef)
    p = coef[n-1]
    for i in range(n-2, -1, -1):
        p = p*(x0 - xs[i]) + coef[i]
    return p


def newton_adicionar_ponto(xs, coef, x_new, y_new):
    """Atualiza coeficientes de Newton com um novo ponto em O(n)."""
    n = len(xs)
    xs_new = list(xs) + [x_new]
    coef_new = list(coef)

    new_coef = y_new 
    
    for i in range(n):
        new_coef = (new_coef - coef[i]) / (x_new - xs[i])

    coef_new.append(new_coef)
    
    return xs_new, coef_new


# ==========================================================
# 4. EXPERIMENTO 1 – ERRO (máx n=40)
# ==========================================================
# ... Inalterado ...
def experimento_erro():
    graus = [5, 10, 20, 40]
    erros_lag = []
    erros_new = []

    a, b = 0, math.pi 

    x_teste = np.linspace(a, b, 2000)
    y_true = f(x_teste)

    for n in graus:
        xs = pontos_chebyshev(a, b, n)
        ys = f(xs)

        y_lag = [lagrange_interpolar(xs, ys, x) for x in x_teste]
        erros_lag.append(np.max(np.abs(y_true - y_lag)))

        coef = newton_coef(xs, ys)
        y_new = [newton_avaliar(xs, coef, x) for x in x_teste]
        erros_new.append(np.max(np.abs(y_true - y_new)))

    return graus, erros_lag, erros_new


# ==========================================================
# 5. EXPERIMENTO 2 – TEMPO (INALTERADO)
# ==========================================================
# ... Inalterado ...
def experimento_tempo():
    graus = [20, 40, 80, 120, 200]
    tempo_lag = []
    tempo_new = []

    a, b = 0, math.pi

    for n in graus:
        xs = pontos_chebyshev(a, b, n)
        ys = f(xs)
        
        x0_testes = [0.1]*500

        ini = time.time()
        for x0 in x0_testes:
            lagrange_interpolar(xs, ys, x0)
        tempo_lag.append((time.time() - ini)*1000)

        coef = newton_coef(xs, ys)
        ini = time.time()
        for x0 in x0_testes:
            newton_avaliar(xs, coef, x0)
        tempo_new.append((time.time() - ini)*1000)

    return graus, tempo_lag, tempo_new


# ==========================================================
# 6. EXPERIMENTO 3 – CUSTO DE INSERÇÃO (AJUSTADO PARA VISUALIZAÇÃO O(n))
# ==========================================================

def experimento_incremental():
    ns = [20, 40, 80]
    tempo_lag = []
    tempo_new = []
    
    # Número de repetições para amortecer o ruído do O(n)
    N_REPETICOES_NEWTON = 100000 

    a, b = 0, math.pi

    for n in ns:
        xs = pontos_chebyshev(a, b, n)
        ys = f(xs)

        x_new = (a+b)/2
        y_new = f(x_new)
        
        # 500 pontos de teste para simular re-interpolação
        x_teste = np.linspace(a, b, 500) 
        
        # ---- Lagrange: custo de reavaliar (O(N_teste * n^2)) -> Sem repetição
        xs_lag = list(xs) + [x_new]
        ys_lag = list(ys) + [y_new]
        
        ini = time.time()
        for x in x_teste:
             lagrange_interpolar(xs_lag, ys_lag, x)
        tempo_lag.append((time.time() - ini)*1000)

        # ---- Newton: atualização incremental (O(n)) -> COM REPETIÇÃO
        coef = newton_coef(xs, ys) 
        
        ini = time.time()
        for _ in range(N_REPETICOES_NEWTON):
            newton_adicionar_ponto(xs, coef, x_new, y_new) 
        
        # O tempo final é o tempo total / número de repetições
        tempo_new_total = (time.time() - ini)*1000 
        tempo_new.append(tempo_new_total / N_REPETICOES_NEWTON)

    return ns, tempo_lag, tempo_new


# ==========================================================
# 7. EXECUÇÃO
# ==========================================================

if __name__ == "__main__":
    
    print("Iniciando Experimentos...")
    
    # -------------------------
    # Experimento 1
    # -------------------------
    graus, eL, eN = experimento_erro()

    print("\n--- Experimento 1 (Erro Máximo) ---")
    print("Graus:", graus)
    print("Lagrange:", [f"{e:.2e}" for e in eL])
    print("Newton:", [f"{e:.2e}" for e in eN])

    plt.figure(figsize=(7, 5))
    plt.semilogy(graus, eL, 'o-', label="Lagrange (erro máximo)")
    plt.semilogy(graus, eN, 'o-', label="Newton (erro máximo)")
    plt.xlabel("Grau do polinômio")
    plt.ylabel("Erro máximo")
    plt.title("Experimento 1 - Erro de Interpolação (Chebyshev)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # -------------------------
    # Experimento 2 (Gráficos Separados)
    # -------------------------
    graus, tL, tN = experimento_tempo()
    
    print("\n--- Experimento 2 (Tempo de Avaliação) ---")
    print("Graus:", graus)
    print("Lagrange (ms):", [f"{t:.2f}" for t in tL])
    print("Newton (ms):", [f"{t:.2f}" for t in tN])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Experimento 2 - Comparação de Tempo de Avaliação")

    ax1.plot(graus, tL, 'o-', color='tab:blue', label="Lagrange (O(n²))")
    ax1.set_xlabel("Grau do polinômio")
    ax1.set_ylabel("Tempo (ms) / 500 avaliações")
    ax1.set_title("Lagrange (Crescimento Quadrático)")
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(graus, tN, 'o-', color='tab:orange', label="Newton (O(n))")
    ax2.set_xlabel("Grau do polinômio")
    ax2.set_ylabel("Tempo (ms) / 500 avaliações")
    ax2.set_title("Newton (Crescimento Linear)")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


    # -------------------------
    # Experimento 3 (Gráficos Separados - Ajustado)
    # -------------------------
    ns, tL2, tN2 = experimento_incremental()

    print("\n--- Experimento 3 (Custo de Inserção) ---")
    print("Tamanho (n):", ns)
    print("Lagrange (Re-avaliação, ms):", [f"{t:.2f}" for t in tL2])
    # Note a mudança de unidade de ms para microssegundos (us) para Newton
    print(f"Newton (Atualização, us): {[f'{t*1000:.2f}' for t in tN2]}")

    fig_3, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))
    fig_3.suptitle("Experimento 3 - Custo de Inserir um Novo Ponto")
    
    # Gráfico Lagrange (Crescimento Alto)
    ax3.plot(ns, tL2, 'o-', color='tab:red', label="Lagrange (Re-interpolação)")
    ax3.set_xlabel("Tamanho do conjunto (n)")
    ax3.set_ylabel("Tempo (ms)")
    ax3.set_title("Lagrange (Crescimento Alto - $O(n^2)$)")
    ax3.grid(True)
    ax3.legend()
    
    # Gráfico Newton (Crescimento Linear - Agora com o tempo em microssegundos)
    # Multiplicamos tN2 por 1000 para converter de ms para microssegundos (us)
    ax4.plot(ns, [t*1000 for t in tN2], 'o-', color='tab:green', label="Newton (Atualização O(n))")
    ax4.set_xlabel("Tamanho do conjunto (n)")
    ax4.set_ylabel("Tempo (µs) / por atualização")
    ax4.set_title("Newton (Crescimento Linear - $O(n)$)")
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()