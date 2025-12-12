import random
import time
from fractions import Fraction 

SEGREDO_REAL = 12343

def gerar_shares(k, n_total):
    coefs = [SEGREDO_REAL] + [random.randint(1, 100) for _ in range(k - 1)]
    shares = []
    for x in range(1, n_total + 1):
        y = sum([c * (x**i) for i, c in enumerate(coefs)])
        shares.append((x, y))
    return shares

class ReconstrutorNewton:
    def __init__(self):
        self.x = []
        self.y = [] 
        self.coeffs = []

    def adicionar_share(self, share):
        xi, yi = share
        self.x.append(Fraction(xi))
        self.y.append(Fraction(yi))
        
        self.coeffs = self._calcular_tabela_completa()
        return self.obter_segredo()

    def _calcular_tabela_completa(self):
        n = len(self.x)
        coef = list(self.y)
        
        for j in range(1, n):
            for i in range(n - 1, j - 1, -1):
                numerador = coef[i] - coef[i - 1]
                denominador = self.x[i] - self.x[i - j]
                coef[i] = numerador / denominador
                
        return coef

    def obter_segredo(self):
        if not self.coeffs:
            return 0
            
        n = len(self.coeffs)
        segredo = Fraction(0)
        x_alvo = Fraction(0)
        
        for i in range(n):
            termo = self.coeffs[i]
            
            for j in range(i):
                termo *= (x_alvo - self.x[j])
            
            segredo += termo
            
        return segredo

# --- 3. EXECUÇÃO DO BENCHMARK ---
tamanhos_k = [5, 10, 20, 50, 100]
tempos_newton = []

print(f"{'K (Shares)':<10} | {'Newton (ms)':<15} | Segredo")
print("-" * 35)

for k in tamanhos_k:
    shares = gerar_shares(k, k)
    
    start = time.time()
    
    reconstrutor = ReconstrutorNewton()
    for s in shares:
        reconstrutor.adicionar_share(s)
    
    res = reconstrutor.obter_segredo()
    
    tempos_newton.append((time.time() - start) * 1000)

    check = "OK" if res == SEGREDO_REAL else "ERRO"

    print(f"{k:<10} | {tempos_newton[-1]:.4f} ({check}) | {res}")