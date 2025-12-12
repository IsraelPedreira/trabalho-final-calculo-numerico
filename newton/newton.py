from fractions import Fraction


class ReconstrutorNewton:
    def __init__(self):
        self.x = []
        self.y = []
        self.coeffs = []

    def adicionar_share(self, share):
        xi, yi = share
        self.x.append(xi)
        self.y.append(yi)

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
        segredo = 0.0
        x_alvo = 0.0

        for i in range(n):
            termo = self.coeffs[i]

            for j in range(i):
                termo *= x_alvo - self.x[j]

            segredo += termo

        return segredo

    @staticmethod
    def run_newton(shares: list[tuple[int, int]]):
        reconstrutor = ReconstrutorNewton()
        for s in shares:
            reconstrutor.adicionar_share(s)
        return reconstrutor.obter_segredo()


# --- 3. EXECUÇÃO DO BENCHMARK ---
# tamanhos_k = [5, 10, 20, 50, 100]
# tempos_newton = []

# print(f"{'K (Shares)':<10} | {'Newton (ms)':<15} | Segredo")
# print("-" * 35)

# for k in tamanhos_k:
#     shares = gerar_shares(k, k)

#     start = time.time()

#     reconstrutor = ReconstrutorNewton()
#     for s in shares:
#         reconstrutor.adicionar_share(s)

#     res = reconstrutor.obter_segredo()

#     tempos_newton.append((time.time() - start) * 1000)

#     check = "OK" if res == SEGREDO_REAL else "ERRO"

#     print(f"{k:<10} | {tempos_newton[-1]:.4f} ({check}) | {res}")
