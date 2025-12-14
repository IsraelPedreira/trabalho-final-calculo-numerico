import json
import os
from datetime import datetime
from typing import Any, Optional, Callable, Type
from time import perf_counter
from matplotlib import pyplot as plt
import numpy as np

class StreamingInterpolator:
    def add_share(self, share: tuple[float, float]) -> None:
        raise NotImplementedError
    def get_secret(self) -> float:
        raise NotImplementedError
    def reset(self) -> None:
        raise NotImplementedError

class InterpBenchmark:
    def __init__(self, segredo_real: Optional[int] = None) -> None:
        self.segredo_real = segredo_real

    def run_for_one_fn(self, shares: np.ndarray, interp_fn: Callable) -> dict[str, float]:
        metrics = {
            "exec_time_ms": float("inf"),
            "output": float("inf"),
            "error": float("inf") 
        }
        
        start_time = perf_counter()
        output = interp_fn(shares)
        end_time = perf_counter()

        metrics["exec_time_ms"] = end_time - start_time
        metrics["output"] = output
        
        if self.segredo_real is not None:
            metrics["error"] = abs(output - self.segredo_real)

        return metrics

    def run(
        self,
        interp_fns: list[Callable],
        k_values: list[int],
        segredo: Optional[int] = None,
        save_to_file: Optional[str] = None,
        plot_results: bool = True,
        metrics_to_plot: list[str] = ["exec_time_ms"],
    ) -> tuple[dict, str]:
        """
        Executa uma benchmark de tempo de execução de todas as funções em interp_fns.

        :param interp_fns: List com funções de interpolação a serem avaliadas. Elas recebem uma lista de pontos (shares) como entrada.
        :type interp_fns: list[Callable]
        :param k_values: Quantidade de shares.
        :type k_values: list[int]
        :param segredo: Segredo a ser solucionado pelos interpoladores.
        :type segredo: Optional[int]
        :param save_to_file: Diretório para resultados para arquivo em formato json.
        :type save_to_file: Optional[str]
        :param plot_results: Salvar gráficos dos resultados ou não.
        :type plot_results: bool
        :param metrics_to_plot: Quais métricas dos resultados devem ser "plottadas".
        :type metrics_to_plot: list[str]
        :return: Retorna um dicionário com os resultados e o caminho para o arquivo json gerado.
        :rtype: tuple[dict[Any, Any], str]
        """
        evaluations = {f.__name__: dict.fromkeys(k_values) for f in interp_fns}
        results_file = ""

        print(f"[!] Evaluating {len(interp_fns)} functions.")
        
        print("[!] Performing warm-up...")
        warmup_shares = InterpBenchmark.generate_shares(k=min(k_values) if k_values else 3)
        for fn in interp_fns:
            try:
                print(f"  => Warming up {fn.__name__}...", end=" ")
                fn(warmup_shares)
                print("done.")
            except Exception as e:
                print(f"skipped ({type(e).__name__}).")
        print("  => Warm-up completed.\n")

        for k in k_values:
            shares = InterpBenchmark.generate_shares(
                k=k,
            )

            for fn_name, fn in zip(evaluations.keys(), interp_fns):
                print(f" => Evaluating {fn_name}... ", end="")
                start_time = perf_counter()
                results = self.run_for_one_fn(
                    interp_fn=fn,
                    shares=shares,
                )
                end_time = perf_counter()
                evaluations[fn_name][k] = results

                print(f"finished in {end_time - start_time}ms.")

        if save_to_file is not None:
            results_file = InterpBenchmark.save_results(save_to_file, evaluations)

        if plot_results:
            InterpBenchmark.plot_results(
                results=evaluations,
                output_path=os.path.join(save_to_file or ".", "plots"),
                metrics_to_plot=metrics_to_plot,
            )

        return evaluations, results_file

    @staticmethod
    def generate_shares(k: int, secret: float = 42.0, degree: Optional[int] = None) -> np.ndarray:
        if degree is None or degree < 1:
            degree = k-1
        
        indices = np.arange(1, k + 1)
        x = np.cos((2 * indices - 1) * np.pi / (2 * k))

        y = secret + (x ** degree)
        
        return np.column_stack((x, y))

    @staticmethod
    def save_results(fpath: str, content: Any) -> str:
        output_path = (
            fpath
            if fpath.endswith(".json")
            else os.path.join(
                fpath, datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + ".json"
            )
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print("[!] Saving results to file:", output_path)
        with open(output_path, "w+") as f:
            f.write(json.dumps(content))
        print("  => Saved.")

        return output_path

    @staticmethod
    def plot_results(
        results: str | dict,
        output_path: str = "./results/plots",
        metrics_to_plot: list[str] = ["exec_time_ms"],
    ) -> None:
        if isinstance(results, str):
            with open(results, 'r') as f:
                results = json.loads(f.read())

        print("[!] Plotting results...")
        os.makedirs(output_path, exist_ok=True)
        
        for metric in metrics_to_plot:
            plt.figure(figsize=(10, 6))
            fpath = os.path.join(output_path, f"{metric}")
            
            for method in results.keys():
                raw_keys = results[method].keys()
                xs = sorted([int(k) for k in raw_keys])
                
                ys = []
                for k in xs:
                    if k in results[method]:
                        val = results[method][k][metric]
                    elif str(k) in results[method]:
                        val = results[method][str(k)][metric]
                    else:
                        print(f"Warning: Key {k} not found for method {method}")
                        continue
                    ys.append(val)

                plt.style.use("ggplot")
                plt.plot(xs, ys, marker='o', label=method)

            plt.xlabel("Número de Shares (K)")
            plt.ylabel("Tempo de Execução (s)")
            plt.title(f"Comparação: {metric}")
        
            if "error" in metric.lower():
                plt.yscale("log")
                plt.ylabel(f"Erro Absoluto (Escala Log)")

            plt.legend()
            plt.grid(True)
            plt.savefig(fpath + ".png")
            plt.close()
            print(f"  => Saved {metric} plot to {fpath}.png")

    def run_streaming(
        self,
        interpolator_classes: list[Type[StreamingInterpolator]],
        max_k: int,
        step: int = 1,
        save_to_file: Optional[str] = None,
        plot_results: bool = True
    ) -> tuple[dict, str]:
        
        all_shares = self.generate_shares(max_k, secret=self.segredo_real)
    
        evaluations = {cls.__name__: {"x": [], "y_time": [], "y_error": []} for cls in interpolator_classes}
        
        print(f"[!] Iniciando Benchmark de Streaming (Max K={max_k})...")

        for cls in interpolator_classes:
            print(f" => Testando {cls.__name__}...", end=" ", flush=True)
            
            interpolator = cls()
            
            total_time = 0.0
            x_axis = []
            y_times = []
            y_errors = []
            
            for i in range(max_k):
                share = (all_shares[i][0], all_shares[i][1])
                
                start = perf_counter()
                interpolator.add_share(share)
                recuperado = interpolator.get_secret()
                dt = perf_counter() - start
                
                total_time += dt
                
                erro_atual = abs(recuperado - self.segredo_real)

                if (i + 1) % step == 0:
                    x_axis.append(i + 1)
                    y_times.append(total_time)
                    y_errors.append(erro_atual) 
            
            evaluations[cls.__name__]["x"] = x_axis
            evaluations[cls.__name__]["y_time"] = y_times
            evaluations[cls.__name__]["y_error"] = y_errors 
            
            print(f"Finalizado (Total: {total_time:.4f}s)")

        if save_to_file:
            self.save_results(save_to_file, evaluations)
            
        if plot_results:
            self._plot_streaming(evaluations, save_to_file or ".")
            
        return evaluations, ""

    def _plot_streaming(self, results: dict, output_path: str):
        print("[!] Plotting Streaming results...")
        os.makedirs(output_path, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.style.use("ggplot")
        
        for name, data in results.items():
            plt.plot(data["x"], data["y_time"], label=name)
            
        plt.xlabel("Número de Shares Processadas")
        plt.ylabel("Tempo Acumulado (s)")
        plt.title("Streaming: Custo Computacional Acumulado")
        plt.legend()
        plt.grid(True)
        
        fpath_time = os.path.join(output_path, "streaming_time.png")
        plt.savefig(fpath_time)
        plt.close()
        print(f"  => Plot de TEMPO salvo em {fpath_time}")

        plt.figure(figsize=(10, 6))
        
        for name, data in results.items():
            erro_plot = [e if e > 0 else 1e-20 for e in data["y_error"]]
            plt.plot(data["x"], erro_plot, label=name, marker='o', markersize=4)
            
        plt.xlabel("Número de Shares Processadas")
        plt.ylabel("Erro Absoluto (Escala Log)")
        plt.yscale("log") # Importante para ver a explosão do erro
        plt.title("Streaming: Evolução do Erro Numérico")
        plt.legend()
        plt.grid(True)
        
        fpath_error = os.path.join(output_path, "streaming_error.png")
        plt.savefig(fpath_error)
        plt.close()
        print(f"  => Plot de ERRO salvo em {fpath_error}")