import json
import os
from datetime import datetime
import random
from typing import Any, Optional, Callable
from time import perf_counter
from matplotlib import pyplot as plt

import torch
import numpy as np


class InterpBenchmark:
    def __init__(self, segredo_real: Optional[int] = None) -> None:
        self.segredo_real = segredo_real

    def run_for_one_fn(
        self, shares: np.ndarray, interp_fn: Callable
    ) -> dict[str, float]:
        metrics = {
            "exec_time_ms": float("inf"),
            "output": float("inf"),
        }
        start_time = perf_counter()
        output = interp_fn(shares)
        end_time = perf_counter()

        metrics["exec_time_ms"] = end_time - start_time
        metrics["output"] = output

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
    def generate_shares(k: int) -> np.ndarray:
        """Gera k pedaços (x, y) de um polinômio onde P(0) = SEGREDO"""
        return np.random.randn(k, 2)

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
            results = json.loads(results)
        if isinstance(results, dict):
            pass
        else:
            raise TypeError(
                f"`results` deve ser o caminho para um arquivo ou um dicionário com os resultados. Recebido: {type(results)}"
            )

        assert isinstance(results, dict), "Unreachable"

        print("[!] Plotting results...")
        os.makedirs(output_path, exist_ok=True)
        for metric in metrics_to_plot:
            fpath = os.path.join(output_path, f"{metric}")
            for method in results.keys():
                xs = list(results[method].keys())
                ys = [results[method][k][metric] for k in xs]
                plt.style.use("ggplot")
                plt.plot(xs, ys)

                plt.xlabel("K")
                plt.ylabel(metric)

            print("[!] Saving plot to file:", output_path)
            plt.legend(results.items())
            plt.savefig(fpath + ".png")
            print("  => Saved.")
