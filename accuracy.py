from benchmark import InterpBenchmark
from lagrange import (lagrange, lagrange_fraction)
from newton import (newton, newton_fraction)
import numpy as np

def main():
    SEGREDO = 42.0

    ks = [2, 3, 5, 7, 10, 12, 15, 20, 25, 30]
    interp_fns = [
        lagrange, 
        #lagrange_fraction,
        #lagrange_parallel, 
        newton,
        #newton_fraction,
        #newton_parallel
    ]

    bench = InterpBenchmark(segredo_real=SEGREDO)
    original_gen = InterpBenchmark.generate_shares
    InterpBenchmark.generate_shares = lambda k: original_gen(k, secret=SEGREDO, degree=k-1)

    bench.run(
        interp_fns=interp_fns,
        k_values=ks,
        segredo=SEGREDO,
        save_to_file="results/accuracy/",
        plot_results=True,
        metrics_to_plot=["exec_time_ms", "error"]
    )
    
    InterpBenchmark.generate_shares = original_gen

if __name__ == "__main__":
    main()