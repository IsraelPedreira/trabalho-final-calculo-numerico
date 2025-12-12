from benchmark import InterpBenchmark
from lagrange import lagrange, lagrange_parallel, lagrange_parallel_batch
from newton import ReconstrutorNewton

def main():
    ks = [5, 10, 20, 50, 100]
    interp_fns = [lagrange, lagrange_parallel, ReconstrutorNewton.run_newton]

    bench = InterpBenchmark()
    bench.run(
        segredo=1456,
        interp_fns=interp_fns,
        k_values=ks,
        save_to_file="results/",
        plot_results=True,
    )


if __name__ == '__main__':
    main()