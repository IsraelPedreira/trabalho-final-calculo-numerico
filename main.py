from benchmark import InterpBenchmark
from lagrange import lagrange, lagrange_parallel, lagrange_parallel_batch
from newton import newton
from newton import newton_parallel
from streaming_wrappers import (
    StreamNewtonCPU,
    StreamLagrangeCPU,
    StreamLagrangeGPU,
    StreamNewtonGPU,
)

def main():
    # Batch benchmark
    ks = [5, 10, 20, 50, 100, 500, 1000, 3000, 5000, 10000]
    interp_fns = [lagrange, lagrange_parallel, newton, newton_parallel]

    bench = InterpBenchmark()
    bench.run(
        segredo=1456,
        interp_fns=interp_fns,
        k_values=ks,
        save_to_file="results/",
        plot_results=True,
    )
    
    # Streaming benchmark
    max_k_streaming = 300  
    
    streaming_classes = [
        #StreamNewtonCPU,  
        StreamLagrangeCPU, 
        #StreamLagrangeGPU,
        StreamNewtonGPU
    ]

    bench.run_streaming(
        interpolator_classes=streaming_classes,
        max_k=max_k_streaming,
        step=10, # Salva ponto no gráfico a cada 10 shares adicionadas
        save_to_file="results/streaming/",
        plot_results=True
    )


if __name__ == '__main__':
    main()