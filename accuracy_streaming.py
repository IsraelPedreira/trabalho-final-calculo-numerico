from benchmark import InterpBenchmark
from newton import NewtonStreaming, NewtonStreamingFraction
from lagrange import LagrangeStreaming, LagrangeStreamingFraction


def main_streaming():
    SEGREDO = 42.0
    MAX_K = 100  

    classes_to_test = [
        NewtonStreaming,
        LagrangeStreaming,
        NewtonStreamingFraction,
        LagrangeStreamingFraction
    ]

    bench = InterpBenchmark(segredo_real=SEGREDO)

    bench.run_streaming(
        interpolator_classes=classes_to_test,
        max_k=MAX_K,
        step=1, 
        save_to_file="results/streaming/",
        plot_results=True
    )

if __name__ == "__main__":
    main_streaming()