import time
from simulator import Simulator

if __name__ == "__main__":
    sim = Simulator()

    # Number of physics steps to benchmark
    N = 10000

    # Run one step first so MuJoCo is fully initialized
    sim.step()

    # Start timing
    start = time.perf_counter()

    for _ in range(N):
        sim.step()

    # Stop timing
    elapsed = time.perf_counter() - start

    # Calculate performance
    simulated_time = N * sim.dt
    fps = N / elapsed
    real_time_factor = simulated_time / elapsed

    print("\n==============================")
    print("MuJoCo Physics Benchmark")
    print("==============================")
    print(f"Steps:             {N}")
    print(f"Simulation timestep: {sim.dt:.6f} s")
    print(f"Simulated time:    {simulated_time:.3f} s")
    print(f"Real time:         {elapsed:.3f} s")
    print(f"Physics FPS:       {fps:.2f}")
    print(f"Real-time factor:  {real_time_factor:.3f}x")
    print("==============================")