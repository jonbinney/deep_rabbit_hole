import multiprocessing as mp

from v2 import benchmarks, load_config_and_setup_run, self_play, train

if __name__ == "__main__":
    config = load_config_and_setup_run(
        "deep_quoridor/experiments/B5W3/base.yaml", "/Users/amarcu/code/deep_rabbit_hole"
    )
    print("hey")
    mp.set_start_method("spawn", force=True)

    processes = []

    p = mp.Process(target=train, args=[config])
    p.start()
    processes.append(p)

    ps = benchmarks.create_benchmark_processes(config)
    [p.start() for p in ps]
    processes.extend(ps)

    for i in range(config.self_play.num_workers):
        p = mp.Process(target=self_play, args=[config])
        p.start()
        processes.append(p)

    for worker in processes:
        worker.join()
