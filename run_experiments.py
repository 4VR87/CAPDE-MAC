from pathlib import Path
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from capde_sim import run_one

PROTOCOLS = ['capde', 'static', 'priority_csma', 'adt_mac', 'mdp_hymac', 'no_prediction', 'no_preemption']
LOADS = [0.6, 1.0, 1.4, 1.8]
SENSOR_CSV = ROOT / 'config' / 'sensor_profiles.csv'
CONFIG_JSON = ROOT / 'config' / 'simulation_parameters.json'


def read_seeds(path):
    with open(path, newline='', encoding='utf-8') as f:
        return [int(r['seed']) for r in csv.DictReader(f)]


def worker(job):
    protocol, load, seed = job
    metrics, _ = run_one(SENSOR_CSV, CONFIG_JSON, protocol, load, seed, trace=False)
    return metrics.as_dict()


def main():
    seeds = read_seeds(ROOT / 'config' / 'seeds.csv')
    jobs = [(p, l, s) for p in PROTOCOLS for l in LOADS for s in seeds]
    out = ROOT / 'data' / 'raw' / 'raw_run_metrics.csv'
    rows = []
    workers = max(1, min(5, os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(worker, job): job for job in jobs}
        for k, fut in enumerate(as_completed(futures), start=1):
            rows.append(fut.result())
            if k % 50 == 0 or k == len(jobs):
                print(f'Completed {k}/{len(jobs)} runs', flush=True)
    rows.sort(key=lambda r: (PROTOCOLS.index(r['protocol']), r['load'], r['seed']))
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f'Wrote {out}')

    # One packet-level diagnostic trace to make event accounting directly inspectable.
    metrics, trace = run_one(SENSOR_CSV, CONFIG_JSON, 'capde', 1.8, seeds[0], trace=True)
    trace_out = ROOT / 'data' / 'raw' / 'example_packet_trace_capde_load1p8_seed1.csv'
    with open(trace_out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(trace[0].keys()) if trace else ['packet_id'])
        w.writeheader(); w.writerows(trace)
    print(f'Wrote {trace_out}')

if __name__ == '__main__':
    main()
