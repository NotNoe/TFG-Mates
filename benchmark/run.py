import argparse
import os
import yaml
import time
import json
from tqdm import tqdm
from typing import Literal
from benchmark.Problems import *
from datetime import datetime
import pandas as pd
import numpy as np

def round_metrics(metrics: dict, sigfigs: int = 3) -> dict:
    out = {}
    for k, v in metrics.items():
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, float):
            out[k] = float(f"{v:.{sigfigs}g}")
        elif isinstance(v, dict):
            out[k] = round_metrics(v, sigfigs)
        else:
            out[k] = v
    return out


ECUACIONES_CLASES = {
    "heat1d": Heat1D,
    "heat2d": Heat2D,
    "wave1d": Wave1D,
    "wave2d": Wave2D,
    "laplace": Laplace
}


def main():
    p = argparse.ArgumentParser(prog = "Benchmark runner", description="Ejecuta el benchmark de solvers en CPU/GPU siguiendo la configuración del archivo YAML")
    p.add_argument("--cfg", default="benchmark/problems.yaml", type=str, help="Ruta del archivo de configuración")
    p.add_argument("--outdir", default="out/benchmark", type=str, help="Ruta del directorio de salida")
    p.add_argument("--equations", nargs="+", default=None, help="Lista de ecuaciones a ejecutar (por defecto todas). Argumento excluyente con cases")
    p.add_argument("--cases", nargs="+", default=None, help = "Ejecuta solamente las ecuaciones ecuación:idx1:gpu,idx2:cpu. Argumento excluyente con equations")
    p.add_argument("--verbose", action="store_true", help="Muestra el report en pantalla")
    p.add_argument("--no_save", action="store_true", help="No guarda el report en disco")
    args = p.parse_args()

    if args.equations is not None and args.cases is not None:
        raise ValueError("No se pueden usar --equations y --cases al mismo tiempo")

    with open(args.cfg, "r") as f:
        problems_yaml = yaml.safe_load(f)
    #Asegurarnos que todas las etiquetas son válidas
    for eq in problems_yaml.keys():
        if eq not in ECUACIONES_CLASES.keys():
            raise ValueError(f"Etiqueta de ecuación inválida: {eq}")
        
    #Aquí las etiquetas del yaml son EXACTAMENTE ECUACIONES
    id = 0
    problems = {}
    if args.cases is None:
        for eq in problems_yaml.keys():
            problems[eq] = []
            if args.equations is None or eq in args.equations: #Hay que meter estas ecuaciones
                for idx, params in enumerate(problems_yaml[eq]):
                    #problems[eq].append(ECUACIONES_CLASES[eq](idx = idx, csv_idx = id, device = "cpu", **params))
                    #id += 1
                    problems[eq].append(ECUACIONES_CLASES[eq](idx = idx, csv_idx = id, device = "gpu", **params))
                    id += 1
    else:
        for case in args.cases:
            eq, idx, device = case.split(":")
            device = device.lower()
            idx = int(idx)
            if eq not in ECUACIONES_CLASES.keys():
                raise ValueError(f"Etiqueta de ecuación inválida: {eq}")
            if len(problems_yaml[eq]) <= idx:
                raise ValueError(f"Índice {idx} fuera de rango para la ecuación {eq}")
            if device not in ["cpu", "gpu"]:
                raise ValueError(f"Dispositivo inválido: {device}")
            if eq not in problems.keys():
                problems[eq] = []
            problems[eq].append(ECUACIONES_CLASES[eq](idx = idx, csv_idx = id, device = device, **problems_yaml[eq][idx]))
            id += 1
    #Problems ya está bien montado.
    
    #Dataframe and report
    report = {}
    ts = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    report["timestamp"] = ts
    report["cases"] = []
    n = sum(len(lista) for lista in problems.values())
    df = pd.DataFrame.from_dict({
    "equation": ["" for _ in range(n)],
    "device": ["" for _ in range(n)],
    "t_wall": [np.nan for _ in range(n)],
    "t_latency": [np.nan for _ in range(n)],
    "t_kernel": [np.nan for _ in range(n)],
    "error": [np.nan for _ in range(n)],
    })



    for eq, lista in tqdm(problems.items(), desc="Ecuaciones"):
        for prob in tqdm(lista, desc=f"{eq}", leave=False):
            metrics = prob.solve()
            df.iloc[prob.csv_idx] = {
                "equation": eq,
                "device": prob.device,
                "t_wall": metrics["t_wall"],
                "t_latency": metrics["t_latency"] if "t_latency" in metrics else np.nan,
                "t_kernel": metrics["t_kernel"] if "t_kernel" in metrics else np.nan,
                "error": metrics["error"]
            }
            report["cases"].append({
                "equation": eq,
                "device": prob.device,
                "csv_idx": prob.csv_idx,
                "metrics": round_metrics(metrics),
                "args": prob.get_params()
            })
            
    if not args.no_save:
        os.makedirs(os.path.join(args.outdir, ts), exist_ok = True)
        #Save report
        with open(os.path.join(args.outdir, ts, "report.json"), "w") as f:
            json.dump(report, f, indent=4)
        #Save dataframe
        df.to_csv(os.path.join(args.outdir, ts, "report.csv"), float_format="%.18e")

    if args.verbose:
        print("Report:")
        print(json.dumps(report, indent=4))



if __name__ == "__main__":
    main()