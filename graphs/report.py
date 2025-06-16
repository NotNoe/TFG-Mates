import json
from matplotlib.lines import Line2D
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ECUATIONS_TO_NAME = {"heat1d": "Ecuación del Calor 1D",
                     "laplace": "Ecuación de Laplace 2D",
                     "heat2d": "Ecuación del Calor 2D",
                     "wave1d": "Ecuación de Onda 1D",
                     "wave2d": "Ecuación de Onda 2D"}

def generate_df(report_path):
    with open(report_path, 'r') as f:
        cases = json.load(f)["cases"]

    df = pd.DataFrame(cases)
    args_df = pd.json_normalize(df["args"]).astype({"n_x": pd.Int32Dtype(),
                                                    "n_y": pd.Int32Dtype(),
                                                    "n_t": pd.Int32Dtype()})
                                        
    metrics_df = pd.json_normalize(df["metrics"])
    metrics_df = metrics_df.astype({"t_wall": pd.Float32Dtype(),
                                    "error": pd.Float32Dtype()})
    df = pd.concat([df.drop(columns=["args", "metrics"]), args_df, metrics_df], axis=1)
    df['mesh_nodes'] = df[['n_x', 'n_y']].prod(1, True)
    return df

def graf_speedup(df, ax=None):
    pivot = df.pivot(index="mesh_nodes", columns="device", values="t_wall").sort_index()
    pivot["speedup"] = pivot["cpu"] / pivot["gpu"]
    x = pivot.index
    y = pivot["speedup"]
    logx = np.log(x)
    logy = np.log(y)
    b, loga = np.polyfit(logx, logy, 1)       # ajuste lineal en log-log
    a = np.exp(loga)

    # luego, sobre tu gráfica:
    xx = np.linspace(x.min(), x.max(), 100)
    yy = a * xx**b
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xx, yy, linestyle=':', label=f'Tendencia $\\propto x^{{{b:.2f}}}$')
    ax.plot(pivot.index, pivot["speedup"], marker="o", linestyle="-")
    ax.set_xscale("log")
    #plt.yscale("log")
    ax.grid(True, which="major", ls="--", lw=0.5)
    return ax

def graf_error(df, ax = None):
    pivot_err = (
        df
        .pivot(index="mesh_nodes", columns="device", values="error")
        .sort_index()
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Dibujar Error vs mesh_nodes
    ax.plot(pivot_err.index, pivot_err["cpu"], marker="o", linestyle="-", label="CPU")
    ax.plot(pivot_err.index, pivot_err["gpu"], marker="o", linestyle="-", label="GPU")
    ax.set_xscale("log")
    ax.set_yscale("log")  # Error often plotted log-log to see rates
    ax.set_xlabel("Número de nodos de malla")
    ax.set_ylabel("Error numérico")
    ax.set_title("Error vs. Número de nodos para la Ecuación del Calor")
    ax.grid(which="major", linestyle="--", linewidth=0.5)
    legend_handles = [
        Line2D([0], [0], color='C0', lw=2),
        Line2D([0], [0], color='C1', lw=2)
    ]
    ax.legend(legend_handles, ["CPU", "GPU"])
    return ax

def graf_time(df, ax=None):
    pivot_time = (
        df
        .pivot(index="mesh_nodes", columns="device", values="t_wall")
        .sort_index()
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # 6) Dibujar tiempo vs mesh_nodes en escala log-log
    ax.plot(pivot_time.index, pivot_time["cpu"], marker="o", linestyle="-", label="CPU")
    ax.plot(pivot_time.index, pivot_time["gpu"], marker="o", linestyle="-", label="GPU")

    ax.set_xscale("log")
    #ax.set_yscale("log")
    ax.grid(which="major", linestyle="--", linewidth=0.5)
    legend_handles = [
        Line2D([0], [0], color='C0', lw=2),
        Line2D([0], [0], color='C1', lw=2)
    ]
    ax.legend(legend_handles, ["CPU", "GPU"])

    return ax



if __name__ == "__main__":
    df = generate_df("out/report.json")
    fig_time, axs_time = plt.subplots(2, 3, figsize=(12, 8))
    axs_time = axs_time.flatten()
    equations = df["equation"].unique()
    for ax, eq in zip(axs_time, equations):
        df_eq = df[df["equation"] == eq]
        graf_time(df_eq, ax=ax)      # Usa tu función para graficar tiempos
        ax.set_title(ECUATIONS_TO_NAME[eq])
    # Ocultar el subplot extra
    axs_time[-1].axis("off")
    fig_time.supxlabel("Número de nodos de malla")
    fig_time.supylabel("Tiempo de ejecución (s)")
    fig_time.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('out/t_wall.png')

    # 2) Figura de speed-up
    fig_sp, axs_sp = plt.subplots(2, 3, figsize=(12, 8))
    axs_sp = axs_sp.flatten()
    for ax, eq in zip(axs_sp, equations):
        df_eq = df[df["equation"] == eq]
        graf_speedup(df_eq, ax=ax)   # Usa tu función para graficar speed-up
        ax.set_title(ECUATIONS_TO_NAME[eq])
    # Ocultar el subplot extra
    axs_sp[-1].axis("off")
    fig_sp.supxlabel("Número de nodos de malla")
    fig_sp.supylabel("Aceleración (CPU / GPU)")
    fig_sp.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('out/speedup.png')

    # 2) Figura de speed-up
    fig_sp, axs_sp = plt.subplots(2, 3, figsize=(12, 8))
    axs_sp = axs_sp.flatten()
    for ax, eq in zip(axs_sp, equations):
        df_eq = df[df["equation"] == eq]
        graf_error(df_eq, ax=ax)   # Usa tu función para graficar speed-up
        ax.set_title(ECUATIONS_TO_NAME[eq])
    axs_sp[-1].axis("off")
    #plt.show()