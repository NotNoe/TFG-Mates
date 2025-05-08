# Resolución de ecuaciones en GPU

Este repositorio contiene el código utilizado para el desarrollo de mi Trabajo de Fin de Grado del **Doble Grado en Matemáticas e Informática** de la **Facultad de Matemáticas** de la **Universidad Complutense de Madrid**.

## Título
Resolución de ecuaciones en GPU

Solving equations in GPUs

## Autor
Noelia Barranco Godoy
## Directora
Ana María Carpio Rodríguez

## Resumen

El estudio y la simulación numérica de fenómenos físicos descritos por ecuaciones en derivadas parciales (EDPs) constituye una herramienta fundamental en múltiples disciplinas científicas y de ingeniería. Sin embargo, la resolución eficiente de estos problemas puede volverse computacionalmente costosa cuando se requieren mallas finas o dominios de gran tamaño. En este contexto, la programación en GPU surge como una alternativa poderosa para acelerar algoritmos numéricos gracias a su arquitectura de paralelismo masivo.

En este trabajo se analizan e implementan métodos numéricos para resolver cinco EDPs clásicas —la ecuación del calor (1D y 2D), la ecuación de onda (1D y 2D) y la ecuación de Laplace (2D)—  mediante esquemas de diferencias finitas. Tras establecer el marco teórico, se desarrollan implementaciones tanto en CPU como en GPU utilizando Python y PyCUDA.

El objetivo principal es cuantificar el aumento de rendimiento obtenido al ejecutar estos algoritmos en GPU. Para ello, se construye un benchmark y se comparan los tiempos de ejecución en ambos algoritmos. Los resultados muestran mejoras significativas al utilizar GPU, con aceleraciones de hasta dos órdenes de magnitud en algunos casos. Finalmente, se discuten las limitaciones observadas y se proponen posibles líneas de optimización y trabajo futuro.

## Abstract
The study and numerical simulation of physical phenomena described by partial differential equations (PDEs) is a fundamental tool in many scientific and engineering disciplines. However, solving these problems efficiently can become computationally expensive when fine meshes or large domains are required. In this context, GPU programming emerges as a powerful alternative to accelerate numerical algorithms, thanks to its massively parallel architecture.

This work analyzes and implements numerical methods to solve five classical PDEs —the heat equation (1D and 2D), the wave equation (1D and 2D), and the Laplace equation (2D)— using finite difference schemes. After establishing the theoretical framework, we develop implementations both on CPU and GPU using Python and PyCUDA.

The main objective is to quantify the performance improvement achieved by executing these algorithms on a GPU. To this end, a benchmark is built and execution times are compared between both approaches. The results show significant gains when using GPU, with speedups of up to two orders of magnitude in some cases. Finally, observed limitations are discussed and possible optimization paths and future work are proposed.


## Estructura del repositorio

- **TFG.pdf**: El pdf con el trabajo.
- **cpu/**: Implementación de los métodos numéricos en CPU.
- **gpu/**: Implementación de los métodos numéricos en GPU.
- **CUDA/**: Archivos .cu con los kernels de CUDA.
- **benchmark/**: Infraestructura para ejecutar tests y almacenar los tiempos.
- **graphs/**: Scripts para generar las gráficas utilizadas en el TFG.
- **tests/**: Scripts para la validación numérica de los algoritmos.
- **demos/**: Ejemplos ilustrativos de programas sencillos de PyCUDA.
- **utils/**: Utilidades para compilar kernels.
- **env.yml/**: Archivo con la configuración del entorno Conda.