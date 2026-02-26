# Taller 1: Análisis Bayesiano en Turismo - Villa de Leyva

Este proyecto contiene la solución detallada de un taller de inferencia bayesiana aplicado al sector turístico. El objetivo es tomar decisiones operativas (cuántos tamales preparar y dónde venderlos) utilizando modelos probabilísticos conjugados.

## 📌 Contexto del Proyecto
El taller simula una salida académica a Villa de Leyva donde un grupo de estudiantes de Ciencia de Datos debe asesorar un puesto de tamales. Se utilizan dos enfoques principales:
1.  **Modelo Gamma-Poisson:** Para estimar el número de turistas diarios (conteos).
2.  **Modelo Beta-Binomial:** Para estimar la proporción de turistas que permanecen en el centro histórico.

## 📁 Estructura de Archivos
- `TallerBinomialPoisson.pdf`: Enunciado original del taller.
- `Solucion_Taller.tex`: Documento principal de la solución redactado en LaTeX.
- `src/generar_graficas.py`: Script en Python que realiza los cálculos estadísticos y genera las figuras obligatorias (G1 a G4).
- `graficas/`: Directorio que contiene las imágenes generadas para el documento final.
- `venv/`: Entorno virtual con las dependencias necesarias (`numpy`, `scipy`, `matplotlib`).

## 🚀 Estado Actual del Proyecto
| Sección | Estado | Tareas Realizadas |
| :--- | :--- | :--- |
| **Parte A (Gamma-Poisson)** | ✅ Completa | Calibración de priors, actualización de posteriors para dos escenarios (informado vs débil), derivación de la predictiva (Binomial Negativa) y toma de decisiones. |
| **Parte B (Beta-Binomial)** | 🚧 En progreso | Se ha completado el punto **B1.1** (calibración de la prior local basada en media y varianza). |
| **Parte C (Integrador)** | ⏳ Pendiente | Análisis final comparativo de todos los escenarios. |

## 🛠️ Cómo continuar el proyecto
Para seguir con el desarrollo, se deben atender los siguientes puntos en orden:
1.  **Actualizar `src/generar_graficas.py`**: Implementar los cálculos para la Parte B (Priors Beta, Likelihood Binomial, Posteriors y Predictiva Beta-Binomial).
2.  **Continuar en `Solucion_Taller.tex`**:
    - **B1.2 y B1.3**: Graficar la prior y explicar las pseudo-observaciones.
    - **B2**: Integrar el dato observado ($x=42, n=100$) y derivar la posterior.
    - **B3 y B4**: Comparar la prior no informativa vs la prior experta (terca).
    - **B5**: Calcular la predictiva para una muestra futura de $m=50$ turistas.
3.  **Redacción Final**: Completar la Parte C con el análisis de riesgos y recomendaciones finales de ubicación.

## 📝 Reglas de la Entrega
- Todas las gráficas (G1-G4) deben tener intervalos de 95% **sombreados**.
- La verosimilitud (Likelihood) debe estar escalada a un máximo de 1.
- Cada paso matemático debe incluir una interpretación de 2-4 líneas sobre "qué le hizo el dato a la prior".

---
**Autores:** Julian Jimenez, Tomas Rincon, Julian Duarte.
**Fecha:** Febrero 2026.
