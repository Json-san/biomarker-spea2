# 🧬 SPEA2 Biomarker Optimizer

**Algoritmo evolutivo multiobjetivo para selección de biomarcadores en datos genómicos de leucemia.**

![Next.js](https://img.shields.io/badge/Next.js-14.0-black?style=flat&logo=next.js)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue?style=flat&logo=typescript)
![TailwindCSS](https://img.shields.io/badge/Tailwind-3.4-38bdf8?style=flat&logo=tailwindcss)

---

## 📖 Descripción

Este proyecto implementa el algoritmo **SPEA2 (Strength Pareto Evolutionary Algorithm 2)** para la selección óptima de biomarcadores en el dataset de Leucemia (Golub et al., 1999). El sistema optimiza dos objetivos en conflicto:

- **Maximizar** la precisión de clasificación
- **Minimizar** el número de genes seleccionados

La visualización web interactiva permite explorar el proceso evolutivo en tiempo real y entender cómo evoluciona el frente de Pareto a través de las generaciones.

---

## 🚀 Cómo Ejecutar el Proyecto

### Prerrequisitos

- **Node.js** 18.0 o superior
- **npm** o **yarn** o **pnpm**

### Instalación

```bash
# Clonar el repositorio (si aplica)
git clone <url-del-repositorio>
cd biomarker-spea2

# Instalar dependencias
npm install
```

### Ejecutar en Desarrollo

```bash
npm run dev
```

Abre [http://localhost:3000](http://localhost:3000) en tu navegador para ver la aplicación.

### Construir para Producción

```bash
# Crear build de producción
npm run build

# Ejecutar build de producción
npm start
```

---

## 📓 Documentación del Algoritmo

### Notebook Jupyter

Este proyecto incluye un **Jupyter Notebook** que explica en detalle:

1. **El algoritmo SPEA2**: Fundamentos teóricos, métricas de fitness, y operadores evolutivos
2. **El experimento**: Configuración, dataset de leucemia, y metodología de evaluación
3. **Resultados**: Análisis del frente de Pareto, genes más frecuentes, y comparación con baseline

> 📁 **Ubicación**: Consulta el notebook `SPEA2_Experiment.ipynb` (o similar) en el directorio raíz o en la carpeta de documentación para una explicación completa del funcionamiento del algoritmo y los resultados experimentales.

### Documentación Técnica

Para documentación técnica detallada del proyecto, consulta:
- [`DOCUMENTATION.md`](./DOCUMENTATION.md) - Documentación completa de arquitectura, componentes y resultados

---

## 🎮 Uso de la Aplicación

1. **Iniciar la evolución**: Haz clic en el botón ▶️ Play
2. **Observar el frente de Pareto**: Evoluciona en el gráfico de dispersión
3. **Explorar selección de genes**: Visualiza la animación de la cadena de ADN
4. **Consultar genes frecuentes**: Tabla de biomarcadores más consistentes
5. **Aprender el algoritmo**: Expande la sección "Cómo Funciona"

---

## 📊 Resultados Principales

| Método | Precisión | Genes | Reducción |
|--------|-----------|-------|-----------|
| Baseline (todos los genes) | 97.2% | 7,129 | 0% |
| SPEA2 (alta precisión) | 94.4% | ~25 | 99.6% |
| SPEA2 (balanceado) | 91.7% | ~12 | 99.8% |
| SPEA2 (mínimo) | 86.1% | ~5 | 99.9% |

---

## 🛠️ Stack Tecnológico

| Componente | Tecnología |
|------------|------------|
| Frontend | Next.js 14, React 18, TypeScript |
| Estilos | Tailwind CSS |
| Gráficos | Chart.js con react-chartjs-2 |
| Animaciones | SVG + CSS transitions |
| Algoritmo | Python (offline), JSON data replay |

---

## 📂 Estructura del Proyecto

```
biomarker-spea2/
├── src/
│   ├── app/                    # Páginas Next.js
│   ├── components/             # Componentes React
│   ├── hooks/                  # Hooks personalizados
│   └── types/                  # Tipos TypeScript
├── public/
│   └── evolution_data.json     # Datos pre-computados
├── DOCUMENTATION.md            # Documentación técnica
└── README.md                   # Este archivo
```

---

## 📜 Referencias

1. Zitzler, E., Laumanns, M., & Thiele, L. (2001). *SPEA2: Improving the Strength Pareto Evolutionary Algorithm.*
2. Golub, T.R., et al. (1999). *Molecular Classification of Cancer: Class Discovery and Class Prediction by Gene Expression Monitoring.* Science.


---

*SPEA2 Biomarker Optimizer v1.0*
