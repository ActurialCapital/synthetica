

<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<p align="center"><img src="https://github.com/ActurialCapital/synthetica/blob/main/docs/static/logo.png" alt="logo" width="90%" height="90%"></p>

| Descripción general | |
|---|---|
| **Código abierto** |  [![BSD 3-clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/ActurialCapital/synthetica/blob/main/LICENSE) |
| **Código** |  [![!pypi](https://img.shields.io/pypi/v/python-synthetica?color=orange)](https://pypi.org/project/python-synthetica/) [![!python-versions](https://img.shields.io/pypi/pyversions/python-synthetica)](https://www.python.org/) |
| **CI/CD** | [![!codecov](https://img.shields.io/codecov/c/github/ActurialCapital/synthetica?label=codecov&logo=codecov)](https://codecov.io/gh/ActurialCapital/synthetica) |
| **Descargas** | ![PyPI - Downloads](https://img.shields.io/pypi/dw/python-synthetica) ![PyPI - Downloads](https://img.shields.io/pypi/dm/python-synthetica) [![Downloads](https://static.pepy.tech/personalized-badge/python-synthetica?period=total&units=international_system&left_color=grey&right_color=blue&left_text=cumulative%20(pypi))](https://pepy.tech/project/python-synthetica) |


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Tabla de contenidos</summary>
  <ol>
    <li>
      <a href="#about-the-project">Acerca del Proyecto</a>
        <ul>
            <li><a href="#introduction">Introducción</a></li>
        </ul>
        <ul>
            <li><a href="#built-with">Construido con</a></li>
        </ul>
    </li>
    <li><a href="#installation">Instalación</a></li>
    <li><a href="#getting-started">Primeros pasos</a></li>
    <li><a href="#notes">Notas</a></li>
    <li><a href="#contributing">Contribuciones</a></li>
    <li><a href="#license">Licencia</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## Acerca del Proyecto

### Introducción

`Synthetica` es una herramienta versátil y robusta para generar datos sintéticos de series temporales. Ya sea que esté involucrado en modelado financiero, simulación de datos IoT o cualquier proyecto que requiera datos realistas de series temporales para crear señales correlacionadas o no correlacionadas, `Synthetica` proporciona conjuntos de datos generados de alta calidad y personalizables. Aprovechando técnicas estadísticas avanzadas y algoritmos de aprendizaje automático, `Synthetica` produce datos sintéticos que replican de cerca las características y patrones de los datos del mundo real.

La última versión del proyecto incorpora una amplia variedad de modelos, ofreciendo un extenso conjunto de herramientas para generar datos sintéticos de series temporales. Esta versión incluye características como:

* `GeometricBrownianMotion`
* `AutoRegressive`
* `NARMA`
* `Heston`
* `CIR`
* `LevyStable`
* `MeanReverting`
* `Merton`
* `Poisson`
* `Seasonal`

Sin embargo, la versión `SyntheticaAdvenced` eleva aún más las capacidades, integrando algoritmos impulsados por datos de aprendizaje profundo más sofisticados, como `TimeGAN`.

<p align="right">(<a href="#readme-top">volver al inicio</a>)</p>

### Construido con

* `numpy = "^1.26.4"`
* `pandas = "^2.2.2"`
* `scipy = "^1.13.1"`

<p align="right">(<a href="#readme-top">volver al inicio</a>)</p>

<!-- GETTING STARTED -->
## Instalación

```sh
$ pip install python-synthetica
```

<p align="right">(<a href="#readme-top">volver al inicio</a>)</p>

<!-- USAGE EXAMPLES -->
## Primeros pasos

Una vez que haya clonado el repositorio, puede comenzar a utilizar `Synthetica` para generar datos sintéticos de series temporales. Aquí hay algunos pasos iniciales para ayudarle a comenzar su exploración:

```python
>>> import synthetica as sth
```

En este ejemplo, utilizamos los siguientes parámetros con fines ilustrativos:

* `length=252`: La longitud de la serie temporal
* `num_paths=5`: El número de trayectorias a generar
* `seed=123`: Reinicializa la instancia singleton `RandomState` de `numpy` para fines de reproducibilidad

**Inicializar el modelo**: Utilizando el modelo `GeometricBrownianMotion` (GBM): Este enfoque inicializa el modelo con una longitud de trayectoria especificada, número de trayectorias y una semilla aleatoria fija:

```python
>>> model = sth.GeometricBrownianMotion(length=252, num_paths=5, seed=123)
```

**Generar señales aleatorias**: El método transform luego genera las señales aleatorias en consecuencia:

```python
>>> model.transform() # Generar señales aleatorias
```

<p align="center"><img src="https://github.com/ActurialCapital/synthetica/blob/main/docs/static/gbm_random_transform.png" alt="chart-1" width="75%" height="75%"></p>

**Generar trayectorias correlacionadas**: Este proceso garantiza que las características resultantes sean altamente correlacionadas positivamente, aprovechando el método de descomposición de Cholesky para lograr la estructura de correlación deseada de `matrix`:

```python
>>> model.transform(matrix) # Produce características altamente correlacionadas positivamente
```

<p align="center"><img src="https://github.com/ActurialCapital/synthetica/blob/main/docs/static/gbm_corr_transform.png" alt="chart-2"  width="75%" height="75%"></p>


<p align="right">(<a href="#readme-top">volver al inicio</a>)</p>

## Notas

### Descomposición de Cholesky

La transformación de Cholesky (o descomposición de Cholesky) es una técnica matemática utilizada para descomponer una matriz definida positiva en el producto de una matriz triangular inferior y su transpuesta. Esto es particularmente útil en diversos campos como el análisis numérico, la optimización y el modelado financiero:

1. **Estabilidad numérica**: La descomposición de Cholesky es más numéricamente estable que otros métodos de descomposición para matrices definidas positivas.
2. **Resolución de sistemas lineales**: Se utiliza para resolver sistemas de ecuaciones lineales de manera eficiente.
3. **Simulación de variables aleatorias correlacionadas**: En finanzas y estadística, se utiliza para generar variables aleatorias correlacionadas a partir de variables no correlacionadas.

#### Definición matemática

Dada una matriz definida positiva $A$, la descomposición de Cholesky es una factorización tal que $A = L L^T$, donde:
- $A$ es una matriz definida positiva.
- $L$ es una matriz triangular inferior.
- $L^T$ es la transpuesta de $L$.


#### Implementación

En el contexto de la generación de datos sintéticos, la transformación de Cholesky se puede utilizar para aplicar una estructura de correlación a un conjunto de variables aleatorias no correlacionadas. `synthetica` utiliza `np.linalg.cholesky` en segundo plano.

### Definitividad positiva

#### Qué significa definitividad positiva en una matriz de covarianza

Una matriz de covarianza se considera definida positiva si cumple las siguientes propiedades clave:

1. Es simétrica, lo que significa que la matriz es igual a su transpuesta.
2. Para cualquier vector no nulo $x$, $x^T * C * x > 0$, donde $C$ es la matriz de covarianza y $x^T$ es la transpuesta de $x$.
3. Todos sus autovalores son estrictamente positivos.

La definitividad positiva en una matriz de covarianza tiene implicaciones importantes:

1. Garantiza que la matriz sea invertible, lo cual es crucial para muchas [técnicas estadísticas](https://stats.stackexchange.com/questions/52976/is-a-sample-covariance-matrix-always-symmetric-and-positive-definite).
2. Garantiza que la matriz representa una [distribución de probabilidad válida](https://statproofbook.github.io/P/covmat-psd.html).
3. Permite soluciones únicas en [problemas de optimización](https://gowrishankar.info/blog/why-covariance-matrix-should-be-positive-semi-definite-tests-using-breast-cancer-dataset/) y asegura la estabilidad de ciertos algoritmos.
4. Indica que ninguna combinación lineal de las variables tiene varianza cero, lo que significa que todas las variables aportan [información significativa](https://math.stackexchange.com/questions/114072/what-is-the-proof-that-covariance-matrices-are-always-semi-definite).

Una matriz de covarianza que es semi-definida positiva (permitiendo que los autovalores sean no negativos en lugar de estrictamente positivos) sigue siendo válida, pero puede indicar dependencias lineales entre variables.

En la práctica, las matrices de covarianza de muestras suelen ser definidas positivas si el número de observaciones supera el número de variables y no existen relaciones lineales perfectas entre las variables.

#### Implementación

`synthetica` encuentra automáticamente la matriz definida positiva más cercana a la entrada utilizando la función de Python `nearest_positive_definite`. Está directamente extraída de [Computing a nearest symmetric positive semidefinite matrix](https://doi.org/10.1016/0024-3795(88)90223-6).

#### Otras fuentes

* [MatLab](https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd)
* [StackOverflow](https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite)
* [Gist](https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd)

<!-- CONTRIBUTING -->
## Contribuciones

Las contribuciones son lo que hacen de la comunidad de código abierto un lugar tan increíble para aprender, inspirarse y crear. Cualquier contribución que realice es **muy apreciada**.

Si tiene una sugerencia que mejoraría esto, bifurque el repositorio y cree una solicitud de extracción (pull request). También puede simplemente abrir un problema con la etiqueta "enhancement".
¡No olvide darle una estrella al proyecto! ¡Gracias de nuevo!

1. Bifurque el Proyecto
2. Cree su Rama de Característica (`git checkout -b feature/AmazingFeature`)
3. Confirme sus Cambios (`git commit -m 'Add some AmazingFeature'`)
4. Envíe a la Rama (`git push origin feature/AmazingFeature`)
5. Abra una Solicitud de Extracción

<p align="right">(<a href="#readme-top">volver al inicio</a>)</p>


<!-- LICENSE -->
## Licencia

Distribuido bajo la Licencia BSD-3. Consulte `LICENSE.txt` para más información.

<p align="right">(<a href="#readme-top">volver al inicio</a>)</p>
