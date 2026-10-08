# 02 — Características y línea base robusta

## De la serie a características por hora

El detector no mira la lectura cruda, sino un pequeño vector de **características**
(features) por cada hora, calculado a partir de su contexto reciente y de la línea
base del propio contador (`features.py`):

| Característica        | Qué captura                                                              |
| -------------------- | ------------------------------------------------------------------------ |
| `norm_consumption`   | lectura / mediana de la casa (su nivel típico)                           |
| `baseline_deviation` | desviación relativa respecto a la línea base: `(x − esperado) / esperado`|
| `robust_z`           | z-score robusto de esa desviación (mediana + MAD) — detecta picos        |
| `rate_of_change`     | cambio respecto a la hora anterior (salto brusco)                        |
| `zero_run`           | longitud de la racha de ceros que termina en esta hora (apagón)          |
| `flat_run`           | longitud de la racha de valores constantes (contador pegado)             |
| `night_day_ratio`    | relación consumo-noche / consumo-día en la ventana reciente              |
| `roll_mean`          | media móvil de la ventana (normalizada)                                  |
| `roll_std`           | desviación móvil (un plano la hunde; un pico la dispara)                 |

## Por qué las características son invariantes a la escala

El modelo opcional se entrena con unas casas y se aplica a **otras nunca vistas**.
Si una característica fuera el consumo absoluto en kWh, el modelo aprendería el
tamaño de las casas de entrenamiento y fallaría con casas de otro tamaño.

Por eso **todas** las características son ratios, desviaciones, z-scores o
longitudes de racha: una casa de 5 kW y otra de 0,5 kW con el **mismo tipo** de
problema producen los **mismos** valores de característica. Esto es lo que hace
que la evaluación con contadores retenidos sea una prueba real de generalización y
no una prueba de memoria (ver 04).

## Por qué la línea base usa la MEDIANA

La línea base de cada contador es la **mediana** de las lecturas de cada celda
`(fin de semana?, hora)`, no la media. La razón es la robustez:

- Las anomalías inyectadas son siempre una **minoría** de las muestras de una
  celda (un pico dura 1-3 horas; un robo afecta unos pocos días de ~28).
- La **mediana ignora** esa minoría, así que la línea base sigue representando el
  comportamiento *normal*.
- La **media**, en cambio, se dejaría arrastrar hacia la propia anomalía que
  queremos medir, subestimando la desviación.

## El z-score robusto (mediana + MAD)

Para los picos usamos un **z-score robusto** basado en la mediana y la desviación
absoluta mediana (MAD, *median absolute deviation*):

```
z_robusto = 0,6745 · (d − mediana(d)) / MAD(d)
```

donde `d` es la desviación respecto a la línea base. La mediana y la MAD son
insensibles a unos pocos valores extremos, así que **el pico sobresale en lugar de
esconderse**. La constante 0,6745 escala la MAD para que, con datos normales, el z
robusto coincida con el z ordinario. El porqué de no usar media y desviación
estándar (el efecto de enmascaramiento) está en el ADR 0004 y en el apartado 04.
