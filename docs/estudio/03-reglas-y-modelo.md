# 03 — Las reglas y el modelo opcional

## Primero las reglas, deterministas y explicables

El corazón del detector son tres **reglas deterministas** (`detector.py`). Cada
una produce un hallazgo (`Finding`) con su tipo, su gravedad, el valor que falla,
la línea base del contador y una **razón en lenguaje claro**. Un analista que
recibe la alerta ve *por qué* saltó y puede actuar o descartarla.

1. **Apagón / plano (`flatline`).** Una racha de lecturas a cero (corte) o
   perfectamente constantes (contador pegado) de al menos `GG_FLATLINE_MIN_RUN`
   horas. El consumo real nunca es exactamente plano durante horas, así que una
   racha plana larga es un contador muerto o averiado.
2. **Pico de consumo (`robust_z_spike`).** Una hora cuyo z-score robusto está por
   encima de `GG_SPIKE_Z_THRESHOLD` **y** que además es materialmente grande (al
   menos el doble de la línea base, `GG_SPIKE_MIN_REL`). La segunda condición
   evita que una sola hora ruidosa en un contador muy estable se confunda con un
   pico.
3. **Robo / pérdida no técnica (`sustained_drop`).** Lecturas mantenidas por
   debajo de `GG_THEFT_FRACTION` de la línea base durante al menos
   `GG_THEFT_MIN_RUN` horas, o lecturas **negativas / puenteadas** (bypass). Es
   el caso difícil (ver 04).

### Prioridad por hora

Las reglas se aplican en orden de prioridad, de modo que cada hora recibe **como
mucho un tipo**: primero negativo/bypass, luego apagón/plano, luego bajón
sostenido, y por último pico. Así un apagón (ceros) no se cuenta además como robo.

## Luego el modelo, como segunda opinión

Encima de las reglas hay un modelo de scikit-learn **opcional**
(`GG_MODEL_KIND`): `logistic` (regresión logística supervisada, por defecto),
`isolation_forest` (no supervisado) o `none`. Se entrena con las casas de
entrenamiento y aporta **solo una bandera global** de anomalía sobre cada hora,
no un tipo.

¿Por qué separarlo así?

- Las reglas son la **fuente de la verdad explicable**: misma serie, mismos
  hallazgos, con una razón auditable.
- El modelo no puede anular las reglas; solo puede **ampliar la red** para
  mejorar la exhaustividad global. La evaluación mide exactamente qué añade
  (recall) y qué cuesta (precisión), y deja ese compromiso al operador (ver ADR
  0002).
- Una anomalía con una forma nueva que las reglas no codifican puede aun así
  aparecer en la bandera global del modelo, aunque sin tipo concreto: una señal
  honesta de que quizá hace falta una regla nueva.

## Severidad

Cada hallazgo lleva una gravedad (`low` / `medium` / `high`) derivada de la
magnitud: el z robusto para los picos, la duración para los apagones, y la
profundidad y duración (o el ser negativo) para el robo. Es una ayuda de
priorización, no una verdad absoluta.
