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
3. **Robo / pérdida no técnica (`sustained_drop`).** El caso interesante. Un
   contador manipulado **sub-registra** el consumo de una casa que sigue habitada:
   el consumo real tiene su forma diaria de siempre (picos de mañana y de tarde),
   pero el contador la reporta **escalada** hacia abajo. La regla lo detecta así:
   - toma el cociente `lectura / línea_base` hora a hora y lo **suaviza** con una
     media móvil corta (6 h). Suavizar importa: el robo es un cambio de régimen
     sostenido, y sin suavizar el ruido de una hora concreta por encima del umbral
     rompería la racha (un robo al 0,70 se sub-detectaría). El cociente suavizado
     debe quedar por debajo de `GG_THEFT_FRACTION` (0,80; el robo inyectado baja
     hasta ~0,72, así que el umbral queda justo por encima con margen);
   - y, sobre esa ventana, la **correlación con la forma diaria** debe ser al menos
     `GG_THEFT_SHAPE_CORR_MIN` (0,5). Aquí está la clave que separa el robo de unas
     **vacaciones**: en vacaciones la casa está vacía, el consumo se aplana a un
     nivel de *standby* (sin picos) y la correlación con la forma se va a ~0; en el
     robo la forma se conserva (solo escalada) y la correlación es alta.
   - además, lecturas **negativas / puenteadas** (bypass) se marcan directamente.

   Correlación de forma, en una línea: `corr = cov(lecturas, esperado) /
   (std(lecturas)·std(esperado))` sobre la ventana. Vale ~1 si la carga sigue el
   perfil diario (normal o robo escalado) y ~0 si se ha aplanado (vacaciones).

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
- El modelo **va con compuerta de confianza** (`GG_MODEL_MIN_PROBA`, 0,9). Esto no
  es un detalle: las reglas ya tienen ~0,99 de precisión, así que un modelo sin
  compuerta (umbral 0,5) inunda de falsos positivos y hunde la precisión combinada
  a ~0,6 — un detector **peor**. Con la compuerta alta, el modelo solo añade las
  horas de las que está seguro, y `reglas + modelo` pasa a ser un **dial de
  recall** real (recall 0,89 → 0,95 a un coste de precisión controlado), no un
  default dañino. El modelo nunca anula las reglas; solo **amplía la red**.
- La evaluación mide exactamente qué añade (recall) y qué cuesta (precisión), y
  deja esa elección precisión/recall al operador (ver ADR 0002).
- Una anomalía con una forma nueva que las reglas no codifican puede aun así
  aparecer en la bandera global del modelo, aunque sin tipo concreto: una señal
  rigurosa de que quizá hace falta una regla nueva.

## Severidad

Cada hallazgo lleva una gravedad (`low` / `medium` / `high`) derivada de la
magnitud: el z robusto para los picos, la duración para los apagones, y la
profundidad y duración (o el ser negativo) para el robo. Es una ayuda de
priorización, no una verdad absoluta.
