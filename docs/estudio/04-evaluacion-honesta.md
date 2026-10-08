# 04 — Evaluación honesta

## El riesgo: fingir que funciona

Como los datos son sintéticos, sería trivial —y deshonesto— reportar un resultado
casi perfecto. Tres trampas habituales:

- un *split* aleatorio que mezcla los mismos contadores en train y test,
- un único número global que esconde que un tipo de anomalía es mucho más difícil
  que otro,
- una única semilla que cae en un split con suerte.

La evaluación (`python -m gridguard.evaluation`) evita las tres.

## Las cuatro decisiones honestas

1. **Retener contadores enteros.** Las casas de test **nunca** se ven en
   entrenamiento (`make_split`). Así el modelo se juzga sobre contadores nuevos,
   no sobre unos que podría haber memorizado.
2. **Puntuar hora a hora (point-wise).** Cada hora se compara con su etiqueta
   real. Un apagón detectado una hora tarde pierde recall en esas horas: no se
   regala crédito por "casi".
3. **Reportar por tipo.** Precisión, recall y F1 se dan por separado para pico,
   apagón y robo, porque un número mezclado ocultaría que el robo es difícil.
4. **Promediar varias semillas.** Se promedian varios splits (y se reporta su
   dispersión) para que un split afortunado no fije el titular.

Además inyectamos **vacaciones** legítimas a propósito. Una vacación no es un robo
suave: la casa está **vacía**, así que el consumo se aplana a un nivel de *standby*
(sin picos de mañana ni de tarde), mientras que el robo **escala** el perfil (lo
baja pero conserva su forma). Mantener la vacación en los datos obliga al detector
a distinguir *forma aplanada* de *forma conservada* — y la evaluación demuestra que
esa distinción aguanta sobre contadores no vistos.

## Los números (medidos, media de 5 semillas)

```
por tipo de anomalía (reglas deterministas)
  consumption_spike   precision 0.90  recall 1.00  f1 0.95
  outage_flatline     precision 0.93  recall 1.00  f1 0.96
  theft_tampering     precision 0.99  recall 0.81  f1 0.89

global cualquier-anomalía (por hora)
  rules only          precision 0.99  recall 0.89  f1 0.94
  model only          precision 0.69  recall 0.65  f1 0.67
  rules + model       precision 0.76  recall 0.95  f1 0.85
```

## Qué dice y qué NO dice

- **Los tres tipos se detectan bien** (F1 0,89–0,96). La clave es el robo: se
  separa de una vacación por la **forma**. El contador que sub-registra una casa
  habitada escala el perfil (la carga sigue correlacionada con la forma diaria);
  una vacación aplana a *standby* y pierde esa forma. La regla de robo solo salta
  si el bajón sostenido **conserva la forma**, de ahí su precisión ~0,99 (no
  confunde vacaciones) y su recall ~0,81 (lee el bajón sobre el cociente suavizado,
  así que también caza el robo sutil de fracción alta). Tratar robo y vacación como
  "el mismo bajón" es justo el error que evita la prueba de forma.
- **Las reglas son la espina dorsal de alta precisión; el modelo es un dial de
  recall.** Las reglas llegan a F1 0,94 con ~0,99 de precisión. El modelo va con
  **compuerta de confianza**, así que `reglas + modelo` sube el recall 0,89 → 0,95
  a un coste de precisión controlado (0,99 → 0,76): una elección precisión/recall
  del operador, no un default roto. (Sin compuerta, la unión hundiría la precisión
  a ~0,6; justo lo que la compuerta evita.)

## Por qué el efecto de enmascaramiento importa

Si para los picos usáramos la **media y la desviación estándar**, el propio pico
inflaría ambas y bajaría su z-score hasta esconderse (enmascaramiento). La
**mediana y la MAD** son insensibles a unos pocos extremos, así que el pico
sobresale. Es la misma filosofía que la línea base por mediana (ADR 0001 y 0004).

## El límite, dicho con claridad

Todo esto es un mundo **sintético** con etiquetas conocidas. Un sistema real
necesitaría datos AMI reales, **etiquetas reales** (casos de robo confirmados,
partes de corte), manejo de huecos y de cambios de hora, y reevaluación bajo
*drift*. El objetivo aquí es el método medido con honestidad, no los números, que
solo valen para este mundo sintético.
