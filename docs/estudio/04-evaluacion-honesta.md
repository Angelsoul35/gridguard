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

Además inyectamos **vacaciones** legítimas (consumo bajo sostenido etiquetado como
*normal*) a propósito, para medir el robo contra un confusor realista y no contra
un espantapájaros.

## Los números (medidos, media de 5 semillas)

```
por tipo de anomalía (reglas deterministas)
  consumption_spike   precision 0.90  recall 1.00  f1 0.95
  outage_flatline     precision 0.93  recall 1.00  f1 0.96
  theft_tampering     precision 0.63  recall 0.46  f1 0.53

global cualquier-anomalía (por hora)
  rules only          precision 0.77  recall 0.64  f1 0.69
  model only          precision 0.60  recall 0.89  f1 0.72
  rules + model       precision 0.61  recall 0.93  f1 0.73
```

## Qué dice y qué NO dice

- **Picos y apagones son fáciles y el detector es bueno** (F1 ~0,95): un salto
  brusco de varias sigmas y una racha plana larga son inequívocos.
- **El robo es difícil de verdad, y el número lo admite.** Un bajón sutil y
  sostenido al ~70% de lo normal es casi indistinguible de un uso bajo legítimo
  —y las vacaciones inyectadas lo demuestran—, así que la regla de robo produce
  honestamente falsos positivos. Un recall de ~0,46 significa que la regla caza
  los casos flagrantes (bajones profundos o lecturas negativas/puenteadas) y se
  deja los sutiles. Reportar ~0,46 en vez de esconder el robo dentro de un número
  global es lo honesto.
- **El modelo es una segunda opinión útil, no magia.** Sobre contadores no vistos
  sube el recall global de 0,64 (reglas) a 0,93 (reglas + modelo), a costa de
  precisión (0,77 → 0,61). Ese compromiso lo decide el operador.

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
