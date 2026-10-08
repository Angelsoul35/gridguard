# 01 — La serie del contador y la estacionalidad

## Qué medimos

Un contador inteligente (smart meter, parte de una red AMI: *Advanced Metering
Infrastructure*) registra el consumo eléctrico de una vivienda a intervalos
regulares. Aquí trabajamos con lecturas **horarias**: para cada casa tenemos una
serie temporal de kWh consumidos cada hora durante varias semanas.

El objetivo de GridGuard es marcar **anomalías** en esa serie:

- **pico de consumo** (consumption spike): un salto brusco y grande,
- **apagón o contador plano** (outage / flatline): lecturas a cero (corte) o
  constantes (contador "pegado" o averiado),
- **robo / manipulación** (theft / non-technical loss): consumo mantenido muy por
  debajo de lo normal, o lecturas negativas / puenteadas (bypass).

## Por qué el consumo tiene forma

El consumo eléctrico doméstico **no** es aleatorio: sigue patrones muy marcados.

- **Perfil diario:** poco consumo de madrugada, una subida por la mañana, un valle
  durante la jornada y el **pico fuerte por la tarde-noche** (18:00-22:00), cuando
  la gente vuelve a casa, cocina, enciende luces y calefacción/aire.
- **Perfil semanal:** los fines de semana el perfil es más plano y con más consumo
  a mediodía, porque la gente está en casa.

En el código, esto son dos vectores de 24 multiplicadores (`_WEEKDAY_PROFILE` y
`_WEEKEND_PROFILE`) que definen la "forma" esperada. La serie sintética de cada
casa es: `escala_de_la_casa × forma_estacional × ruido`.

## Por qué cada contador se compara CONSIGO MISMO

Las casas son muy distintas: una vivienda grande gasta diez veces más que un
estudio. Y todo el mundo gasta más a las 19:00 que a las 04:00. Si usáramos un
umbral global, marcaríamos como anómala cada casa grande y cada pico de tarde.

La pregunta correcta no es "¿esta lectura es alta?", sino **"¿esta lectura es rara
para ESTE contador, a ESTA hora de ESTE tipo de día?"**. Por eso la línea base se
estima por contador: la **mediana** de todas las lecturas que caen en la misma
celda `(fin de semana?, hora del día)` de su propia serie (48 celdas). Así el
detector se adapta a cada casa sin configuración por vivienda y sin necesitar
datos de entrenamiento (ver ADR 0001).

## Los datos son sintéticos (y etiquetados)

Generamos los datos nosotros (`generate.py`), lo que nos da dos cosas:

1. **Etiquetas verdaderas:** sabemos exactamente qué horas son anómalas y de qué
   tipo, así que podemos medir precisión y exhaustividad de verdad.
2. **Confusor a propósito:** inyectamos también **vacaciones** legítimas
   (etiquetadas *normal*). Una vacación es una casa **vacía**: el consumo se aplana
   a un nivel de *standby* sin picos diarios, a diferencia del robo, que **escala**
   el perfil pero conserva su forma. Dejarlas dentro obliga al detector a
   distinguir *forma aplanada* de *forma conservada*, en vez de marcar todo bajón
   como robo (ver 03 y 04).

Todo es offline y determinista: misma semilla, mismos datos.
