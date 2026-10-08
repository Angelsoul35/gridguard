# Guía de estudio — GridGuard

Una guía en español para entender el proyecto a fondo y poder defenderlo en una
entrevista. El código, los comentarios y la documentación técnica están en
inglés (como en un entorno profesional real); esta guía explica el *porqué* de
cada decisión.

## Orden sugerido

1. [01 — La serie del contador y la estacionalidad](01-serie-del-contador-y-estacionalidad.md)
   Qué es una serie de contador inteligente, por qué tiene forma diaria y semanal,
   y por qué cada contador se compara consigo mismo.
2. [02 — Características y línea base robusta](02-caracteristicas-y-linea-base.md)
   Las características por hora, por qué son invariantes a la escala, y por qué la
   línea base usa la mediana.
3. [03 — Las reglas y el modelo opcional](03-reglas-y-modelo.md)
   Las tres reglas deterministas (pico, apagón/plano, robo) y el modelo como
   segunda opinión.
4. [04 — Evaluación honesta](04-evaluacion-honesta.md)
   Por qué se evalúa con contadores no vistos, por tipo, hora a hora, y por qué el
   robo es el caso difícil.
5. [05 — Preguntas de entrevista](05-preguntas-de-entrevista.md)
   Preguntas típicas y cómo responderlas con este proyecto.

## La idea en una frase

Detección de anomalías en series de contadores inteligentes **determinista y
explicable** (picos de consumo, apagones y contadores "pegados", y pérdida no
técnica / robo de electricidad), donde cada alerta dice **por qué** ha saltado, y
que se mide con **honestidad** sobre contadores nunca vistos — sin fingir un
resultado perfecto.
