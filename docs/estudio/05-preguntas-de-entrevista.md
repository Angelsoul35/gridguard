# 05 — Preguntas de entrevista

Preguntas típicas sobre este proyecto y cómo responderlas.

### "¿Por qué no usas solo un modelo de machine learning para detectarlo todo?"
Porque un analista que recibe una alerta necesita saber **por qué** ha saltado:
"el contador estuvo a cero 14 horas" es accionable; "el modelo dio 0,87" no. Las
reglas deterministas dan hallazgos explicables y auditables, y el modelo queda
como **suplemento de recall con compuerta de confianza** que solo amplía la red
global cuando está seguro. Separar las dos responsabilidades hace el sistema
defendible ante una cuadrilla de campo.

### "¿Por qué comparas cada contador consigo mismo y no con un umbral global?"
Porque el consumo es heterogéneo: una casa grande gasta diez veces más que un
estudio, y todos gastamos más a las 19:00 que a las 04:00. Un umbral global
marcaría cada casa grande y cada pico de tarde. La línea base por contador —la
mediana por celda `(fin de semana?, hora)`— responde a la pregunta correcta: "¿es
raro para ESTE contador a ESTA hora?".

### "¿Por qué la mediana y no la media para la línea base y los picos?"
Por el **enmascaramiento**: la propia anomalía arrastra la media y la desviación
estándar, bajando su z-score hasta esconderse. La mediana y la MAD son
insensibles a unos pocos extremos, así que la anomalía sobresale en lugar de
camuflarse. La constante 0,6745 escala la MAD para que coincida con el z ordinario
en datos normales.

### "El robo parece el caso difícil, ¿cómo lo distingues de un uso bajo legítimo?"
Por la **forma**. Un contador manipulado sub-registra una casa **habitada**: la
carga sigue su perfil diario (picos de mañana y tarde), solo que escalado. Una
**vacación** es una casa **vacía**: se aplana a un *standby* sin picos. Así que la
regla de robo exige dos cosas sobre el bajón sostenido: que el cociente
`lectura/base` **suavizado** caiga bajo el umbral, y que la ventana **conserve la
forma** (correlación con el perfil ≥ 0,5). La vacación, al aplanarse, tiene
correlación ~0 y no se marca. Resultado medido: robo precisión ~0,99, recall ~0,81,
F1 ~0,89 — en la liga de picos y apagones. Tratar robo y vacación como "el mismo
bajón" es justo el error que evita la prueba de forma. (Suavizar el cociente es lo
que recupera el robo de fracción alta, ~0,70,
cuyas horas ruidosas un umbral por hora dejaba escapar.)

### "¿Cómo garantizas que la evaluación no se engaña a sí misma?"
Reteniendo **contadores enteros** (test nunca visto en train), puntuando **hora a
hora** (un apagón tardío pierde recall), reportando **por tipo** (picos ~0,95,
apagones ~0,96, robo ~0,89 F1) y **promediando varias semillas**. Y dejo
vacaciones reales dentro de los datos para que la evaluación demuestre que la
separación robo-vs-vacación aguanta sobre contadores no vistos, no contra un
espantapájaros.

### "¿Por qué las características son invariantes a la escala?"
Porque el modelo se entrena con unas casas y se aplica a otras **no vistas**. Si
una característica fuera el consumo absoluto, el modelo aprendería el tamaño de
las casas de entrenamiento y fallaría con casas de otro tamaño. Al ser todo
ratios, desviaciones y longitudes de racha, el problema de una casa grande y una
pequeña se ve igual.

### "¿Qué harías para llevarlo a producción?"
Ingerir datos AMI reales, entrenar y ajustar umbrales con **etiquetas reales**
(robos confirmados, partes de corte, mantenimiento), manejar huecos y cambios de
hora/DST, añadir persistencia y usuarios, vigilar el **drift** y reevaluar con la
misma forma de evaluación sin fuga. Y seguramente añadir una regla de "consumo alto
sostenido", que hoy es una zona gris que el pico solo cubre a medias.
