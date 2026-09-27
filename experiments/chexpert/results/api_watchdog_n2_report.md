# Reporte de imposibilidad de ejecución CheXpert con `N_2`

**Fecha de cierre:** 2026-09-26 12:47:01 UTC
**Modelo:** `meta/llama-3.2-90b-vision-instruct`
**Fuente de credencial:** `N_2` (el valor de la clave no se incluye)
**Manifest previsto:** `experiments/chexpert/results/solo_600.csv`

## Resultado

La ejecución no pudo iniciar el primer bloque de 20 casos. El watchdog fue detenido después de registrar 23 pruebas consecutivas de disponibilidad; todas terminaron en timeout durante el preflight del modelo.

La ventana observada fue:

- Inicio del primer preflight: `2026-09-26T04:18:04.924928+00:00`
- Último preflight: `2026-09-26T12:46:33.508258+00:00`
- Bloque alcanzado: `20` solicitado, `0` iniciado

## Conteos

| Métrica | Conteo | Observación |
|---|---:|---|
| Casos completados | 0 | No se inició ningún caso |
| Llamadas de inferencia fallidas | 0 | El runner no llegó a ejecutarse |
| Preflight intentados | 23 | Todos para el bloque `20` |
| Timeouts de preflight | 23 | Sin respuesta dentro de 30 segundos |
| Respuestas HTTP exitosas | 0 | Ninguna respuesta válida del modelo |
| Respuestas `403` en esta vigilancia | 0 | No observadas en el log N2 |
| Respuestas `429` en esta vigilancia | 0 | No observadas en el log N2 |
| Resultados servidos desde cache | 0 | Cache N2 no creada |
| Filas de cache N2 | 0 | `img_cache_llama_n2.jsonl` ausente |
| Bloques válidos | 0 de 5 | `20`, `40`, `60`, `80`, `100` |
| Checkpoints válidos | 0 de 15 | Tres fases por cada bloque |

## Evidencia de la API

La fuente de evidencia es:

`experiments/chexpert/results/api_watchdog_n2_incremental.jsonl`

El log contiene 23 eventos `api_unavailable`, todos con:

```text
error=timeout
http=0
target=20
cache_rows=0
```

El `http=0` indica que no se recibió una respuesta HTTP; por tanto, estos eventos no deben contabilizarse como créditos consumidos ni como resultados del modelo.

## Validación de resultados

No se encontró un bloque completo en `phase2_n20`, `phase2_n40`, `phase2_n60`, `phase2_n80` ni `phase2_n100`.

El directorio histórico `phase2_n100` contiene únicamente un checkpoint baseline vacío y logs de intentos anteriores. No se contabiliza porque no pertenece a una corrida N2 válida y no contiene las tres fases completas.

Los siguientes artefactos permanecieron sin modificación:

- Cache anterior y sus resultados.
- Archivo `.env`.
- Manifest `solo_600.csv`.
- Resultados históricos de `phase2_n100`.

## Causa y recomendación

La causa principal es que el endpoint de NVIDIA no respondió al preflight del modelo exacto dentro del timeout configurado. Al no existir una respuesta real del modelo, el runner nunca inició una muestra y no se generaron resultados parciales.

Se recomienda no continuar automáticamente con los bloques `20`–`100` hasta validar la disponibilidad de la cuenta o de la API N2 mediante una prueba manual que devuelva una respuesta real. El watchdog quedó detenido para evitar reintentos indefinidos.
