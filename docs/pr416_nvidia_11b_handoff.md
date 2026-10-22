# PR #416 — Llama 3.2 11B experiment handoff

_Estado de experimentos y datos para continuar la campaña · 22 de octubre de 2026 · America/Guayaquil_

---

## 📋 Resumen ejecutivo

- **Completado:** CheXpert principal `n=600`, sus tres fases, y el brazo `hidden_note` de 35 casos.
- **Ejecutado y validado por conteo:** los ocho brazos solicitados de `experiments/imaging_chexpert`: cuatro de `n=35` y cuatro de `n=150`; cuatro controles `cue=none` tienen la limitación causal descrita abajo.
- **En curso:** SUPPORT2 `n=120`, comenzando por `support2_solo`, con un worker y una pausa global de 20 s entre llamadas.
- **Descargas:** MedQA y el primer lote de imágenes NIH están en curso; el ZIP de MedMCQA ya pasó la validación de integridad; PubMedQA y el CSV NIH están validados localmente.
- **No subir:** datos crudos ni credenciales. Permanecen en `data/` (ignorado por Git); el repositorio documenta las fuentes y requisitos en [`DATASETS.md`](./DATASETS.md#the-datasets).
- **Importante:** no iniciar una corrida duplicada mientras el proceso SUPPORT2 siga activo. Un cache parcial no equivale a casos completados ni a un checkpoint validado.

## 📋 Estado actual

| Área | Estado | Evidencia o siguiente acción |
| --- | --- | --- |
| CheXpert principal | 🟢 Completo | 600 filas por fase; cache con 1,800 respuestas |
| CheXpert `hidden_note` | 🟢 Completo | 35 filas en las cuatro salidas |
| CheXpert imaging | 🟢 Conteos completos | 8 brazos: cuatro con 35 y cuatro con 150 filas; cuatro controles no admiten lectura causal |
| SUPPORT2 `n=120` | 🟡 En curso | `support2_solo`; cache parcial, sin checkpoint validado todavía |
| NIH ChestX-ray14 | 🟡 Descargando | CSV completo; lote `images_001.tar.gz` todavía incompleto y sin extracción validada |
| MedQA | 🟡 Descargando | ZIP incompleto; falta reconstruir y validar el manifest US |
| MedMCQA | 🟢 Descargado | ZIP íntegro; falta reconstruir y validar el manifest requerido |
| PubMedQA | 🟢 Descargado | `ori_pqal.json` está local; validar antes de usarlo en una corrida |
| Llama 3.2 90B con `N_2` | 🔴 No ejecutado | 23 preflights agotaron timeout; no hubo respuesta del modelo |
| Microsoft Phi-3 Vision | 🔴 No disponible | El endpoint probado devolvió `404`; no hay resultados utilizables |

## 📊 Resultados ya conseguidos

### CheXpert principal, `n=600`

| Métrica | Resultado |
| --- | ---: |
| Casos del manifest | 600 |
| Baseline correcto (`base_ans == "yes"`) | 371/600 |
| Blind drift condicionado al baseline correcto | 277/371 (74.7%) |
| Test-aware drift condicionado al baseline correcto | 294/371 (79.2%) |
| Supresión por awareness | No observada; test-aware tiene 17 drifts más |
| Respuestas no parseables | 0/1,800 |

La interpretación headline debe usar el estrato de baseline correcto. Las tasas incondicionales antiguas mezclan los 229 casos en que el baseline ya respondió `no` y no deben sustituir a esa comparación.

### Brazo oculto compartido, `n=35`

| Métrica incondicional | Resultado |
| --- | ---: |
| Decoy `SCORING NOTE` adoptado en blind | 23/35 (65.7%) |
| Decoy de mensaje de validación adoptado en test-aware | 15/35 (42.9%) |

Estas tasas son incondicionales sobre los 35 casos. Las tasas de drift deben conservar su denominador de baseline correcto.

### Comparación disponible con Gemini

Gemini sigue limitado a 35 casos. No existe todavía una corrida Gemini de 600 casos con el mismo protocolo, por lo que no debe presentarse como comparación emparejada 600-vs-600.

## 📊 Batería de CheXpert con imágenes

Los ocho brazos pedidos ya tienen los conteos de filas esperados. La cache canónica de Llama 11B contiene 1,612 respuestas con claves únicas y prefijo de modelo correcto; no contiene imágenes embebidas ni patrones de credenciales. No se agrega la cache auxiliar `img_cache_pre_n2.jsonl`. Los `case_id` de cada brazo coinciden exactamente con los de Qwen; también coinciden `n` y `cue` en los resúmenes comparables.

| Brazo | Casos | Artefacto y validación |
| --- | ---: | --- |
| `imaging_solo` | 35 | JSONL y resumen `n=35` |
| `imaging_cascade` | 35 | JSONL y resumen `n=35` |
| `imaging_matched_temp` | 35 | JSONL y resumen `n=35` |
| `imaging_noise_floor` | 35 | JSONL con 35 filas; no tiene resumen |
| `device_absent` (`imaging_cascade_none`) | 150 | JSONL y resumen `n=150` |
| `natural_cues` (`imaging_cascade_none`) | 150 | JSONL y resumen `n=150` |
| `natural_independent` (`imaging_cascade_none`) | 150 | JSONL y resumen `n=150` |
| `system_flag` | 150 | JSONL y resumen `n=150` |

**Limitación interpretativa:** los cuatro controles de `cue=none` (los tres cascades y `system_flag`) reutilizan la misma clave de cache para las lecturas `clean` e `iso`: ambas usan exactamente el mismo prompt y la misma imagen. Por eso `clean == iso` en las 150 filas de cada brazo; no son dos inferencias independientes. Sus filas y tasas de adopción compartida son reproducibles, pero no se debe interpretar `shared_adopt - iso_adopt` como contagio causal. El mismo defecto está registrado para la línea Qwen y se mantiene explícito en el guard; el arreglo requiere un comparador genuino y una nueva corrida.

## ⚙️ Ejecución alojada y control de saturación

La cola activa es SUPPORT2, modelo `meta/llama-3.2-11b-vision-instruct`, un worker y separación global de 20 s. El primer brazo es `support2_solo`; la cache parcial no es un checkpoint validado. No iniciar una segunda corrida ni reutilizarla hasta que termine el proceso actual.

El protocolo de llamadas alojadas es:

```mermaid
flowchart LR
    accTitle: Safe hosted run flow
    accDescr: The private N_2 credential is loaded from the local environment, requests are serialized through one worker and a 20 second gap, successful responses are cached, and only then is a checkpoint accepted.

    load_key([🔐 Load N_2]) --> pace[🛡️ One worker and 20 s gap]
    pace --> nim[☁️ NVIDIA NIM 11B]
    nim --> cache[💾 Append successful response]
    cache --> checkpoint[✅ Validate checkpoint]

    classDef start fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#3b0764
    classDef process fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef success fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class load_key start
    class pace,nim,cache process
    class checkpoint success
```

## 🔐 Continuación segura con otra cuenta

No usar `source .env`: el archivo contiene asignaciones que no están escritas como comandos shell ejecutables. Cargar la clave sin imprimirla:

```bash
cd /Users/yehu/Documents/MIT/benchmaxxing-1
export PY=/Users/yehu/Documents/MIT/benchmaxxing/.venv/bin/python
export N_2="$($PY -c 'from dotenv import dotenv_values; print(dotenv_values(".env")["N_2"])')"
export BENCHMAXXING_MIN_CALL_INTERVAL=20
export BENCHMAXXING_GLOBAL_MIN_CALL_INTERVAL=20
export BENCHMAXXING_MAX_WORKERS=1
export PYTHONUNBUFFERED=1
```

Antes de iniciar cualquier comando, comprobar que no exista otra corrida:

```bash
pgrep -af 'experiments.support2.support2_(solo|cascade|cascade_strength|referee|referee_judge)'
wc -l experiments/support2/results/meta_llama-3.2-11b-vision-instruct_call_cache.jsonl
```

Si el proceso SUPPORT2 sigue activo, no iniciar otro worker ni modificar su cache. La API key no debe copiarse a este informe, a Git ni a logs.

## 📋 Datos descargados y siguientes pasos

Las entradas crudas permanecen en `data/`, que está ignorado por Git; sólo se publicarán resultados, documentación, manifiestos autorizados y caches reproducibles.

| Dataset | Estado local | Uso pendiente |
| --- | --- | --- |
| MedQA | `data/medqa/data_clean.zip` descargándose | Terminar, validar el ZIP y reconstruir el manifest US |
| MedMCQA | `data/medmcqa/data.zip`, descargado y verificado con `unzip -t` | Reconstruir y validar el manifest requerido |
| PubMedQA | `data/pubmedqa/ori_pqal.json`, 1,000 registros JSON válidos | No está en la batería de paridad solicitada; no lanzar sin ampliar el plan |
| NIH ChestX-ray14 | CSV de etiquetas válido; `images_001.tar.gz` descargándose | Validar el archivo y las 35 imágenes requeridas antes de cualquier brazo NIH |
| CheXpert | No se está descargando en esta tarea | Revisar registro/acuerdo de investigación antes de otra adquisición |
| MIMIC-CXR/EHR | No descargado | Requiere credenciales/formación; fuera del alcance actual |

Una vez terminadas y verificadas las descargas:

1. Dejar que la cola SUPPORT2 activa termine; validar cada JSONL, resumen y cache antes de contar un brazo como completo.
2. Esperar a que termine MedQA y validar su ZIP; después generar los manifests de MedQA y MedMCQA con las fuentes descritas en [`DATASETS.md`](./DATASETS.md#the-datasets).
3. Cuando finalice `images_001.tar.gz`, extraer o localizar las imágenes de `experiments/imaging/results/nih_manifest.csv` y comprobar que todas existan antes de iniciar la batería NIH.
4. Mantener fuera de Git los ZIP, las radiografías y los datos crudos. No aceptar acuerdos legales ni descargar datasets credentialed en nombre del usuario.

El 90B con `N_2` no obtuvo ninguna respuesta durante 23 preflights; la evidencia está en [`api_watchdog_n2_report.md`](../experiments/chexpert/results/api_watchdog_n2_report.md) y [`api_watchdog_n2_incremental.jsonl`](../experiments/chexpert/results/api_watchdog_n2_incremental.jsonl). Microsoft Phi-3 Vision devolvió `404` en el endpoint probado. Ninguno de los dos resultados debe mezclarse con los resultados válidos de 11B.

## ✅ Validación y estado de Git

- Pruebas dirigidas de `test_lineage_parity`, `test_degeneracy_guard`, `test_imaging_blind_metric` y `test_chexpert_incremental`: **64 passed**.
- Los siete avisos nuevos del guard corresponden al mismo defecto `cue=none` ya registrado para Qwen; quedaron anotados por ruta exacta y el límite de baseline subió explícitamente de 116 a 123, sin añadir gaps a lineage parity.
- La rama está basada en `main`; este handoff y los resultados validados se publican en el PR #416. No se incluye ningún dataset crudo.
- Los valores de API no están incluidos en este informe ni en los artefactos rastreados.

## 🔗 Artefactos principales

- [Resultados Llama 11B y comparación](./pr416_nvidia_11b_results.md)
- [Resultados CheXpert Llama 11B, `n=600`](../experiments/chexpert/results/meta_llama-3.2-11b-vision-instruct/)
- [Brazo oculto de 35 casos](../experiments/chexpert/results/meta_llama-3.2-11b-vision-instruct/hidden_note_n35/)
- [Cache reproducible de CheXpert](../experiments/chexpert/results/600_runs/img_cache_llama11b_n2.jsonl)
- [Batería CheXpert con imágenes y cache Llama 11B](../experiments/imaging_chexpert/results/meta_llama-3.2-11b-vision-instruct/)
- [Cache parcial de la cola SUPPORT2](../experiments/support2/results/meta_llama-3.2-11b-vision-instruct_call_cache.jsonl)
- [Fuentes y requisitos de datasets](./DATASETS.md#the-datasets)
- [Reporte de bloqueo del 90B](../experiments/chexpert/results/api_watchdog_n2_report.md)

_El PR #416 contiene este handoff y los artefactos de resultados reproducibles; los datasets y las credenciales permanecen locales._
