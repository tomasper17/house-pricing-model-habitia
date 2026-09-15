# Datos

Todas las rutas de esta carpeta se definen en [`src/rutas.py`](../src/rutas.py).

```
data/
├── raw/                          fuentes externas, tal como se descargan
│   ├── idealista18/              Madrid_Sale.rda, Madrid_Polygons.rda, Madrid_POIS.rda
│   ├── ine_seccionado_2018/      SECC_CE_20180101.* (shapefile)
│   ├── madrid_seccionado/        Secciones_Censales_mappings.json, secciones_mapeos_post_2016.xlsx
│   ├── madrid_alquiler/          alquiler_madrid_historico.xlsx
│   ├── madrid_compraventa/       serie_compraventa_madrid.xlsx
│   ├── madrid_incidencias_2026/  837676-*-incidencias-...-policia-municipal.csv
│   ├── madrid_poblacion/         poblacion_distrito_barrio_2026.csv
│   ├── madrid_vulnerabilidad/    vulnerabilidad_barrios_2020.csv
│   └── madrid_equipamientos/     *.geo (GeoRSS de datos.madrid.es)
├── interim/                      salida del notebook 00
│   ├── habitia_madrid_2018.csv   dataset base (separador ;)
│   ├── diccionario_datos.csv     tipo, nulos y cobertura por columna
│   └── asset_distrito.parquet    ASSETID → barrio_code, distrito_code
├── processed/                    salida del notebook 04
│   ├── X_{train,test}_<variante>.parquet
│   └── y_{train,test}.parquet    log(PRICE)
├── results/                      salidas de 05 y 06
│   ├── resultados_baseline.csv
│   ├── resultados_fine_tuning.csv
│   └── modelos_seleccionados_06.json
├── models/                       salida de 07 y del predictor exportable
│   ├── ganador_global.joblib
│   ├── produccion.joblib
│   └── paquete_produccion/       `python -m src.predictor exportar`
└── api_idealista/                respuestas guardadas de la API y predicciones (08)
```

`raw/` no se modifica nunca. Todo lo demás se regenera ejecutando los notebooks en orden.

## Fuentes (`raw/`)

| carpeta | fuente | contenido | lo usa |
|---|---|---|---|
| `idealista18/` | [idealista18](https://github.com/paezha/idealista18) (Rey-Blanco et al., 2024) | anuncios de venta de Madrid en 2018 con catastro y distancias; zonas; puntos de interés (Sol, metro, Castellana) | 00, 08, `produccion` |
| `ine_seccionado_2018/` | INE | seccionado censal a 1-1-2018 (ETRS89 / UTM 30N) | 00, `produccion` |
| `madrid_seccionado/` | Ayuntamiento de Madrid | sección censal → barrio y distrito (2026) y altas/bajas de secciones desde 2016 | 00, `produccion` |
| `madrid_alquiler/` | Ayuntamiento de Madrid (SERPAVI) | nº de viviendas en alquiler y renta por sección censal, una hoja por año (2011–2024) | 00, `indices_precio`, `produccion` |
| `madrid_compraventa/` | Ayuntamiento de Madrid (estadística registral) | precio medio declarado de compraventa (€/m²) por distrito y barrio, 2007–2025 | 07, 08, `indices_precio` |
| `madrid_incidencias_2026/` | Ayuntamiento de Madrid (datos.madrid.es, 837676) | incidencias recibidas por la Policía Municipal, 2026 | 00, `crime_indices` |
| `madrid_poblacion/` | Ayuntamiento de Madrid | población por distrito y barrio, 2026 | 00, `crime_indices` |
| `madrid_vulnerabilidad/` | Ayuntamiento de Madrid | índice de vulnerabilidad por barrio, 2020 | 00, `vulnerability_indices` |
| `madrid_equipamientos/` | datos.madrid.es (200761, 212769, 212808, 300614) | parques y jardines, atención médica, espacios deportivos y centros educativos | 00, `poi_distances` |

## Dataset base (`interim/habitia_madrid_2018.csv`)

Una fila por inmueble de venta en Madrid capital (75.469 filas, 43 columnas, llave `ASSETID`).
Lo construye [`notebooks/00_construccion_dataset.ipynb`](../notebooks/00_construccion_dataset.ipynb).
Tres decisiones de construcción condicionan todo lo que viene después (detalladas en 01 §0):

1. **Deduplicación por `ASSETID` con el `PRICE` mínimo.** `idealista18` tiene 94.815 anuncios
   de 75.804 inmuebles (uno por trimestre en venta). Se conserva el precio más bajo, el más
   cercano al de cierre: el modelo predice el mínimo anunciado en 2018, no el precio de venta.
2. **Barrio (131 unidades) como unidad geográfica**, asignado por unión espacial contra las
   secciones censales del INE disueltas a barrio. La sección censal es demasiado fina para el
   ruido de las coordenadas. El `LOCATIONID` de idealista *parece* un código de sección del INE
   pero no lo es, y no se usa.
3. **Coordenadas con ruido de anonimización.** Rey-Blanco et al. reportan un desplazamiento
   medio de 45 m (σ ≈ 38 m por eje, p95 ≈ 94 m): la resolución espacial fiable es de ~100 m.

Se descartan además 334 anuncios del barrio 194 (El Cañaveral), sin dato de alquiler porque
el barrio no existía como tal en la hoja de alquiler usada.

| columnas | descripción |
|---|---|
| `ASSETID` | identificador del inmueble |
| `PRICE`, `UNITPRICE` | precio de venta anunciado (€) y precio por m² (fuga de información: no es *feature*) |
| `CONSTRUCTEDAREA`, `ROOMNUMBER`, `BATHNUMBER` | superficie construida (m²), habitaciones, baños |
| `FLOORCLEAN` | planta (96,1 % de cobertura) |
| `ISDUPLEX`, `ISSTUDIO`, `ISINTOPFLOOR` | tipología (0/1) |
| `HASTERRACE`, `HASLIFT`, `HASAIRCONDITIONING`, `HASPARKINGSPACE`, `HASBOXROOM`, `HASWARDROBE`, `HASSWIMMINGPOOL`, `HASDOORMAN`, `HASGARDEN` | instalaciones declaradas por el anunciante (0/1) |
| `ISPARKINGSPACEINCLUDEDINPRICE`, `PARKINGSPACEPRICE` | detalle de la plaza de garaje |
| `HASNORTHORIENTATION`, `HASSOUTHORIENTATION`, `HASEASTORIENTATION`, `HASWESTORIENTATION` | orientación (0/1, no excluyentes) |
| `CONSTRUCTIONYEAR` | año de construcción declarado (40,1 % de cobertura) |
| `CADCONSTRUCTIONYEAR`, `CADMAXBUILDINGFLOOR`, `CADDWELLINGCOUNT`, `CADASTRALQUALITYID` | catastro: año, plantas del edificio, viviendas del edificio, calidad catastral |
| `BUILTTYPEID_1`, `BUILTTYPEID_2`, `BUILTTYPEID_3` | tipología constructiva catastral (excluyentes) |
| `DISTANCE_TO_CITY_CENTER`, `DISTANCE_TO_METRO`, `DISTANCE_TO_CASTELLANA` | distancia (km) a Puerta del Sol, al metro más cercano y al eje de la Castellana (`idealista18`) |
| `alq_mediana_eur_m2_barrio` | renta mediana de alquiler (€/m²·mes) de las secciones del barrio, ponderada por nº de viviendas en alquiler |
| `dist_centro_educativo_m`, `dist_parque_m`, `dist_espacio_deporte_m`, `dist_centro_medico_m` | distancia (m) al equipamiento municipal más cercano de cada tipo |
| `delitos_per_10k_barrio` | incidencias de policía por 10.000 habitantes del barrio |
| `indice_vulnerabilidad` | índice de vulnerabilidad del barrio |

Los campos catastrales (`CAD*`, `BUILTTYPEID_*`) vienen precalculados en `idealista18` a partir
de la dirección real. No existen para un anuncio de la API de idealista, por eso el modelo de
producción (`arboles_desplegable`) prescinde de ellos.

`asset_distrito.parquet` guarda `barrio_code` y `distrito_code` por `ASSETID` aparte, porque no
son *features* del modelo: los usan el 07 (error por distrito) y el 08 (comprobación de la
asignación de barrio y variables de barrio en producción).

## Variantes de modelado (`processed/`)

El 04 escribe un split 80/20 (semilla 42) y estas variantes de la matriz de variables:

| variante | contenido |
|---|---|
| `lineal`, `lineal_completo` | vía lineal (formas funcionales del 02), sin / con POI y orientación |
| `incluido`, `desplegable` | vía lineal podada por el 03, con / sin catastro |
| `arboles`, `arboles_completo`, `arboles_podado` | vía árboles, sin POI y orientación / con ellas / con la poda del 03 |
| `*_desplegable` | la variante de árboles correspondiente sin catastro, `CADDWELLINGCOUNT` ni `ISINTOPFLOOR` |
