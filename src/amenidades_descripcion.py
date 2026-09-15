"""
Extracción de instalaciones desde el texto libre `description` de la API de idealista.

La API permite filtrar por varias instalaciones pero no las devuelve en la respuesta, así
que se reconstruyen leyendo la descripción del anuncio. Criterio: son rasgos que suman
valor, así que quien los tiene tiende a mencionarlos -- lo no mencionado se toma como
ausente (`False`).

Cada amenity queda en uno de cuatro estados: "mencionado", "negado" (el anuncio subraya
que no lo tiene: "sin ascensor", "no dispone de trastero"), "sin_mencion" o
"sin_descripcion" (el anuncio no trae texto). Para el modelo, "sin_descripcion" se trata
igual que "sin_mencion": ausente. El estado se conserva para poder auditar esos anuncios.
Si el mismo texto lo menciona y lo niega, gana la mención y `conflicto` queda a True.

Cada regla es un patrón positivo más exclusiones que miran solo la ventana de texto
alrededor de la coincidencia. `extraer_amenidades` devuelve, además del estado, el
fragmento que lo justifica y los fragmentos descartados con su motivo, para poder
auditar las reglas a mano.
"""

import re
import unicodedata
from dataclasses import dataclass

import pandas as pd

VENTANA = 45

# La negación debe quedar pegada al término: "sin compromiso y con terraza" no niega la terraza.
NEGACION = re.compile(
    r"\b(sin|no tiene|no dispone de|no cuenta con|no incluye|carece de|no hay|ni|ausencia de|falta de)\s"
    r"(?:(?!\b(?:con|y|e|pero|aunque|mas)\b)[^.,;:()!?\n•\-]){0,20}$"
)

ESTADOS = ("mencionado", "negado", "sin_mencion", "sin_descripcion")


@dataclass(frozen=True)
class Regla:
    patron: str
    no_antes: str | None = None
    no_despues: str | None = None
    no_contexto: str | None = None
    requiere_contexto: str | None = None


OPCIONAL_ANTES = (r"(posibilidad|opcion) de( adquirir| comprar| anadir| alquilar)?( una| un)?( plaza de)?\s*$"
                  r"|opcional( plaza de)?\s*$")
OPCIONAL_DESPUES = r"^[\s,:]*(\(?opcional|no incluid|aparte|por separado|a elegir|con coste)"

REGLAS: dict[str, list[Regla]] = {
    "HASTERRACE": [
        Regla(r"\bterraza(s)?\b",
              no_antes=r"(bares|restaurantes|restauracion|hosteleria|oferta de|comercios)[^.]{0,30}$",
              no_despues=r"^[\s,]*(comunitari|de (los )?bares|y restaurantes|de ocio)"),
    ],
    "HASAIRCONDITIONING": [
        Regla(r"aire acondicionado|\bclimatizacion\b|\bclimatizad[oa]s?\b|bomba de calor|\bsplits?\b|\ba/a\b",
              no_antes=r"pre-? ?instalacion( de)?\s*$|preinstalad[oa]s?( de| para)?\s*$"),
    ],
    "HASBOXROOM": [
        Regla(r"\btrastero(s)?\b", no_antes=OPCIONAL_ANTES, no_despues=OPCIONAL_DESPUES),
    ],
    "HASWARDROBE": [
        Regla(r"\barmario(s)?\b[^.;:]{0,25}\bempotrad|\bempotrados?\b[^.;:]{0,15}\barmario(s)?\b"),
    ],
    "HASSWIMMINGPOOL": [
        Regla(r"\bpiscinas?\b",
              no_antes=r"(cerca|proxim\w*|junto|al lado|a pocos (metros|minutos))( de| a| del| al)?( la| las| una)?\s*$",
              no_despues=r"^\s*(municipal|publica|climatizada municipal)"),
    ],
    "HASDOORMAN": [
        Regla(r"\bporter[oa]s?\b|\bporteria\b|\bconserjes?\b|\bconserjeria\b",
              no_antes=r"antigua\s*$",
              no_despues=r"^\s*(automatico|electronico)"),
    ],
    "HASGARDEN": [
        Regla(r"\bjardin(es)?\b",
              no_antes=r"(real|ciudad|orientad[oa]s?( a| hacia)( el| un| los)?|vistas?( a| al| hacia)( el| un| los)?|rodead[oa]s? de|parques y)\s*$",
              no_despues=r"^\s*(botanico|de (sabatini|la villa|el capricho))"),
        Regla(r"\bajardinad[oa]s?\b",
              requiere_contexto=r"urbanizacion|comunitari|zonas? comunes|piscina",
              no_antes=r"vistas?( a| al| hacia)( las| los| unas)?( zonas?| espacios?)?\s*$"),
    ],
    # Calibración: la API sí devuelve estos dos, así que permiten medir el método.
    "HASLIFT": [
        Regla(r"\bascensor(es)?\b",
              no_antes=r"(instalacion|instalar|proyecto|poner|colocar|futuro|previsto)[^.]{0,35}$"),
    ],
    "HASPARKINGSPACE": [
        Regla(r"\bgarajes?\b|\bparking\b|plazas? de aparcamiento|aparcamiento (privado|propio|incluido)",
              no_antes=OPCIONAL_ANTES + r"|(facil|zona de)\s*$",
              no_despues=OPCIONAL_DESPUES + r"|^\s*publico"),
    ],
}

AMENIDADES_API_NO_DEVUELVE = ["HASTERRACE", "HASAIRCONDITIONING", "HASBOXROOM", "HASWARDROBE",
                              "HASSWIMMINGPOOL", "HASDOORMAN", "HASGARDEN"]


def normalizar(texto: str | None) -> str:
    if not isinstance(texto, str):
        texto = ""
    t = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", t)


def _motivo_descarte(t: str, inicio: int, fin: int, regla: Regla) -> str | None:
    antes = t[max(0, inicio - VENTANA):inicio]
    despues = t[fin:fin + VENTANA]
    contexto = antes + t[inicio:fin] + despues
    if NEGACION.search(antes):
        return "negacion"
    if regla.no_antes and re.search(regla.no_antes, antes):
        return "no_antes"
    if regla.no_despues and re.search(regla.no_despues, despues):
        return "no_despues"
    if regla.no_contexto and re.search(regla.no_contexto, contexto):
        return "no_contexto"
    if regla.requiere_contexto and not re.search(regla.requiere_contexto, contexto):
        return "falta_contexto"
    return None


def extraer_amenidades(texto: str | None, reglas: dict[str, list[Regla]] = REGLAS) -> dict[str, dict]:
    t = normalizar(texto)
    resultado: dict[str, dict] = {}
    if not t.strip():
        for amenidad in reglas:
            resultado[amenidad] = {"estado": "sin_descripcion", "conflicto": False,
                                   "evidencia": None, "negacion": None, "descartes": []}
        return resultado
    for amenidad, lista in reglas.items():
        positivos, negaciones, descartes = [], [], []
        for regla in lista:
            for m in re.finditer(regla.patron, t):
                fragmento = t[max(0, m.start() - VENTANA):m.end() + VENTANA]
                motivo = _motivo_descarte(t, m.start(), m.end(), regla)
                if motivo is None:
                    positivos.append(fragmento)
                elif motivo == "negacion":
                    negaciones.append(fragmento)
                else:
                    descartes.append((motivo, fragmento))
        estado = "mencionado" if positivos else "negado" if negaciones else "sin_mencion"
        resultado[amenidad] = {
            "estado": estado,
            "conflicto": bool(positivos and negaciones),
            "evidencia": positivos[0] if positivos else None,
            "negacion": negaciones[0] if negaciones else None,
            "descartes": descartes,
        }
    return resultado


def aplicar(df: pd.DataFrame, columna: str = "description",
            reglas: dict[str, list[Regla]] = REGLAS) -> pd.DataFrame:
    """
    Dos columnas booleanas por amenity: `X` (mencionado) y `X_negado` (negado
    explícitamente), más `sin_descripcion` para identificar los anuncios sin texto, en los
    que todas las instalaciones quedan a False.
    """
    extraido: list[dict[str, dict]] = [extraer_amenidades(x, reglas) for x in df[columna]]
    columnas = {}
    for a in reglas:
        estados: list[str] = [r[a]["estado"] for r in extraido]
        columnas[a] = [e == "mencionado" for e in estados]
        columnas[f"{a}_negado"] = [e == "negado" for e in estados]
    primera = next(iter(reglas))
    columnas["sin_descripcion"] = [r[primera]["estado"] == "sin_descripcion" for r in extraido]
    return pd.DataFrame(columnas, index=df.index)
