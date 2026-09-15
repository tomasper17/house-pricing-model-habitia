import pandas as pd
from typing import Dict, Tuple
import glob
from pathlib import Path


def mapa_censo_2026_a_2016(json_mapping: Dict, xlsx_changes_path: str) -> Dict:
    """
    Actualización del mapeo de secciones censales del INE de 2026 a 2018, basado en un archivo JSON original y un archivo XLSX con los cambios.
    
    Args:
        json_mapping: Original JSON mapping dictionary
        xlsx_changes_path: Path to XLSX file with changes
    
    Returns:
        Updated mapping dictionary
    
    Example:
        updated_mapping = mapa_censo_2026_a_2016(
            'Secciones_Censales_mappings.json',
            'secciones_mapeos_post_2016.xlsx'
        )
    """
    
    
    # Load XLSX
    xlsx_df = pd.read_excel(xlsx_changes_path)
    xlsx_df = xlsx_df[:-2]  # Remove last two rows

    new_mapping = json_mapping.copy()  # Create a copy of the original mapping to modify
    
    # Process each row
    for idx, row in xlsx_df.iterrows():
        distrito = str(int(row['Distrito'])).zfill(2)
        seccion = str(int(row['Sección'])).zfill(3)
        barrio = str(int(row['Barrio'])).zfill(3)
        
        new_entry = {"COD_DIS": distrito, "COD_BAR": barrio}
        
        if pd.notna(row['Fecha de Baja']):

            key_to_add = distrito + seccion
        else:
            # New section created
            old_section_num = str(int(row['Procedencia'])).zfill(3)
            key_to_add = distrito + old_section_num

        new_mapping[key_to_add] = new_entry


    return new_mapping


def crear_indice_criminalidad(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga y procesa datos de incidencias desde archivos CSV, filtrando por tipos de
    incidencia específicos y agregando por barrio para crear un índice de criminalidad.

    El campo Distrito de este fichero no es fiable (barrios de un distrito
    aparecen bajo otros distritos distintos); Barrio es la única
    georreferencia utilizable, así que la agregación se hace por barrio.

    Args:
        data_dir: Ruta al directorio que contiene los archivos CSV de incidencias

    Returns:
        Tupla de (df_agregado_por_barrio, df_desglose_detallado)
        - df_agregado_por_barrio: DataFrame con total de incidentes por barrio,
          ordenado de mayor a menor
        - df_desglose_detallado: DataFrame con desglose por tipo de incidente y barrio

    Ejemplo:
        agg, detallado = crear_indice_criminalidad("data/raw/madrid_incidencias_2026")
    """

    TIPOS_INCIDENCIA = {
        "CONSUMO ALCOHOL/DROGAS EN VIA PUBLICA",
        "FALLECIDOS POR DELITO O CAUSA DESCONOCIDA",
        "HURTOS",
        "INFRACCIONES LEY SEGURIDAD CIUDADANA",
        "RESOLUCION CONFLICTOS PRIVADOS",
        "REYERTAS / AGRESIONES",
        "ROBOS CON FUERZA",
        "ROBOS CON VIOLENCIA / INTIMIDACION",
        "SUSTRACCION DE VEHICULO"
    }

    # Buscar todos los archivos CSV en el directorio
    csv_files = glob.glob(str(Path(data_dir) / "*.csv"))
    print(f"Se encontraron {len(csv_files)} archivos CSV\n")

    # Cargar y concatenar todos los archivos
    dfs = []
    for file in csv_files:
        print(f"Cargando {Path(file).name}...")
        df = pd.read_csv(file, sep=";")
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Total de registros cargados: {len(combined_df)}\n")

    BARRIOS_DESCONOCIDOS = {"#null", "Desconocido"}

    # Filtrar por tipos de incidencia
    filtered_df = combined_df[
        combined_df["Descripcíon tipo de apertura"].isin(TIPOS_INCIDENCIA)
    ]

    # Descartar registros sin barrio identificable
    filtered_df = filtered_df[~filtered_df["Barrio"].isin(BARRIOS_DESCONOCIDOS)]
    filtered_df = filtered_df.dropna(subset=["Barrio"])

    print(f"Registros después del filtro por tipo: {len(filtered_df)}")
    print(f"Tipos de incidencia únicos en datos filtrados: {filtered_df['Descripcíon tipo de apertura'].nunique()}\n")

    # Agregar por barrio
    agg_by_barrio = filtered_df.groupby("Barrio").agg({
        "Incidentes": "sum"
    }).reset_index()

    agg_by_barrio.columns = ["Barrio", "Total_Incidentes"]
    agg_by_barrio = agg_by_barrio.sort_values("Total_Incidentes", ascending=False)

    print("=" * 60)
    print("INCIDENCIAS AGREGADAS POR BARRIO")
    print("=" * 60)
    print(f"Total de incidencias: {agg_by_barrio['Total_Incidentes'].sum()}")
    print(f"Total de barrios: {len(agg_by_barrio)}\n")

    # Desglose detallado por tipo de incidencia y barrio
    detailed_df = filtered_df.groupby(
        ["Barrio", "Descripcíon tipo de apertura"]
    ).agg({
        "Incidentes": "sum"
    }).reset_index()

    detailed_df.columns = ["Barrio", "Tipo_Incidente", "Total"]
    detailed_df = detailed_df.sort_values(["Barrio", "Total"], ascending=[True, False])

    return agg_by_barrio, detailed_df