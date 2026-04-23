class Config:
    """
    Variables de configuración de la librería.
    
    Attributes
    ---
    projected_crs : int or str
        CRS proyectado a utilizar cuando se necesite realizar operaciones
        geométricas (p. ej. encontrar centroies o calcular áreas). Por defecto
        es 5361 (el CRS proyectado para Chile).
    quackosm_working_directory : str
        Directorio en el que QuackOSM guarda archivos cacheados. Por defecto es
        `files`.
    osmconvert_path : str
        Ruta al ejecutable de osmconvert. Por defecto es `osmconvert`.
    alpha_when_roads_shown : float
        Opacidad a utilizar por defecto cuando se muestran los caminos en una
        visualización. Puede ser útil para asegurarse que el color de las
        celdas en el gráfico sea igual al color de la leyenda correspondiente,
        si así se desea. Por defecto es 0.8.  
    """
    
    projected_crs = 5361
    quackosm_working_directory = "files"
    osmconvert_path = "osmconvert"
    alpha_when_roads_shown = 0.8
    
config = Config()