import rich
from rich.console import Console
from rich.table import Table

class Tabla:
    console: Console = None
    table: Table = None
    PASO: int = 0
    ultimo_no_sup: int = 0
    ultimo_sup: int = 0


    def __init__(self):
        self.console = Console()
        self.table = Table(title="Resultados", show_header=True, header_style="bold magenta")
        self.table.add_column("Paso", justify="center", style="green")
        self.table.add_column("Secuencias evaluadas", style="dim", width=12)
        self.table.add_column("Aciertos nosup")
        self.table.add_column("Aciertos sup")
        self.table.add_column("Precisión nosup", justify="center")
        self.table.add_column("Precisión sup", justify="center")

    def imprimir_resultados(self, secuencias_evaluadas: int, aciertos_nosup: int, aciertos_sup: int):

        if self.ultimo_no_sup >= aciertos_nosup: color_nosup = "red"
        else: color_nosup = "green"
        if self.ultimo_sup >= aciertos_sup: color_sup = "red"
        else: color_sup = "green"

        self.ultimo_no_sup = aciertos_nosup
        self.ultimo_sup = aciertos_sup


        porcentaje_nosup = str(round(float(aciertos_nosup)/secuencias_evaluadas * 100, 2)) + "%"
        porcentaje_sup = str(round(float(aciertos_sup)/secuencias_evaluadas * 100, 2)) + "%"

        self.table.add_row(
            str(self.PASO),
            str(secuencias_evaluadas),
            f"[{color_nosup}]{str(aciertos_nosup)}[/{color_nosup}]",
            f"[{color_sup}]{str(aciertos_sup)}[/{color_sup}]",
            f"[{color_nosup}]{str(porcentaje_nosup)}[/{color_nosup}]",
            f"[{color_sup}]{str(porcentaje_sup)}[/{color_sup}]"
        )

        self.PASO += 1
        self.console.print(self.table)

# tabla = Tabla()
# tabla.imprimir_resultados(450, 5, 5)
# tabla.imprimir_resultados(450, 20, 75)
# tabla.imprimir_resultados(450, 5, 5)