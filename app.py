"""
Explorador-IA (Scout-AI) - App Streamlit

Mismo método que los notebooks: StandardScaler + k-NN con distancia coseno,
usando los perfiles de métricas (A/B/C) de los experimentos de cada posición.
Opcional: ajuste por contexto del equipo (src/build_contexto.py) y valor de
mercado con filtro de presupuesto (src/build_valores.py).

Uso (desde la carpeta scout-ai, con el venv activado):
    streamlit run app.py
"""
import difflib
import re
import unicodedata
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"

# ----------------------------------------------------------------------------
# CONFIGURACIÓN POR POSICIÓN (sale de los 12 experimentos)
# ----------------------------------------------------------------------------
CONFIG = {
    "Delantero": {
        "fuente": "Premier League 2022/23 (FBref)",
        "archivos": ("players_full.csv",),
        "col_pos": "position",
        "valores_pos": ("FW", "FW,MF"),
        "col_min": "minutes",
        "minimo": 600,
        "perfiles": {
            "A - Completo": (
                "Goles, asistencias, tiros y centros.",
                ["goals_90", "assists_90", "goals_assists_90",
                 "shots_on_target", "goals_per_shot", "crosses"],
            ),
            "B - Goleador puro": (
                "Solo finalización.",
                ["goals_90", "shots_on_target", "goals_per_shot"],
            ),
            "C - Creativo": (
                "Creación y asociación.",
                ["assists_90", "goals_assists_90", "crosses", "goals_per_shot"],
            ),
        },
    },
    "Mediocampista": {
        "fuente": "La Liga 2015/16 (StatsBomb)",
        "archivos": ("statsbomb_players.csv",),
        "col_pos": "position",
        "valores_pos": ("Left Defensive Midfield", "Right Defensive Midfield",
                        "Center Defensive Midfield"),
        "col_min": None,
        "minimo": None,
        "perfiles": {
            "A - Completo": (
                "Visión general del mediocampista defensivo.",
                ["interceptions", "ball_recoveries", "duel_win_pct",
                 "pass_completion_pct", "pressures"],
            ),
            "B - Destructor": (
                "Quita la pelota con el cuerpo.",
                ["interceptions", "duels_won", "duel_win_pct"],
            ),
            "C - Pivote moderno": (
                "Recupera, presiona y distribuye.",
                ["ball_recoveries", "pass_completion_pct", "pressures"],
            ),
        },
    },
    "Defensor": {
        "fuente": "La Liga 2015/16 (StatsBomb)",
        "archivos": ("statsbomb_players.csv",),
        "col_pos": "position",
        "valores_pos": ("Left Center Back", "Right Center Back"),
        "col_min": None,
        "minimo": None,
        "perfiles": {
            "A - Completo": (
                "Visión general del central.",
                ["interceptions", "ball_recoveries", "duel_win_pct",
                 "duels_won", "pressures"],
            ),
            "B - Clásico": (
                "Interceptar y ganar duelos.",
                ["interceptions", "duels_won", "duel_win_pct"],
            ),
            "C - Moderno": (
                "Recuperar y presionar alto.",
                ["ball_recoveries", "pressures", "duel_win_pct"],
            ),
        },
    },
    "Portero": {
        "fuente": "La Liga 2015/16 (StatsBomb, eventos Goal Keeper)",
        # Se busca primero en el CSV general y después en uno propio de porteros
        "archivos": ("statsbomb_players.csv", "statsbomb_porteros.csv"),
        "col_pos": "position",
        "valores_pos": ("Goalkeeper",),
        "col_min": "total_intervenciones",
        "minimo": 50,
        "perfiles": {
            "A - Completo": (
                "Volumen y efectividad.",
                ["pct_paradas", "paradas_exitosas", "salidas",
                 "total_intervenciones"],
            ),
            "B - Bajo los palos": (
                "Calidad bajo los tres palos (el más útil).",
                ["pct_paradas", "pct_goles_recibidos", "paradas_exitosas"],
            ),
            "C - Moderno": (
                "Participación y salidas.",
                ["salidas", "pct_paradas", "total_intervenciones"],
            ),
        },
    },
}


# ----------------------------------------------------------------------------
# DATOS
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cargar(archivos, col_pos, valores_pos, col_min, minimo, necesarias):
    """Devuelve (df, None) o (None, lista_de_errores)."""
    errores = []
    for nombre in archivos:
        ruta = PROCESSED / nombre
        if not ruta.exists():
            errores.append(f"No existe el archivo {ruta}")
            continue
        df = pd.read_csv(ruta)
        if "player" not in df.columns:
            errores.append(f"{nombre} no tiene la columna 'player'.")
            continue
        faltan = [m for m in necesarias if m not in df.columns]
        if faltan:
            errores.append(f"{nombre} no tiene las columnas: {', '.join(faltan)}")
            continue
        if col_pos in df.columns:
            valores_disp = sorted(df[col_pos].dropna().astype(str).unique())
            df = df[df[col_pos].isin(valores_pos)]
            if df.empty:
                errores.append(
                    f"{nombre}: ningún jugador con posición {list(valores_pos)}. "
                    f"Valores disponibles: {valores_disp}"
                )
                continue
        if col_min and col_min in df.columns:
            df = df[df[col_min] >= minimo]
        df = df.drop_duplicates(subset="player").reset_index(drop=True)
        if df.empty:
            errores.append(f"{nombre}: no quedan jugadores tras el filtro de mínimo.")
            continue
        return df, None
    return None, errores


def buscar_similares(df, metricas, idx):
    """StandardScaler + k-NN coseno (igual que src/similarity.py).
    Devuelve TODOS los jugadores ordenados por similitud; el filtro de
    presupuesto y el top-k se aplican después."""
    X = df[metricas].fillna(0)
    Xs = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(metric="cosine", n_neighbors=len(df))
    nn.fit(Xs)
    dist, ind = nn.kneighbors([Xs[idx]])
    filas = [(i, (1 - d) * 100) for i, d in zip(ind[0], dist[0]) if i != idx]
    return pd.DataFrame(filas, columns=["idx", "similitud"])


@st.cache_data(show_spinner=False)
def cargar_valores():
    """Valores de mercado de Transfermarkt (generados con src/build_valores.py)."""
    ruta = PROCESSED / "valores_mercado.csv"
    if not ruta.exists():
        return None
    return pd.read_csv(ruta)


def millones(valor):
    return "Sin dato" if pd.isna(valor) else f"{valor / 1e6:,.1f} M€"


# ----------------------------------------------------------------------------
# VALIDACIÓN DE LA BÚSQUEDA (misma lógica que src/validator.py)
#   1) nombre encontrado en la posición elegida   -> se usa
#   2) encontrado en OTRA posición o fuente        -> se explica por qué no se compara
#   3) nombre parecido (mal escrito)               -> "¿Quisiste decir...?"
#   4) nada                                        -> no está en la base (retirado, otra liga...)
# ----------------------------------------------------------------------------
def normalizar(texto):
    texto = unicodedata.normalize("NFKD", str(texto))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"[^a-z ]", " ", texto.lower()).split()


def coincide(consulta, nombre, umbral):
    """True si cada palabra escrita se parece (>= umbral) a alguna palabra del nombre."""
    toks = normalizar(nombre)
    if not consulta or not toks:
        return False
    return all(
        any(q == t or t.startswith(q) and len(q) >= 3
            or difflib.SequenceMatcher(None, q, t).ratio() >= umbral for t in toks)
        for q in consulta
    )


def posicion_de(posicion_cruda):
    """Traduce la posición cruda del CSV a las posiciones de la app (o None si no se cubre)."""
    for nombre_pos, c in CONFIG.items():
        if posicion_cruda in c["valores_pos"]:
            return nombre_pos
    return None


@st.cache_data(show_spinner=False)
def base_completa():
    """Todos los jugadores de todas las fuentes, para explicar por qué alguien no aparece."""
    fuentes = {
        "players_full.csv": "Premier League 2022/23",
        "statsbomb_players.csv": "La Liga 2015/16",
        "statsbomb_porteros.csv": "La Liga 2015/16",
    }
    partes = []
    for archivo, liga in fuentes.items():
        ruta = PROCESSED / archivo
        if ruta.exists():
            d = pd.read_csv(ruta, usecols=lambda c: c in ("player", "position"))
            d["liga"] = liga
            partes.append(d)
    if not partes:
        return pd.DataFrame(columns=["player", "position", "liga"])
    return pd.concat(partes).dropna(subset=["player"]).drop_duplicates(["player", "liga"])


def validar_busqueda(texto, df, posicion):
    """Devuelve (estado, datos). Estados: ok, varios, otra_posicion, sugerencias, no_encontrado."""
    consulta = normalizar(texto)
    nombres = df["player"].astype(str)

    base = base_completa()
    base_nombres = base["player"].astype(str)

    def palabra_completa(n):
        toks = normalizar(n)
        return bool(consulta) and all(q in toks for q in consulta)

    # Orden de prioridad (no adivina en silencio):
    #   1) palabra completa en esta posición, 2) palabra completa en otra posición/liga,
    #   3) comienzo de palabra en esta posición, 4) comienzo de palabra en otra posición/liga.
    # Así "Rodri" encuentra a Rodri (Premier) antes que a "Rodrigo" (La Liga).
    for criterio in (palabra_completa, lambda n: coincide(consulta, n, 1.01)):
        exactos = [n for n in nombres if criterio(n)]
        if len(exactos) == 1:
            return "ok", exactos[0]
        if len(exactos) > 1:
            return "varios", sorted(exactos)
        fuera = base[[criterio(n) for n in base_nombres]]
        if not fuera.empty:
            return "otra_posicion", fuera

    parecidos = [n for n in nombres if coincide(consulta, n, 0.75)]
    if parecidos:
        return "sugerencias", sorted(parecidos)[:5]
    return "no_encontrado", None


def grafico_radar(df, metricas, jugadores: dict):
    """Radar con percentiles (0-100) dentro de la posición."""
    pct = df[metricas].fillna(0).rank(pct=True) * 100
    fig = go.Figure()
    for idx, nombre in jugadores.items():
        vals = pct.loc[idx, metricas].tolist()
        fig.add_trace(go.Scatterpolar(
            r=vals + vals[:1], theta=[etiqueta(m) for m in metricas + metricas[:1]],
            fill="toself", name=nombre, opacity=0.55,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=110, r=110, t=40, b=40),
        height=520,
    )
    return fig



# ----------------------------------------------------------------------------
# NOMBRES PARA EL USUARIO Y AJUSTE POR CONTEXTO
# ----------------------------------------------------------------------------
ETIQUETAS = {
    "goals_90": "Goles por 90'", "assists_90": "Asistencias por 90'",
    "goals_assists_90": "Goles + asist. por 90'", "shots_on_target": "Remates al arco",
    "goals_per_shot": "Goles por remate", "crosses": "Centros",
    "interceptions": "Intercepciones", "ball_recoveries": "Recuperaciones",
    "duel_win_pct": "% duelos ganados", "pass_completion_pct": "% pases completados",
    "pressures": "Presiones", "duels_won": "Duelos ganados",
    "pct_paradas": "% de paradas", "paradas_exitosas": "Paradas",
    "salidas": "Salidas", "total_intervenciones": "Intervenciones",
    "pct_goles_recibidos": "% goles recibidos",
    # ajustadas por contexto
    "pct_goles_equipo": "Goles (% del equipo)", "pct_asist_equipo": "Asistencias (% del equipo)",
    "pct_ga_equipo": "Goles + asist. (% del equipo)", "pct_tiros_arco_equipo": "Remates al arco (% del equipo)",
    "pct_centros_equipo": "Centros (% del equipo)",
    "interceptions_aj": "Intercepciones (aj. posesión)", "ball_recoveries_aj": "Recuperaciones (aj. posesión)",
    "pressures_aj": "Presiones (aj. posesión)", "duels_won_aj": "Duelos ganados (aj. posesión)",
    "paradas_por_100_tiros": "Paradas c/100 remates recibidos",
    "intervenciones_por_partido": "Intervenciones por partido", "salidas_por_partido": "Salidas por partido",
}


def etiqueta(m):
    return ETIQUETAS.get(m, m)


# Qué métrica reemplaza a cada una cuando se ajusta por contexto (las de
# porcentaje o eficiencia no dependen del volumen del equipo y quedan igual).
AJUSTE = {
    "goals_90": "pct_goles_equipo", "assists_90": "pct_asist_equipo",
    "goals_assists_90": "pct_ga_equipo", "shots_on_target": "pct_tiros_arco_equipo",
    "crosses": "pct_centros_equipo",
    "interceptions": "interceptions_aj", "ball_recoveries": "ball_recoveries_aj",
    "pressures": "pressures_aj", "duels_won": "duels_won_aj",
    "paradas_exitosas": "paradas_por_100_tiros",
    "total_intervenciones": "intervenciones_por_partido", "salidas": "salidas_por_partido",
}


@st.cache_data(show_spinner=False)
def cargar_contexto():
    """Contexto de equipo y métricas ajustadas (generados con src/build_contexto.py)."""
    ruta = PROCESSED / "contexto_jugadores.csv"
    return pd.read_csv(ruta) if ruta.exists() else None


def oportunidad(sim_pct, valor, valor_ref):
    """Regla de negocio: muy parecido y mucho más barato que la referencia."""
    return (pd.notna(valor) and pd.notna(valor_ref) and valor_ref > 0
            and sim_pct >= 85 and valor <= 0.30 * valor_ref)


EJEMPLOS = {
    "Haaland": ("Delantero", "A - Completo", "Haaland"),
    "Kroos": ("Mediocampista", "C - Pivote moderno", "Kroos"),
    "Keylor Navas": ("Portero", "B - Bajo los palos", "Navas"),
}


def cargar_ejemplo(nombre_ejemplo):
    pos, perfil_ej, texto_ej = EJEMPLOS[nombre_ejemplo]
    st.session_state["pos"] = pos
    st.session_state[f"perfil_{pos}"] = perfil_ej
    st.session_state[f"busqueda_{pos}"] = texto_ej


# ----------------------------------------------------------------------------
# INTERFAZ
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Explorador-IA · Scout-AI", page_icon="⚽", layout="wide")

st.markdown("""
<style>
.hero {padding: 1.4rem 1.6rem; border-radius: 14px; margin-bottom: 0.8rem;
       background: linear-gradient(120deg, #0B3D2E 0%, #14532D 55%, #1F7A4D 100%); color: #F4F7F2;}
.hero h1 {margin: 0; font-size: 2.1rem; color: #FFFFFF;}
.hero p {margin: 0.35rem 0 0 0; font-size: 1.05rem; opacity: 0.92;}
.chip {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem;
       font-weight: 600; margin-right: 4px;}
.chip-grande {background: #FDE2E1; color: #9B1C1C;}
.chip-medio {background: #FEF3C7; color: #92400E;}
.chip-chico {background: #DCFCE7; color: #166534;}
.chip-moneyball {background: #1F7A4D; color: #FFFFFF;}
.ficha-nombre {font-weight: 700; font-size: 1.02rem; line-height: 1.2;}
.ficha-equipo {opacity: 0.75; font-size: 0.85rem; margin-bottom: 6px;}
.ficha-sim {font-size: 1.7rem; font-weight: 800; color: #1F7A4D; line-height: 1.1;}
.ficha-valor {font-size: 0.95rem; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>⚽ Explorador-IA</h1>
  <p><b>Encontrá al próximo crack por una fracción de su precio.</b>
  Elegí un jugador de élite y la IA te muestra quién juega como él, dentro de tu presupuesto.</p>
</div>
""", unsafe_allow_html=True)

c_ej = st.columns([1.1, 1, 1, 1, 3])
c_ej[0].markdown("**Probá con:**")
for col, nombre_ej in zip(c_ej[1:4], EJEMPLOS):
    col.button(nombre_ej, on_click=cargar_ejemplo, args=(nombre_ej,), width="stretch")


def chip_nivel(n):
    if not isinstance(n, str):
        return ""
    return f'<span class="chip chip-{n.lower()}">Equipo {n.lower()}</span>'


# ---------------- barra lateral ----------------
with st.sidebar:
    st.header("Búsqueda")
    posicion = st.selectbox("Posición", list(CONFIG), key="pos")
    cfg = CONFIG[posicion]
    perfil = st.selectbox("Perfil de juego", list(cfg["perfiles"]), key=f"perfil_{posicion}")
    descripcion, metricas_perfil = cfg["perfiles"][perfil]
    st.caption(descripcion)
    k = st.slider("Cantidad de similares", 3, 15, 5)
    st.caption(f"Fuente: {cfg['fuente']}")

    todas = list(dict.fromkeys(m for p in cfg["perfiles"].values() for m in p[1]))
    with st.expander("Personalizar métricas"):
        metricas_base = st.multiselect(
            "Métricas usadas", todas, default=metricas_perfil, format_func=etiqueta,
            key=f"ms_{posicion}_{perfil}",
        )

if len(metricas_base) < 3:
    st.warning("Elegí al menos 3 métricas en 'Personalizar métricas'.")
    st.stop()

df, errores = cargar(
    cfg["archivos"], cfg["col_pos"], cfg["valores_pos"],
    cfg["col_min"], cfg["minimo"], tuple(todas),
)
if df is None:
    st.error("No pude cargar los datos de esta posición.")
    for e in errores:
        st.write(f"- {e}")
    st.stop()

if len(df) < 5:
    st.warning(f"Solo hay {len(df)} jugadores disponibles para esta posición.")
    st.stop()

clave = "premier_2223" if posicion == "Delantero" else "laliga_1516"

# --- Valor de mercado (Transfermarkt) ---
valores = cargar_valores()
hay_valores = valores is not None
if hay_valores:
    v = valores[valores["fuente"] == clave][["player", "valor_eur"]]
    df = df.drop(columns=["valor_eur"], errors="ignore").merge(v, on="player", how="left")
else:
    df["valor_eur"] = float("nan")

# --- Contexto de equipo ---
contexto = cargar_contexto()
hay_contexto = contexto is not None
if hay_contexto:
    ctx = contexto[contexto["fuente"] == clave].drop(columns=["fuente", "team"], errors="ignore")
    ctx = ctx.drop_duplicates("player")
    df = df.drop(columns=[c for c in ctx.columns if c != "player" and c in df.columns]).merge(
        ctx, on="player", how="left")
else:
    df["nivel_equipo"] = None

with st.sidebar:
    st.header("Contexto del equipo")
    ajustables = [m for m in metricas_base if m in AJUSTE and AJUSTE[m] in df.columns]
    if hay_contexto and ajustables:
        ajustar = st.toggle("Ajustar por contexto del equipo", value=False,
                            help="Compara lo que cada jugador aporta en relación con su equipo, "
                                 "no los números en bruto.")
        st.caption("Delanteros: % de la producción de su equipo. Defensas y mediocampistas: "
                   "acciones por partido ajustadas por posesión. Porteros: paradas cada 100 "
                   "remates recibidos.")
    else:
        ajustar = False
        st.caption("Corré src/build_contexto.py para activar el ajuste por contexto.")

    st.header("Presupuesto")
    if hay_valores and df["valor_eur"].notna().any():
        tope = int(df["valor_eur"].max() // 1e6) + 1
        presupuesto = st.slider("Valor de mercado máximo (M€)", 0, tope, tope)
        incluir_sin_valor = st.checkbox("Incluir jugadores sin valor conocido", value=True)
        st.caption("Valores de Transfermarkt de esa temporada (transfermarkt-datasets).")
    else:
        presupuesto, incluir_sin_valor = None, True
        st.caption("Corré src/build_valores.py para activar el presupuesto.")


def metricas_para(ajustado):
    return [AJUSTE[m] if ajustado and m in AJUSTE and AJUSTE[m] in df.columns else m
            for m in metricas_base]


metricas = metricas_para(ajustar)
# Jugadores sin dato de contexto: se completa con la mediana de la posición
# (en vez de 0, que los haría parecer los peores en esa métrica).
for m in metricas:
    df[m] = df[m].fillna(df[m].median())

# ---------------- búsqueda ----------------
texto = st.text_input(
    f"Jugador de referencia ({len(df)} {posicion.lower()}s en la base)",
    placeholder="Escribí un nombre o apellido. Ej: Kroos, Haaland, Navas...",
    key=f"busqueda_{posicion}",
)
with st.expander(f"Ver los {len(df)} {posicion.lower()}s disponibles"):
    cols_lista = ["player"] + [c for c in ("team", "nivel_equipo") if c in df.columns]
    st.dataframe(df[cols_lista].rename(columns={"player": "Jugador", "team": "Equipo",
                                                "nivel_equipo": "Nivel del equipo"})
                 .sort_values("Jugador"), hide_index=True, width="stretch")

if not texto.strip():
    st.info("Escribí el nombre de un jugador o tocá uno de los ejemplos de arriba.")
    st.stop()

estado, datos = validar_busqueda(texto, df, posicion)

if estado == "ok":
    nombre = datos
elif estado == "varios":
    nombre = st.selectbox(f"Hay {len(datos)} jugadores que coinciden con '{texto}'. Elegí uno:",
                          datos, index=None, placeholder="Elegí un jugador")
    if nombre is None:
        st.stop()
elif estado == "sugerencias":
    st.warning(f"No encontré a '{texto}' entre los {posicion.lower()}s. ¿Quisiste decir...?")
    nombre = st.radio("Sugerencias", datos, index=None, label_visibility="collapsed")
    if nombre is None:
        st.stop()
elif estado == "otra_posicion":
    st.warning(f"'{texto}' está en la base, pero no se puede comparar como {posicion.lower()}.")
    for _, fila in datos.head(3).iterrows():
        pos_app = posicion_de(fila.get("position"))
        if pos_app == posicion:
            st.write(f"- **{fila['player']}** es {posicion.lower()}, pero no cumple el mínimo "
                     f"de participación de la base (minutos o intervenciones), así que no se "
                     f"compara: con tan pocos datos, sus estadísticas no son confiables.")
        elif pos_app:
            st.write(f"- **{fila['player']}** figura como **{pos_app}** "
                     f"({fila['liga']}). El sistema solo compara jugadores de la misma "
                     f"posición: cambiá la posición en la barra lateral.")
        else:
            st.write(f"- **{fila['player']}** figura en {fila['liga']} con posición "
                     f"'{fila.get('position')}', que está fuera del alcance de esta posición. "
                     f"Recordá: delanteros = Premier 2022/23; mediocampistas defensivos, "
                     f"centrales y porteros = La Liga 2015/16.")
    st.stop()
else:
    st.error(
        f"'{texto}' no está en la base. El sistema solo incluye jugadores de la Premier "
        f"League 2022/23 (delanteros, con 600+ minutos) y de La Liga 2015/16 "
        f"(mediocampistas defensivos, centrales y porteros). Puede ser un jugador retirado, "
        f"de otra liga o temporada, o con pocos minutos."
    )
    st.stop()

idx = int(df.index[df["player"].astype(str) == nombre][0])


def top_similares(mets):
    s = buscar_similares(df, mets, idx)
    if presupuesto is not None:
        valor_sim = df.loc[s["idx"], "valor_eur"].values
        dentro = valor_sim <= presupuesto * 1e6
        if incluir_sin_valor:
            dentro = dentro | pd.isna(valor_sim)
        s = s[dentro]
    return s[s["similitud"] > 0].head(k)  # nunca mostrar perfiles opuestos


sim = top_similares(metricas)
if sim.empty:
    st.warning("Ningún jugador parecido entra en ese presupuesto. Subí el valor máximo.")
    st.stop()

valor_ref = df.loc[idx, "valor_eur"]
nivel_ref = df.loc[idx, "nivel_equipo"] if "nivel_equipo" in df.columns else None
equipo_ref = df.loc[idx, "team"] if "team" in df.columns else ""

# ---------------- encabezado del jugador ----------------
st.markdown(f"## Jugadores como **{nombre}**")
st.markdown(
    f"{equipo_ref} {chip_nivel(nivel_ref)} · Valor de mercado: **{millones(valor_ref)}** · "
    f"Perfil: {perfil}{' · **ajustado por contexto**' if ajustar else ''}",
    unsafe_allow_html=True,
)

if nivel_ref == "Grande" and not ajustar and ajustables:
    st.info(f"**{nombre}** juega en un equipo grande: parte de sus números puede venir del "
            f"contexto (más pelota, más ocasiones). Activá **Ajustar por contexto del equipo** "
            f"para comparar lo que aporta en relación con su equipo.")

# ---------------- tarjetas de resumen ----------------
mejor = sim.iloc[0]
v_mejor = df.loc[mejor["idx"], "valor_eur"]
m1, m2, m3 = st.columns(3)
m1.metric("Más parecido", df.loc[mejor["idx"], "player"])
m1.caption(f"{mejor['similitud']:.1f}% de similitud")
if pd.notna(valor_ref) and pd.notna(v_mejor) and valor_ref > 0:
    ahorro = valor_ref - v_mejor
    m2.metric("Su precio", millones(v_mejor))
    m2.caption(f"{v_mejor / valor_ref * 100:.0f}% del valor de {nombre}")
    m3.metric("Ahorro estimado", millones(ahorro) if ahorro > 0 else "—")
    m3.caption(f"{ahorro / valor_ref * 100:.0f}% menos que {nombre}" if ahorro > 0 else "no es más barato")
else:
    m2.metric("Su precio", millones(v_mejor))
    m3.metric("Ahorro estimado", "—")
    m3.caption("falta el valor de mercado")

n_oport = sum(oportunidad(s, df.loc[i, "valor_eur"], valor_ref) for i, s in zip(sim["idx"], sim["similitud"]))
if n_oport:
    st.success(f"⭐ **{n_oport} oportunidad{'es' if n_oport > 1 else ''} Moneyball**: "
               f"más de 85% parecido{'s' if n_oport > 1 else ''} por menos del 30% del precio de {nombre}.")

# ---------------- fichas de jugadores ----------------
POR_FILA = 5
filas_sim = list(sim.itertuples(index=False))
for inicio in range(0, len(filas_sim), POR_FILA):
    cols = st.columns(POR_FILA)
    for col, fila in zip(cols, filas_sim[inicio:inicio + POR_FILA]):
        i = int(fila.idx)
        val = df.loc[i, "valor_eur"]
        chips = chip_nivel(df.loc[i, "nivel_equipo"])
        if oportunidad(fila.similitud, val, valor_ref):
            chips += '<span class="chip chip-moneyball">⭐ Moneyball</span>'
        with col.container(border=True):
            st.markdown(
                f'<div class="ficha-nombre">{df.loc[i, "player"]}</div>'
                f'<div class="ficha-equipo">{df.loc[i, "team"] if "team" in df.columns else ""}</div>'
                f'<div class="ficha-sim">{fila.similitud:.1f}%</div>'
                f'<div style="opacity:.7;font-size:.8rem">similitud</div>'
                f'<div class="ficha-valor">{millones(val)}</div>'
                f'<div style="margin-top:6px">{chips}</div>',
                unsafe_allow_html=True,
            )

# ---------------- qué cambió con el ajuste ----------------
if ajustar:
    sin_ajuste = top_similares(metricas_para(False))
    with st.expander("¿Qué cambió al ajustar por contexto?", expanded=True):
        a, b = st.columns(2)
        a.markdown("**Sin ajuste (números en bruto)**")
        a.dataframe(pd.DataFrame({
            "Jugador": df.loc[sin_ajuste["idx"], "player"].values,
            "Nivel": df.loc[sin_ajuste["idx"], "nivel_equipo"].values,
            "Similitud %": sin_ajuste["similitud"].round(1).values}), hide_index=True, width="stretch")
        b.markdown("**Con ajuste por contexto**")
        b.dataframe(pd.DataFrame({
            "Jugador": df.loc[sim["idx"], "player"].values,
            "Nivel": df.loc[sim["idx"], "nivel_equipo"].values,
            "Similitud %": sim["similitud"].round(1).values}), hide_index=True, width="stretch")
        nuevos = set(sim["idx"]) - set(sin_ajuste["idx"])
        st.caption(f"{len(nuevos)} de {len(sim)} jugadores cambian al descontar el efecto del equipo.")

# ---------------- radar y detalle ----------------
izq, der = st.columns([1, 1.3])
nombres_sim = df.loc[sim["idx"], "player"].tolist()
with izq:
    st.subheader("Comparación cara a cara")
    elegidos = st.multiselect("Comparar con", nombres_sim, default=nombres_sim[:2])
    with st.expander("Ver los números de cada métrica"):
        idxs = [idx] + sim["idx"].tolist()
        tabla_val = df.loc[idxs, ["player"] + metricas].copy()
        tabla_val.columns = ["Jugador"] + [etiqueta(m) for m in metricas]
        st.dataframe(tabla_val.round(2), hide_index=True, width="stretch")
    with st.expander("Ver como tabla"):
        st.dataframe(pd.DataFrame({
            "Jugador": nombres_sim,
            "Equipo": df.loc[sim["idx"], "team"].values if "team" in df.columns else "",
            "Nivel del equipo": df.loc[sim["idx"], "nivel_equipo"].values,
            "Similitud %": sim["similitud"].round(1).values,
            "Valor de mercado": [millones(x) for x in df.loc[sim["idx"], "valor_eur"]],
        }), hide_index=True, width="stretch")
with der:
    jugadores = {idx: nombre}
    for i, nom in zip(sim["idx"], nombres_sim):
        if nom in elegidos:
            jugadores[int(i)] = nom
    st.plotly_chart(grafico_radar(df, metricas, jugadores), width="stretch")
    st.caption("Percentiles dentro de la posición: 100 = el mejor de la base en esa métrica.")

if posicion == "Portero" and not ajustar:
    st.info("Ojo con el sesgo de contexto: los porteros de equipos chicos suelen tener más "
            "intervenciones. Activá el ajuste por contexto para compararlos por cada 100 "
            "remates recibidos.")

st.divider()
st.caption("Explorador-IA · Scout-AI · UCA 2026 — Similitud con k-NN (distancia coseno) sobre "
           "métricas estandarizadas. Los valores de mercado son de Transfermarkt. "
           "La herramienta propone candidatos: la decisión final es del scout y del director deportivo.")
