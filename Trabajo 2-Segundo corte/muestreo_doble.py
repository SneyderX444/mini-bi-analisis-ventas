"""
╔══════════════════════════════════════════════════════════════════════╗
║         MUESTREO DOBLE (EN DOS FASES) — VERSIÓN AVANZADA            ║
║      Control de Calidad · Fábrica de Tornillos · Python 3            ║
║                                                                      ║
║  Caso Real: Una fábrica produce 10,000 tornillos diarios.            ║
║  Especificación: diámetro 10 mm ± 0.5 mm (9.5 mm – 10.5 mm)        ║
║  Objetivo: Estimar la tasa de defectos con el menor costo posible.   ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import scipy.stats as stats
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════
# PALETA DE COLORES Y ESTILO GLOBAL
# ══════════════════════════════════════════════════════════════════════
COLOR = {
    'azul':      '#1A3A5C',
    'azul_med':  '#2E6DA4',
    'azul_cla':  '#7FBBE0',
    'verde':     '#1B5E20',
    'verde_med': '#388E3C',
    'verde_cla': '#81C784',
    'rojo':      '#B71C1C',
    'rojo_med':  '#E53935',
    'rojo_cla':  '#EF9A9A',
    'naranja':   '#E65100',
    'naranja_m': '#F57C00',
    'amarillo':  '#F9A825',
    'gris_osc':  '#263238',
    'gris_med':  '#546E7A',
    'gris_cla':  '#ECEFF1',
    'blanco':    '#FFFFFF',
    'crema':     '#FFFDE7',
    'fase1':     '#1565C0',
    'fase2':     '#6A1B9A',
}

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        10,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.25,
    'grid.linestyle':   '--',
    'figure.facecolor': '#F8FAFC',
    'axes.facecolor':   '#F8FAFC',
})

# ══════════════════════════════════════════════════════════════════════
# PARÁMETROS DEL PROBLEMA
# ══════════════════════════════════════════════════════════════════════
SEMILLA       = 2026
POBLACION     = 10_000      # N
N1            = 200         # Fase 1
N2            = 50          # Fase 2
SPEC_NOMINAL  = 10.0        # mm (valor objetivo)
SPEC_TOL      = 0.5         # mm (tolerancia ± )
SPEC_MIN      = SPEC_NOMINAL - SPEC_TOL   # 9.5 mm
SPEC_MAX      = SPEC_NOMINAL + SPEC_TOL   # 10.5 mm
VISUAL_MARGEN = 0.7         # tolerancia visual (± margen adicional)
COSTO_VISUAL  = 0.50        # $ por tornillo (Fase 1)
COSTO_CALIB   = 5.00        # $ por tornillo (Fase 2)
NIVEL_CONF    = 0.95        # 95 % de confianza

np.random.seed(SEMILLA)

# ══════════════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════════════
def separador(titulo="", ancho=62, char="═"):
    if titulo:
        pad = (ancho - len(titulo) - 2) // 2
        print(f"\n{char*pad} {titulo} {char*(ancho - pad - len(titulo) - 2)}")
    else:
        print(char * ancho)

def fila(etiqueta, valor, unidad="", ancho_et=38):
    et = etiqueta.ljust(ancho_et)
    print(f"  {et} {valor} {unidad}")

def barra_texto(valor, maximo, largo=30, char_lleno="█", char_vacio="░"):
    llenos = int(round(valor / maximo * largo))
    return char_lleno * llenos + char_vacio * (largo - llenos)

# ══════════════════════════════════════════════════════════════════════
# 1. GENERACIÓN DE POBLACIÓN
# ══════════════════════════════════════════════════════════════════════
separador("GENERACIÓN DE LA POBLACIÓN", char="═")

# Población realista: mezcla de tornillos buenos y con desviaciones
# 90% producción normal, 8% desviada, 2% muy defectuosa
n_buenos   = int(POBLACION * 0.90)
n_desv     = int(POBLACION * 0.08)
n_malos    = POBLACION - n_buenos - n_desv

pop_buenos  = np.random.normal(SPEC_NOMINAL, 0.18, n_buenos)
pop_desv    = np.random.normal(SPEC_NOMINAL, 0.45, n_desv)
pop_malos   = np.random.normal(SPEC_NOMINAL + np.random.choice([-1, 1], n_malos) * 0.6, 0.15, n_malos)

diametros_poblacion = np.concatenate([pop_buenos, pop_desv, pop_malos])
np.random.shuffle(diametros_poblacion)

real_defectos  = np.sum((diametros_poblacion < SPEC_MIN) | (diametros_poblacion > SPEC_MAX))
real_tasa      = real_defectos / POBLACION * 100

fila("Tamaño de la población (N):",        f"{POBLACION:,} tornillos")
fila("Especificación nominal:",             f"{SPEC_NOMINAL} mm  ±  {SPEC_TOL} mm")
fila("Rango aceptable:",                   f"[{SPEC_MIN} mm  —  {SPEC_MAX} mm]")
fila("Defectos reales (verdad oculta):",   f"{real_defectos} tornillos  ({real_tasa:.2f}%)")
print("  (en la práctica, este valor es desconocido — el muestreo lo estimará)")

# ══════════════════════════════════════════════════════════════════════
# 2. FASE 1 — MUESTRA GRANDE · INSPECCIÓN VISUAL
# ══════════════════════════════════════════════════════════════════════
separador("FASE 1 — INSPECCIÓN VISUAL  (n₁ = 200)", char="─")

indices_f1   = np.random.choice(POBLACION, N1, replace=False)
muestra_f1   = diametros_poblacion[indices_f1]

# Inspección visual: clasifica con tolerancia más amplia (ojo humano)
visual_ok    = muestra_f1[(muestra_f1 >= SPEC_MIN - 0.2) & (muestra_f1 <= SPEC_MAX + 0.2)]
visual_sosp  = muestra_f1[(muestra_f1 < SPEC_MIN - 0.2) | (muestra_f1 > SPEC_MAX + 0.2)]

p_ok_f1      = len(visual_ok) / N1 * 100
p_sosp_f1    = len(visual_sosp) / N1 * 100

fila("Muestra Fase 1 (n₁):",              f"{N1} tornillos")
fila("Método:",                            "Inspección visual rápida")
fila("Costo unitario:",                   f"$ {COSTO_VISUAL:.2f} por tornillo")
fila("Costo total Fase 1:",               f"$ {N1 * COSTO_VISUAL:,.2f}")
print()
fila("  Clasificados como 'correctos':",  f"{len(visual_ok):>4}  ({p_ok_f1:.1f}%)")
fila("  Clasificados como 'sospechosos':",f"{len(visual_sosp):>4}  ({p_sosp_f1:.1f}%)")
print()
fila("  Media Fase 1 (x̄₁):",            f"{np.mean(muestra_f1):.4f} mm")
fila("  Desv. Est. Fase 1 (s₁):",        f"{np.std(muestra_f1, ddof=1):.4f} mm")
print()
print(f"  {'Distribución visual':}")
print(f"  Correctos   {barra_texto(len(visual_ok), N1)} {len(visual_ok)}")
print(f"  Sospechosos {barra_texto(len(visual_sosp), N1)} {len(visual_sosp)}")

# ══════════════════════════════════════════════════════════════════════
# 3. ANÁLISIS INTERMEDIO
# ══════════════════════════════════════════════════════════════════════
separador("ANÁLISIS INTERMEDIO — CLASIFICACIÓN POR ESTRATOS", char="─")

# Estratificar los 200 tornillos de F1 en 3 grupos
estrato_A = muestra_f1[muestra_f1 >= 9.8]                                  # excelente
estrato_B = muestra_f1[(muestra_f1 >= 9.5) & (muestra_f1 < 9.8)]           # aceptable
estrato_C = muestra_f1[(muestra_f1 < 9.5) | (muestra_f1 > 10.5)]           # defectuoso

estratos = {
    'A — Excelente  (≥ 9.8 mm)':     estrato_A,
    'B — Aceptable  (9.5–9.8 mm)':   estrato_B,
    'C — Defectuoso (< 9.5 mm)':     estrato_C,
}

print()
for nombre, est in estratos.items():
    pct = len(est) / N1 * 100
    print(f"  Estrato {nombre}")
    print(f"           n = {len(est):>3}  ({pct:.1f}%)  {barra_texto(len(est), N1, 25)}")
    print()

print("  → La Fase 2 aplicará muestreo proporcional a cada estrato.")

# ══════════════════════════════════════════════════════════════════════
# 4. FASE 2 — SUBMUESTRA · MEDICIÓN PRECISA CON CALIBRADOR
# ══════════════════════════════════════════════════════════════════════
separador("FASE 2 — MEDICIÓN PRECISA CON CALIBRADOR  (n₂ = 50)", char="─")

# Muestreo estratificado proporcional para Fase 2
def muestra_estratificada(estrato, total_estrato, n_total_f1, n2_total):
    n_aloc = max(1, round(len(estrato) / n_total_f1 * n2_total))
    n_aloc = min(n_aloc, len(estrato))
    idx    = np.random.choice(len(estrato), n_aloc, replace=False)
    return estrato[idx]

mf2_A = muestra_estratificada(estrato_A, len(estrato_A), N1, N2)
mf2_B = muestra_estratificada(estrato_B, len(estrato_B), N1, N2)
mf2_C = muestra_estratificada(estrato_C, len(estrato_C), N1, N2)
muestra_f2 = np.concatenate([mf2_A, mf2_B, mf2_C])[:N2]

conformes    = np.sum((muestra_f2 >= SPEC_MIN) & (muestra_f2 <= SPEC_MAX))
no_conf      = len(muestra_f2) - conformes
tasa_def_est = (no_conf / len(muestra_f2)) * 100

media_f2  = np.mean(muestra_f2)
mediana_f2= np.median(muestra_f2)
desv_f2   = np.std(muestra_f2, ddof=1)
cv_f2     = desv_f2 / media_f2 * 100
q1, q3    = np.percentile(muestra_f2, [25, 75])
iqr       = q3 - q1
asimetria = stats.skew(muestra_f2)
curtosis  = stats.kurtosis(muestra_f2)

fila("Submuestra Fase 2 (n₂):",           f"{len(muestra_f2)} tornillos")
fila("Método:",                            "Muestreo estratificado proporcional")
fila("Instrumento:",                       "Calibrador digital (±0.001 mm)")
fila("Costo unitario:",                   f"$ {COSTO_CALIB:.2f} por tornillo")
fila("Costo total Fase 2:",               f"$ {len(muestra_f2) * COSTO_CALIB:,.2f}")
print()
print("  ── MEDIDAS DE TENDENCIA CENTRAL ──")
fila("  Media aritmética (x̄):",           f"{media_f2:.4f} mm")
fila("  Mediana (Me):",                    f"{mediana_f2:.4f} mm")
fila("  Desviación estándar (s):",         f"{desv_f2:.4f} mm")
fila("  Coeficiente de variación (CV):",   f"{cv_f2:.2f} %")
fila("  Mínimo:",                          f"{np.min(muestra_f2):.4f} mm")
fila("  Máximo:",                          f"{np.max(muestra_f2):.4f} mm")
fila("  Q1 — Q3:",                        f"{q1:.4f} mm — {q3:.4f} mm")
fila("  Rango intercuartil (IQR):",        f"{iqr:.4f} mm")
fila("  Asimetría (skewness):",            f"{asimetria:.4f}")
fila("  Curtosis (excess):",               f"{curtosis:.4f}")
print()
print("  ── CONTROL DE CALIDAD ──")
fila("  Conformes (dentro de especif.):", f"{conformes:>3} tornillos  ({conformes/len(muestra_f2)*100:.1f}%)")
fila("  No conformes (fuera de especif.):",f"{no_conf:>3} tornillos  ({tasa_def_est:.1f}%)")
print()
print(f"  Conformes   {barra_texto(conformes, len(muestra_f2))} {conformes}")
print(f"  Defectuosos {barra_texto(no_conf,   len(muestra_f2))} {no_conf}")

# ══════════════════════════════════════════════════════════════════════
# 5. INFERENCIA ESTADÍSTICA
# ══════════════════════════════════════════════════════════════════════
separador("INFERENCIA SOBRE LA POBLACIÓN", char="─")

n2_real  = len(muestra_f2)
p_hat    = no_conf / n2_real                        # proporción muestral de defectos
z_alpha  = stats.norm.ppf(1 - (1 - NIVEL_CONF) / 2) # z = 1.96 para 95%
se_p     = np.sqrt(p_hat * (1 - p_hat) / n2_real)   # error estándar de la proporción
ic_inf   = max(0, p_hat - z_alpha * se_p)
ic_sup   = min(1, p_hat + z_alpha * se_p)

# Estimador de la media con corrección de población finita
se_media = desv_f2 / np.sqrt(n2_real) * np.sqrt(1 - n2_real / POBLACION)
ic_m_inf = media_f2 - z_alpha * se_media
ic_m_sup = media_f2 + z_alpha * se_media

defectos_est    = int(p_hat * POBLACION)
defectos_ic_inf = int(ic_inf * POBLACION)
defectos_ic_sup = int(ic_sup * POBLACION)

print()
fila("Proporción estimada de defectos (p̂):", f"{p_hat*100:.2f}%")
fila("Error estándar de p̂:",                f"{se_p*100:.4f}%")
fila("Intervalo de confianza (95%) para p:",
     f"[{ic_inf*100:.2f}%  —  {ic_sup*100:.2f}%]")
print()
fila("Defectos estimados en producción:",    f"~{defectos_est:,} tornillos")
fila("Intervalo de defectos estimados:",     f"[{defectos_ic_inf:,} — {defectos_ic_sup:,}]")
print()
fila("Media estimada de la población:",      f"{media_f2:.4f} mm")
fila("IC 95% para la media:",               f"[{ic_m_inf:.4f} — {ic_m_sup:.4f}] mm")
print()
fila("Defectos reales (verificación):",      f"{real_defectos:,} ({real_tasa:.2f}%)")
error_abs = abs(p_hat * POBLACION - real_defectos)
fila("Error absoluto de estimación:",        f"{error_abs:.0f} tornillos")
fila("Error relativo:",                      f"{error_abs/real_defectos*100:.2f}%")

# ══════════════════════════════════════════════════════════════════════
# 6. PRUEBA DE HIPÓTESIS
# ══════════════════════════════════════════════════════════════════════
separador("PRUEBA DE HIPÓTESIS (α = 0.05)", char="─")

# H₀: p ≤ 0.05 (tasa de defectos aceptable ≤ 5%)
# H₁: p > 0.05 (tasa de defectos inaceptable)
p0       = 0.05
z_prueba = (p_hat - p0) / np.sqrt(p0 * (1 - p0) / n2_real)
p_valor  = 1 - stats.norm.cdf(z_prueba)
z_critico= stats.norm.ppf(0.95)  # cola derecha, α = 0.05

print()
print("  H₀: p ≤ 0.05  (proceso bajo control — máx 5% defectos)")
print("  H₁: p > 0.05  (proceso fuera de control)")
print()
fila("  Estadístico de prueba Z:",           f"{z_prueba:.4f}")
fila("  Valor crítico Z (α = 0.05):",        f"{z_critico:.4f}")
fila("  p-valor:",                           f"{p_valor:.4f}")
print()
if p_valor < 0.05:
    print("  ⚠  DECISIÓN: RECHAZAR H₀")
    print("     La tasa de defectos supera el 5% aceptable.")
    print("     → Se recomienda revisión inmediata del proceso.")
else:
    print("  ✔  DECISIÓN: NO RECHAZAR H₀")
    print("     No hay evidencia suficiente de que la tasa supere 5%.")
    print("     → El proceso está bajo control estadístico.")

# ══════════════════════════════════════════════════════════════════════
# 7. ANÁLISIS DE COSTOS
# ══════════════════════════════════════════════════════════════════════
separador("ANÁLISIS COMPARATIVO DE COSTOS", char="─")

costo_f1     = N1 * COSTO_VISUAL
costo_f2     = N2 * COSTO_CALIB
costo_doble  = costo_f1 + costo_f2
costo_total  = POBLACION * COSTO_CALIB          # medir todo
costo_simple = N2 * COSTO_CALIB                 # solo muestra sin Fase 1
ahorro_vs_total = (1 - costo_doble / costo_total) * 100
ahorro_dolares  = costo_total - costo_doble

print()
print("  ┌─────────────────────────────────────────────────┐")
print("  │  ESCENARIO           PIEZAS     COSTO TOTAL     │")
print("  ├─────────────────────────────────────────────────┤")
print(f"  │  Medir toda la planta {POBLACION:>7,}   $ {costo_total:>10,.2f}    │")
print(f"  │  Muestra simple (n₂)  {N2:>7,}   $ {costo_simple:>10,.2f}    │")
print(f"  │  Muestreo doble                              │")
print(f"  │    Fase 1 visual     {N1:>7,}   $ {costo_f1:>10,.2f}    │")
print(f"  │    Fase 2 calibrador {N2:>7,}   $ {costo_f2:>10,.2f}    │")
print(f"  │    TOTAL DOBLE       {N1+N2:>7,}   $ {costo_doble:>10,.2f}    │")
print("  ├─────────────────────────────────────────────────┤")
print(f"  │  AHORRO vs. total:        $ {ahorro_dolares:>10,.2f}             │")
print(f"  │  AHORRO %:                  {ahorro_vs_total:>9.1f}%             │")
print("  └─────────────────────────────────────────────────┘")

# ══════════════════════════════════════════════════════════════════════
# 8. RESUMEN FINAL
# ══════════════════════════════════════════════════════════════════════
separador("RESUMEN EJECUTIVO", char="═")
print()
print(f"  ✦ Caso:        Control de calidad — {POBLACION:,} tornillos/día")
print(f"  ✦ Fase 1:      {N1} tornillos inspeccionados visualmente   ($ {costo_f1:.2f})")
print(f"  ✦ Fase 2:      {N2} tornillos medidos con calibrador        ($ {costo_f2:.2f})")
print(f"  ✦ Costo total: $ {costo_doble:.2f}  (ahorro del {ahorro_vs_total:.1f}% vs. inspección completa)")
print()
print(f"  ✦ Media estimada:  {media_f2:.4f} mm  (nominal: {SPEC_NOMINAL} mm)")
print(f"  ✦ Desv. Estándar:  {desv_f2:.4f} mm")
print(f"  ✦ Tasa de defectos estimada:  {tasa_def_est:.1f}%  "
      f"(IC 95%: [{ic_inf*100:.1f}%, {ic_sup*100:.1f}%])")
print(f"  ✦ Defectos estimados:  ~{defectos_est:,} tornillos/día")
print(f"  ✦ Prueba H₀ (p≤5%): {'RECHAZADA ⚠' if p_valor < 0.05 else 'No rechazada ✔'}  (p-valor = {p_valor:.4f})")
print()
separador(char="═")

# ══════════════════════════════════════════════════════════════════════
# 9. FIGURAS
# ══════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(20, 22), facecolor='#F0F4F8')
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.42, wspace=0.35)

TITULO_KW = dict(fontsize=11, fontweight='bold', color=COLOR['azul'], pad=10)
LABEL_KW  = dict(fontsize=9, color=COLOR['gris_med'])

# ── Paleta para fases ──
c_f1 = COLOR['fase1']
c_f2 = COLOR['fase2']

# ╔══ FILA 0: Diagrama de flujo del proceso ══╗
ax_flow = fig.add_subplot(gs[0, :])
ax_flow.set_xlim(0, 10)
ax_flow.set_ylim(0, 1)
ax_flow.axis('off')
ax_flow.set_facecolor('#F0F4F8')

def caja(ax, x, y, w, h, txt, color, txt_color='white', fs=9):
    rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.05", facecolor=color, edgecolor='white',
        linewidth=2, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y, txt, ha='center', va='center', fontsize=fs,
            color=txt_color, fontweight='bold', zorder=4,
            multialignment='center')

caja(ax_flow, 1.0, 0.5, 1.6, 0.55, f"POBLACIÓN\nN = {POBLACION:,}", COLOR['gris_osc'], fs=8)
caja(ax_flow, 3.0, 0.5, 1.8, 0.55, f"FASE 1\nn₁ = {N1}\nInspección Visual", c_f1, fs=8)
caja(ax_flow, 5.3, 0.5, 1.8, 0.55, f"ANÁLISIS\nINTERMEDIO\nEstratificación", COLOR['naranja_m'], fs=8)
caja(ax_flow, 7.5, 0.5, 1.8, 0.55, f"FASE 2\nn₂ = {N2}\nCalibrador Digital", c_f2, fs=8)
caja(ax_flow, 9.4, 0.5, 1.0, 0.55, f"ESTIMACIÓN\nFINAL", COLOR['verde'], fs=7.5)

for x0, x1 in [(1.8, 2.1), (3.9, 4.4), (6.2, 6.6), (8.4, 8.9)]:
    ax_flow.annotate("", xy=(x1, 0.5), xytext=(x0, 0.5),
        arrowprops=dict(arrowstyle="->", color=COLOR['gris_med'], lw=2))

ax_flow.text(5, 0.96, "MUESTREO DOBLE EN DOS FASES — Control de Calidad de Tornillos",
             ha='center', va='top', fontsize=13, fontweight='bold', color=COLOR['azul'])

# ╔══ FILA 1, col 0: Distribución de la POBLACIÓN ══╗
ax1 = fig.add_subplot(gs[1, 0])
bins_pop = np.linspace(8.0, 12.0, 60)
ax1.hist(diametros_poblacion, bins=bins_pop, color=COLOR['gris_med'],
         alpha=0.6, edgecolor='white', lw=0.4, label='Población')
ax1.axvline(SPEC_MIN, color=COLOR['rojo_med'], lw=2, ls='--', label=f'Mín {SPEC_MIN} mm')
ax1.axvline(SPEC_MAX, color=COLOR['rojo_med'], lw=2, ls=':',  label=f'Máx {SPEC_MAX} mm')
ax1.axvline(SPEC_NOMINAL, color=COLOR['verde_med'], lw=2, label=f'Nominal {SPEC_NOMINAL} mm')
ax1.fill_betweenx([0, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 800],
                   SPEC_MIN, SPEC_MAX, alpha=0.08, color=COLOR['verde_med'])
ax1.set_title("Distribución Real de la Población", **TITULO_KW)
ax1.set_xlabel("Diámetro (mm)", **LABEL_KW)
ax1.set_ylabel("Frecuencia", **LABEL_KW)
ax1.legend(fontsize=7.5)

# ╔══ FILA 1, col 1: Distribución Fase 1 ══╗
ax2 = fig.add_subplot(gs[1, 1])
ax2.hist(muestra_f1, bins=20, color=c_f1, alpha=0.75, edgecolor='white', lw=0.5)
ax2.axvline(np.mean(muestra_f1), color=COLOR['amarillo'], lw=2.5,
            label=f'x̄₁ = {np.mean(muestra_f1):.3f} mm')
ax2.axvline(SPEC_MIN, color=COLOR['rojo_med'], lw=1.8, ls='--')
ax2.axvline(SPEC_MAX, color=COLOR['rojo_med'], lw=1.8, ls='--')
ax2.set_title(f"Fase 1 — Inspección Visual  (n₁={N1})", **TITULO_KW)
ax2.set_xlabel("Diámetro (mm)", **LABEL_KW)
ax2.set_ylabel("Frecuencia", **LABEL_KW)
ax2.legend(fontsize=8)

# ╔══ FILA 1, col 2: Distribución Fase 2 ══╗
ax3 = fig.add_subplot(gs[1, 2])
ax3.hist(muestra_f2, bins=14, color=c_f2, alpha=0.80, edgecolor='white', lw=0.5)
ax3.axvline(media_f2,   color=COLOR['amarillo'], lw=2.5, label=f'x̄₂ = {media_f2:.3f} mm')
ax3.axvline(mediana_f2, color=COLOR['verde_cla'], lw=2.5, ls='-.', label=f'Me = {mediana_f2:.3f} mm')
ax3.axvline(SPEC_MIN, color=COLOR['rojo_med'], lw=1.8, ls='--', label='Especif.')
ax3.axvline(SPEC_MAX, color=COLOR['rojo_med'], lw=1.8, ls='--')
# Sombrear zona de defectos
x_range = np.linspace(muestra_f2.min(), SPEC_MIN, 100)
ax3.fill_between(x_range, 0, 1, transform=ax3.get_xaxis_transform(),
                  alpha=0.15, color=COLOR['rojo_med'])
x_range2 = np.linspace(SPEC_MAX, muestra_f2.max(), 100)
ax3.fill_between(x_range2, 0, 1, transform=ax3.get_xaxis_transform(),
                  alpha=0.15, color=COLOR['rojo_med'])
ax3.set_title(f"Fase 2 — Medición Calibrador  (n₂={N2})", **TITULO_KW)
ax3.set_xlabel("Diámetro (mm)", **LABEL_KW)
ax3.set_ylabel("Frecuencia", **LABEL_KW)
ax3.legend(fontsize=7.5)

# ╔══ FILA 2, col 0: Torta Conformes/No conformes ══╗
ax4 = fig.add_subplot(gs[2, 0])
sizes  = [conformes, no_conf]
labels = [f'Conformes\n{conformes} ({conformes/N2*100:.1f}%)',
          f'Defectuosos\n{no_conf} ({no_conf/N2*100:.1f}%)']
colors_pie = [COLOR['verde_med'], COLOR['rojo_med']]
wedges, texts = ax4.pie(sizes, labels=labels, colors=colors_pie,
    startangle=140, wedgeprops=dict(edgecolor='white', linewidth=2),
    textprops=dict(fontsize=8.5))
ax4.set_title("Resultado Control de Calidad\n(Fase 2)", **TITULO_KW)

# ╔══ FILA 2, col 1: Box-plot comparativo F1 vs F2 ══╗
ax5 = fig.add_subplot(gs[2, 1])
bp = ax5.boxplot([muestra_f1, muestra_f2],
    labels=[f'Fase 1\n(n₁={N1})', f'Fase 2\n(n₂={N2})'],
    patch_artist=True, notch=False, widths=0.5,
    medianprops=dict(color=COLOR['amarillo'], lw=2.5),
    whiskerprops=dict(color=COLOR['gris_med']),
    capprops=dict(color=COLOR['gris_med']))
bp['boxes'][0].set_facecolor(c_f1 + '99')
bp['boxes'][1].set_facecolor(c_f2 + '99')
ax5.axhline(SPEC_MIN, color=COLOR['rojo_med'], lw=1.5, ls='--', label='Límites especif.')
ax5.axhline(SPEC_MAX, color=COLOR['rojo_med'], lw=1.5, ls='--')
ax5.axhline(SPEC_NOMINAL, color=COLOR['verde_med'], lw=1.5, ls=':', label='Nominal')
ax5.set_title("Box-Plot: Fase 1 vs Fase 2", **TITULO_KW)
ax5.set_ylabel("Diámetro (mm)", **LABEL_KW)
ax5.legend(fontsize=7.5)

# ╔══ FILA 2, col 2: Estimación con IC ══╗
ax6 = fig.add_subplot(gs[2, 2])
categorias = ['Proporción\ndefectuosos', 'Media\nestimada (mm)']
valores    = [p_hat * 100, media_f2]
ic_bajos   = [ic_inf * 100, ic_m_inf]
ic_altos   = [ic_sup * 100, ic_m_sup]
errores_b  = [valores[0] - ic_bajos[0], valores[1] - ic_bajos[1]]
errores_a  = [ic_altos[0] - valores[0], ic_altos[1] - valores[1]]

ax6_twin = ax6.twinx()
ax6.bar(0, valores[0], color=COLOR['rojo_med'], alpha=0.75, width=0.5,
        yerr=[[errores_b[0]], [errores_a[0]]], capsize=10,
        error_kw=dict(ecolor=COLOR['gris_osc'], lw=2, capthick=2))
ax6_twin.bar(1, valores[1], color=c_f2, alpha=0.75, width=0.5,
        yerr=[[errores_b[1]], [errores_a[1]]], capsize=10,
        error_kw=dict(ecolor=COLOR['gris_osc'], lw=2, capthick=2))
ax6.set_xticks([0, 1])
ax6.set_xticklabels(categorias, fontsize=8)
ax6.set_ylabel("% Defectuosos", fontsize=9, color=COLOR['rojo_med'])
ax6_twin.set_ylabel("Diámetro (mm)", fontsize=9, color=c_f2)
ax6.set_title("Estimaciones con IC 95%", **TITULO_KW)
ax6.text(0, valores[0] + errores_a[0] + 0.3, f"{valores[0]:.1f}%", ha='center', fontsize=8.5, fontweight='bold')
ax6_twin.text(1, valores[1] + errores_a[1] + 0.002, f"{valores[1]:.4f}", ha='center', fontsize=8.5, fontweight='bold')

# ╔══ FILA 3, col 0: Estratos ══╗
ax7 = fig.add_subplot(gs[3, 0])
nombres_est = ['A — Excelente', 'B — Aceptable', 'C — Defectuoso']
conteos_est = [len(estrato_A), len(estrato_B), len(estrato_C)]
colores_est = [COLOR['verde_med'], COLOR['azul_med'], COLOR['rojo_med']]
bars = ax7.barh(nombres_est, conteos_est, color=colores_est, edgecolor='white', lw=1.5, height=0.5)
for bar, v in zip(bars, conteos_est):
    ax7.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
             f'{v}  ({v/N1*100:.1f}%)', va='center', fontsize=8.5, fontweight='bold')
ax7.set_title("Estratificación Fase 1", **TITULO_KW)
ax7.set_xlabel("Tornillos", **LABEL_KW)
ax7.set_xlim(0, max(conteos_est) * 1.25)

# ╔══ FILA 3, col 1: Curva normal + zona rechazo prueba H ══╗
ax8 = fig.add_subplot(gs[3, 1])
x_norm = np.linspace(-4, 5, 400)
y_norm = stats.norm.pdf(x_norm)
ax8.plot(x_norm, y_norm, color=COLOR['azul_med'], lw=2.5)
ax8.fill_between(x_norm, y_norm, where=(x_norm >= z_critico),
                  color=COLOR['rojo_med'], alpha=0.4, label=f'Zona rechazo (α=0.05)')
ax8.axvline(z_prueba, color=COLOR['naranja'], lw=2.5, ls='--',
            label=f'Z prueba = {z_prueba:.3f}')
ax8.axvline(z_critico, color=COLOR['rojo_med'], lw=1.8, ls=':',
            label=f'Z crítico = {z_critico:.3f}')
ax8.set_title("Prueba de Hipótesis\nH₀: p ≤ 5% defectos", **TITULO_KW)
ax8.set_xlabel("Estadístico Z", **LABEL_KW)
ax8.set_ylabel("Densidad", **LABEL_KW)
ax8.legend(fontsize=7.5)
decision = "RECHAZAR H₀ ⚠" if p_valor < 0.05 else "No rechazar H₀ ✔"
color_dec = COLOR['rojo_med'] if p_valor < 0.05 else COLOR['verde']
ax8.text(0.5, 0.92, decision, transform=ax8.transAxes, ha='center',
         fontsize=9, fontweight='bold', color=color_dec,
         bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR['crema'], edgecolor=color_dec))

# ╔══ FILA 3, col 2: Comparación de costos ══╗
ax9 = fig.add_subplot(gs[3, 2])
esc_labels = ['Inspección\ncompleta', 'Solo\nFase 2', 'Muestreo\nDoble']
esc_costos = [costo_total, costo_simple, costo_doble]
esc_colors = [COLOR['rojo_med'], COLOR['azul_med'], COLOR['verde_med']]
bars_c = ax9.bar(esc_labels, esc_costos, color=esc_colors, edgecolor='white', lw=1.5, width=0.55)
for bar, v in zip(bars_c, esc_costos):
    ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 400,
             f'$ {v:,.0f}', ha='center', fontsize=8.5, fontweight='bold')
ax9.set_title("Comparación de Costos ($)", **TITULO_KW)
ax9.set_ylabel("Costo Total ($)", **LABEL_KW)
ax9.set_ylim(0, costo_total * 1.18)
# Flecha de ahorro
ax9.annotate("", xy=(2, costo_doble + 1500), xytext=(0, costo_total - 1500),
    arrowprops=dict(arrowstyle="<->", color=COLOR['amarillo'], lw=2))
ax9.text(1.0, (costo_total + costo_doble) / 2 + 1000,
         f"Ahorro\n{ahorro_vs_total:.1f}%",
         ha='center', fontsize=8, fontweight='bold', color=COLOR['amarillo'])

# ── Firma ──
fig.text(0.5, 0.005,
         "Muestreo Doble · Control de Calidad · Probabilidad y Estadística 2026 · "
         f"Semilla: {SEMILLA}  |  n₁={N1}  n₂={N2}  N={POBLACION:,}",
         ha='center', fontsize=7.5, color=COLOR['gris_med'])

plt.savefig(r'C:/Users/Usuario/Downloads/muestreo_doble_resultado.png',
            dpi=180, bbox_inches='tight', facecolor='#F0F4F8')
print("\n  Figura guardada: muestreo_doble_resultado.png")
print()
separador("FIN DEL ANÁLISIS", char="═")
