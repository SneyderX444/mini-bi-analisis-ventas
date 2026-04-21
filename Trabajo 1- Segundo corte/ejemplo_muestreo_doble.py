"""
EJEMPLO APLICADO: MUESTREO DOBLE (EN DOS FASES)
Control de Calidad en una Fábrica de Tornillos

Contexto: Una fábrica produce 10,000 tornillos diarios.
Se necesita verificar si el diámetro cumple la especificación (10mm ± 0.5mm).
Medir con calibrador digital es costoso (requiere técnico especializado).

Fase 1: Seleccionar 200 tornillos al azar → inspección visual rápida
         (clasificar: "parece bien" o "sospechoso")
Fase 2: De los 200, seleccionar 50 → medición precisa con calibrador
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

np.random.seed(2026)

# ══════════════════════════════════════════════════
# FASE 1: Muestra grande - Inspección visual rápida
# ══════════════════════════════════════════════════
print("=" * 55)
print("  MUESTREO DOBLE — FÁBRICA DE TORNILLOS")
print("=" * 55)

poblacion = 10000
n1 = 200  # Muestra fase 1

# Simular diámetros reales de toda la población
diametros_poblacion = np.random.normal(loc=10.0, scale=0.3, size=poblacion)

# Fase 1: seleccionar 200 al azar
indices_fase1 = np.random.choice(poblacion, n1, replace=False)
muestra_fase1 = diametros_poblacion[indices_fase1]

# Inspección visual (clasificación rápida y barata)
visual_ok = np.sum((muestra_fase1 >= 9.3) & (muestra_fase1 <= 10.7))
visual_sosp = n1 - visual_ok

print(f"\nFASE 1: Inspección visual de {n1} tornillos")
print(f"  Parecen correctos: {visual_ok}")
print(f"  Sospechosos:       {visual_sosp}")
print(f"  Costo: BAJO (solo observación)")

# ══════════════════════════════════════════════════
# FASE 2: Submuestra - Medición precisa
# ══════════════════════════════════════════════════
n2 = 50  # Muestra fase 2

indices_fase2 = np.random.choice(n1, n2, replace=False)
muestra_fase2 = muestra_fase1[indices_fase2]

# Medición precisa con calibrador digital
especificacion_min = 9.5
especificacion_max = 10.5
conformes = np.sum((muestra_fase2 >= especificacion_min) &
                    (muestra_fase2 <= especificacion_max))
no_conformes = n2 - conformes
tasa_defectos = (no_conformes / n2) * 100

print(f"\nFASE 2: Medición precisa de {n2} tornillos")
print(f"  Conformes (9.5-10.5mm):    {conformes}")
print(f"  No conformes:              {no_conformes}")
print(f"  Tasa de defectos:          {tasa_defectos:.1f}%")
print(f"  Costo: ALTO (calibrador + técnico)")

# ══════════════════════════════════════════════════
# ANÁLISIS ESTADÍSTICO DE LA FASE 2
# ══════════════════════════════════════════════════
media = np.mean(muestra_fase2)
mediana = np.median(muestra_fase2)
desv = np.std(muestra_fase2, ddof=1)

print(f"\nANÁLISIS DE LA MUESTRA FINAL (n={n2}):")
print(f"  Media:     {media:.4f} mm")
print(f"  Mediana:   {mediana:.4f} mm")
print(f"  Desv. Est: {desv:.4f} mm")
print(f"  Mínimo:    {np.min(muestra_fase2):.4f} mm")
print(f"  Máximo:    {np.max(muestra_fase2):.4f} mm")

# ══════════════════════════════════════════════════
# INFERENCIA: Estimar defectos en toda la producción
# ══════════════════════════════════════════════════
defectos_estimados = int(tasa_defectos / 100 * poblacion)
print(f"\nINFERENCIA SOBRE LA POBLACIÓN:")
print(f"  Producción diaria:         {poblacion} tornillos")
print(f"  Defectos estimados:        ~{defectos_estimados} tornillos")
print(f"  Solo se midieron:          {n2} (en vez de {poblacion})")
print(f"  Ahorro:                    {((1 - n2/poblacion)*100):.1f}% del costo")

# ══════════════════════════════════════════════════
# COMPARACIÓN: Sin muestreo doble vs Con muestreo doble
# ══════════════════════════════════════════════════
costo_visual = 0.5    # $ por tornillo (inspección visual)
costo_calibrador = 5  # $ por tornillo (medición precisa)

# Sin muestreo doble: medir todos con calibrador
costo_sin = poblacion * costo_calibrador
# Con muestreo doble
costo_con = n1 * costo_visual + n2 * costo_calibrador

print(f"\nCOMPARACIÓN DE COSTOS:")
print(f"  Sin muestreo doble: ${costo_sin:,.0f}")
print(f"  Con muestreo doble: ${costo_con:,.0f}")
print(f"  Ahorro:             ${costo_sin - costo_con:,.0f} ({((1-costo_con/costo_sin)*100):.1f}%)")

# ══════════════════════════════════════════════════
# GRÁFICA
# ══════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Muestreo Doble — Control de Calidad de Tornillos',
             fontsize=14, fontweight='bold')

# 1. Distribución fase 2
axes[0].hist(muestra_fase2, bins=12, color='#2E86AB', edgecolor='black', alpha=0.85)
axes[0].axvline(especificacion_min, color='red', linestyle='--', lw=2, label=f'Mín={especificacion_min}mm')
axes[0].axvline(especificacion_max, color='red', linestyle='--', lw=2, label=f'Máx={especificacion_max}mm')
axes[0].axvline(media, color='green', linestyle='-', lw=2, label=f'Media={media:.2f}mm')
axes[0].set_title('Distribución de Diámetros (Fase 2)', fontweight='bold')
axes[0].set_xlabel('Diámetro (mm)'); axes[0].set_ylabel('Frecuencia')
axes[0].legend(fontsize=8); axes[0].grid(axis='y', alpha=0.3)

# 2. Conformes vs No conformes
axes[1].bar(['Conformes', 'No conformes'], [conformes, no_conformes],
            color=['#2E7D32', '#C62828'], edgecolor='black')
for i, v in enumerate([conformes, no_conformes]):
    axes[1].text(i, v+0.5, str(v), ha='center', fontweight='bold', fontsize=14)
axes[1].set_title('Resultado del Control de Calidad', fontweight='bold')
axes[1].set_ylabel('Cantidad de Tornillos')
axes[1].grid(axis='y', alpha=0.3)

# 3. Comparación de costos
axes[2].bar(['Sin muestreo\ndoble', 'Con muestreo\ndoble'],
            [costo_sin, costo_con], color=['#C62828', '#2E7D32'], edgecolor='black')
axes[2].text(0, costo_sin+500, f'${costo_sin:,.0f}', ha='center', fontweight='bold', fontsize=11)
axes[2].text(1, costo_con+500, f'${costo_con:,.0f}', ha='center', fontweight='bold', fontsize=11)
axes[2].set_title('Comparación de Costos', fontweight='bold')
axes[2].set_ylabel('Costo ($)')
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('muestreo_doble_ejemplo.png', dpi=200, bbox_inches='tight')
plt.show()

print("\n" + "=" * 55)
print("  EJEMPLO COMPLETADO EXITOSAMENTE")
print("=" * 55)
