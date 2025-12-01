# Instrucciones para Ejecutar Demo NAS

## 📋 Checklist Pre-Ejecución

- ✅ Entorno virtual creado (`.venv`)
- ✅ Dependencias instaladas
- ✅ Implementación actualizada con cambios del paper
- ✅ Configuración `demo` disponible

## 🚀 Ejecución Paso a Paso

### 1. Activar Entorno Virtual

**macOS/Linux**:
```bash
source .venv/bin/activate
```

**Windows**:
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Navegar al Directorio de la Aplicación

```bash
cd app
```

### 3. Ejecutar Demo NAS

```bash
python main.py --mode nas --config demo
```

**Qué sucederá**:
- Se cargarán los datos CIFAR-10 (primera vez descargará ~170MB)
- Iniciará la búsqueda NAS con 10 episodes × 3 arquitecturas = 30 total
- Cada arquitectura se entrenará por 10 épocas
- El schedule progresivo de capas se activará automáticamente
- Todo quedará documentado en logs

**Tiempo estimado**: 2-3 horas (depende del hardware)

### 4. Monitorear Progreso

**En otra terminal** (mientras corre):

```bash
# Ver últimas líneas del log
tail -f logs/nas_demo/nas_search_*.log

# Ver solo cambios de capas
grep "LAYER SCHEDULE" logs/nas_demo/nas_search_*.log

# Ver resúmenes de episodes
grep "EPISODE.*SUMMARY" -A 8 logs/nas_demo/nas_search_*.log
```

### 5. Después de Completar

```bash
# Analizar resultados
python analyze_nas_logs.py logs/nas_demo/nas_search_*.log

# Ver mejor arquitectura
cat checkpoints/nas_demo/best_architecture.json

# Ver estructura de checkpoints
ls -lR checkpoints/nas_demo/
```

## 📊 Qué Verás Durante la Ejecución

### Inicio
```
======================================================================
🔍 MODO: NEURAL ARCHITECTURE SEARCH
======================================================================
Configuración: demo
Episodios: 10
Children/episodio: 3
======================================================================

✓ GPU Apple Silicon (MPS) disponible
✓ Tensor de prueba creado en: mps:0

======================================================================
CARGANDO DATOS
======================================================================
...
✓ Train: 45000 imágenes
✓ Val: 5000 imágenes
✓ Batch size: 128
======================================================================
```

### Durante la Búsqueda
```
━━━ EPISODE 1/10 ━━━
Current depth: 6 layers
Generating and evaluating 3 architectures...

→ Evaluating architecture 1/3
DNA: [3, 48, 1, 1, 5, 24, 1, 1, 7, 36, 1, 1, ...]
Child ep1_child1 - Architecture built: 125,482 parameters, 6 conv layers, 0.48 MB

ENTRENAMIENTO DEL MODELO
Épocas: 10
...
Epoch 01 | Train Loss: 2.1234 | Val Loss: 2.0123 | Val Acc: 15.23%
Epoch 02 | Train Loss: 1.9876 | Val Loss: 1.8765 | Val Acc: 22.45%
...
Epoch 10 | Train Loss: 1.5432 | Val Loss: 1.6123 | Val Acc: 31.23%

Child ep1_child1 - Training completed:
  Max Val Acc (last 3 epochs) = 0.3123
  Reward = 0.030456
```

### Schedule Progresivo
```
🔼 LAYER SCHEDULE: Increasing depth to 8 layers (after 12 architectures)
```

### Resumen de Episode
```
━━━ EPISODE 1 SUMMARY ━━━
  • Mean reward: 0.028456 ± 0.004123
  • Best child this episode: 0.032789
  • Worst child this episode: 0.024123
  • Baseline EMA: 0.028456
  • Global best architecture: 0.032789
  • Controller learning rate: 0.000600
  • Mean advantage: 0.0000

  Progress: 1/10 (10.0%)
```

### Al Finalizar
```
======================================================================
🏁 BÚSQUEDA NAS FINALIZADA
======================================================================
Tiempo total: 2h 15m 34s
Mejor reward: 0.045678
Checkpoints: checkpoints/nas_demo
Logs: logs/nas_demo/nas_search_20231130_143022.log
======================================================================
```

## 🔍 Análisis de Resultados

Después de ejecutar, el script `analyze_nas_logs.py` generará:

```
======================================================================
RESUMEN DE BÚSQUEDA NAS
======================================================================

CONFIGURACIÓN:
  • Device: mps
  • Total arquitecturas: 30
  • Episodes completados: 10
  • Arquitecturas evaluadas: 30

SCHEDULE PROGRESIVO DE CAPAS:
  • Inicio: 6 capas
  • Después de 12 arquitecturas → 8 capas
  • Después de 24 arquitecturas → 10 capas

EVOLUCIÓN DE REWARDS:
  Episode    Mean Reward     Best Child      Global Best    
  ---------- --------------- --------------- ---------------
  1          0.028456        0.032789        0.032789       
  2          0.031234        0.036890        0.036890       
  3          0.033567        0.039123        0.039123       
  ...

TOP 5 ARQUITECTURAS:
  ID                   Val Acc         Reward         
  -------------------- --------------- ---------------
  ep7_child2           0.3623          0.047567       
  ep5_child1           0.3512          0.043298       
  ep8_child3           0.3489          0.042456       
  ...

ESTADÍSTICAS FINALES:
  • Mejor reward encontrado: 0.047567
  • Reward promedio: 0.032145
  • Peor reward: 0.018234
```

## 🛑 Si Algo Sale Mal

### Error: "ModuleNotFoundError"
```bash
# Reinstalar dependencias
pip install -r requirements.txt
```

### Error: "CUDA/MPS not available"
No es problema, el código funcionará en CPU (más lento):
```
✓ Device: cpu
```

### Demo toma demasiado tiempo
Usa menos arquitecturas:
```bash
python main.py --mode nas --config demo --episodes 3 --children 2
# Solo 6 arquitecturas, ~30-40 minutos
```

### Quedó sin memoria
Reduce batch size editando `configs.py`:
```python
'child_batch_size': 64,  # En lugar de 128
```

## 📝 Para Presentación/Demo

### Comandos Rápidos de Demostración

```bash
# 1. Mostrar schedule progresivo
grep "LAYER SCHEDULE" logs/nas_demo/nas_search_*.log

# 2. Mostrar DNAs generados
grep "DNA:" logs/nas_demo/nas_search_*.log | head -10

# 3. Mostrar rewards
grep "Reward =" logs/nas_demo/nas_search_*.log

# 4. Mostrar solo mejores encontradas
grep "NEW BEST" logs/nas_demo/nas_search_*.log

# 5. Resumen ejecutivo
python analyze_nas_logs.py logs/nas_demo/nas_search_*.log
```

### Archivos Clave para Mostrar

1. **Log completo**: `logs/nas_demo/nas_search_*.log`
2. **Mejor arquitectura**: `checkpoints/nas_demo/best_architecture.json`
3. **Análisis**: Salida de `analyze_nas_logs.py`

## ✅ Verificación de Éxito

Al finalizar deberías tener:

- ✅ Log completo en `logs/nas_demo/`
- ✅ Checkpoints en `checkpoints/nas_demo/`
- ✅ `best_architecture.json` creado
- ✅ 30 subdirectorios en `checkpoints/nas_demo/children/`
- ✅ Schedule progresivo visible en logs (6→8→10 capas)
- ✅ Rewards documentados para cada arquitectura
- ✅ Evolución observable (rewards tienden a mejorar)

## 🎯 Próximos Pasos

Después del demo:

1. **Analizar logs** con el script provisto
2. **Revisar mejor arquitectura** encontrada
3. **Comparar** con otros episodes
4. **Documentar** hallazgos para presentación

## 📚 Referencias

- `NAS_PAPER_IMPLEMENTATION.md` - Detalles técnicos
- `DEMO_NAS.md` - Guía completa del demo
- `README_DEMO.md` - Contexto y explicación
- `RESUMEN_IMPLEMENTACION.md` - Cambios realizados

---

**¡Todo listo para ejecutar el demo NAS!** 🚀
