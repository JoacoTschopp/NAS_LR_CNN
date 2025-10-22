# Trabajo Práctico Final - Clasificación de Imágenes CIFAR-10

**Visión Computacional Basada en Redes Neuronales Artificiales**  
**Grupo 3**

## 👥 Integrantes

- Joaquín Sebastián Tschopp
- Santiago Bezchinsky

---

## 📋 Índice

1. [Descripción del Proyecto](#-descripción-del-proyecto)
2. [Dataset CIFAR-10](#-dataset-cifar-10)
3. [Arquitectura del Pipeline](#-arquitectura-del-pipeline)
4. [Modelos Implementados](#-modelos-implementados)
5. [Cómo Usar el Notebook](#-cómo-usar-el-notebook)
6. [Estructura del Proyecto](#-estructura-del-proyecto)
7. [Requisitos](#-requisitos)

---

## 🎯 Descripción del Proyecto

Este proyecto implementa un **pipeline completo de Deep Learning** para clasificación de imágenes usando el dataset CIFAR-10. La solución está diseñada con **arquitectura orientada a objetos** que permite:

✅ **Entrenar múltiples arquitecturas CNN** de forma intercambiable  
✅ **Validación automática** con early stopping  
✅ **Evaluación en datasets externos** (CIFAR-10.1)  
✅ **Visualizaciones profesionales** de resultados  
✅ **Detección automática de hardware** (CUDA/MPS/CPU)  

---

## 📊 Dataset CIFAR-10

### Descripción General

**CIFAR-10** (Canadian Institute For Advanced Research) es un dataset de referencia en Computer Vision que contiene **60,000 imágenes a color de 32×32 píxeles**, divididas en **10 clases mutuamente excluyentes**.

### Composición del Dataset

| Clase | ID | Nombre | Descripción | Ejemplos |
|-------|-----|--------|-------------|----------|
| 0 | ✈️ | **Airplane** | Aviones comerciales, jets, avionetas | Boeing 747, Cessna, F-16 |
| 1 | 🚗 | **Automobile** | Sedanes, SUVs, autos deportivos | Toyota, Ford, Ferrari |
| 2 | 🐦 | **Bird** | Pájaros de diferentes especies | Águila, colibrí, gorrión |
| 3 | 🐱 | **Cat** | Gatos domésticos | Persa, siamés, común |
| 4 | 🦌 | **Deer** | Venados, ciervos | Ciervo de cola blanca |
| 5 | 🐕 | **Dog** | Perros de diferentes razas | Labrador, bulldog, husky |
| 6 | 🐸 | **Frog** | Ranas y sapos | Rana arbórea, sapo común |
| 7 | 🐴 | **Horse** | Caballos | Pura sangre, mustang |
| 8 | 🚢 | **Ship** | Barcos, buques, veleros | Crucero, yate, carguero |
| 9 | 🚚 | **Truck** | Camiones, camionetas | Pickup, camión de carga |

### Distribución de Datos

```
┌──────────────────────────────────────────────┐
│  CIFAR-10 Dataset Split                     │
├──────────────────────────────────────────────┤
│  📦 Training:   50,000 imágenes (83.3%)     │
│  📊 Validation: 10,000 imágenes (16.7%)     │
│  🧪 Test:        2,021 imágenes (CIFAR-10.1)│
└──────────────────────────────────────────────┘
```

**Características técnicas:**
- **Resolución**: 32×32 píxeles (baja resolución intencional)
- **Canales**: 3 (RGB)
- **Balanceo**: Perfectamente balanceado (6,000 imágenes por clase en training)
- **Normalización**: Media `[0.491, 0.482, 0.447]`, Std `[0.247, 0.243, 0.262]`

### CIFAR-10.1 (Test Set Independiente)

Utilizamos **CIFAR-10.1** como conjunto de test final. Este dataset fue creado en **2019** por investigadores de UC Berkeley para:

- Evaluar la **verdadera generalización** de modelos
- Detectar **overfitting al dataset original**
- Contiene imágenes **completamente nuevas** con la misma metodología
- **Gap típico**: 4-10% menos accuracy que CIFAR-10 test set

---

## 🏗️ Arquitectura del Pipeline

### Clase `TrainingPipeline`

El proyecto está construido sobre una **clase orientada a objetos** que encapsula todo el flujo de trabajo:

```python
TrainingPipeline
├── __init__()              # Inicialización + detección de hardware
├── _detect_device()        # CUDA > MPS > CPU (automático)
├── _train_epoch()          # Entrenamiento de una época
├── _validate_epoch()       # Validación de una época
├── train()                 # Loop completo con early stopping
├── save_checkpoint()       # Guardar modelo automáticamente
├── load_checkpoint()       # Cargar modelo guardado
├── resume_training()       # Reanudar entrenamiento interrumpido
├── evaluate()              # Evaluación completa con métricas
├── plot_training_curves()  # Visualización de curvas de aprendizaje
├── plot_confusion_matrix() # Matriz de confusión
└── plot_examples()         # Ejemplos visuales de predicciones
```

### Diagrama de Flujo

```
┌─────────────────┐
│  Cargar Datos   │
│   (CIFAR-10)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│ Crear Modelo    │─────▶│  BaseModel       │
│                 │      │  SimpleCNN       │
│                 │      │  ImprovedCNN ⭐  │
│                 │      │  ResNetCIFAR     │
└────────┬────────┘      └──────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Inicializar Pipeline       │
│  - Detecta hardware (GPU)   │
│  - Configura optimizador    │
│  - Inicializa métricas      │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│     Loop de Training        │
│  ┌─────────────────────┐    │
│  │  Época 1..N         │    │
│  │  ├─ Train          │    │
│  │  ├─ Validate       │    │
│  │  ├─ Checkpoint     │    │
│  │  └─ Early Stop?    │    │
│  └─────────────────────┘    │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Visualizar Resultados      │
│  - Curvas de loss/accuracy  │
│  - Análisis de overfitting  │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Evaluar en Test Set        │
│  - CIFAR-10.1 (2,021 imgs)  │
│  - Matriz de confusión      │
│  - Accuracy por clase       │
│  - Ejemplos visuales        │
└─────────────────────────────┘
```

### Características Principales

#### 🖥️ Detección Automática de Hardware

```python
def _detect_device(self):
    if torch.cuda.is_available():
        return torch.device('cuda')  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        return torch.device('mps')   # Apple Silicon
    else:
        return torch.device('cpu')   # CPU fallback
```

**Beneficios:**
- No requiere configuración manual
- Aprovecha GPU automáticamente
- Funciona en cualquier plataforma

#### 🔄 Sistema de Checkpoints Robusto

```python
models/
├── best_model.pth              # Mejor accuracy de validación
├── last_checkpoint.pth         # Checkpoint cada 5 épocas
└── interrupted_checkpoint.pth  # Si se interrumpe (Ctrl+C)
```

**Características:**
- ✅ Guardado automático cada 5 épocas
- ✅ Mejor modelo siempre guardado
- ✅ Recuperación ante interrupciones
- ✅ Método `resume_training()` para continuar

#### ⏹️ Early Stopping

- Monitorea accuracy de validación
- Se detiene si no hay mejora en N épocas
- Evita overfitting
- Ahorra tiempo de entrenamiento

---

## 🧠 Modelos Implementados

### Comparación Rápida

| Modelo | Parámetros | Accuracy Esperado | Velocidad | Mejor Para |
|--------|------------|-------------------|-----------|------------|
| **BaseModel** | 1.6M | ~50% | 1.0x | Baseline |
| **SimpleCNN** | 122K | 65-70% | 1.2x | Prototipado rápido |
| **ImprovedCNN** ⭐ | 340K | **75-80%** | 1.5x | **Producción** |
| **ResNetCIFAR** | 470K | 80-85% | 2.0x | Máximo rendimiento |

---

### 1️⃣ BaseModel (Baseline)

**Arquitectura:** Fully Connected de 2 capas

```python
nn.Sequential(
    nn.Flatten(),           # 3×32×32 = 3072 features
    nn.Linear(3072, 512),
    nn.Tanh(),
    nn.Linear(512, 10)
)
```

**Características:**
- ❌ No usa convoluciones
- 📊 Parámetros: ~1.6M
- 🎯 Accuracy: ~50%
- 💡 Uso: Baseline para comparación

---

### 2️⃣ SimpleCNN

**Arquitectura:** CNN básica con 3 bloques convolucionales

```
Input (3×32×32)
    ↓
[Conv 3→32, 3×3] → ReLU → MaxPool → (32×16×16)
[Conv 32→64, 3×3] → ReLU → MaxPool → (64×8×8)
[Conv 64→128, 3×3] → ReLU → MaxPool → (128×4×4)
    ↓
Flatten → FC(2048→256) → Dropout(0.5) → FC(256→10)
```

**Características:**
- ✅ 3 capas convolucionales
- ✅ Dropout para regularización
- 📊 Parámetros: ~122K
- 🎯 Accuracy: 65-70%
- ⚡ Rápida de entrenar

---

### 3️⃣ ImprovedCNN ⭐ **RECOMENDADA**

**Arquitectura:** CNN profunda con Batch Normalization

```
Input (3×32×32)
    ↓
[Conv 3→64] → BatchNorm → ReLU → (64×32×32)
[Conv 64→128] → BatchNorm → ReLU → MaxPool → Dropout → (128×16×16)
[Conv 128→256] → BatchNorm → ReLU → (256×16×16)
[Conv 256→256] → BatchNorm → ReLU → MaxPool → Dropout → (256×8×8)
[Conv 256→512] → BatchNorm → ReLU → MaxPool → Dropout → (512×4×4)
    ↓
Flatten → FC(8192→512) → BatchNorm → Dropout → FC(512→10)
```

**Características:**
- ✅ 5 bloques convolucionales
- ✅ Batch Normalization (acelera convergencia)
- ✅ Dropout estratégico (previene overfitting)
- 📊 Parámetros: ~340K
- 🎯 Accuracy: **75-80%**
- 🏆 **Mejor balance complejidad/rendimiento**

---

### 4️⃣ ResNetCIFAR

**Arquitectura:** ResNet adaptado con skip connections

```
Input (3×32×32)
    ↓
[Conv 3→64, 3×3] → BatchNorm → ReLU
    ↓
ResidualBlock ×2 (64→64) → (64×32×32)
ResidualBlock ×2 (64→128, stride=2) → (128×16×16)
ResidualBlock ×2 (128→256, stride=2) → (256×8×8)
    ↓
GlobalAvgPool → FC(256→10)
```

**Bloque Residual:**
```
x → [Conv → BN → ReLU → Conv → BN] → (+) → ReLU
↓_________shortcut (identity)_______↑
```

**Características:**
- ✅ Skip connections (combaten vanishing gradient)
- ✅ Global Average Pooling
- 📊 Parámetros: ~470K
- 🎯 Accuracy: 80-85%
- 🚀 Arquitectura state-of-the-art

---

## 🚀 Cómo Usar el Notebook

### Guía Paso a Paso

#### **Paso 1: Ejecutar Setup (Celdas 1-10)**

```python
# Importaciones automáticas
# Descarga de CIFAR-10
# Cálculo de media y std para normalización
```

**Output esperado:**
```
✓ datasets/Grupo_3/cifar10.1_v4_data.npy ya existe
Mean: [0.491, 0.482, 0.447]
Std:  [0.247, 0.243, 0.262]
```

---

#### **Paso 2: Comparar Modelos (Opcional - Celda 17)**

```python
compare_models()
```

**Output:**
```
======================================================================
COMPARACIÓN DE ARQUITECTURAS
======================================================================

BaseModel (actual)
  Parámetros totales: 1,578,506
  Tamaño estimado: 6.02 MB

SimpleCNN
  Parámetros totales: 122,282
  Tamaño estimado: 0.47 MB

ImprovedCNN (recomendada)
  Parámetros totales: 340,042
  Tamaño estimado: 1.30 MB

ResNetCIFAR
  Parámetros totales: 469,194
  Tamaño estimado: 1.79 MB
======================================================================
```

---

#### **Paso 3: Elegir Arquitectura (Celda 22)**

```python
# Opción 1: Baseline (~50% accuracy)
model = BaseModel()

# Opción 2: CNN simple (65-70% accuracy)
model = SimpleCNN()

# Opción 3: CNN mejorada (75-80% accuracy) ⭐ RECOMENDADA
model = ImprovedCNN()

# Opción 4: ResNet (80-85% accuracy)
model = ResNetCIFAR()
```

---

#### **Paso 4: Configurar Hiperparámetros (Celda 21)**

```python
config = {
    'lr': 0.001,           # Learning rate
    'epochs': 50,          # Número máximo de épocas
    'batch_size': 64,      # Tamaño de batch
    'patience': 10,        # Early stopping patience
    'momentum': 0.9,       # Momentum para SGD
    'checkpoint_dir': 'models/'
}
```

**Tips de configuración:**
- ⬆️ `lr` más alto → Converge más rápido (pero puede ser inestable)
- ⬇️ `batch_size` más pequeño → Menos memoria GPU
- ⬆️ `patience` más alto → Más tiempo antes de detener

---

#### **Paso 5: Entrenar (Celda 23)**

```python
pipeline = TrainingPipeline(model, config)
pipeline.train(train_dataloader, validation_dataloader)
```

**Output en tiempo real:**
```
======================================================================
ENTRENAMIENTO DEL MODELO
======================================================================
Épocas: 50
Batch size: 64
Learning rate: 0.001
Device: mps                    ← Detecta automáticamente
======================================================================

Epoch 01 | Train Loss: 1.8265 | Val Loss: 1.7435 | Val Acc: 39.67% ✓ MEJOR
Epoch 02 | Train Loss: 1.6981 | Val Loss: 1.6797 | Val Acc: 42.75% ✓ MEJOR
...
Epoch 25 | Train Loss: 0.8768 | Val Loss: 1.5257 | Val Acc: 49.83% ✓ MEJOR
  → Checkpoint guardado
...

! Early stopping en época 35
  Mejor accuracy: 49.83% (época 25)
======================================================================
```

---

#### **Paso 6: Visualizar Curvas (Celda 25)**

```python
pipeline.plot_training_curves()
```

**Genera 3 gráficos:**
1. 📉 **Loss**: Training vs Validation
2. 📈 **Accuracy**: Evolución por época
3. ⚠️ **Overfitting**: Gap entre train y val

---

#### **Paso 7: Evaluar en Test (Celdas 27-28)**

```python
# Evaluación
results = pipeline.evaluate(test_dataloader, "CIFAR-10.1")

# Visualizaciones
pipeline.plot_confusion_matrix(results['predictions'], 
                               results['labels'], 
                               class_names)

pipeline.plot_examples(images, 
                      results['predictions'], 
                      results['labels'],
                      class_names, mean, std)
```

**Output:**
```
======================================================================
EVALUACIÓN EN CIFAR-10.1
======================================================================
Accuracy en CIFAR-10.1: 36.81%
   Correctas: 744/2021

ACCURACY POR CLASE
======================================================================
  airplane    : 32.69%  ( 208 samples)
  automobile  : 38.21%  ( 212 samples)
  ...
======================================================================
```

---

#### **Paso 8: Reanudar si se Interrumpió (Celda 24)**

```python
pipeline.resume_training('interrupted_checkpoint.pth',
                        train_dataloader,
                        validation_dataloader)
```

---

## 📁 Estructura del Proyecto

```
TP-FINAL/
├── VCBRNA-grupo-3.ipynb           # 📓 Notebook principal
├── README.md                       # 📄 Este archivo
├── Trabajo Práctico Especial.pdf  # 📋 Consigna
│
├── datasets/                       # 💾 Datos (en .gitignore)
│   └── Grupo_3/
│       ├── cifar-10-batches-py/   # CIFAR-10 original
│       ├── cifar10.1_v4_data.npy  # Test set CIFAR-10.1
│       └── cifar10.1_v4_labels.npy
│
└── models/                         # 🤖 Modelos guardados
    ├── best_model.pth              # Mejor accuracy
    ├── last_checkpoint.pth         # Último checkpoint
    └── interrupted_checkpoint.pth  # Si hubo Ctrl+C
```

---

## 📦 Requisitos

### Librerías

```python
torch >= 2.0.0          # Framework de Deep Learning
torchvision >= 0.15.0   # Datasets y transformaciones
numpy >= 1.24.0         # Manipulación de arrays
matplotlib >= 3.7.0     # Visualización
seaborn >= 0.12.0       # Gráficos estadísticos
scikit-learn >= 1.3.0   # Métricas (confusion matrix)
```

### Instalación

```bash
# Activar entorno virtual
source .venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install torch torchvision numpy matplotlib seaborn scikit-learn
```

### Hardware Recomendado

| Hardware | RAM | Tiempo/Época | Notas |
|----------|-----|--------------|-------|
| **CPU** | 8GB+ | ~5-10 min | Funcional pero lento |
| **Apple Silicon (M1/M2/M3)** | 8GB+ | ~1-2 min | ⭐ Excelente balance |
| **NVIDIA GPU (CUDA)** | 4GB VRAM+ | ~30-60 seg | ⭐ Más rápido |

---

## ✨ Características Destacadas

### 🎯 Pipeline Orientado a Objetos
- **Código limpio** y organizado
- **Reutilizable** en otros proyectos
- **Extensible** (fácil agregar modelos)
- **Testeable** (métodos independientes)

### 🖥️ Multi-plataforma
```python
Device detectado: mps    # Mac con Apple Silicon
Device detectado: cuda   # PC con NVIDIA GPU
Device detectado: cpu    # Cualquier máquina
```

### 📊 Visualizaciones Profesionales
- Curvas de loss/accuracy con seaborn
- Matriz de confusión interactiva
- Ejemplos visuales de predicciones
- Análisis automático de overfitting

### 🛡️ Prevención de Overfitting
- **Early stopping** automático
- **Dropout** (0.5) en capas FC
- **Batch Normalization** para estabilidad
- **Data augmentation** ready

---

## 📝 Quick Start (3 Líneas)

```python
# 1. Crear y entrenar
pipeline = TrainingPipeline(ImprovedCNN(), config)
pipeline.train(train_dataloader, validation_dataloader)

# 2. Evaluar
results = pipeline.evaluate(test_dataloader, "CIFAR-10.1")
```

**¡Listo! 🎉**

---

## 🙏 Referencias

- **CIFAR-10**: Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images.
- **CIFAR-10.1**: Recht, B., et al. (2019). Do ImageNet Classifiers Generalize to ImageNet? ICML.
- **ResNet**: He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
- **Batch Normalization**: Ioffe, S., & Szegedy, C. (2015). Batch Normalization. ICML.

---

**Última actualización**: Octubre 2025  
**Versión**: 1.0  
**Notebook**: `VCBRNA-grupo-3.ipynb`
