from pathlib import Path

import torch

from src.arqui_cnn import BaseModel, SimpleCNN, ImprovedCNN, ResNetCIFAR, NASCNN15
from src.auxiliares import compare_models, draw_model, que_fierro_tengo
from src.load import load_cifar10, load_data
from src.pre_processed import config_augmentation
from src.test import run_cifar101_evaluation
from src.train_pipeline import TrainingPipeline







def main():
    print("Iniciando el proyecto")

    #Que fierro tengo??
    que_fierro_tengo()

    # Experimento Nombre y rutas de salida
    experiment_name = "Grupo_3_V4"
    experiments_root = Path("../experiments")
    experiment_dir = experiments_root / experiment_name

    checkpoints_dir = experiment_dir / "checkpoints"
    plots_dir = experiment_dir / "plots"
    artifacts_dir = experiment_dir / "artifacts"

    for directory in (experiment_dir, checkpoints_dir, plots_dir, artifacts_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Crear carpeta datasets
    datasets = Path("../datasets")
    
    # 1. Cargar datos
    augmentation_configs = config_augmentation()
    datasets_folder = load_data(datasets_folder=str(datasets))


    # Comparar arquitecturas
    compare_models()

    # Dibujar arquitectura
    draw_model(NASCNN15(), output_dir=artifacts_dir)

    # 2. Preprocesamiento de Datos

    # Cargamos los datos de entrenamiento, calculamos media y desvio para normalizar

    augmentation_configs = config_augmentation()
    cifar10_training, cifar10_validation, training_transformations, test_transformations = load_cifar10(datasets_folder, config=augmentation_configs.config_sin_augmentation)

    # 3. Entrenamiento
    # ==============================================================================
    # CONFIGURACIÓN DE HIPERPARÁMETROS
    # ==============================================================================

    config = {
        'experiment_name': experiment_name,
        'lr': 0.1,
        'epochs': 5,
        'batch_size': 128,
        'es_patience': 10,
        'lr_scheduler': True,
        'lr_patience': 3,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'nesterov': True,
        'use_scheduler': True,
        #'warmup_epochs': 5,   #No utilizado
        'label_smoothing': 0.05,
        'optimizer': 'SGD',
        'base_dir': str(experiment_dir),
        'checkpoint_dir': str(checkpoints_dir),
        'experiment_dir': str(experiment_dir),
        'plots_dir': str(plots_dir),
        'artifacts_dir': str(artifacts_dir),
    }

    # Actualizar variables globales para compatibilidad
    LR = config['lr']
    EPOCHS = config['epochs']
    BATCH_SIZE = config['batch_size']

    print("="*70)
    print("CONFIGURACIÓN")
    print("="*70)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*70)


    # ==============================================================================
    # PREPARACIÓN DE DATOS Y MODELO
    # ==============================================================================

    # Crear DataLoaders
    train_dataloader = torch.utils.data.DataLoader(
        cifar10_training, 
        batch_size=config['batch_size'], 
        shuffle=True
    )

    validation_dataloader = torch.utils.data.DataLoader(
        cifar10_validation, 
        batch_size=config['batch_size'], 
        shuffle=False
    )
    print("="*70)
    print("DATASET")
    print("="*70)
    print(f"✓ Train set: {len(train_dataloader.dataset)} imágenes")
    print(f"✓ Validation set: {len(validation_dataloader.dataset)} imágenes")

    # Crear modelo
    #model = BaseModel()
    #model = SimpleCNN()
    #model = ImprovedCNN()
    #model = ResNetCIFAR()
    model = NASCNN15()

    print("="*70)
    print("SELECCIÓN DE MODELO")
    print("="*70)
    print(" ")
    print(f"✓ {model.__class__.__name__}")
    print("="*70)



    # Crear pipeline de entrenamiento
    pipeline = TrainingPipeline(model, config)

    print(f"✓ Pipeline inicializado")
    print(f"✓ Total de parámetros: {sum(p.numel() for p in model.parameters()):,}")

    # ==============================================================================
    # ENTRENAMIENTO
    # ==============================================================================

    pipeline.train(train_dataloader, validation_dataloader)
    
    # ==============================================================================
    # REANUDAR ENTRENAMIENTO (si fue interrumpido)
    # ==============================================================================

    # Descomenta y ejecuta si fue interrumpido:
    # pipeline.resume_training('interrupted_checkpoint.pth', train_dataloader, validation_dataloader)

    print("! Para reanudar, descomenta la línea anterior y ejecuta esta celda")

    # ==============================================================================
    # VISUALIZACIÓN DE PREPROCESAMIENTOS E HIPERPARÁMETROS DEL ENTRENAMIENTO
    # ==============================================================================
    pipeline.register_experiment(training_transformations, test_transformations)

    # ==============================================================================
    # VISUALIZACIÓN DE CURVAS DE ENTRENAMIENTO
    # ==============================================================================

    pipeline.plot_training_curves()

    # ==============================================================================
    # SUMARIZACIÓN DE EXPERIMENTOS
    # ==============================================================================
    summary = pipeline.summarize_experiments(sort_by="results.best_val_acc", top_k=5)

    # ==============================================================================
    # TEST
    # ==============================================================================
    run_cifar101_evaluation(pipeline, datasets_folder)
    
if __name__ == "__main__":
    main()
