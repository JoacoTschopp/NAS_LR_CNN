import os
import textwrap
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix

# ==============================================================================
# PIPELINE DE ENTRENAMIENTO
# ==============================================================================


class TrainingPipeline:
    """
    Pipeline completo de entrenamiento, validación y evaluación
    Incluye:
    - Detección automática de device (CUDA/MPS/CPU)
    - Entrenamiento con early stopping
    - Sistema de checkpoints
    - Evaluación y métricas
    - Visualizaciones profesionales
    """

    def __init__(self, model, config):
        """
        Args:
            model: Modelo de PyTorch (nn.Module)
            config: Dict con configuración (lr, epochs, batch_size, patience, etc.)
        """
        # Detección automática de device
        self.device = self._detect_device()
        print(f"Device detectado: {self.device}")

        # Modelo y configuración
        self.model = model.to(self.device)
        self.config = config

        # Optimizador y loss
        optimizer_type = config.get("optimizer", "SGD")
        for optimizer in optim.__dict__:
            if optimizer == optimizer_type:
                self.optimizer = optim.__dict__[optimizer](
                    self.model.parameters(),
                    lr=config["lr"],
                    # momentum=config.get('momentum', 0.9) # NO for Adam
                )
                break

        # Mantenemos solo esta loss por las clases y el label smoothing para pruebas
        self.loss_function = nn.CrossEntropyLoss(
            # https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#:~:text=Default%3A%20%27mean%27-,label_smoothing,-(float%2C
            label_smoothing=0.05  # Por defecto es 0.0
        ).to(self.device)

        # Estado del entrenamiento
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.current_epoch = 0

        # Directorio de checkpoints
        self.checkpoint_dir = config.get("checkpoint_dir", "models/")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _detect_device(self):
        """Detecta el mejor dispositivo disponible"""
        if torch.cuda.is_available():
            return torch.device("cuda")  # NVIDIA GPU
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")  # Apple Silicon (M1/M2/M3)
        else:
            return torch.device("cpu")  # CPU fallback

    def _train_epoch(self, train_loader):
        """Entrena una época completa"""
        self.model.train()
        running_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_function(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * data.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def _validate_epoch(self, val_loader):
        """Valida una época completa"""
        self.model.eval()
        running_loss = 0.0
        correct = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.loss_function(output, target)
                running_loss += loss.item() * data.size(0)

                probs = self.model.final_activation(output)
                preds = probs.argmax(dim=1)
                correct += (preds == target).sum().item()

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = correct / len(val_loader.dataset)
        return epoch_loss, epoch_acc

    def train(self, train_loader, val_loader):
        """
        Entrenamiento completo con early stopping y checkpoints

        Args:
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
        """
        print("\n" + "=" * 70)
        print("ENTRENAMIENTO DEL MODELO")
        print("=" * 70)
        print(f"Épocas: {self.config['epochs']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Learning rate: {self.config['lr']}")
        print(f"Device: {self.device}")
        print(f"Early stopping patience: {self.config['patience']}")
        print("=" * 70 + "\n")

        patience_counter = 0

        try:
            for epoch in range(1, self.config["epochs"] + 1):
                self.current_epoch = epoch

                # Entrenar
                train_loss = self._train_epoch(train_loader)
                self.train_losses.append(train_loss)

                # Validar
                val_loss, val_acc = self._validate_epoch(val_loader)
                self.val_losses.append(val_loss)
                self.val_metrics.append(val_acc)

                # Logging
                print(
                    f"Epoch {epoch:02d} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.2%}",
                    end="",
                )

                # Guardar mejor modelo
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_epoch = epoch
                    patience_counter = 0
                    self.save_checkpoint("best_model.pth", is_best=True)
                    print(" ✓ MEJOR", end="")
                else:
                    patience_counter += 1

                print()

                # Checkpoint periódico
                if epoch % 5 == 0:
                    self.save_checkpoint("last_checkpoint.pth")
                    print("  → Checkpoint guardado")

                # Early stopping
                if patience_counter >= self.config["patience"]:
                    print(f"\n! Early stopping en época {epoch}")
                    print(f"  No hubo mejora en {self.config['patience']} épocas")
                    print(
                        f"  Mejor accuracy: {self.best_val_acc:.2%} (época {self.best_epoch})"
                    )
                    break

        except KeyboardInterrupt:
            print("\n" + "=" * 70)
            print("! ENTRENAMIENTO INTERRUMPIDO")
            print("=" * 70)
            self.save_checkpoint("interrupted_checkpoint.pth")
            print(
                f"Estado guardado en: {self.checkpoint_dir}interrupted_checkpoint.pth"
            )
            print("=" * 70)

        # Resumen final
        print("\n" + "=" * 70)
        print("RESUMEN DEL ENTRENAMIENTO")
        print("=" * 70)
        print(f"Épocas completadas: {len(self.train_losses)}")
        print(
            f"Mejor accuracy de validación: {self.best_val_acc:.2%} (época {self.best_epoch})"
        )
        print(f"Accuracy final: {self.val_metrics[-1]:.2%}")
        print("=" * 70)

        # Cargar mejor modelo
        self.load_checkpoint("best_model.pth")
        print("\n✓ Mejor modelo cargado automáticamente")

    def save_checkpoint(self, filename, is_best=False):
        """Guarda checkpoint del estado actual"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_metrics": self.val_metrics,
            "config": self.config,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filename):
        """Carga checkpoint desde archivo"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        if not os.path.exists(filepath):
            print(f"! Checkpoint no encontrado: {filepath}")
            return False

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_acc = checkpoint["best_val_acc"]
        self.best_epoch = checkpoint["best_epoch"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.val_metrics = checkpoint["val_metrics"]

        print(f"✓ Checkpoint cargado: {filename}")
        print(f"  Época: {self.current_epoch}, Acc: {self.best_val_acc:.2%}")
        return True

    def resume_training(self, checkpoint_file, train_loader, val_loader):
        """Reanuda entrenamiento desde checkpoint"""
        if not self.load_checkpoint(checkpoint_file):
            print("No se puede reanudar el entrenamiento")
            return

        print(f"\nReanudando desde época {self.current_epoch + 1}...")

        # Ajustar configuración para continuar
        remaining_epochs = self.config["epochs"] - self.current_epoch
        if remaining_epochs <= 0:
            print("El entrenamiento ya se completó")
            return

        # Continuar entrenamiento
        self.train(train_loader, val_loader)

    def evaluate(self, test_loader, dataset_name="Test"):
        """
        Evalúa el modelo en un conjunto de datos

        Args:
            test_loader: DataLoader de test
            dataset_name: Nombre del dataset para logging

        Returns:
            Dict con resultados: accuracy, predictions, labels, probabilities
        """
        print("\n" + "=" * 70)
        print(f"EVALUACIÓN EN {dataset_name.upper()}")
        print("=" * 70)

        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                probs = self.model.final_activation(output)
                preds = probs.argmax(dim=1)

                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)

        # Calcular accuracy
        accuracy = (all_predictions == all_labels).sum() / len(all_labels)

        print(f"\nAccuracy en {dataset_name}: {accuracy:.2%}")
        print(
            f"   Correctas: {(all_predictions == all_labels).sum()}/{len(all_labels)}"
        )
        print("=" * 70)

        return {
            "accuracy": accuracy,
            "predictions": all_predictions,
            "labels": all_labels,
            "probabilities": all_probabilities,
        }

    def describe_pipeline(
        self,
        train_transforms,
        val_transforms,
        max_model_lines=40,
        max_transforms_lines=40,
    ):
        """
        Genera una imagen-resumen del experimento y la guarda en checkpoint_dir.

        Incluye:
        - Arquitectura del modelo
        - Hiperparámetros y optimizador
        - Transformaciones de preprocesamiento

        Args:
            train_transforms, val_transforms:
                Transformaciones (p.ej. torchvision.transforms.Compose) o strings.
            max_model_lines:
                Máximo de líneas a mostrar de la arquitectura del modelo.
            max_transforms_lines:
                Máximo de líneas totales para la sección de transformaciones.
        """
        # ---------------------------------------------------------------------
        # 1) Preparar texto de arquitectura del modelo
        # ---------------------------------------------------------------------
        model_str = str(self.model)
        model_lines = model_str.splitlines()
        if len(model_lines) > max_model_lines:
            model_lines = model_lines[:max_model_lines] + ["...", "(salteado)"]
        # # Wrap para que no se salga del gráfico
        # wrapped_model_lines = []
        # for line in model_lines:
        #     wrapped_model_lines.extend(textwrap.wrap(line, width=80) or [""])

        # ---------------------------------------------------------------------
        # 2) Preparar texto de hiperparámetros y optimizador
        # ---------------------------------------------------------------------
        hp_lines = []
        hp_lines.append(f"Device: {self.device}")
        hp_lines.append(f"Épocas: {self.config.get('epochs', 'N/A')}")
        hp_lines.append(f"Batch size: {self.config.get('batch_size', 'N/A')}")
        hp_lines.append(f"Learning rate (lr): {self.config.get('lr', 'N/A')}")
        hp_lines.append(
            f"Patience (early stopping): {self.config.get('patience', 'N/A')}"
        )
        hp_lines.append("")

        # Info del optimizador
        opt_name = type(self.optimizer).__name__
        opt_state = self.optimizer.state_dict()
        opt_params = opt_state.get("param_groups", [{}])[0]

        hp_lines.append(f"Optimizador: {opt_name}")
        # Filtrar claves numéricas útiles
        for key in ["lr", "momentum", "weight_decay", "nesterov"]:
            if key in opt_params:
                hp_lines.append(f"  - {key}: {opt_params[key]}")

        # ---------------------------------------------------------------------
        # 3) Transformaciones de preprocesamiento
        # ---------------------------------------------------------------------
        def _transforms_to_lines(name, tf_obj):
            lines = []
            if tf_obj is None:
                lines.append(f"{name}: (no especificado)")
            else:
                if isinstance(tf_obj, str):
                    tf_str = tf_obj
                else:
                    # Para objetos Compose / listas, usar su representación en texto
                    tf_str = str(tf_obj)
                lines.append(f"{name}:")
                for str_l in tf_str.splitlines():
                    wrapped = textwrap.wrap(str_l, width=80) or [""]
                    lines.extend([f"  {w}" for w in wrapped])
            lines.append("")  # línea en blanco al final
            return lines

        tf_lines = []
        tf_lines.extend(_transforms_to_lines("Train transforms", train_transforms))
        tf_lines.extend(_transforms_to_lines("Validation transforms", val_transforms))

        if len(tf_lines) > max_transforms_lines:
            tf_lines = tf_lines[:max_transforms_lines] + ["...", "(salteado)"]

        # ---------------------------------------------------------------------
        # 4) Crear la figura tipo “poster” con Matplotlib
        # ---------------------------------------------------------------------
        fig = plt.figure(figsize=(14, 10))

        # Título global
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        experiment_title = self.config.get("experiment_name", "Experimento")
        fig.suptitle(
            f"{experiment_title} | Resumen del pipeline\n{now_str}",
            fontsize=16,
            fontweight="bold",
            y=0.97,
        )

        # Ejes: modelo (arriba izquierda), hparams (arriba derecha),
        # transforms (abajo a la derecha)
        ax_model = fig.add_axes(
            [0.05, 0.50, 0.42, 0.40]
        )  # [left, bottom, width, height]
        ax_hp = fig.add_axes([0.53, 0.50, 0.42, 0.40])
        ax_tf = fig.add_axes([0.53, 0.05, 0.42, 0.40])  # [0.05, 0.07, 0.90, 0.35]

        for ax in (ax_model, ax_hp, ax_tf):
            ax.axis("off")

        def draw_block(ax, title, lines, line_height=0.045):
            y = 1.0
            ax.text(
                0.0,
                y,
                title,
                fontsize=13,
                fontweight="bold",
                ha="left",
                va="top",
                transform=ax.transAxes,
            )
            y -= line_height * 1.3

            for line in lines:
                wrapped = textwrap.wrap(line, width=65) or [""]
                for w in wrapped:
                    ax.text(
                        0.0,
                        y,
                        w,
                        fontsize=9,
                        family="monospace",
                        ha="left",
                        va="top",
                        transform=ax.transAxes,
                    )
                    y -= line_height

        draw_block(
            ax_model, "Arquitectura del modelo", model_lines
        )  # wrapped_model_lines
        draw_block(ax_hp, "Hiperparámetros y optimizador", hp_lines)
        draw_block(
            ax_tf, "Transformaciones de preprocesamiento", tf_lines, line_height=0.035
        )
        plt.show()

    def plot_training_curves(self):
        """Genera gráficos de curvas de entrenamiento"""
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.2)

        colors = sns.color_palette("husl", 3)

        print("\n" + "=" * 70)
        print("CURVAS DE APRENDIZAJE")
        print("=" * 70)
        print(f"Épocas: {len(self.train_losses)}")
        print(
            f"Mejor accuracy: {max(self.val_metrics):.2%} (época {np.argmax(self.val_metrics)+1})"
        )
        print(f"Overfitting gap: {self.val_losses[-1] - self.train_losses[-1]:.4f}")
        print("=" * 70)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        epochs = np.arange(1, len(self.train_losses) + 1)

        # Loss
        axes[0].plot(
            epochs,
            self.train_losses,
            color=colors[0],
            linewidth=2.5,
            label="Train",
            alpha=0.8,
        )
        axes[0].plot(
            epochs,
            self.val_losses,
            color=colors[1],
            linewidth=2.5,
            label="Validation",
            alpha=0.8,
        )
        best_epoch = np.argmin(self.val_losses)
        axes[0].scatter(
            best_epoch + 1,
            self.val_losses[best_epoch],
            color="green",
            s=150,
            marker="*",
            zorder=5,
        )
        axes[0].set_xlabel("Época", fontweight="bold")
        axes[0].set_ylabel("Loss", fontweight="bold")
        axes[0].set_title("Evolución de la Loss", fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(
            epochs,
            np.array(self.val_metrics) * 100,
            color=colors[1],
            linewidth=2.5,
            alpha=0.8,
        )
        best_acc_epoch = np.argmax(self.val_metrics)
        axes[1].scatter(
            best_acc_epoch + 1,
            self.val_metrics[best_acc_epoch] * 100,
            color="green",
            s=150,
            marker="*",
            zorder=5,
        )
        axes[1].axhline(y=10, color="red", linestyle="--", alpha=0.5, label="Baseline")
        axes[1].set_xlabel("Época", fontweight="bold")
        axes[1].set_ylabel("Accuracy (%)", fontweight="bold")
        axes[1].set_title("Evolución del Accuracy", fontweight="bold")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 100)

        # Overfitting gap
        gap = np.array(self.val_losses) - np.array(self.train_losses)
        axes[2].plot(epochs, gap, color="purple", linewidth=2.5, alpha=0.8)
        axes[2].fill_between(epochs, 0, gap, color="purple", alpha=0.2)
        axes[2].axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)
        axes[2].set_xlabel("Época", fontweight="bold")
        axes[2].set_ylabel("Gap (Val - Train)", fontweight="bold")
        axes[2].set_title("Medida de Overfitting", fontweight="bold")
        axes[2].grid(True, alpha=0.3)

        if gap[-1] > 0.5:
            axes[2].text(
                0.5,
                0.95,
                "! OVERFITTING",
                transform=axes[2].transAxes,
                fontsize=11,
                color="red",
                fontweight="bold",
                ha="center",
                bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.3),
            )

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, predictions, labels, class_names):
        """Genera matriz de confusión"""
        cm = confusion_matrix(labels, predictions)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_xlabel("Predicción", fontweight="bold")
        ax.set_ylabel("Real", fontweight="bold")
        ax.set_title("Matriz de Confusión", fontweight="bold", pad=15)
        plt.tight_layout()
        plt.show()

    def plot_examples(
        self,
        images,
        predictions,
        labels,
        class_names,
        mean,
        std,
        n_correct=10,
        n_incorrect=10,
    ):
        """Muestra ejemplos de predicciones correctas e incorrectas"""

        def denormalize(img, mean, std):
            img_denorm = img.copy()
            for c in range(3):
                img_denorm[c] = img_denorm[c] * std[c] + mean[c]
            img_denorm = np.clip(img_denorm, 0, 1)
            return np.transpose(img_denorm, (1, 2, 0))

        correct_mask = predictions == labels
        correct_idx = np.where(correct_mask)[0]
        incorrect_idx = np.where(~correct_mask)[0]

        # Ejemplos correctos
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle("Predicciones CORRECTAS", fontsize=16, fontweight="bold")

        sample_correct = np.random.choice(
            correct_idx, size=min(n_correct, len(correct_idx)), replace=False
        )

        for idx, ax in enumerate(axes.flat):
            if idx < len(sample_correct):
                i = sample_correct[idx]
                img = denormalize(images[i], mean, std)
                ax.imshow(img)
                ax.set_title(
                    f"Real: {class_names[labels[i]]}\n"
                    f"Pred: {class_names[predictions[i]]}",
                    color="green",
                    fontsize=9,
                )
                ax.axis("off")
            else:
                ax.axis("off")

        plt.tight_layout()
        plt.show()

        # Ejemplos incorrectos
        if len(incorrect_idx) > 0:
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            fig.suptitle("Predicciones INCORRECTAS", fontsize=16, fontweight="bold")

            sample_incorrect = np.random.choice(
                incorrect_idx, size=min(n_incorrect, len(incorrect_idx)), replace=False
            )

            for idx, ax in enumerate(axes.flat):
                if idx < len(sample_incorrect):
                    i = sample_incorrect[idx]
                    img = denormalize(images[i], mean, std)
                    ax.imshow(img)
                    ax.set_title(
                        f"Real: {class_names[labels[i]]}\n"
                        f"Pred: {class_names[predictions[i]]}",
                        color="red",
                        fontsize=9,
                    )
                    ax.axis("off")
                else:
                    ax.axis("off")

            plt.tight_layout()
            plt.show()


print("✓ Clase TrainingPipeline cargada exitosamente")
