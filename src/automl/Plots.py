# utils/visualize.py

import optuna
import matplotlib.pyplot as plt

def save_optuna_visualizations(study, prefix=""):
    try:
        # Optimization history
        fig = optuna.visualization.plot_optimization_history(
            study,
            target=lambda t: t.values[0],
            target_name="Accuracy"
        )
        fig.write_html(f"{prefix}optuna_optimization_history.html")
        print(f"✅ Saved: {prefix}optuna_optimization_history.html")

        # Pareto front
        fig = optuna.visualization.plot_pareto_front(
            study,
            target_names=["Accuracy", "F1", "Training Time"],
            include_dominated_trials=False
        )
        fig.write_html(f"{prefix}pareto_front.html")
        print(f"✅ Saved: {prefix}pareto_front.html")

        # Param importances
        fig = optuna.visualization.plot_param_importances(
            study,
            target=lambda t: t.values[0],
            target_name="Accuracy"
        )
        fig.write_html(f"{prefix}param_importance.html")
        print(f"✅ Saved: {prefix}param_importance.html")

        # Parallel coordinate
        fig = optuna.visualization.plot_parallel_coordinate(
            study,
            target=lambda t: t.values[0],
            target_name="Accuracy"
        )
        fig.write_html(f"{prefix}parallel_coords.html")
        print(f"✅ Saved: {prefix}parallel_coords.html")

    except Exception as e:
        print(f"⚠️ Optuna visualization error: {e}")


def save_accuracy_histogram(study, path="accuracy_hist.png"):
    try:
        accs = [t.values[0] for t in study.trials if t.values is not None]
        plt.figure()
        plt.hist(accs, bins=20, color='skyblue')
        plt.xlabel("Accuracy")
        plt.ylabel("Count")
        plt.title("Distribution of Accuracy Across Trials")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"✅ Saved: {path}")
    except Exception as e:
        print(f"⚠️ Histogram plot error: {e}")


def save_metric_curves(study, acc_path="trials_accuracy.png", loss_path="trials_loss.png"):
    try:
        # Accuracy Curves
        plt.figure(figsize=(10, 5))
        for t in study.trials:
            if "history" in t.user_attrs:
                plt.plot(t.user_attrs["history"]["acc"], alpha=0.3, label='train_acc' if t == study.trials[0] else "")
                plt.plot(t.user_attrs["history"]["val_acc"], alpha=0.3, linestyle='--', label='val_acc' if t == study.trials[0] else "")
        plt.title("Accuracy per Epoch (All Trials)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(acc_path)
        plt.close()
        print(f"✅ Saved: {acc_path}")

        # Loss Curves
        plt.figure(figsize=(10, 5))
        for t in study.trials:
            if "history" in t.user_attrs:
                plt.plot(t.user_attrs["history"]["loss"], alpha=0.3, label='train_loss' if t == study.trials[0] else "")
                plt.plot(t.user_attrs["history"]["val_loss"], alpha=0.3, linestyle='--', label='val_loss' if t == study.trials[0] else "")
        plt.title("Loss per Epoch (All Trials)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(loss_path)
        plt.close()
        print(f"✅ Saved: {loss_path}")

    except Exception as e:
        print(f"⚠️ Accuracy/Loss curve error: {e}")
