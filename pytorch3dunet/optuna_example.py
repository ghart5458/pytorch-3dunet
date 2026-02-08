#!/usr/bin/env python3
"""
Hyperparameter Tuning with Optuna

HYPERPARAMETER TUNING
Hyperparameter tuning is the process of automatically finding the best settings for your model.
Think of it like adjusting the knobs on a radio to get the clearest signal - except here we're
adjusting learning rates, model sizes, and data augmentation settings to get the best accuracy.

WHAT THIS SCRIPT DOES:
1. Tries different combinations of hyperparameters (like learning rate, model size)
2. Trains the model briefly with each combination
3. Records which combinations work best
4. Saves all results to MLflow so you can compare them visually

OPTUNA (There are a lot of options out there, so don't feel stuck with this one!)
- Automatically suggests smart parameter combinations (not random guessing)
- Stops bad experiments early to save time
- Integrates with MLflow to track everything
- Much faster than manually testing different settings

This shows how to tune transforms and model parameters with minimal code changes.
"""
import copy
import yaml
import optuna
from optuna.integration.mlflow import MLflowCallback
import mlflow
from pytorch3dunet.unet3d.config import load_config
from pytorch3dunet.unet3d.trainer import create_trainer


def objective(trial):
    """
    The 'objective function' - this is what Optuna calls to test each hyperparameter combination.

    WHAT HAPPENS HERE:
    1. Optuna gives us a 'trial' object with suggested parameter values
    2. We modify our training config with these suggestions
    3. We run training with the modified config
    4. We return a score that tells Optuna how good this combination was

    Optuna uses this score to decide what to try next (it learns from previous trials).
    """

    # Load our base configuration file
    # This is like loading our "default settings" that we'll modify
    import sys
    #sys.argv = ['optuna_example.py', '--config', r'C:\guineafowl_model_training\00_pilot\train_config.yml']
    sys.argv = ['optuna_example.py', '--config', r'C:\3dunet_trainings\00_gfowl_only\train_config_multiclass.yaml']
    config, config_path = load_config()

    # HYPERPARAMETER SUGGESTIONS
    # Here Optuna suggests values to try. We replace the defaults with Optuna's suggestions.

    # Learning rate: How fast the model learns
    # Too high = model doesn't converge, too low = takes forever to learn
    # 'log=True' means search logarithmically (good for learning rates)
    config['optimizer']['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)

    # Model size: Number of feature maps (affects model capacity and memory usage)
    # More features = more powerful model but uses more GPU memory
    config['model']['f_maps'] = trial.suggest_categorical('f_maps', [8, 16, 32])

    # Patch depth: How deep (Z-dimension) our training patches are
    # Larger patches see more context but use more memory
    config['loaders']['train']['slice_builder']['patch_shape'][0] = trial.suggest_categorical('patch_depth', [16, 24, 32])

    # DATA AUGMENTATION TUNING
    # Augmentation adds variation to training data to prevent overfitting
    # But too much augmentation can hurt performance - we need to find the sweet spot
    for transform in config['loaders']['train']['transformer']['raw']:
        if transform['name'] == 'GaussianBlur3D':
            # How often should we blur images? (10% to 80% of the time)
            transform['execution_probability'] = trial.suggest_float('gaussian_blur_prob', 0.1, 0.8)
        elif transform['name'] == 'AdditiveGaussianNoise':
            # How often should we add random noise? (5% to 50% of the time)
            transform['execution_probability'] = trial.suggest_float('gaussian_noise_prob', 0.05, 0.5)
        elif transform['name'] == 'AdditivePoissonNoise':
            # How often should we add Poisson noise? (5% to 50% of the time)
            transform['execution_probability'] = trial.suggest_float('poisson_noise_prob', 0.05, 0.5)
        elif transform['name'] == 'RandomRotate':
            # Maximum rotation angle in degrees (15° to 90°)
            # Larger angles = more variation but might distort important features
            transform['angle_spectrum'] = trial.suggest_int('rotation_angle', 15, 90)

    # CHECKPOINT DIRECTORY FOR THIS TRIAL
    # Create a unique checkpoint folder for each trial so they don't overwrite each other
    # This makes it easy to load specific trial results later
    base_checkpoint_dir = config['trainer']['checkpoint_dir']
    trial_checkpoint_dir = f"{base_checkpoint_dir}/trial_{trial.number}"
    config['trainer']['checkpoint_dir'] = trial_checkpoint_dir

    # FAST TRAINING FOR TESTING
    # We don't need full training to compare hyperparameters
    # Short runs give us enough info to rank different combinations
    config['trainer']['max_num_iterations'] = 5000      # Only train for 10000 steps
    config['trainer']['validate_after_iters'] = 25     # Check validation every 25 steps

    # MLFLOW RUN FOR THIS TRIAL
    # Create a custom run name with the trial number for easy identification
    run_name = f"trial_{trial.number}"

    mlflow.autolog()                        # Enable automatic PyTorch logging
    mlflow.enable_system_metrics_logging()  # Track CPU, GPU, memory usage

    with mlflow.start_run(run_name=run_name):
        # Log all the settings we're using for this trial
        mlflow.log_params(config)                                           # Log the full config
        mlflow.log_params({f"optuna_{k}": v for k, v in trial.params.items()})  # Log Optuna's suggestions

        # Create the trainer with our modified config and train the model
        trainer = create_trainer(config)
        trainer.fit()  # This runs the actual training

        # Log the final result
        mlflow.log_metric("final_validation_score", trainer.best_eval_score)

    # RETURN THE RESULT
    # This is the key: we return a single number that represents "how good" this trial was
    # Optuna will try to minimize this number (lower = better)
    # For AdaptedRandError, lower values mean better segmentation
    return trainer.best_eval_score


def run_hyperparameter_tuning():
    """
    Main function that coordinates the hyperparameter search.

    WHAT THIS FUNCTION DOES:
    1. Sets up MLflow to track all our experiments
    2. Creates an Optuna "study" (think of it as an experiment manager)
    3. Tells Optuna to run multiple trials with different parameter combinations
    4. Reports the best combination found
    """

    # MLFLOW SETUP
    # Connect to your MLflow server and create/use an experiment for this tuning session
    mlflow.set_tracking_uri("http://localhost:8080")         # Your MLflow server
    mlflow.set_experiment("3dunet_gfowl_optuna")            # Separate experiment for hyperparameter tuning

    # OPTUNA STUDY CREATION
    # A "study" is Optuna's way of managing a hyperparameter search
    # Think of it as the "experiment coordinator" that decides what to try next
    study = optuna.create_study(
        direction='minimize',                       # We want LOWER scores (better segmentation)
        study_name='pytorch3dunet_hyperopt',       # Name for this search session
        storage='sqlite:///optuna_study.db',       # Save results to database (persists between runs)
        load_if_exists=True                        # Resume if we've run this before
    )

    # RUN THE HYPERPARAMETER SEARCH
    # This is where the magic happens:
    # - Optuna will call our 'objective' function 20 times
    # - Each time with different hyperparameter suggestions
    # - It learns from each trial to make better suggestions (not random!)
    # - Each trial creates its own MLflow run with a clear "trial_X" name
    study.optimize(objective, n_trials=150)

    # REPORT RESULTS
    # After all trials finish, create a summary of what we discovered
    with mlflow.start_run():
        mlflow.log_params(study.best_params)        # Best hyperparameter combination found
        mlflow.log_metric("best_objective", study.best_value)  # Best score achieved

        print(f"\nHYPERPARAMETER TUNING COMPLETE!")
        print(f"Best validation score: {study.best_trial.value}")
        print(f"Best hyperparameters found:")
        for param, value in study.best_params.items():
            print(f"  {param}: {value}")
        print(f"\nView all trials in MLflow: http://localhost:8080")


if __name__ == "__main__":
    run_hyperparameter_tuning()