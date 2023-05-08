import argparse
import os

from transformers.trainer_utils import get_last_checkpoint

from finetune import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--warmup_step_ratio", type=str, default=0.2)

    # Data, model, and output directories
    parser.add_argument(
        "--checkpoints", type=str, default="/opt/ml/checkpoints/"
    )
    parser.add_argument("--checkpoint_index", type=int, default=-1)
    parser.add_argument(
        "--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"]
    )
    parser.add_argument(
        "--model_dir", type=str, default=os.environ["SM_MODEL_DIR"]
    )
    parser.add_argument(
        "--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"]
    )
    parser.add_argument(
        "--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"]
    )

    parser.add_argument("--project_name", type=str)
    parser.add_argument("--group_name", type=str)

    args, _ = parser.parse_known_args()

    last_checkpoint = get_last_checkpoint(args.checkpoints)

    train(
        base_model=args.model_name,
        data_path=args.training_dir,
        output_dir=args.model_dir,
        checkpoints_dir=args.checkpoints,
        num_epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        wandb_project=args.project_name,
        wandb_run_name=args.group_name,
        resume_from_checkpoint=last_checkpoint,
    )
