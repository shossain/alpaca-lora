import argparse
import os

from transformers.trainer_utils import get_last_checkpoint

from finetune import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_loader_workers", type=int)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--warmup_step_ratio", type=str, default=0.2)
    parser.add_argument("--do_swa", default=False, action="store_true")
    parser.add_argument("--swa_start_step_ratio", type=str, default=0.75)

    # Data, model, and output directories
    # TODO remove this
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
    # parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--config_bucket", type=str)
    parser.add_argument("--config_key", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--project_name", type=str)
    parser.add_argument("--group_name", type=str)
    # TODO: Fix boolean parse
    parser.add_argument("--show_progress", default=False, action="store_true")
    parser.add_argument(
        "--freeze_base_model", default=False, action="store_true"
    )
    parser.add_argument("--do_full_train", default=False, action="store_true")
    parser.add_argument(
        "--do_loss_weight_averaging", default=False, action="store_true"
    )
    parser.add_argument(
        "--loss_type", type=str, default="cross_entropy"
    )  # cross_entropy, focal, generalized_dice, soft_dice TODO: enum
    parser.add_argument("--label_smoothing_factor", type=str, default=0.0)

    args, _ = parser.parse_known_args()

    last_checkpoint = get_last_checkpoint(args.checkpoints)

    train(
        base_model=args.model_name,
        data_path="yahma/alpaca-cleaned",
        output_dir=args.model_dir,
        checkpoints_dir=args.checkpoints,
        wandb_project=args.project_name,
        wandb_run_name=args.group_name,
        resume_from_checkpoint=last_checkpoint,
    )
