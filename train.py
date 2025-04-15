import argparse
import sys

def main():
    model, unknown = model_parser().parse_known_args(sys.argv[1:])
    parser = argparse.ArgumentParser()

    if 'llama3' in model.model_type.lower():
        from train_wrappers.llama3_train_wrapper import LlamaSFTTrainerWrapper
        LlamaSFTTrainerWrapper.add_arguments(parser)
        trainer = LlamaSFTTrainerWrapper(parser.parse_args(unknown))
    else:
        raise NotImplementedError

    if model.train:
        trainer.train()
    if model.evaluate:
        raise NotImplementedError

def model_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        required=True,
        help="The type of model to fine-tune/train/evaluate",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true"
    )
    parser.add_argument(
        "--train",
        action="store_true"
    )
    return parser

if __name__ == '__main__':
    main()