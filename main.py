"""
ML Pipeline
"""
import argparse
import source.clean_data
import source.train_model
import logging


def execute(args):
    """
    Execute the pipeline
    """
    logging.basicConfig(level=logging.INFO)

    # if args.action == "all" or args.action == "basic_cleaning":
    #     logging.info("Basic cleaning procedure started")
    #     source.clean_data.clean_data()

    # if args.action == "all" or args.action == "train_test_model":
    #     logging.info("Train/Test model procedure started")
    #     source.train_model.train_test_model()

    if args.action == "all" or args.action == "check_score":
        logging.info("Score check procedure started")
        source.train_model.evaluate()


if __name__ == "__main__":
    """
    Main entrypoint
    """
    parser = argparse.ArgumentParser(description="ML training pipeline")

    parser.add_argument(
        "--action",
        type=str,
        choices=["basic_cleaning",
                 "train_test_model",
                 "check_score",
                 "all"],
        default="all",
        help="Pipeline action"
    )

    main_args = parser.parse_args()

    execute(main_args)