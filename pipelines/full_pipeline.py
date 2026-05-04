"""
End-to-end MLOps pipeline: data validation → train classifier → train segmentor → evaluate → register.
Run with: python pipelines/full_pipeline.py
"""
import logging
import mlflow

from src.data.dataset import load_config, build_dataframe, split_data, validate_dataset
from src.data.preprocessing import get_classifier_generators
from src.training.train_classifier import train as train_classifier
from src.training.train_segmentation import train as train_segmentor
from src.evaluation.evaluate import evaluate_classifier, evaluate_segmentation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)
    mlflow.set_tracking_uri(config["project"]["mlflow_tracking_uri"])
    mlflow.set_experiment(config["project"]["experiment_name"])

    with mlflow.start_run(run_name="full_pipeline"):
        logger.info("Step 1/4: Data validation")
        df = build_dataframe(config["paths"]["raw_data"])
        validate_dataset(df)
        train_df, val_df, test_df = split_data(df, config)

        logger.info("Step 2/4: Train classifier")
        clf_model, clf_history = train_classifier(config_path)

        logger.info("Step 3/4: Train segmentation model")
        seg_model, seg_history = train_segmentor(config_path)

        logger.info("Step 4/4: Evaluate both models")
        _, val_gen, test_gen = get_classifier_generators(train_df, val_df, test_df, config)
        evaluate_classifier(clf_model, test_gen, test_df)

        from src.training.train_segmentation import SegmentationDataGenerator
        test_seg_gen = SegmentationDataGenerator(test_df[test_df["has_mask"] == 1], config)
        metrics = evaluate_segmentation(seg_model, test_seg_gen)

        logger.info(f"Pipeline complete. Avg Dice={metrics['avg_dice']:.4f}")


if __name__ == "__main__":
    run_pipeline()
