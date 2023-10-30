import argparse
import os
import cv2
from tqdm import tqdm
import lama_engine as LE

REF_DIR = os.path.dirname(__file__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, default=f"{REF_DIR}/configs/predict_config.yaml"
    )

    return parser.parse_args()


def main(args: argparse.Namespace):
    print(args)
    config_path = args.config_path

    sample_images_dir = os.path.join(REF_DIR, "assets", "sampes_for_export")
    image_original_file_name = "image_original.png"
    image_mask_1_file_name = "image_mask_1.png"
    image_mask_2_file_name = "image_mask_2.png"
    image_mask_3_file_name = "image_mask_3.png"

    # create engine
    engine = LE.LaMaEngine(config_path=config_path)

    # make image paths
    image_pair_paths = [
        (
            os.path.join(sample_images_dir, image_original_file_name),
            os.path.join(sample_images_dir, fn),
        )
        for fn in [
            image_mask_1_file_name,
            image_mask_2_file_name,
            image_mask_3_file_name,
        ]
    ]

    # get inference result for original model
    for ip, mp in tqdm(image_pair_paths):
        assert os.path.isfile(ip)
        assert os.path.isfile(mp)
        image_original = cv2.imread(ip, cv2.IMREAD_COLOR)
        image_mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        assert image_original is not None
        assert image_mask is not None
        output_data = engine.run([(image_original, image_mask)])[0]
        cv2.imwrite(mp.replace("sampes_for_export", "outputs"), output_data)


if __name__ == "__main__":
    args = parse_args()
    main(args)
