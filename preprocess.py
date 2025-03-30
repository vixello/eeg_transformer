import os, sys
import scripts.preprocessing.bci2a as bci2a
from eeg_logger import logger
from download import DATA_BASE_DIR

PREPROCESSED_DATA_BASE_DIR: str = "./preprocessed_data"


def main() -> None:
    dataset_name: str = sys.argv[1] if len(sys.argv) > 1 else ""

    if not os.path.exists(PREPROCESSED_DATA_BASE_DIR):
        os.makedirs(PREPROCESSED_DATA_BASE_DIR)

    match dataset_name:
        case "bci3a":
            pass
        case "bci2a":
            bci2a.extract_epochs(data_path=f"{DATA_BASE_DIR}/BCI_IV_2a", save_path_root=PREPROCESSED_DATA_BASE_DIR)
        case "bci2b":
            pass
        case "physionet":
            pass
        case _:
            logger.warning("No dataset to preprocess provided")


if __name__ == "__main__":
    main()
