import os, mne, shutil
from eeg_logger import logger

"""
# Event | Type  | Description
_________________________________________
# 768   | 0x0300 | Start of a trial
# 769   | 0x0301 | Left hand (class 1)
# 770   | 0x0302 | Right hand (class 2)
# 771   | 0x0303 | Foot (class 3)
# 772   | 0x0304 | Tongue (class 4)
# 783   | 0x030F | Cue unknown
# 1023  | 0x03FF | Rejected trial
# 1072  | 0x0430 | Eye movements
# 32766 | 0x7FFE | Start of a new run
"""

def extract_epochs(data_path: str, save_path_root: str) -> None:

    if not os.path.exists(data_path):
        logger.error(f"No data to preprocess in {data_path}")
        return

    save_directory: str = __create_save_directory(save_path_root)
    subject_folders = [f for f in os.listdir(data_path)]

    for subject in subject_folders:

        data_file = os.path.join(data_path, subject, f"A{subject[1:3]}T.gdf")

        logger.info(f"Reading data from {subject}...")
        raw = mne.io.read_raw_gdf(data_file, eog=["EOG-left", "EOG-central", "EOG-right"], preload=True)
        epochs = __extract(raw)

        os.makedirs(os.path.join(save_directory, subject))

        filename = os.path.join(save_directory, subject, f"PA{subject[1:3]}T-epo.fif")
        epochs.save(filename)
        logger.info(f"Preprocessed data for subject {subject[1:3]} saved as {filename}")


def __extract(raw_data: mne.io.BaseRaw) -> mne.Epochs:

    events, event_ids = mne.events_from_annotations(raw_data)  # EXTRACT EVENTS
    logger.info(f"Event ids: {event_ids}")  # THIS IS IMPORTANT BECAUSE IT PROVIDES MAPPING TO EVENT IDS
    selected_event_id = {"left_hand": 769, "right_hand": 770, "none": 1023}  # EVENT MAPPINGS BASED ON BCI3A

    # PICK ONLY EEG
    picks = mne.pick_types(raw_data.info, meg=False, eeg=True, eog=False, stim=False, exclude="bads")
    tmin, tmax = 1.0, 4.0

    epochs = mne.Epochs(
        raw_data,
        events,
        event_id=selected_event_id,
        tmin=tmin,
        tmax=tmax,
        picks=picks,
        baseline=None,
        preload=True,
    )
    logger.info(f"Epochs: {epochs}")
    return epochs


def __create_save_directory(save_path_root: str) -> str:

    path: str = f"{save_path_root}/BCI_IV_3a"

    if os.path.exists(path):
        logger.info("Removing old preprocess directory for BCI_IV_3a")
        shutil.rmtree(path)

    os.makedirs(path)

    return path