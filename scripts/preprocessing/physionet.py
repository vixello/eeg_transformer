import os, mne, shutil
from eeg_logger import logger

"""
In Physionet dataset, runs regarding motor imagery are [4, 8, 12].
Each annotation includes one of three codes (T0, T1, or T2):

- T0 corresponds to rest
- T1 corresponds to onset of motion (real or imagined) of the left fist (in runs 3, 4, 7, 8, 11, and 12)
- T2 corresponds to onset of motion (real or imagined) of the right fist (in runs 3, 4, 7, 8, 11, and 12)

"""


def extract_epochs(data_path: str, save_path_root: str) -> None:

    if not os.path.exists(data_path):
        logger.error(f"No data to preprocess in {data_path}")
        return

    save_directory: str = __create_save_directory(save_path_root)
    subject_folders = [f for f in os.listdir(data_path)]
    bad_subjects = [88, 92, 100, 104]  # THESE SUBJECTS HAVE INCOMPLETE ANNOTATIONS

    for idx, subject in enumerate(subject_folders, start=1):

        if idx in bad_subjects:
            continue

        data_file_run_4 = os.path.join(data_path, subject, f"S{subject[1:4]}R04.edf")
        data_file_run_8 = os.path.join(data_path, subject, f"S{subject[1:4]}R08.edf")
        data_file_run_12 = os.path.join(data_path, subject, f"S{subject[1:4]}R12.edf")

        logger.info(f"Reading data from {subject}...")

        raw_run_4 = mne.io.read_raw_edf(data_file_run_4, preload=True)
        raw_run_8 = mne.io.read_raw_edf(data_file_run_8, preload=True)
        raw_run_12 = mne.io.read_raw_edf(data_file_run_12, preload=True)

        epochs_run_4 = __extract(raw_run_4)
        epochs_run_8 = __extract(raw_run_8)
        epochs_run_12 = __extract(raw_run_12)

        os.makedirs(os.path.join(save_directory, subject))

        filename_run_4 = os.path.join(save_directory, subject, f"PA{subject[1:4]}R04-epo.fif")
        filename_run_8 = os.path.join(save_directory, subject, f"PA{subject[1:4]}R08-epo.fif")
        filename_run_12 = os.path.join(save_directory, subject, f"PA{subject[1:4]}R12-epo.fif")

        epochs_run_4.save(filename_run_4)
        epochs_run_8.save(filename_run_8)
        epochs_run_12.save(filename_run_12)
        logger.info(f"Preprocessed data for subject {subject[1:4]} saved")


def __extract(raw_data: mne.io.BaseRaw) -> mne.Epochs:

    events, event_ids = mne.events_from_annotations(raw_data)  # EXTRACT EVENTS
    logger.info(f"Event ids: {event_ids}")  # THIS IS IMPORTANT BECAUSE IT PROVIDES MAPPING TO EVENT IDS
    selected_event_id = {"rest": 1, "left_hand": 2, "right_hand": 3}  # BASED ON EVENT_IDS

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

    path: str = f"{save_path_root}/Physionet"

    if os.path.exists(path):
        logger.info("Removing old preprocess directory for Physionet")
        shutil.rmtree(path)

    os.makedirs(path)

    return path
