"""Open an interactive MNE browser for one EEGLAB recording or chunk.

Example:
    python tools/data_vis.py model-data/train/sub-001_eeg_chunk_1.set
"""
import sys

import mne

path = sys.argv[1] if len(sys.argv) > 1 else 'eeg-data/sub-001/eeg/sub-001_task-eyesclosed_eeg.set'
raw = mne.io.read_raw_eeglab(path, preload=False)
raw.plot(block=True)
