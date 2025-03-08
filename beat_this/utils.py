from itertools import chain
from pathlib import Path

import numpy as np

import torch
import torchaudio.transforms as T

def index_to_framewise(index, length):
    """Convert an index to a framewise sequence"""
    sequence = np.zeros(length, dtype=bool)
    sequence[index] = True
    return sequence


def filename_to_augmentation(filename):
    """Convert a filename to an augmentation factor."""
    parts = Path(filename).stem.split("_")
    augmentations = {}
    for part in parts[1:]:
        if part.startswith("ps"):
            augmentations["shift"] = int(part[2:])
        elif part.startswith("ts"):
            augmentations["stretch"] = int(part[2:])
    return augmentations


def save_beat_tsv(beats: np.ndarray, downbeats: np.ndarray, outpath: str) -> None:
    """
    Save beat information to a tab-separated file in the standard .beats format:
    each line has a time in seconds, a tab, and a beat number (1 = downbeat).
    The function requires that all downbeats are also listed as beats.

    Args:
        beats (numpy.ndarray): Array of beat positions in seconds (including downbeats).
        downbeats (numpy.ndarray): Array of downbeat positions in seconds.
        outpath (str): Path to the output TSV file.

    Returns:
        None
    """
    # check if all downbeats are beats
    if not np.all(np.isin(downbeats, beats)):
        raise ValueError("Not all downbeats are beats.")

    # handle pickup measure, by considering the beat count of the first full measure
    if len(downbeats) >= 2:
        # find the number of beats between the first two downbeats
        first_downbeat, second_downbeat = np.searchsorted(beats, downbeats[:2])
        beats_in_first_measure = second_downbeat - first_downbeat
        # find the number of beats before the first downbeat
        pickup_beats = first_downbeat
        # derive where to start counting
        if pickup_beats < beats_in_first_measure:
            start_counter = beats_in_first_measure - pickup_beats
        else:
            print(
                "WARNING: There are more beats in the pickup measure than in the first measure. The beat count will start from 2 without trying to estimate the length of the pickup measure."
            )
            start_counter = 1
    else:
        print(
            "WARNING: There are less than two downbeats in the predictions. Something may be wrong. The beat count will start from 2 without trying to estimate the length of the pickup measure."
        )
        start_counter = 1

    # write the beat file
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    counter = start_counter
    downbeats = chain(downbeats, [-1])
    next_downbeat = next(downbeats)
    try:
        with open(outpath, "w") as f:
            for beat in beats:
                if beat == next_downbeat:
                    counter = 1
                    next_downbeat = next(downbeats)
                else:
                    counter += 1
                f.write(f"{beat}\t{counter}\n")
    except KeyboardInterrupt:
        outpath.unlink()  # avoid half-written files


def replace_state_dict_key(state_dict: dict, old: str, new: str):
    """Replaces `old` in all keys of `state_dict` with `new`."""
    keys = list(state_dict.keys())  # take snapshot of the keys
    for key in keys:
        if old in key:
            state_dict[key.replace(old, new)] = state_dict.pop(key)
    return state_dict

def inverse_mel_spectrogram(x, sr=22050, n_fft=1024, hop_length=441, n_mels=128, device="cuda", gain=100.0):
    # Convert input to float32 and move to device
    if x.dtype == torch.float16:
        melspect = x.float().to(device)
    else:
        melspect = torch.tensor(x, dtype=torch.float32, device=device)

    # Step 1: Undo the ln(1 + 1000x) scaling
    linear_melspect = (torch.expm1(melspect)) / 1000  # [n_mels, T]
    linear_melspect = linear_melspect * gain  # Boost scale

    # Step 2: Define the original Mel filterbank
    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        f_min=30,
        f_max=11000,
        n_mels=n_mels,
        mel_scale='slaney',
        normalized='frame_length',
        power=1
    ).to(device)

    # Get Mel filterbank weights
    mel_filterbank = mel_transform.mel_scale.fb.T
    mel_filterbank_pinv = torch.pinverse(mel_filterbank)

    # Map Mel spectrogram to linear spectrogram
    spectrogram = torch.matmul(mel_filterbank_pinv, linear_melspect)  # [n_fft / 2 + 1, T]

    # Step 3: Griffin-Lim
    griffin_lim = T.GriffinLim(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        power=1.0,
        n_iter=32
    ).to(device)

    # Reconstruct waveform
    waveform = griffin_lim(spectrogram)  # [L]

    # Step 4: Normalize waveform
    waveform = waveform / torch.max(torch.abs(waveform))  # [-1, 1]


    return waveform.cpu().numpy()

