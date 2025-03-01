import os
import numpy as np
import soundfile as sf
from tqdm import tqdm
import torch
import torchaudio
import torchaudio.transforms as T

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

if __name__ == "__main__":

    # set the paths
    NPZ_DIR = "E:/data/audio/spectrograms"
    OUTPUT_DIR = "E:/data/audio/mono_tracks"
    METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.tsv")

    # create the output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # get all the .npz files
    npz_files = [os.path.join(NPZ_DIR, f) for f in os.listdir(NPZ_DIR) if f.endswith(".npz")]
    metadata = []

    for npz_path in npz_files:
        # load the data
        data = np.load(npz_path, allow_pickle=True)
        print(f"正在处理 {npz_path}")
        
        dataset_name = os.path.basename(npz_path).replace(".npz", "")
        os.makedirs(os.path.join(OUTPUT_DIR, dataset_name), exist_ok=True)
        keys = list(data.keys())

        # get the spectrograms and filenames
        spectrograms = [data[key] for key in data if key.endswith("/track")]
        filenames = [key.replace("/track", "") for key in data if key.endswith("/track")]

        batch = []
        # reconstruct the audio tracks
        for i, spec in tqdm(enumerate(spectrograms), total=len(spectrograms)):
            
            waveform = inverse_mel_spectrogram(spec.T, sr=22050, n_fft=2048, hop_length=441, n_mels=128, device="cuda")

            # get the filename
            filename = os.path.splitext(filenames[i])[0] + ".wav"
            output_path = os.path.join(OUTPUT_DIR, dataset_name, filename)

            # save the waveform
            sf.write(output_path, np.squeeze(waveform).astype("float32"), samplerate=22050)  # 22kHz 采样率

            # record the metadata
            metadata.append(f"{dataset_name}\t{dataset_name}/{filename}")

    # save the metadata
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata))

    print(f"Reconstructed audio tracks have been saved to: {OUTPUT_DIR}")
    print(f"Metadata has been saved to: {METADATA_FILE}")

