import os
import numpy as np
import soundfile as sf
from tqdm import tqdm
import torch
import torchaudio
import torch

def inverse_mel_spectrogram(x, sr=22050, n_fft=2048, hop_length=441, n_mels=128, device="cuda"):
    
    dtype = torch.float32  # use float32
    x = torch.tensor(x, dtype=dtype, device=device)  
    
    # reverse log1p
    mel_spectrogram =(torch.exp(x) - 1)**4 / 10

    # mel filter
    mel_filter = torchaudio.transforms.MelScale(
        n_mels=n_mels, 
        sample_rate=sr, 
        n_stft=n_fft//2 + 1,
        f_max=11000,
        f_min=30,
        mel_scale="slaney",
    ).to(device, dtype=dtype)  
    
    mel_filter_matrix = mel_filter(torch.eye(n_fft//2 + 1, dtype=dtype, device=device))  
    mel_filter_pinv = torch.linalg.pinv(mel_filter_matrix)

    # reverse melspectrogram to spectrogram
    spectrogram = torch.matmul(mel_filter_pinv, mel_spectrogram)
    spectrogram = torch.clamp(spectrogram, min=1e-5)  

    # Griffin-Lim
    spectrogram_tensor = spectrogram.unsqueeze(0)  
    waveform = torchaudio.transforms.GriffinLim(n_iter=20, n_fft=n_fft, hop_length=hop_length).to(device, dtype=dtype)(spectrogram_tensor)

    waveform = waveform.squeeze().cpu().numpy()

    # standardize the waveform to -3dB
    waveform = waveform / np.max(np.abs(waveform)) * 0.707

    return waveform

if __name__ == "__main__":

    # set the paths
    NPZ_DIR = "data/audio/spectrograms"
    OUTPUT_DIR = "data/audio/mono_tracks"
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

