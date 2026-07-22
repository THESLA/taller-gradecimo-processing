import torch
import torchaudio
import soundfile as sf
import numpy as np
from diffusers import StableDiffusionPipeline
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando: {device}")

pipe = StableDiffusionPipeline.from_pretrained(
    "riffusion/riffusion-model-v1",
    torch_dtype=torch.float16,
    safety_checker=None,
).to(device)

SR = 22050
N_FFT = 1024
HOP = 512
N_MELS = 128
N_ITER = 64

inv_mel = torchaudio.transforms.InverseMelScale(
    n_stft=N_FFT // 2 + 1, n_mels=N_MELS, sample_rate=SR
).to(device)

griffin = torchaudio.transforms.GriffinLim(
    n_fft=N_FFT, hop_length=HOP, power=1.0, n_iter=N_ITER
).to(device)

def generate_music(prompt, filename):
    print(f"Generando: {prompt[:50]}...")
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.0,
                 width=512, height=512).images[0]
    img = np.array(image.convert("L"), dtype=np.float32).T / 255.0
    img = img * 2.0 - 1.0
    img = np.clip(img, -10.0, 10.0)
    mel_spec = torch.from_numpy(img).unsqueeze(0).to(device)
    mel_spec = torchaudio.functional.DB_to_amplitude(mel_spec, 1.0, 1.0)
    lin_spec = inv_mel(mel_spec)
    audio = griffin(lin_spec).cpu().numpy()
    sf.write(filename, audio, SR)
    print(f"  -> {filename} ({len(audio)/SR:.1f}s)")

generate_music("cumbia colombiana alegre con acordeon y marimba", "riffusion_1.wav")
generate_music("vallenato romantico con acordeon caja y guacharaca", "riffusion_2.wav")
print("Listo!")
