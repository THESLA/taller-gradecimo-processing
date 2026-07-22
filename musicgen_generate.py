import argparse, sys, torch, soundfile as sf
from transformers import AutoProcessor, MusicgenForConditionalGeneration

parser = argparse.ArgumentParser(description="Genera musica con MusicGen")
parser.add_argument("prompts", nargs="*", help="Descripciones de la musica a generar")
parser.add_argument("-o", "--output", default="musica", help="Nombre base para archivos")
parser.add_argument("-t", "--tokens", type=int, default=512, help="Tokens (~10s por 512)")
parser.add_argument("-m", "--model", default="facebook/musicgen-small", help="Modelo MusicGen")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando: {device} | Modelo: {args.model}")

processor = AutoProcessor.from_pretrained(args.model)
model = MusicgenForConditionalGeneration.from_pretrained(
    args.model, torch_dtype=torch.float16
).to(device)

prompts = args.prompts if args.prompts else [
    "melodia tranquila de piano y cuerdas, atmosferica",
    "ritmo electronico moderno con bajo y bateria, energico",
    "guitarra acustica suave con percusion ligera, relajante",
]

for i, prompt in enumerate(prompts):
    print(f"Generando: {prompt[:60]}...")
    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
    audio_values = model.generate(**inputs, max_new_tokens=args.tokens)
    audio = audio_values[0, 0].cpu().float().numpy()
    filename = f"{args.output}_{i+1}.wav"
    sf.write(filename, audio, samplerate=model.config.audio_encoder.sampling_rate)
    duracion = len(audio) / model.config.audio_encoder.sampling_rate
    print(f"  -> {filename} ({duracion:.1f}s)")

print("Listo!")
