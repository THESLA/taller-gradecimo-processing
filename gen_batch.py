import requests, base64, os, sys, time

API_KEY = "TU_API_KEY_AQUI"  # Pega tu key de https://build.nvidia.com (no subir keys reales al repo)
URL = "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-dev"
H = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

PROMPTS = [
    ("01_cable_utp.png",
     "Cross section view of a gray UTP Cat5e ethernet network cable cut open, revealing 4 twisted pairs of thin colored copper wires in orange green blue and brown, clean technical educational illustration, isolated on pure white background, sharp detail"),
    ("02_conector_rj45.png",
     "Closeup macro photo of a transparent plastic RJ-45 ethernet network connector showing 8 shiny gold contact pins in a row and the plastic locking clip, with colorful wires entering from behind, isolated on pure white background, studio product photography"),
    ("03_t568b.png",
     "Educational diagram of 8 parallel straight colored electrical wires side by side in a row, colors from left to right: light orange, orange, light green, blue, light blue, green, light brown, brown, each wire clearly separated, clean minimal technical illustration on white background"),
    ("04_herramientas.png",
     "Flat lay overhead photo of computer network cable tools on a white table: a yellow wire stripping tool, a blue handled metal crimping pliers, two small transparent RJ-45 connectors, a coil of gray network cable, and a small black cable tester box, neat organized arrangement, bright studio lighting"),
    ("05_pelar.png",
     "Closeup photo of hands using a yellow wire stripping tool to peel the gray outer jacket off the end of a thick network cable, colorful twisted wires visible inside, step by step tutorial style, white background, bright lighting"),
    ("06_ordenar.png",
     "Closeup photo of fingers holding 8 thin straight parallel colored wires, orange green blue and brown colors with striped white ones, arranged flat in a neat row ready to insert into a connector, tutorial style, white background, bright lighting"),
    ("07_crimpear.png",
     "Closeup photo of a transparent RJ-45 connector with cable inserted into the jaws of a blue handled metal crimping tool, demonstrating the crimping action, tutorial style, white background, bright lighting"),
    ("08_tester.png",
     "Photo of a small black network cable tester device with a gray ethernet cable plugged into it, a row of small glowing green LED indicator lights turned on, on a white table, bright studio lighting"),
]

os.makedirs("imagenes_taller4", exist_ok=True)

start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
end = int(sys.argv[2]) if len(sys.argv) > 2 else len(PROMPTS)

for i in range(start, min(end, len(PROMPTS))):
    name, prompt = PROMPTS[i]
    print(f"[{i+1}/{len(PROMPTS)}] Generando {name}...", flush=True)
    try:
        r = requests.post(URL, headers=H, json={
            "prompt": prompt, "seed": 100 + i, "width": 1024, "height": 1024
        }, timeout=240)
        if r.status_code != 200:
            print(f"  ERROR {r.status_code}: {r.text[:200]}", flush=True)
            continue
        b64 = r.json()["artifacts"][0].get("base64")
        if not b64:
            print("  Sin imagen en respuesta", flush=True)
            continue
        data = base64.b64decode(b64)
        with open(f"imagenes_taller4/{name}", "wb") as f:
            f.write(data)
        print(f"  OK ({len(data)//1024}KB)", flush=True)
    except Exception as e:
        print(f"  EXCEPCION: {type(e).__name__}", flush=True)
    time.sleep(1)

print("Lote terminado.", flush=True)
