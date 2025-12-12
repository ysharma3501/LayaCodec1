# LayaCodec
A highly efficient neural audio codec for TTS models

code
```
from IPython.display import Audio
file = r"C:\Users\Nitin\Downloads\hello-fellow-diamonds-in-a-rough-welcome-to-real-barcha-fcb-group-b-you-must-be-at-a-60-or-70-overall-otherwise-lock-off.mp3"
file = r"C:\Users\Nitin\Downloads\ElevenLabs_2025-11-02T22_31_30_Jessica_pre_sp100_s35_sb80_v3.mp3"
file = file.replace("C:\\Users\\Nitin\\Downloads\\", "/mnt/c/Users/Nitin/Downloads/")
import torch

with torch.no_grad():
    codes = model.encode_audio(file)
    wav = model.decode_codes(codes)
```
