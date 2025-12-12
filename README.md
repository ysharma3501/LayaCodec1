# 🚀 LayaCodec
LayaCodec: Rapid, High-Fidelity Audio Compression: Reaching the Pareto Frontier in Neural Audio Codecs

This is the github repo for LayaCodec, a fast high quality neural audio codec best suitable for TTS models. 
It has several benefits over other codecs and offers significant speed, efficiency, and quality improvements for TTS models in both the training and inference phases.

### 🔥 Key benefits
- Extremely fast: over 40x faster then similar compressive diffusion based codecs!
- Extreme compression: compresses from rates of 12.5 tokens per second(0.16kpbs) to 50 tokens per second(0.65kpbs)
- Better quality: most neural audio codecs produce only 16khz or 24khz sampling rate audio, not capturing the full range of human hearing, but Laya produces 44.1khz audio.
- Multiple functions: Laya can act as an incredibly fast audio super-resolution model and speech enhancement model as well!

## ⚙️ Installing repo:
```
git clone https://github.com/ysharma3501/LayaCodec.git
cd LayaCodec
pip install requirements.txt
```
Model usage:
```python
from codec.model import LayaCodec
from IPython.display import Audio
import torch

model = LayaCodec.from_pretrained("YatharthS/LayaCodec").cuda().eval()

file = "audio_path"
with torch.no_grad():
    codes = model.encode_audio(file) ## returns discrete codes for TTS, Audio models, etc.
    wav = model.decode_codes(codes) ## returns wav at 44.1khz sampling rate

Audio(wav.cpu(), rate=44100)
```

## 🔨 The architecture
The basic architecture can be summarized as below, heavily inspired by focalcodec.
<img width="1647" height="321" alt="image" src="https://github.com/user-attachments/assets/369bbbb5-8b1d-4c3e-a745-95ed47f9722e" />
1. 6 layers of [wavlm-large](https://huggingface.co/microsoft/wavlm-large) encodes 16khz audio into 50hz features.
2. FocalCompressor compresses these features
3. The BSQ(binary spherical quantizer) model maps the features to discrete codes
4. The FocalDecompressor reconstructs wavlm embeddings from the codes
5. The modified vocos built from a vocos and hifigan hybrid architecture decodes the wavlm embeddings into 44.1khz audio.

## 🗓️ Next steps
- [x] Clean up code
- [ ] Further train model(only trained on several 100 hours of data so undertrained right now)
- [ ] Python package
- [ ] Provide training code
- [ ] Release paper

## 📝 Final notes and Contact
I am seeking GPU Contributions to help improve this model considerably and build new projects, any such contributions will be very helpful. 
I am also happy to solve any issues through email or github issues. 

Email: yatharthsharma350@gmail.com

This is also heavily inspired by [FocalCodec](https://github.com/lucadellalib/focalcodec) and [NandemoGHS's xcodec2 model](https://huggingface.co/NandemoGHS/Anime-XCodec2-44.1kHz-v2), so thanks very much to them.

## 📚 Citation
If you find this work helpful, please leave a star and cite our repo. Thanks very much!
```
@misc{sharma2025layacodec,
    title = {{LayaCodec}},
    author = {Yatharth Sharma},
    howpublished = {\url{https://github.com/ysharma3501/LayaCodec}},
    year = {2025},
}
```

