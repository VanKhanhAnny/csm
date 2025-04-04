from generator import load_csm_1b
import torchaudio
import torch

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

generator = load_csm_1b(device=device)

audio = generator.generate(
    text="Hello, my name is Linda. Today, I am providing a comprehensive voice sample designed to capture every detail of my speech patterns. In this recording, you will hear me speak naturally about a variety of topics, varying my pace, pitch, and tone to showcase the full depth of my vocal expression. I will discuss the beauty of a calm sunrise, the excitement of exploring new ideas, and the reflective moments that define my day-to-day experiences. Listen carefully as I articulate complex sentences, ask rhetorical questions, and use emphatic expressions to bring my words to life. This sample is intended to serve as a rich reference for advanced text-to-speech synthesis, ensuring that every subtle inflection and nuance of my voice is accurately captured. Whether I’m narrating a story about a quiet forest or describing the buzz of a lively city street, my goal is to convey both clarity and emotion. Thank you for listening, and I look forward to hearing the synthesized version of my voice.",
    speaker=0,
    context=[],
    max_audio_length_ms=1_200_000,
)

torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)