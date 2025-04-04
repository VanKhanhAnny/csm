from generator import load_csm_1b, Segment
import torchaudio

# Load the generator model
generator = load_csm_1b(device="cuda")

speakers = [0, 1]
transcripts = [
    "Hey how are you doing.",
    "Pretty good, pretty good.",
    "I'm great.",
    "So happy to be speaking to you.",
]
# Adjust audio_paths to match the number of segments if needed.
audio_paths = [
    "prompt1.wav",
    "audio1.wav",
]

def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    return audio_tensor

# Create segments from the provided prompts
segments = [
    Segment(text=transcript, speaker=speaker, audio=load_audio(audio_path))
    for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
]

# Generate new speech using the context from the segments
audio = generator.generate(
    text="I go to school this morning, nah, so boring",
    speaker=1,
    context=segments,
    max_audio_length_ms=10_000,
)

# Save the generated audio to a file
torchaudio.save("audio2.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
