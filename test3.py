import torch
import torchaudio
import threading
import queue
import time
import sounddevice as sd
from generator import load_csm_1b, Segment

class RealTimeVoiceSystem:
    def __init__(self, device="cuda"):
        print(f"Initializing voice system on {device}...")
        
        # Check if CUDA is actually working
        if device == "cuda" and torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            # Set to maximum performance
            torch.cuda.set_device(0)
            # Fix the deprecated autocast
            self.autocast = lambda: torch.amp.autocast(device_type='cuda')
        else:
            print("CUDA not available, falling back to CPU")
            device = "cpu"
            self.autocast = lambda: nullcontext()
            
        # Load the CSM model
        self.generator = load_csm_1b(device=device)
        self.sample_rate = self.generator.sample_rate
        self.device = device
        
        # Audio playback queue and thread
        self.audio_queue = queue.Queue()
        self.playback_thread = None
        self.is_playing = False
        
        # Store voice prompts
        self.voice_prompts = {}
        self.conversation_history = []
        
    def add_voice_prompt(self, voice_id, name, text, audio_path):
        """Add a voice prompt to the system"""
        audio_tensor, sr = torchaudio.load(audio_path)
        audio_tensor = audio_tensor.squeeze(0)
        if sr != self.sample_rate:
            audio_tensor = torchaudio.functional.resample(audio_tensor, orig_freq=sr, new_freq=self.sample_rate)
        
        # Create a segment from the prompt
        prompt = Segment(text=text, speaker=voice_id, audio=audio_tensor)
        
        self.voice_prompts[voice_id] = {
            "name": name,
            "prompt": prompt
        }
        print(f"Added voice prompt: {name} (ID: {voice_id})")
    
    def _audio_playback_worker(self):
        """Worker thread to play audio chunks as they're generated"""
        while self.is_playing:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
                if audio_chunk is not None:
                    # Play the audio chunk
                    audio_np = audio_chunk.cpu().numpy()
                    sd.play(audio_np, self.sample_rate)
                    sd.wait()
                self.audio_queue.task_done()
            except queue.Empty:
                pass
    
    def start_playback(self):
        """Start the audio playback thread"""
        if self.playback_thread is None or not self.playback_thread.is_alive():
            self.is_playing = True
            self.playback_thread = threading.Thread(target=self._audio_playback_worker)
            self.playback_thread.daemon = True
            self.playback_thread.start()
    
    def stop_playback(self):
        """Stop the audio playback thread"""
        self.is_playing = False
        if self.playback_thread:
            self.playback_thread.join(timeout=1.0)
    
    def generate_speech(self, text, voice_id, chunk_size_ms=2000, streaming=True, max_context_turns=2):
        """Generate speech with streaming capability"""
        if voice_id not in self.voice_prompts:
            raise ValueError(f"Voice ID {voice_id} not found")
        
        # Prepare context with the voice prompt and conversation history
        # Use only the voice prompt and last few turns to reduce context window
        context = [self.voice_prompts[voice_id]["prompt"]]
        if max_context_turns > 0 and self.conversation_history:
            context.extend(self.conversation_history[-max_context_turns:])
        
        # Start playback thread for streaming
        if streaming:
            self.start_playback()
        
        # Optimize for smaller max_audio_length to improve performance
        # 50ms per character is a reasonable estimate for speech
        max_audio_length_ms = min(len(text) * 50, 10000)  # Cap at 10 seconds
        
        print(f"Generating speech for: {text}")
        start_time = time.time()
        
        # Clear CUDA cache before generation
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        # Use autocast for mixed precision where available
        with torch.inference_mode():
            if self.device == "cuda":
                with self.autocast():
                    audio = self.generator.generate(
                        text=text,
                        speaker=voice_id,
                        context=context,
                        max_audio_length_ms=max_audio_length_ms,
                        temperature=0.7,
                        topk=20,  # Reduce topk for faster generation
                    )
            else:
                audio = self.generator.generate(
                    text=text,
                    speaker=voice_id,
                    context=context,
                    max_audio_length_ms=max_audio_length_ms,
                    temperature=0.7,
                    topk=20,
                )
        
        generation_time = time.time() - start_time
        print(f"Generation completed in {generation_time:.2f} seconds")
        
        # Store in conversation history
        segment = Segment(text=text, speaker=voice_id, audio=audio)
        self.conversation_history.append(segment)
        
        if streaming:
            # Put the audio in the queue for playback
            self.audio_queue.put(audio)
            # Wait for playback to complete
            self.audio_queue.join()
        
        return audio
    
    def save_conversation(self, filename="conversation.wav"):
        """Save the entire conversation history to a file"""
        if not self.conversation_history:
            print("No conversation to save")
            return
        
        all_audio = torch.cat([seg.audio for seg in self.conversation_history], dim=0)
        torchaudio.save(
            filename, 
            all_audio.unsqueeze(0).cpu(),
            self.sample_rate
        )
        print(f"Conversation saved to {filename}")

# Context manager for when torch.cuda.amp isn't available
class nullcontext:
    def __enter__(self): return None
    def __exit__(self, *args): return None

# Example usage
if __name__ == "__main__":
    # Use CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize the system
    voice_system = RealTimeVoiceSystem(device=device)
    
    # Add voice prompts (you would need to create these prompt files)
    try:
        voice_system.add_voice_prompt(
            voice_id=0,
            name="Voice A",
            text="This is a sample voice prompt.",  # Shortened to reduce context
            audio_path="audio.wav"
        )
        
        # Optional second voice
        # voice_system.add_voice_prompt(
        #     voice_id=1,
        #     name="Voice B",
        #     text="This is another sample voice prompt.",
        #     audio_path="audio1.wav"
        # )
    except Exception as e:
        print(f"Error adding voice prompts: {e}")
        # Continue anyway with empty prompts dict
    
    # Generate speech in real-time
    try:
        # Interactive conversation loop
        while True:
            text = input("Enter text to speak (or 'q' to quit): ")
            if text.lower() == 'q':
                break
            
            # Default to voice 0 if only one voice is available
            if len(voice_system.voice_prompts) == 1:
                voice_id = 0
            else:    
                voice_id = int(input("Choose voice (0 or 1): "))
            
            voice_system.generate_speech(
                text, 
                voice_id, 
                streaming=True, 
                max_context_turns=1  # Reduce context for faster generation
            )
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        voice_system.stop_playback()
        voice_system.save_conversation()