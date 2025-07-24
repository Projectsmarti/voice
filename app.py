import os
import gradio as gr
import numpy as np
import torch
import librosa
import soundfile as sf
from google.cloud import texttospeech
from resemblyzer import VoiceEncoder, preprocess_wav
try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:
    speechsdk = None
    print("Azure Speech SDK not available - continuing without Azure support")

try:
    from TTS.api import TTS
except ImportError:
    TTS = None
    print("TTS library not available - continuing with limited functionality")

try:
    import whisper
except ImportError:
    whisper = None
    print("Whisper not available - continuing without voice analysis")

from pydub import AudioSegment
import io
import json
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceCloningSystem:
    def __init__(self):
        self.google_client = None
        self.voice_encoder = VoiceEncoder()
        self.tts_models = {}
        self.speaker_embeddings = {}
        self.whisper_model = None
        
    def setup_google_tts(self, credentials_path: str):
        """Setup Google Cloud TTS client"""
        try:
            if os.path.exists(credentials_path):
                self.google_client = texttospeech.TextToSpeechClient.from_service_account_json(credentials_path)
                logger.info("Google TTS client initialized successfully")
                return True
            else:
                logger.error(f"Credentials file not found: {credentials_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to initialize Google TTS: {e}")
            return False
            
    def setup_azure_tts(self, subscription_key: str, region: str):
        """Setup Azure Cognitive Services TTS"""
        if not speechsdk:
            logger.error("Azure Speech SDK not available")
            return False
            
        try:
            speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)
            self.azure_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
            logger.info("Azure TTS initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Azure TTS: {e}")
            return False
    
    def load_tts_models(self, force_cpu=False):
        """Load multiple TTS models for better accuracy"""
        if not TTS:
            logger.error("TTS library not available. Install with: pip install TTS")
            return False
            
        try:
            device = "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"
            logger.info(f"Using device: {device}")
            
            # Try to load Arabic models with fallbacks
            try:
                # Try different Arabic model configurations
                arabic_models = [
                    "tts_models/ar/css10/tacotron2-DDC",
                    "tts_models/multilingual/multi-dataset/your_tts",
                    "tts_models/en/ljspeech/tacotron2-DDC"  # Fallback to English
                ]
                
                for model_name in arabic_models:
                    try:
                        logger.info(f"Attempting to load: {model_name}")
                        self.tts_models['arabic'] = TTS(model_name).to(device)
                        logger.info(f"Successfully loaded: {model_name}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load {model_name}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Failed to load any Arabic models: {e}")
            
            # Try to load multilingual model
            try:
                if torch.cuda.is_available() and not force_cpu:
                    logger.info("Attempting to load XTTS with GPU")
                    self.tts_models['xtts'] = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
                else:
                    logger.info("Loading lightweight model for CPU")
                    self.tts_models['lightweight'] = TTS("tts_models/en/ljspeech/tacotron2-DDC").to(device)
            except Exception as e:
                logger.warning(f"Failed to load advanced models: {e}")
            
            # Load Whisper if available
            if whisper:
                try:
                    model_size = "tiny" if force_cpu else "base"
                    self.whisper_model = whisper.load_model(model_size)
                    logger.info(f"Whisper {model_size} model loaded")
                except Exception as e:
                    logger.warning(f"Failed to load Whisper: {e}")
            
            if self.tts_models:
                logger.info("TTS models loaded successfully")
                return True
            else:
                logger.error("No TTS models could be loaded")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load TTS models: {e}")
            return False
    
    def extract_voice_features(self, audio_file: str) -> Dict:
        """Extract comprehensive voice features from audio sample"""
        try:
            if not audio_file or not os.path.exists(audio_file):
                return {"error": "Audio file not found"}
                
            # Load audio
            wav, sr = librosa.load(audio_file, sr=22050)
            
            # Extract speaker embedding using Resemblyzer
            processed_wav = preprocess_wav(audio_file)
            speaker_embedding = self.voice_encoder.embed_utterance(processed_wav)
            
            # Extract prosodic features
            pitch, _ = librosa.piptrack(y=wav, sr=sr)
            tempo, _ = librosa.beat.beat_track(y=wav, sr=sr)
            
            # Extract spectral features
            mfccs = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=wav, sr=sr)
            
            # Voice analysis using Whisper
            detected_language = 'unknown'
            if self.whisper_model:
                try:
                    result = self.whisper_model.transcribe(audio_file)
                    detected_language = result.get('language', 'unknown')
                except Exception as e:
                    logger.warning(f"Whisper transcription failed: {e}")
            
            features = {
                'speaker_embedding': speaker_embedding.tolist(),  # Convert to list for JSON serialization
                'pitch_mean': float(np.mean(pitch[pitch > 0])) if np.any(pitch > 0) else 0.0,
                'pitch_std': float(np.std(pitch[pitch > 0])) if np.any(pitch > 0) else 0.0,
                'tempo': float(tempo),
                'mfccs_mean': np.mean(mfccs, axis=1).tolist(),
                'spectral_centroid_mean': float(np.mean(spectral_centroid)),
                'detected_language': detected_language,
                'duration': float(len(wav) / sr)
            }
            
            return features
        except Exception as e:
            logger.error(f"Failed to extract voice features: {e}")
            return {"error": str(e)}
    
    def get_google_voices(self, language_code: str = 'ar-XA') -> List[str]:
        """Get available Google TTS voices for specified language"""
        if not self.google_client:
            return []
        
        try:
            voices = self.google_client.list_voices()
            voice_names = []
            
            for voice in voices.voices:
                for lang_code in voice.language_codes:
                    if lang_code.startswith(language_code.split('-')[0]):
                        voice_names.append(voice.name)
            
            return sorted(list(set(voice_names)))
        except Exception as e:
            logger.error(f"Failed to get Google voices: {e}")
            return []
    
    def generate_google_tts(self, text: str, language_code: str, voice_name: str, 
                           pitch: float, rate: float, volume: float, ssml_mode: bool) -> str:
        """Generate speech using Google TTS with advanced SSML"""
        if not self.google_client:
            raise Exception("Google TTS client not initialized")
        
        try:
            if ssml_mode:
                # Enhanced SSML with prosody control
                ssml_text = f"""
                <speak>
                    <prosody rate="{rate}" pitch="{pitch:+.1f}st" volume="{volume:+.1f}dB">
                        <emphasis level="moderate">{text}</emphasis>
                    </prosody>
                </speak>
                """
                synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)
            else:
                synthesis_input = texttospeech.SynthesisInput(text=text)

            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name
            )

            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=rate,
                pitch=pitch,
                volume_gain_db=volume,
                sample_rate_hertz=22050
            )

            response = self.google_client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )

            output_path = "google_tts_output.mp3"
            with open(output_path, "wb") as out:
                out.write(response.audio_content)

            return output_path
        except Exception as e:
            logger.error(f"Google TTS generation failed: {e}")
            raise
    
    def generate_cloned_voice(self, text: str, reference_audio: str, model_type: str = 'arabic') -> str:
        """Generate speech using voice cloning with reference audio"""
        try:
            if not self.tts_models:
                raise Exception("No TTS models loaded. Please initialize models first.")
            
            # Extract features from reference audio
            features = self.extract_voice_features(reference_audio)
            
            # Select the best available model
            if model_type == 'xtts' and 'xtts' in self.tts_models:
                # Use XTTS for voice cloning
                output_path = "cloned_voice_output.wav"
                self.tts_models['xtts'].tts_to_file(
                    text=text,
                    file_path=output_path,
                    speaker_wav=reference_audio,
                    language="en"  # Use English as fallback since Arabic might not be supported
                )
                return output_path
                
            elif 'arabic' in self.tts_models:
                # Use Arabic model
                output_path = "arabic_output.wav"
                self.tts_models['arabic'].tts_to_file(
                    text=text,
                    file_path=output_path
                )
                return output_path
                
            elif 'lightweight' in self.tts_models:
                # Use lightweight model
                output_path = "lightweight_output.wav"
                self.tts_models['lightweight'].tts_to_file(
                    text=text,
                    file_path=output_path
                )
                return output_path
            else:
                raise Exception("No suitable TTS model available")
                
        except Exception as e:
            logger.error(f"Voice cloning failed: {e}")
            raise
    
    def blend_voices(self, google_audio: str, cloned_audio: str, blend_ratio: float = 0.5) -> str:
        """Blend Google TTS with cloned voice for better results"""
        try:
            # Load both audio files
            google_segment = AudioSegment.from_file(google_audio)
            cloned_segment = AudioSegment.from_file(cloned_audio)
            
            # Normalize lengths
            min_length = min(len(google_segment), len(cloned_segment))
            google_segment = google_segment[:min_length]
            cloned_segment = cloned_segment[:min_length]
            
            # Blend the audio
            blended = google_segment.overlay(cloned_segment, gain_during_overlay=-10 * (1 - blend_ratio))
            
            # Apply voice characteristics from reference
            blended = blended.apply_gain(5)  # Boost volume
            
            output_path = "blended_voice_output.wav"
            blended.export(output_path, format="wav")
            
            return output_path
        except Exception as e:
            logger.error(f"Voice blending failed: {e}")
            return google_audio  # Return original if blending fails

# Initialize the voice cloning system
voice_system = VoiceCloningSystem()

def create_advanced_gradio_ui():
    """Create comprehensive Gradio interface"""
    
    with gr.Blocks(title="Advanced Arabic Voice Cloning System") as demo:
        gr.Markdown("# 🎙️ Advanced Arabic Voice Cloning System")
        gr.Markdown("*Clone voices with high accuracy using multiple TTS models and Google Cloud integration*")
        
        with gr.Tab("Voice Cloning"):
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(
                        label="Text to Synthesize (Arabic/English)",
                        placeholder="أدخل النص هنا أو Enter text here...",
                        lines=3
                    )
                    reference_audio = gr.Audio(
                        label="Reference Voice Sample - Upload a clear audio sample",
                        type="filepath"
                    )
                    
                with gr.Column():
                    language_code = gr.Dropdown(
                        choices=["ar-XA", "en-US", "ar-EG", "ar-SA"],
                        label="Language Code",
                        value="ar-XA"
                    )
                    model_type = gr.Dropdown(
                        choices=["xtts", "arabic", "lightweight", "google_blend"],
                        label="TTS Model",
                        value="arabic"
                    )
                    
            with gr.Row():
                pitch = gr.Slider(-20, 20, 0, label="Pitch Adjustment")
                rate = gr.Slider(0.5, 2.0, 1.0, label="Speaking Rate")
                volume = gr.Slider(-6.0, 6.0, 0.0, label="Volume")
                blend_ratio = gr.Slider(0.0, 1.0, 0.7, label="Voice Blend Ratio")
                
            with gr.Row():
                ssml_mode = gr.Checkbox(label="Enhanced SSML", value=True)
                high_quality = gr.Checkbox(label="High Quality Mode", value=True)
                
            generate_btn = gr.Button("🎵 Generate Cloned Voice", variant="primary")
            
            with gr.Row():
                output_audio = gr.Audio(label="Generated Speech", type="filepath")
                voice_analysis = gr.JSON(label="Voice Analysis Results")
        
        with gr.Tab("Setup & Configuration"):
            gr.Markdown("### System Setup")
            
            with gr.Row():
                force_cpu = gr.Checkbox(label="Force CPU Mode", value=False)
                load_models_btn = gr.Button("🔄 Load TTS Models", variant="secondary")
            
            model_status = gr.Textbox(label="Model Status", interactive=False)
            
            gr.Markdown("### Google Cloud TTS Configuration")
            
            with gr.Row():
                credentials_path = gr.Textbox(
                    label="Google Cloud Credentials Path",
                    value="gemini-lissa-5ae3b8e87082.json"
                )
                setup_google_btn = gr.Button("Setup Google TTS")
                
            available_voices = gr.Dropdown(label="Available Google Voices", choices=[])
            google_status = gr.Textbox(label="Google TTS Status", interactive=False)
        
        # Event handlers
        def load_models_handler(force_cpu):
            try:
                success = voice_system.load_tts_models(force_cpu=force_cpu)
                if success:
                    models_loaded = list(voice_system.tts_models.keys())
                    return f"✅ Models loaded successfully: {', '.join(models_loaded)}"
                else:
                    return "❌ Failed to load models. Check console for details."
            except Exception as e:
                return f"❌ Error loading models: {str(e)}"
        
        def setup_google_tts_handler(credentials_path):
            try:
                success = voice_system.setup_google_tts(credentials_path)
                if success:
                    voices = voice_system.get_google_voices()
                    return gr.update(choices=voices), "✅ Google TTS setup successful"
                else:
                    return gr.update(choices=[]), "❌ Google TTS setup failed"
            except Exception as e:
                return gr.update(choices=[]), f"❌ Setup failed: {str(e)}"
        
        def generate_speech_handler(text, reference_audio, language_code, model_type, 
                                   pitch, rate, volume, blend_ratio, ssml_mode, high_quality):
            try:
                if not text.strip():
                    return None, {"error": "Please enter text to synthesize"}
                
                # Load models if not already loaded
                if not voice_system.tts_models:
                    voice_system.load_tts_models()
                
                # Extract voice features from reference
                voice_features = {}
                if reference_audio:
                    voice_features = voice_system.extract_voice_features(reference_audio)
                
                # Generate speech based on selected model
                if model_type == "google_blend" and voice_system.google_client:
                    # Generate with Google TTS
                    google_voices = voice_system.get_google_voices(language_code)
                    voice_name = google_voices[0] if google_voices else "ar-XA-Wavenet-A"
                    
                    google_audio = voice_system.generate_google_tts(
                        text, language_code, voice_name, pitch, rate, volume, ssml_mode
                    )
                    
                    # Generate cloned version if reference audio provided
                    if reference_audio:
                        cloned_audio = voice_system.generate_cloned_voice(text, reference_audio, "arabic")
                        output_path = voice_system.blend_voices(google_audio, cloned_audio, blend_ratio)
                    else:
                        output_path = google_audio
                else:
                    # Use selected TTS model for cloning
                    if reference_audio:
                        output_path = voice_system.generate_cloned_voice(text, reference_audio, model_type)
                    else:
                        # Generate without reference audio
                        output_path = voice_system.generate_cloned_voice(text, None, model_type)
                
                return output_path, voice_features
                
            except Exception as e:
                logger.error(f"Speech generation failed: {e}")
                return None, {"error": str(e)}
        
        # Connect event handlers
        load_models_btn.click(
            fn=load_models_handler,
            inputs=[force_cpu],
            outputs=[model_status]
        )
        
        setup_google_btn.click(
            fn=setup_google_tts_handler,
            inputs=[credentials_path],
            outputs=[available_voices, google_status]
        )
        
        generate_btn.click(
            fn=generate_speech_handler,
            inputs=[text_input, reference_audio, language_code, model_type,
                   pitch, rate, volume, blend_ratio, ssml_mode, high_quality],
            outputs=[output_audio, voice_analysis]
        )
    
    return demo

# Main execution
if __name__ == "__main__":
    # Initialize the system
    print("Initializing Advanced Arabic Voice Cloning System...")
    
    # Create and launch UI
    demo = create_advanced_gradio_ui()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )