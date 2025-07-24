import os
import io
import json
import logging
import uuid # For unique filenames
import base64 # For sending audio as base64 if desired

import numpy as np
import torch
import librosa
import soundfile as sf
from google.cloud import texttospeech
from resemblyzer import VoiceEncoder, preprocess_wav
from pydub import AudioSegment

# Flask specific imports
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename

# Suppress DeprecationWarning from NumPy for float conversion
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:
    speechsdk = None
    print("Azure Speech SDK not available - continuing without Azure support")

try:
    from TTS.api import TTS
except ImportError:
    TTS = None
    print("TTS library not available - install with: pip install TTS")

try:
    import whisper
except ImportError:
    whisper = None
    print("Whisper not available - install with: pip install openai-whisper")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads' # Directory to save uploaded audio files
OUTPUT_FOLDER = 'outputs' # Directory to save generated audio files
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac'} # Allowed audio file extensions
GOOGLE_CREDENTIALS_PATH = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', 'gemini-lissa-5ae3b8e87082.json')

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

class VoiceCloningSystem:
    def __init__(self):
        self.google_client = None
        self.voice_encoder = VoiceEncoder()
        self.tts_models = {}
        self.whisper_model = None
        self.device = "cpu" # Default, will be updated based on CUDA availability

        # Initialize models and Google TTS on startup
        self._initial_setup()

    def _initial_setup(self):
        """Perform initial setup for models and Google TTS."""
        logger.info("Performing initial setup for VoiceCloningSystem...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Detected device: {self.device}")

        # Load TTS models
        try:
            self.load_tts_models(force_cpu=(self.device == "cpu"))
        except Exception as e:
            logger.error(f"Initial TTS model loading failed: {e}")

        # Setup Google TTS
        try:
            self.setup_google_tts(GOOGLE_CREDENTIALS_PATH)
        except Exception as e:
            logger.error(f"Initial Google TTS setup failed: {e}")

    def setup_google_tts(self, credentials_path: str):
        """Setup Google Cloud TTS client"""
        try:
            if not credentials_path:
                logger.warning("Google Cloud credentials path is empty.")
                return False

            if not os.path.exists(credentials_path):
                logger.error(f"Google Cloud credentials file not found: {credentials_path}")
                return False

            self.google_client = texttospeech.TextToSpeechClient.from_service_account_json(credentials_path)
            logger.info("Google TTS client initialized successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Google TTS: {e}")
            self.google_client = None
            return False

    def setup_azure_tts(self, subscription_key: str, region: str):
        """Setup Azure Cognitive Services TTS (currently not fully integrated in generation logic)"""
        if not speechsdk:
            logger.error("Azure Speech SDK not available.")
            return False
        if not subscription_key or not region:
            logger.warning("Azure subscription key or region is empty.")
            return False

        try:
            speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)
            self.azure_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
            logger.info("Azure TTS initialized successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Azure TTS: {e}")
            return False

    def load_tts_models(self, force_cpu: bool = False):
        """Load multiple TTS models for better accuracy, prioritizing Arabic."""
        if not TTS:
            raise ImportError("Coqui TTS library not available. Cannot load TTS models.")

        self.tts_models = {}
        self.device = "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"
        logger.info(f"Loading TTS models on device: {self.device}")

        arabic_model_names = [
            "tts_models/ar/css10/tacotron2-DDC",
            "tts_models/multilingual/multi-dataset/your_tts" # Often good for cloning, but speaker_wav is key
        ]
        xtts_model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        fallback_model_name = "tts_models/en/ljspeech/tacotron2-DDC"

        try:
            logger.info(f"Attempting to load XTTS model: {xtts_model_name}")
            self.tts_models['xtts'] = TTS(model_name=xtts_model_name).to(self.device)
            logger.info(f"Successfully loaded XTTS model: {xtts_model_name}")
        except Exception as e:
            logger.warning(f"Failed to load XTTS model {xtts_model_name}: {e}")

        for model_name in arabic_model_names:
            try:
                if 'arabic' not in self.tts_models:
                    logger.info(f"Attempting to load Arabic model: {model_name}")
                    self.tts_models['arabic'] = TTS(model_name=model_name).to(self.device)
                    logger.info(f"Successfully loaded Arabic model: {model_name}")
                    break
            except Exception as e:
                logger.warning(f"Failed to load Arabic model {model_name}: {e}")

        try:
            if 'lightweight' not in self.tts_models and not self.tts_models:
                logger.info(f"Attempting to load lightweight fallback model: {fallback_model_name}")
                self.tts_models['lightweight'] = TTS(model_name=fallback_model_name).to(self.device)
                logger.info(f"Successfully loaded lightweight fallback model: {fallback_model_name}")
        except Exception as e:
            logger.warning(f"Failed to load lightweight fallback model: {e}")

        if whisper:
            try:
                model_size = "tiny" if force_cpu else "base"
                logger.info(f"Attempting to load Whisper {model_size} model.")
                self.whisper_model = whisper.load_model(model_size, device=self.device)
                logger.info(f"Whisper {model_size} model loaded successfully.")
            except Exception as e:
                logger.warning(f"Failed to load Whisper: {e}. Check installation and GPU drivers.")
                self.whisper_model = None

        if self.tts_models or self.whisper_model:
            logger.info("TTS and/or Whisper models loaded successfully.")
            return True
        else:
            logger.error("No TTS or Whisper models could be loaded. Check installations and network.")
            return False

    def extract_voice_features(self, audio_file_path: str) -> Dict:
        """Extract comprehensive voice features from audio sample."""
        try:
            if not audio_file_path or not os.path.exists(audio_file_path):
                logger.error(f"Audio file not found for feature extraction: {audio_file_path}")
                return {"error": "Audio file not found"}

            wav_16k, sr_16k = librosa.load(audio_file_path, sr=16000)
            wav_22k, sr_22k = librosa.load(audio_file_path, sr=22050)

            processed_wav_16k = preprocess_wav(wav_16k, source_sr=sr_16k)
            speaker_embedding = self.voice_encoder.embed_utterance(processed_wav_16k)

            pitch, _ = librosa.piptrack(y=wav_22k, sr=sr_22k)
            non_zero_pitch = pitch[pitch > 0]
            pitch_mean = float(np.mean(non_zero_pitch)) if np.any(non_zero_pitch) else 0.0
            pitch_std = float(np.std(non_zero_pitch)) if np.any(non_zero_pitch) else 0.0
            tempo, _ = librosa.beat.beat_track(y=wav_22k, sr=sr_22k)

            mfccs = librosa.feature.mfcc(y=wav_22k, sr=sr_22k, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=wav_22k, sr=sr_22k)

            detected_language = 'unknown'
            transcription = ''
            if self.whisper_model:
                try:
                    result = self.whisper_model.transcribe(audio_file_path, fp16=(self.device == 'cuda'))
                    detected_language = result.get('language', 'unknown')
                    transcription = result.get('text', '')
                except Exception as e:
                    logger.warning(f"Whisper transcription failed: {e}")

            features = {
                'speaker_embedding': speaker_embedding.tolist(),
                'pitch_mean': pitch_mean,
                'pitch_std': pitch_std,
                'tempo': float(tempo),
                'mfccs_mean': np.mean(mfccs, axis=1).tolist(),
                'spectral_centroid_mean': float(np.mean(spectral_centroid)),
                'detected_language': detected_language,
                'transcription': transcription,
                'duration': float(len(wav_22k) / sr_22k)
            }
            logger.info("Voice features extracted successfully.")
            return features
        except Exception as e:
            logger.error(f"Failed to extract voice features: {e}")
            return {"error": str(e)}

    def get_google_voices(self, language_code: str = 'ar-XA') -> List[str]:
        """Get available Google TTS voices for specified language."""
        if not self.google_client:
            logger.warning("Google TTS client not initialized. Cannot get voices.")
            return []
        try:
            voices = self.google_client.list_voices()
            voice_names = []
            for voice in voices.voices:
                for lang_code in voice.language_codes:
                    if lang_code.startswith(language_code.split('-')[0]):
                        voice_names.append(voice.name)
            unique_voices = sorted(list(set(voice_names)))
            logger.info(f"Found {len(unique_voices)} Google voices for {language_code.split('-')[0]}.")
            return unique_voices
        except Exception as e:
            logger.error(f"Failed to get Google voices: {e}")
            return []

    def generate_google_tts(self, text: str, language_code: str, voice_name: str,
                           pitch: float, rate: float, volume: float, ssml_mode: bool) -> str:
        """Generate speech using Google TTS with advanced SSML."""
        if not self.google_client:
            raise Exception("Google TTS client not initialized.")
        if not text.strip():
            raise ValueError("Text for Google TTS cannot be empty.")

        try:
            pitch = max(-20.0, min(20.0, pitch))
            volume = max(-96.0, min(16.0, volume))

            if ssml_mode:
                ssml_text = f"""
                <speak>
                    <prosody rate="{rate:.2f}" pitch="{pitch:+.1f}st" volume="{volume:+.1f}dB">
                        {text}
                    </prosody>
                </speak>
                """
                synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)
                logger.info(f"Google TTS SSML input: {ssml_text}")
            else:
                synthesis_input = texttospeech.SynthesisInput(text=text)
                logger.info(f"Google TTS text input: {text}")

            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name
            )

            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=rate,
                pitch=pitch,
                volume_gain_db=volume,
                sample_rate_hertz=24000
            )

            response = self.google_client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )

            output_filename = f"google_tts_{uuid.uuid4()}.mp3"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            with open(output_path, "wb") as out:
                out.write(response.audio_content)
            logger.info(f"Google TTS audio saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Google TTS generation failed: {e}")
            raise

    def generate_cloned_voice(self, text: str, reference_audio_path: str, model_type: str) -> str:
        """
        Generate speech using voice cloning with reference audio or a standard model.
        Handles multi-speaker models by requiring reference_audio for cloning.
        """
        if not self.tts_models:
            raise Exception("No Coqui TTS models loaded. Please initialize models first.")
        if not text.strip():
            raise ValueError("Text for cloned voice cannot be empty.")

        output_filename = f"cloned_voice_{uuid.uuid4()}.wav"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        if model_type == 'xtts' and 'xtts' in self.tts_models:
            if not reference_audio_path or not os.path.exists(reference_audio_path):
                raise ValueError("XTTS model requires a reference audio for cloning.")
            try:
                logger.info(f"Generating with XTTS (language='ar') for text: {text[:50]}...")
                self.tts_models['xtts'].tts_to_file(
                    text=text,
                    file_path=output_path,
                    speaker_wav=reference_audio_path,
                    language="ar" # Explicitly set language to Arabic
                )
                logger.info(f"XTTS generated audio saved to {output_path}")
                return output_path
            except Exception as e:
                logger.error(f"XTTS voice cloning failed: {e}")
                if 'arabic' in self.tts_models:
                    logger.warning("Falling back to generic Arabic model due to XTTS failure.")
                    return self._generate_with_generic_tts(text, 'arabic', output_path)
                else:
                    raise

        if 'arabic' in self.tts_models and model_type == 'arabic':
            logger.info(f"Generating with generic Arabic model for text: {text[:50]}...")
            return self._generate_with_generic_tts(text, 'arabic', output_path)
        elif 'lightweight' in self.tts_models and model_type == 'lightweight':
            logger.info(f"Generating with lightweight model for text: {text[:50]}...")
            return self._generate_with_generic_tts(text, 'lightweight', output_path)
        else:
            raise Exception(f"No suitable Coqui TTS model available for selected type '{model_type}'.")


    def _generate_with_generic_tts(self, text: str, model_key: str, output_path: str) -> str:
        """Helper to generate speech using non-cloning Coqui TTS models."""
        try:
            model = self.tts_models.get(model_key)
            if model:
                model.tts_to_file(text=text, file_path=output_path)
                logger.info(f"{model_key} generated audio saved to {output_path}")
                return output_path
            else:
                raise Exception(f"Model '{model_key}' not found.")
        except Exception as e:
            logger.error(f"Error generating with {model_key} model: {e}")
            raise

    def blend_voices(self, audio_paths: List[str], blend_ratio: float = 0.5) -> str:
        """
        Blend multiple audio files. Assumes `audio_paths[0]` is the primary (e.g., Google)
        and `audio_paths[1]` is the secondary (e.g., cloned).
        """
        if not audio_paths or len(audio_paths) < 2:
            raise ValueError("Blending requires at least two audio paths.")
        if not all(os.path.exists(p) for p in audio_paths):
            raise FileNotFoundError("One or more audio files for blending not found.")

        try:
            audio_segments = [AudioSegment.from_file(p) for p in audio_paths]
            min_length = min(len(s) for s in audio_segments)
            audio_segments = [s[:min_length] for s in audio_segments]

            audios_np = []
            for seg in audio_segments:
                seg = seg.set_frame_rate(22050).set_channels(1)
                samples = np.array(seg.get_array_of_samples()).astype(np.float32)
                max_val = np.iinfo(seg.array_type).max
                audios_np.append(samples / max_val)

            google_audio_np = audios_np[0]
            cloned_audio_np = audios_np[1]

            blended_np = (google_audio_np * (1 - blend_ratio)) + (cloned_audio_np * blend_ratio)
            blended_np = np.clip(blended_np, -1.0, 1.0)

            output_amplitude = np.iinfo(np.int16).max
            blended_samples_int16 = (blended_np * output_amplitude).astype(np.int16)

            blended_segment = AudioSegment(
                blended_samples_int16.tobytes(),
                frame_rate=22050,
                sample_width=blended_samples_int16.dtype.itemsize,
                channels=1
            )

            output_filename = f"blended_voice_{uuid.uuid4()}.wav"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            blended_segment.export(output_path, format="wav")
            logger.info(f"Blended audio saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Voice blending failed: {e}")
            return audio_paths[0] if audio_paths else None


# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Max upload size: 16MB

# Initialize the voice cloning system globally (loaded once on app startup)
voice_system = VoiceCloningSystem()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Flask Routes ---

@app.route('/')
def index():
    return "Welcome to the Advanced Arabic Voice Cloning API. Use /api/v1/ to access endpoints."

@app.route('/api/v1/status', methods=['GET'])
def get_status():
    """Returns the current status of loaded models and Google TTS client."""
    models_loaded = list(voice_system.tts_models.keys())
    whisper_status = "Loaded" if voice_system.whisper_model else "Not Loaded/Failed"
    google_status = "Initialized" if voice_system.google_client else "Not Initialized/Failed"
    device_status = voice_system.device

    return jsonify({
        "status": "online",
        "device": device_status,
        "coqui_tts_models": models_loaded,
        "whisper_model": whisper_status,
        "google_tts_client": google_status
    })

@app.route('/api/v1/load_models', methods=['POST'])
def load_models():
    """Endpoint to explicitly load/reload TTS models and Whisper."""
    data = request.json
    force_cpu = data.get('force_cpu', False)
    try:
        success = voice_system.load_tts_models(force_cpu=force_cpu)
        if success:
            models_loaded = list(voice_system.tts_models.keys())
            whisper_status = "Whisper Loaded" if voice_system.whisper_model else "Whisper Failed/Not Available"
            return jsonify({
                "message": "Models loaded successfully.",
                "coqui_tts_models": models_loaded,
                "whisper_model_status": whisper_status,
                "device": voice_system.device
            }), 200
        else:
            return jsonify({"error": "Failed to load models. Check server logs."}), 500
    except Exception as e:
        logger.exception("Error loading models via API")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/setup_google_tts', methods=['POST'])
def setup_google_tts():
    """Endpoint to setup Google TTS client."""
    data = request.json
    credentials_path = data.get('credentials_path', GOOGLE_CREDENTIALS_PATH)
    try:
        success = voice_system.setup_google_tts(credentials_path)
        if success:
            voices = voice_system.get_google_voices("ar-XA") # Default to Arabic voices
            return jsonify({
                "message": "Google TTS setup successful.",
                "available_voices": voices
            }), 200
        else:
            return jsonify({"error": "Google TTS setup failed. Check credentials and permissions."}), 500
    except Exception as e:
        logger.exception("Error setting up Google TTS via API")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/get_google_voices', methods=['GET'])
def get_google_voices():
    """Endpoint to get available Google TTS voices for a given language code."""
    language_code = request.args.get('language_code', 'ar-XA')
    try:
        voices = voice_system.get_google_voices(language_code)
        return jsonify({"voices": voices}), 200
    except Exception as e:
        logger.exception("Error fetching Google voices via API")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/generate_speech', methods=['POST'])
def generate_speech():
    """
    Main endpoint for speech generation (cloning, Google TTS, blending).
    Expects text, and optionally a reference_audio file.
    """
    if 'text' not in request.form:
        return jsonify({"error": "Missing 'text' in form data."}), 400

    text = request.form['text']
    language_code = request.form.get('language_code', 'ar-XA')
    google_voice_name = request.form.get('google_voice_name')
    model_type = request.form.get('model_type', 'xtts')
    pitch = float(request.form.get('pitch', 0.0))
    rate = float(request.form.get('rate', 1.0))
    volume = float(request.form.get('volume', 0.0))
    blend_ratio = float(request.form.get('blend_ratio', 0.7))
    ssml_mode = request.form.get('ssml_mode', 'false').lower() == 'true'
    use_ref_for_embedding = request.form.get('use_ref_for_embedding', 'true').lower() == 'true'

    reference_audio_path = None
    if 'reference_audio' in request.files:
        file = request.files['reference_audio']
        if file.filename == '':
            return jsonify({"error": "No selected reference audio file."}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            reference_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(reference_audio_path)
            logger.info(f"Reference audio saved to: {reference_audio_path}")
        else:
            return jsonify({"error": "Invalid file type for reference audio. Allowed: .wav, .mp3, .flac"}), 400

    output_audio_path = None
    voice_analysis_results = {}
    cleanup_paths = [] # List to track temporary files for cleanup

    try:
        if reference_audio_path and use_ref_for_embedding:
            analysis_results = voice_system.extract_voice_features(reference_audio_path)
            voice_analysis_results = analysis_results if "error" not in analysis_results else {"analysis_error": analysis_results["error"]}

        google_audio_path = None
        cloned_audio_path = None

        # Logic similar to Gradio handler
        if model_type == "google_blend" or (model_type not in ["xtts", "arabic", "lightweight"] and voice_system.google_client):
            if not voice_system.google_client:
                raise Exception("Google TTS client not initialized for Google blend/primary generation.")
            if not google_voice_name:
                # Attempt to get a default voice if none provided and Google TTS is used
                google_voices_available = voice_system.get_google_voices(language_code)
                google_voice_name = google_voices_available[0] if google_voices_available else "ar-XA-Wavenet-A"
                logger.warning(f"No Google voice name provided, falling back to: {google_voice_name}")

            google_audio_path = voice_system.generate_google_tts(
                text, language_code, google_voice_name, pitch, rate, volume, ssml_mode
            )
            cleanup_paths.append(google_audio_path)

        if reference_audio_path and (model_type in ["xtts", "arabic", "google_blend"]):
            try:
                cloned_audio_path = voice_system.generate_cloned_voice(text, reference_audio_path, model_type)
                cleanup_paths.append(cloned_audio_path)
            except Exception as e:
                logger.error(f"Specific cloning model '{model_type}' failed: {e}. Attempting fallback.")
                voice_analysis_results["cloning_error"] = str(e)
                cloned_audio_path = None

        if model_type == "google_blend":
            if google_audio_path and cloned_audio_path:
                output_audio_path = voice_system.blend_voices([google_audio_path, cloned_audio_path], blend_ratio)
                if output_audio_path: cleanup_paths.append(output_audio_path)
            elif google_audio_path:
                output_audio_path = google_audio_path
                voice_analysis_results["blend_warning"] = "Could not clone voice for blending; returning Google TTS output."
            else:
                raise Exception("Neither Google TTS nor cloned audio could be generated for blending.")
        elif model_type in ["xtts", "arabic", "lightweight"]:
            if cloned_audio_path: # If it was successfully generated above
                output_audio_path = cloned_audio_path
            else: # If cloned_audio_path is None, try generic generation
                try:
                    output_audio_path = voice_system._generate_with_generic_tts(text, model_type, os.path.join(OUTPUT_FOLDER, f"generic_tts_{uuid.uuid4()}.wav"))
                    if output_audio_path: cleanup_paths.append(output_audio_path)
                except Exception as e:
                    raise Exception(f"Failed to generate with selected model '{model_type}' without cloning: {e}")
        else:
            raise ValueError(f"Invalid model_type: {model_type}")

        if not output_audio_path or not os.path.exists(output_audio_path):
            raise Exception("No audio output could be generated.")

        # Return the audio file directly
        # You can set a more user-friendly filename for download
        response_filename = f"generated_speech_{uuid.uuid4()}.wav" if not model_type.startswith("google") else f"generated_speech_{uuid.uuid4()}.mp3"
        return send_file(output_audio_path, mimetype='audio/wav' if output_audio_path.endswith('.wav') else 'audio/mpeg',
                         as_attachment=True, download_name=response_filename)

    except Exception as e:
        logger.exception("Error during speech generation API call")
        return jsonify({"error": str(e), "analysis_results": voice_analysis_results}), 500
    finally:
        # Clean up temporary files
        if reference_audio_path and os.path.exists(reference_audio_path):
            try:
                os.remove(reference_audio_path)
                logger.info(f"Cleaned up reference audio: {reference_audio_path}")
            except OSError as e:
                logger.warning(f"Error cleaning up reference audio {reference_audio_path}: {e}")
        # We don't clean up generated output_audio_path immediately here if sending via send_file.
        # It's better to have a separate cleanup job or policy for the OUTPUT_FOLDER.
        # If sending base64, then cleanup_paths should be cleared.
        # For this example, send_file manages cleanup effectively.


if __name__ == '__main__':
    # Flask app will run on 0.0.0.0:5000 by default.
    # For production, use a WSGI server like Gunicorn or uWSGI.
    # For development, you can run directly.
    # Example for Gunicorn: gunicorn -w 1 -b 0.0.0.0:5000 app:app
    logger.info("Starting Flask application...")
    app.run(host='0.0.0.0', port=5000, debug=True) # debug=True is for development only
