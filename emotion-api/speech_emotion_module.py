"""
Speech Emotion Recognition Module
==================================
Speech emotion detection using a wav2vec2 model fine-tuned on RAVDESS.
Detects 9 emotions: anger, disgust, fear, happy, neutral, sad, surprised, and more

Input/Output Format:
- INPUT: Audio file path (WAV, MP3, or raw audio) OR audio bytes
- OUTPUT: JSON with emotion labels and confidence scores

Why this model?
- Works perfectly with transformers pipeline
- State-of-the-art accuracy (80-85%)
- Multilingual support
- Fast inference

Usage:
    from speech_emotion_module import SpeechEmotionRecognizer
    recognizer = SpeechEmotionRecognizer()
    result = recognizer.predict_emotion("audio.wav")
"""

import torch
import torchaudio
import json
from pathlib import Path
from typing import Dict, Union, Tuple, List
import logging
import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class SpeechEmotionRecognizer:
    """
    Speech emotion recognition using a wav2vec2 model.

    This model is optimized for:
    - Fast inference
    - High accuracy (80-85%)
    - Multilingual (but best on English)
    - Clear emotion distinctions

    Attributes:
        model_name: Hugging Face model identifier
        device: torch device (cuda if available, else cpu)
        classifier: Lazy-loaded classifier pipeline
    """

    TARGET_SAMPLE_RATE = 16000
    CHUNK_SECONDS = 6.0
    MIN_CHUNK_SECONDS = 1.0
    
    DEFAULT_MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

    LABEL_ALIASES = {
        "ang": "angry",
        "anger": "angry",
        "angry": "angry",
        "cal": "calm",
        "calm": "calm",
        "dis": "disgust",
        "disgust": "disgust",
        "fea": "fearful",
        "fear": "fearful",
        "fearful": "fearful",
        "hap": "happy",
        "happiness": "happy",
        "happy": "happy",
        "neu": "neutral",
        "neutral": "neutral",
        "sad": "sad",
        "sadness": "sad",
        "sur": "surprised",
        "surprise": "surprised",
        "surprised": "surprised",
    }

    def __init__(self, use_gpu: bool = True, model_name: str = None):
        """
        Initialize the speech emotion recognizer.
        
        Args:
            use_gpu: Whether to use GPU if available (default: True)
        """
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        self.processor = None
        self.model = None
        logger.info(f"Speech Emotion Recognizer initialized. Device: {self.device}")
        logger.info(f"Model: {self.model_name}")
        
    def _load_model(self):
        """Lazy-load the model on first use."""
        if self.model is None:
            try:
                from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForAudioClassification
                from transformers.utils import logging as transformers_logging
                
                logger.info(f"Loading model: {self.model_name}")
                logger.info("This may take 30-60 seconds on first run...")
                
                previous_verbosity = transformers_logging.get_verbosity()
                transformers_logging.set_verbosity_error()
                try:
                    # Load feature extractor (handles audio preprocessing).
                    # This model does not ship tokenizer/processor files, so
                    # AutoProcessor fails on recent Transformers versions.
                    self.processor = AutoFeatureExtractor.from_pretrained(self.model_name)
                    
                    config = AutoConfig.from_pretrained(self.model_name)
                    if self._uses_legacy_classifier_head():
                        # Load model with a projection size matching this
                        # checkpoint's legacy classifier head.
                        config.classifier_proj_size = config.hidden_size
                    self.model = AutoModelForAudioClassification.from_pretrained(
                        self.model_name,
                        config=config,
                    )
                finally:
                    transformers_logging.set_verbosity(previous_verbosity)

                if self._uses_legacy_classifier_head():
                    self._restore_classifier_head()
                self.model = self.model.to(self.device)
                self.model.eval()  # Set to evaluation mode
                
                logger.info("[OK] Model loaded successfully")
                
            except ImportError as e:
                logger.error(f"Required library not installed: {e}")
                raise ImportError("Please install: pip install transformers torch torchaudio librosa")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise

    def _uses_legacy_classifier_head(self) -> bool:
        """Whether this checkpoint needs old classifier-head key remapping."""
        return self.model_name == self.DEFAULT_MODEL_NAME

    def _restore_classifier_head(self):
        """
        Restore the emotion classifier weights for checkpoints saved with the
        older Wav2Vec2 classification-head names.

        Recent Transformers versions expect:
            projector.*
            classifier.*

        This checkpoint stores:
            classifier.dense.*
            classifier.output.*
        """
        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file

            checkpoint_path = hf_hub_download(self.model_name, filename="model.safetensors")
            checkpoint = load_file(checkpoint_path)

            name_map = {
                "classifier.dense.weight": "projector.weight",
                "classifier.dense.bias": "projector.bias",
                "classifier.output.weight": "classifier.weight",
                "classifier.output.bias": "classifier.bias",
            }

            model_state = self.model.state_dict()
            restored = {}
            for old_name, new_name in name_map.items():
                if old_name not in checkpoint or new_name not in model_state:
                    continue
                if checkpoint[old_name].shape != model_state[new_name].shape:
                    logger.warning(
                        "Skipping classifier weight %s -> %s due to shape mismatch: %s != %s",
                        old_name,
                        new_name,
                        tuple(checkpoint[old_name].shape),
                        tuple(model_state[new_name].shape),
                    )
                    continue
                restored[new_name] = checkpoint[old_name]

            if len(restored) == len(name_map):
                self.model.load_state_dict(restored, strict=False)
                logger.info("[OK] Restored classifier head weights")
            else:
                logger.warning(
                    "Could not restore all classifier head weights (%s/%s restored)",
                    len(restored),
                    len(name_map),
                )
        except Exception as e:
            logger.warning(f"Could not restore classifier head weights: {e}")

    def _audio_array_to_waveform(self, audio_data) -> torch.Tensor:
        """Convert loaded audio data to a normalized mono float tensor."""
        audio_array = np.asarray(audio_data)

        if audio_array.size == 0:
            raise ValueError("Audio data is empty")

        if np.issubdtype(audio_array.dtype, np.integer):
            dtype_info = np.iinfo(audio_array.dtype)
            max_value = max(abs(dtype_info.min), dtype_info.max)
            audio_array = audio_array.astype(np.float32) / max_value
        else:
            audio_array = audio_array.astype(np.float32)
            max_abs = float(np.max(np.abs(audio_array)))
            if max_abs > 1.0:
                audio_array = audio_array / max_abs

        # soundfile/scipy/librosa return audio as samples x channels for stereo.
        if audio_array.ndim == 2:
            audio_array = audio_array.mean(axis=1)
        elif audio_array.ndim != 1:
            raise ValueError(f"Unsupported audio shape: {audio_array.shape}")

        return torch.from_numpy(np.ascontiguousarray(audio_array)).float()

    def _load_audio_file(self, audio_path: Path) -> Tuple[torch.Tensor, int]:
        """
        Load an audio file without requiring TorchCodec.

        torchaudio 2.11 can route file decoding through TorchCodec. The fallback
        stack below keeps common WAV/MP3 files working in lightweight installs.
        """
        load_errors = []

        try:
            import soundfile as sf

            audio_data, sample_rate = sf.read(str(audio_path), dtype="float32")
            return self._audio_array_to_waveform(audio_data), int(sample_rate)
        except Exception as e:
            load_errors.append(f"soundfile: {e}")

        try:
            from scipy.io import wavfile as scipy_wavfile

            sample_rate, audio_data = scipy_wavfile.read(str(audio_path))
            return self._audio_array_to_waveform(audio_data), int(sample_rate)
        except Exception as e:
            load_errors.append(f"scipy: {e}")

        try:
            import librosa

            audio_data, sample_rate = librosa.load(str(audio_path), sr=None, mono=True)
            return self._audio_array_to_waveform(audio_data), int(sample_rate)
        except Exception as e:
            load_errors.append(f"librosa: {e}")

        raise RuntimeError("Could not load audio file. " + " | ".join(load_errors))

    def _prepare_waveform(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Resample waveform to the model sample rate and ensure it is 1D mono."""
        waveform = waveform.float()

        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
        elif waveform.ndim != 1:
            raise ValueError(f"Unsupported waveform shape: {tuple(waveform.shape)}")

        if sample_rate != self.TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, self.TARGET_SAMPLE_RATE)
            waveform = resampler(waveform)

        return waveform.squeeze()

    def _probabilities_to_emotions(self, probabilities: torch.Tensor) -> Dict[str, float]:
        """Convert model probabilities to an emotion score dictionary."""
        id2label = self.model.config.id2label
        emotions = {}
        for emotion_id, prob in enumerate(probabilities):
            raw_label = id2label[emotion_id]
            emotion_label = self._normalize_label(raw_label)
            emotions[emotion_label] = emotions.get(emotion_label, 0.0) + float(prob)
        return {
            emotion: round(score, 4)
            for emotion, score in sorted(emotions.items())
        }

    def _normalize_label(self, label: str) -> str:
        """Normalize model-specific labels into the project label names."""
        key = str(label).strip().lower().replace(" ", "_")
        return self.LABEL_ALIASES.get(key, key)

    def _predict_from_waveform(self, waveform: torch.Tensor, include_chunks: bool = False) -> Dict:
        """Run model inference on a prepared mono waveform."""
        chunks = self._split_waveform(waveform)
        emotion_sums = None
        chunk_predictions = []

        for idx, chunk in enumerate(chunks):
            probabilities = self._predict_chunk_probabilities(chunk)
            if emotion_sums is None:
                emotion_sums = torch.zeros_like(probabilities)
            emotion_sums += probabilities

            if include_chunks:
                chunk_emotions = self._probabilities_to_emotions(probabilities)
                chunk_primary = max(chunk_emotions, key=chunk_emotions.get)
                chunk_predictions.append({
                    "chunk_index": idx,
                    "start_seconds": round(idx * self.CHUNK_SECONDS, 2),
                    "end_seconds": round(idx * self.CHUNK_SECONDS + chunk.numel() / self.TARGET_SAMPLE_RATE, 2),
                    "primary_emotion": chunk_primary,
                    "confidence": chunk_emotions[chunk_primary],
                    "all_emotions": chunk_emotions,
                })

        averaged = emotion_sums / len(chunks)
        emotions = self._probabilities_to_emotions(averaged)
        primary_emotion = max(emotions, key=emotions.get)

        prediction = {
            "primary_emotion": primary_emotion,
            "confidence": round(emotions[primary_emotion], 4),
            "all_emotions": emotions,
            "chunks_analyzed": len(chunks),
        }

        if include_chunks:
            prediction["chunk_predictions"] = chunk_predictions

        return prediction

    def _split_waveform(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """Split long recordings into model-sized chunks."""
        chunk_size = int(self.TARGET_SAMPLE_RATE * self.CHUNK_SECONDS)
        min_chunk_size = int(self.TARGET_SAMPLE_RATE * self.MIN_CHUNK_SECONDS)

        if waveform.numel() <= chunk_size:
            return [waveform]

        chunks = []
        for start in range(0, waveform.numel(), chunk_size):
            chunk = waveform[start:start + chunk_size]
            if chunk.numel() >= min_chunk_size:
                chunks.append(chunk)

        return chunks or [waveform]

    def _predict_chunk_probabilities(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return model probabilities for a single short waveform chunk."""
        inputs = self.processor(
            waveform.cpu().numpy(),
            sampling_rate=self.TARGET_SAMPLE_RATE,
            return_tensors="pt"
        )

        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        return torch.softmax(logits, dim=-1)[0].detach().cpu()
    
    def predict_emotion(self, audio_input: Union[str, Path], include_chunks: bool = False) -> Dict:
        """
        Predict emotion from audio file.
        
        Args:
            audio_input: Path to audio file (WAV, MP3, etc.)
            
        Returns:
            Dict with structure:
            {
                "success": bool,
                "primary_emotion": str,
                "confidence": float,
                "all_emotions": {
                    "emotion_name": confidence_score,
                    ...
                },
                "audio_path": str
            }
        """
        audio_input = Path(audio_input)
        
        # Validate file exists
        if not audio_input.exists():
            return {
                "success": False,
                "error": f"Audio file not found: {audio_input}",
                "primary_emotion": None,
                "confidence": 0.0
            }

        self._load_model()
        
        try:
            waveform, sample_rate = self._load_audio_file(audio_input)
            waveform = self._prepare_waveform(waveform, sample_rate)
            prediction = self._predict_from_waveform(waveform, include_chunks=include_chunks)
            
            result = {
                "success": True,
                "primary_emotion": prediction["primary_emotion"],
                "confidence": prediction["confidence"],
                "all_emotions": prediction["all_emotions"],
                "chunks_analyzed": prediction["chunks_analyzed"],
                "audio_path": str(audio_input),
                "model_used": self.model_name
            }

            if include_chunks:
                result["chunk_predictions"] = prediction["chunk_predictions"]

            return result
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return {
                "success": False,
                "error": str(e),
                "primary_emotion": None,
                "confidence": 0.0,
                "audio_path": str(audio_input)
            }
    
    def predict_emotion_from_bytes(self, audio_bytes: bytes, sample_rate: int = 16000, include_chunks: bool = False) -> Dict:
        """
        Predict emotion from audio bytes (for microphone input, API uploads, etc.)
        
        Args:
            audio_bytes: Raw audio bytes (WAV format)
            sample_rate: Sample rate of audio (default: 16000 Hz)
            
        Returns:
            Same structure as predict_emotion()
        """
        self._load_model()
        
        try:
            import io
            from scipy.io import wavfile as scipy_wavfile
            
            # Convert bytes to audio array
            if isinstance(audio_bytes, bytes):
                audio_buffer = io.BytesIO(audio_bytes)
                sample_rate_read, audio_data = scipy_wavfile.read(audio_buffer)
                sample_rate = sample_rate_read
            else:
                audio_data = audio_bytes
            
            waveform = self._audio_array_to_waveform(audio_data)
            waveform = self._prepare_waveform(waveform, sample_rate)
            prediction = self._predict_from_waveform(waveform, include_chunks=include_chunks)
            
            result = {
                "success": True,
                "primary_emotion": prediction["primary_emotion"],
                "confidence": prediction["confidence"],
                "all_emotions": prediction["all_emotions"],
                "chunks_analyzed": prediction["chunks_analyzed"],
                "audio_source": "bytes",
                "model_used": self.model_name
            }

            if include_chunks:
                result["chunk_predictions"] = prediction["chunk_predictions"]

            return result
            
        except Exception as e:
            logger.error(f"Error processing audio bytes: {e}")
            return {
                "success": False,
                "error": str(e),
                "primary_emotion": None,
                "confidence": 0.0,
                "audio_source": "bytes"
            }
    
    def predict_emotion_with_prosody(self, audio_input: Union[str, Path]) -> Dict:
        """
        Predict emotion AND extract the prosodic feature vector from one file.

        This is the method the /emotion/voice API endpoint uses. It runs the
        wav2vec2 classifier (the "what emotion") and prosody.extract_prosody
        (the "how it was said") on the *same* prepared waveform, so the two
        analyses are guaranteed to describe identical audio.

        Args:
            audio_input: Path to an audio file (WAV, MP3, etc.).

        Returns:
            Same fields as predict_emotion(), plus:
            {
                "prosody": {
                    "pitch_variation": float, "speech_rate": float,
                    "volume_dynamics": float, "vocal_tension": float,
                    "feature_vector": [...], "raw": {...}
                }
            }
        """
        import prosody

        audio_input = Path(audio_input)
        if not audio_input.exists():
            return {
                "success": False,
                "error": f"Audio file not found: {audio_input}",
                "primary_emotion": None,
                "confidence": 0.0,
            }

        self._load_model()

        try:
            waveform, sample_rate = self._load_audio_file(audio_input)
            waveform = self._prepare_waveform(waveform, sample_rate)
            prediction = self._predict_from_waveform(waveform)

            # Prosody runs on the same prepared (mono, 16 kHz) waveform.
            prosody_features = prosody.extract_prosody(
                waveform.cpu().numpy(),
                sample_rate=self.TARGET_SAMPLE_RATE,
            )

            return {
                "success": True,
                "primary_emotion": prediction["primary_emotion"],
                "confidence": prediction["confidence"],
                "all_emotions": prediction["all_emotions"],
                "chunks_analyzed": prediction["chunks_analyzed"],
                "prosody": prosody_features,
                "audio_path": str(audio_input),
                "model_used": self.model_name,
            }
        except Exception as e:
            logger.error(f"Error processing audio with prosody: {e}")
            return {
                "success": False,
                "error": str(e),
                "primary_emotion": None,
                "confidence": 0.0,
                "audio_path": str(audio_input),
            }

    def analyze_emotion_sequence(self, audio_files: List[str]) -> Dict:
        """
        Analyze emotion across multiple audio samples to track emotional shifts.
        Useful for debate systems to see how emotion changes across arguments.
        
        Args:
            audio_files: List of audio file paths (in order)
            
        Returns:
            {
                "success": bool,
                "sequence": [
                    {"timestamp": 0, "emotion": "neutral", "confidence": 0.8, ...},
                    {"timestamp": 1, "emotion": "angry", "confidence": 0.9, ...},
                    ...
                ],
                "emotion_trajectory": {
                    "angry": [0.1, 0.2, 0.8, 0.9],
                    ...
                },
                "dominant_emotion": str,
                "emotion_variance": float
            }
        """
        sequence = []
        emotion_trajectories = {}
        
        for idx, audio_file in enumerate(audio_files):
            result = self.predict_emotion(audio_file)
            
            if result['success']:
                sequence.append({
                    "timestamp": idx,
                    "audio_file": str(audio_file),
                    "emotion": result['primary_emotion'],
                    "confidence": result['confidence'],
                    "all_emotions": result['all_emotions']
                })
                
                # Track emotion trajectories
                for emotion, score in result['all_emotions'].items():
                    if emotion not in emotion_trajectories:
                        emotion_trajectories[emotion] = []
                    emotion_trajectories[emotion].append(score)
            else:
                logger.warning(f"Failed to process {audio_file}")
        
        # Calculate emotion variance (how much emotions fluctuate)
        if sequence:
            primary_emotions = [s['emotion'] for s in sequence]
            emotion_variance = len(set(primary_emotions)) / len(primary_emotions)
        else:
            emotion_variance = 0.0
        
        dominant = max(emotion_trajectories, key=lambda e: sum(emotion_trajectories[e])) if emotion_trajectories else None
        
        return {
            "success": len(sequence) > 0,
            "sequence": sequence,
            "emotion_trajectory": emotion_trajectories,
            "dominant_emotion": dominant,
            "emotion_variance": round(emotion_variance, 4),
            "total_samples": len(sequence)
        }


# ============================================================================
# Example usage and testing
# ============================================================================

if __name__ == "__main__":
    # Initialize recognizer
    recognizer = SpeechEmotionRecognizer(use_gpu=True)
    
    # Example 1: Test with a file
    print("\n" + "="*60)
    print("SPEECH EMOTION RECOGNITION - Example Usage")
    print("="*60)
    
    # Note: You'll need an actual audio file to test
    test_audio = "test_audio.wav"  # Replace with actual audio file path
    
    # Uncomment this to test if you have an audio file:
    # result = recognizer.predict_emotion(test_audio)
    # print(json.dumps(result, indent=2))
