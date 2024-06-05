import os
import json
import jiwer
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline, WhisperTokenizer, WhisperFeatureExtractor
import torch
from tqdm import tqdm
import io

class ASRManager:
    def __init__(self):
        # initialize the model here
        # pass
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        model_path = "src/wsp-trained"
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="en", task="transcribe")
        
        self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            chunk_length_s=30,
            device=self.device,
            tokenizer=self.tokenizer,
            feature_extractor=self.feature_extractor,
        )
        
        
    @staticmethod
    def number_to_words(number):
        number_mapping = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
        }
        return ' '.join([number_mapping[char] for char in number])

    def modify_transcript(self, transcript: str) -> str:
        modified_transcript = ""
        i = 0
        while i < len(transcript):
            if transcript[i].isdigit():
                number = ""
                while i < len(transcript) and transcript[i].isdigit():
                    number += transcript[i]
                    i += 1
                modified_transcript += self.number_to_words(number)
            else:
                modified_transcript += transcript[i]
                i += 1
        return modified_transcript
        
    def transcribe(self, audio_bytes: bytes) -> str:        
        # perform ASR transcription
        audio_file = io.BytesIO(audio_bytes)
        waveform, sample_rate = torchaudio.load(audio_file)
        audio_data = {"array": waveform.squeeze().numpy(), "sampling_rate": sample_rate}
        
        result = self.pipe(audio_data["array"], batch_size=8)
        generated_transcript = result["text"]
        
        modified_transcript = self.modify_transcript(generated_transcript)
        return modified_transcript
