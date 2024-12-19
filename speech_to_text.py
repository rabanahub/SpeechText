import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load Wav2Vec2 model and processor
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# Load WAV file
wav_file_path = "C:\\felixPythan\\Rohit Sir\\speechText\\test1.wav"
speech, original_sampling_rate = torchaudio.load(wav_file_path)

# Resample the input audio to match the model's sampling rate (16000 Hz)
resampler = torchaudio.transforms.Resample(orig_freq=original_sampling_rate, new_freq=16000)
speech_resampled = resampler(speech)

# Preprocess the resampled input
inputs = processor(speech_resampled, sampling_rate=16000, return_tensors="pt", padding=True)

# Get input values and reshape to match the expected shape [num_channels, num_samples]
input_values = inputs.input_values.squeeze(0)  # Remove the first dimension (batch size of 1)

# Pass the input tensor through the model
with torch.no_grad():
    logits = model(input_values).logits

# Decode the output to obtain the transcription
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)[0]

# Print the transcription
print("Transcription:", transcription)