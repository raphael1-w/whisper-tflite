import tensorflow as tf
import librosa
import numpy as np
from tokenizer_v2 import SimpleWhisperTokenizer
from difflib import SequenceMatcher

# Audio processing parameters
SAMPLE_RATE = 16000
N_FFT = 512
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = SAMPLE_RATE * CHUNK_LENGTH
CHUNK_OVERLAP = 6  # seconds overlap between chunks
N_SAMPLES_OVERLAP = SAMPLE_RATE * CHUNK_OVERLAP

# Define the path to the TFLite model
tflite_model_path = r"C:\Users\Raphael\FYP tensorflow lite\whisper-tiny.tflite"

# Create an interpreter to run the TFLite model
interpreter = tf.lite.Interpreter(tflite_model_path)
interpreter.allocate_tensors()

# Get the input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input Details:", input_details)
print("Output Details:", output_details)

def process_audio_chunk(interpreter, audio_chunk):

    # Calculate mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio_chunk,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    
    print("Audio Chunk length:", len(audio_chunk))
    print("Mel Spectrogram Shape:", mel_spec.shape)
    print("mel spec:", mel_spec[0])

    target_length = 3000
    # Pad or trim the mel spectrogram
    if mel_spec.shape[1] < target_length:
        pad_width = ((0, 0), (0, target_length - mel_spec.shape[1]))
        mel_spec = np.pad(mel_spec, pad_width)
    elif mel_spec.shape[1] > target_length:
        mel_spec = mel_spec[:, :target_length]

    # Convert to log mel spectrogram and normalize
    log_mel = np.log10(np.clip(mel_spec, a_min=1e-10, a_max=None))
    log_mel = (log_mel + 4.0) / 4.0
    
    print("Log Mel: ", log_mel[0])

    # Prepare input data
    input_data = np.expand_dims(log_mel, 0).astype(np.float32)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Process tokens
    tokens = output_data[0]
    print("Tokens:", tokens)
    valid_tokens = [int(token) for token in tokens if 0 <= int(token) < 50258]
    
    if valid_tokens:
        tokenizer = SimpleWhisperTokenizer(r"C:\Users\Raphael\FYP tensorflow lite\whisper-base")
        text = tokenizer.decode(valid_tokens, skip_special_tokens=True)
        return text.strip()
    return ""

def find_overlap(text1, text2):
    words1 = text1.split()
    words2 = text2.split()
    
    # Look for overlapping sequences of words
    for i in range(min(len(words1), len(words2)), 0, -1):
        if words1[-i:] == words2[:i]:
            return " ".join(words2[i:])
    return text2

def find_overlap_fuzzy(text1, text2, threshold=0.85):
    words1 = text1.split()
    words2 = text2.split()
    
    # Try different window sizes for matching
    max_window = min(len(words1), len(words2))
    best_ratio = 0
    best_overlap_len = 0
    
    # Look for overlapping sequences of words
    for window in range(max_window, 3, -1):  # minimum 4 words overlap
        # Compare end of text1 with start of text2
        text1_end = " ".join(words1[-window:])
        text2_start = " ".join(words2[:window])
        
        # Calculate similarity ratio
        ratio = SequenceMatcher(None, text1_end.lower(), text2_start.lower()).ratio()
        
        if ratio > threshold and ratio > best_ratio:
            best_ratio = ratio
            best_overlap_len = window
    
    if best_overlap_len > 0:
        # Return text2 without the overlapping part
        return " ".join(words2[best_overlap_len:])
    return text2

def transcribe_audio(audio_path):
    print(f'Loading audio file: {audio_path}')
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    
    # Calculate number of chunks and overlap
    audio_length = len(audio)
    chunk_samples = N_SAMPLES
    overlap_samples = N_SAMPLES_OVERLAP
    
    chunks = []
    start = 0
    while start < audio_length:
        end = min(start + chunk_samples, audio_length)
        chunk = audio[start:end]
        
        # Only add chunk if it's substantial (at least 25% of max length)
        # if len(chunk) > N_SAMPLES * 0.25:
        #     chunks.append(chunk)
        
        chunks.append(chunk)
            
        start = start + chunk_samples - overlap_samples
    
    print(f"Processing {len(chunks)} chunks...")
    
    # Process each chunk
    transcriptions = []
    previous_transcript = ""
    
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")
        transcript = process_audio_chunk(interpreter, chunk)
        print(f"Raw transcript chunk {i+1}: ", transcript)
        
        if transcript:
            # Handle overlapping text with fuzzy matching
            if previous_transcript:
                cleaned_transcript = find_overlap_fuzzy(previous_transcript, transcript)
                if cleaned_transcript:  # Only add if there's text after overlap removal
                    transcriptions.append(cleaned_transcript)
            else:
                transcriptions.append(transcript)
            previous_transcript = transcript
    
    # Combine transcriptions
    final_transcript = " ".join(transcriptions)
    return final_transcript

# Replace your main processing code with:
final_transcript = transcribe_audio(r"C:\Users\Raphael\FYP tensorflow lite\Test_audio_GPT.m4a")

# Print the final transcription
print("\n" + "Final Transcription: " + final_transcript + "\n")

# Save the result
with open('transcription.txt', 'w', encoding='utf-8') as f:
    f.write(final_transcript)
    print("Transcription has been saved to transcription.txt")