import os
import json
import wave
import tempfile
import zipfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from vosk import Model, KaldiRecognizer, SetLogLevel
from pydub import AudioSegment
import numpy as np

# Suppress Vosk logs for cleaner output
SetLogLevel(-1)
os.environ['PATH'] = '/opt/homebrew/bin:' + os.environ.get('PATH', '')

def convert_to_wav(audio_path, target_path=None):
    """Convert any audio file to optimized WAV format for speech recognition."""
    try:
        if not target_path:
            fd, target_path = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
        
        # Single pass conversion with all effects
        sound = (
            AudioSegment.from_file(audio_path)
            .set_channels(1)
            .set_frame_rate(16000)
            .normalize()
            .low_pass_filter(3000)
            .high_pass_filter(300)
        )
        
        sound.export(
            target_path,
            format='wav',
            parameters=['-ac', '1', '-ar', '16000', '-sample_fmt', 's16']
        )
        return target_path
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None

def transcribe_audio(audio_path, model):
    """Transcribe audio file using Vosk with optimized processing."""
    temp_wav = None
    try:
        if not (temp_wav := convert_to_wav(audio_path)) or not os.path.exists(temp_wav):
            print(f"Failed to convert {os.path.basename(audio_path)} to WAV format")
            return None
        
        rec = KaldiRecognizer(model, 16000)
        rec.SetWords(True)
        results = []
        
        with wave.open(temp_wav, 'rb') as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                print(f"Unsupported audio format: {os.path.basename(audio_path)}")
                return None
            
            while chunk := wf.readframes(8000):
                if rec.AcceptWaveform(chunk):
                    if text := json.loads(rec.Result()).get('text', '').strip():
                        results.append(text)
            
            if final_text := json.loads(rec.FinalResult()).get('text', '').strip():
                results.append(final_text)
        
        return ' '.join(results).strip()
        
    except Exception as e:
        print(f"Error transcribing {os.path.basename(audio_path)}: {str(e)}")
        return None
        
    finally:
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except OSError:
                pass

def process_audio_file(args):
    """Process a single audio file and return its transcription."""
    idx, audio_file, audio_dir, model = args
    audio_path = os.path.join(audio_dir, audio_file)
    print(f"\n{'='*50}\nProcessing {idx[0]+1}/{idx[1]}: {audio_file}")
    
    for attempt in range(3):  # 3 retry attempts
        if transcription := transcribe_audio(audio_path, model):
            # Create new filename with prefix and first 50 chars of transcription
            safe_name = "".join(c if c.isalnum() or c in ' -_.' else '_' for c in transcription[:50])
            new_name = f"{idx[0]+1:02d} - {safe_name}{os.path.splitext(audio_file)[1]}"
            return idx[0], audio_file, new_name, transcription
        print(f"Attempt {attempt + 1} failed, retrying...")
    
    return idx[0], audio_file, None, None

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(base_dir, 'audio')
    model_path = os.path.join(base_dir, 'vosk-model-en-us-0.22')
    os.makedirs(audio_dir, exist_ok=True)
    
    # Load model
    try:
        model = Model(model_path)
    except Exception as e:
        print(f"Failed to load Vosk model: {str(e)}")
        return
    
    # Get audio files
    audio_ext = ('.m4a', '.mp3', '.wav', '.ogg', '.flac')
    audio_files = [
        f for f in os.listdir(audio_dir)
        if os.path.isfile(os.path.join(audio_dir, f)) and f.lower().endswith(audio_ext)
    ]
    
    if not audio_files:
        print(f"No audio files found in: {audio_dir}")
        print(f"Supported formats: {', '.join(audio_ext)}")
        return
    
    # Create transcripts directory with timestamp
    from datetime import datetime
    transcripts_dir = os.path.join(base_dir, 'transcripts')
    os.makedirs(transcripts_dir, exist_ok=True)
    
    # Create transcript filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    transcript_file = os.path.join(transcripts_dir, f'transcript_{timestamp}.txt')
    
    # Process files in parallel
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_audio_file,
                (i, f, audio_dir, model)
            )
            for i, f in enumerate(audio_files)
        ]
        
        for future in as_completed(futures):
            idx, orig_name, new_name, transcription = future.result()
            if transcription:
                results.append((idx, f"{idx+1:02d}. {transcription}"))
                if new_name:
                    try:
                        os.rename(
                            os.path.join(audio_dir, orig_name),
                            os.path.join(audio_dir, new_name)
                        )
                        print(f"✅ Renamed: {new_name}")
                    except Exception as e:
                        print(f"⚠️  Failed to rename {orig_name}: {str(e)}")
    
    # Write results to transcript file
    if results:
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write("this is a \"Horal History Project\" from the class \"SSN103\" class\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for _, line in sorted(results, key=lambda x: x[0]):
                f.write(f"{line}\n\n")
        print(f"\n✅ Saved {len(results)} transcriptions to {transcript_file}")
    else:
        print("\n❌ No transcriptions were generated.")

if __name__ == "__main__":
    main()
