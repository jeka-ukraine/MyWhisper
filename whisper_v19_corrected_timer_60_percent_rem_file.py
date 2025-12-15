import os
import subprocess
import warnings
import sys
import select
# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# FFmpeg setup
import platform
if platform.system() == "Windows":
    ffmpeg_path = r"C:\ffmpeg\bin"
    os.environ["PATH"] += os.pathsep + ffmpeg_path

import whisper
try:
    from pywhispercpp.model import Model as CppModel
except ImportError:
    print("Ошибка: pywhispercpp не установлен.")
    CppModel = None

import ffmpeg
from pydub import AudioSegment
import glob
import time
from tqdm import tqdm
from datetime import datetime, timedelta
import argparse
import threading
import shutil

try:
    subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
except FileNotFoundError:
    print("FFmpeg не найден! Убедись, что путь указан верно.")

# --- Helper Functions ---

def print_step(message, progress=None):
    if not message.endswith('.'):
        message += '.'
    if progress is not None:
        print(f"{message} - {progress}%")
    else:
        print(message)

def move_to_remove_folder(audio_file):
    try:
        # Use script directory as base for files_to_remove
        script_dir = os.path.dirname(os.path.abspath(__file__))
        remove_folder = os.path.join(script_dir, "files_to_remove")
        
        if not os.path.exists(remove_folder):
            os.makedirs(remove_folder)
            print_step(f"Создана папка: {remove_folder}")
        filename = os.path.basename(audio_file)
        destination = os.path.join(remove_folder, filename)
        shutil.move(audio_file, destination)
        print_step(f"Файл {filename} перемещен в 'files_to_remove'")
        return True
    except Exception as e:
        print_step(f"Ошибка при перемещении файла {audio_file}: {e}")
        return False

def estimate_transcription_time(audio_length, model_name, engine='py'):
    # Факторы скорости (Audio Duration / Processing Time)
    # Скорректировано на основе реальных тестов пользователя (M1 Metal)
    
    speed_factor = 0.4 # Базовый fallback для тяжелых моделей
    
    if 'large-v3-turbo' in model_name:
        if engine == 'cpp':
            # Среднее по 5 файлам: ~5.9x от реального времени
            speed_factor = 5.9
        else:
            # Python версия large-v3-turbo (предположительно медленнее CPP)
            speed_factor = 4.0
    elif 'large' in model_name:
        speed_factor = 0.6 
        if engine == 'cpp':
            speed_factor *= 1.5

    estimated_seconds = audio_length / speed_factor
    return estimated_seconds

def get_audio_duration_seconds(audio_file):
    try:
        # Try quick probe with ffmpeg first to avoid full decode
        probe = ffmpeg.probe(audio_file)
        return float(probe['format']['duration'])
    except:
        # Fallback to loading via pydub (slower)
        try:
            audio = AudioSegment.from_file(audio_file)
            return len(audio) / 1000.0
        except:
            return 0

def convert_audio_if_needed(audio_file):
    file_format = audio_file.split('.')[-1].lower()
    temp_wav = None
    
    # Direct support check (whisper ffmpeg usually handles these, but let's stick to safe wav for consistency if needed)
    # Actually, both engines handle mp3/m4a/wav fine via ffmpeg backend.
    # We only convert if it's a video file or problematic format.
    
    if file_format in ['mp4', 'avi', 'mov', 'webm', 'mkv']:
        print_step(f"Извлечение аудио из видео {file_format}")
        audio_output = f"/tmp/temp_audio_{os.getpid()}_{int(time.time())}.wav"
        try:
            ffmpeg.input(audio_file).output(audio_output, acodec='pcm_s16le', ac=1, ar='16000').run(quiet=True, overwrite_output=True)
            return audio_output, True # True means it's a temp file
        except Exception as e:
            print_step(f"Error converting video: {e}")
            return None, False
    
    return audio_file, False

# --- Core Transcription Logic ---

def transcribe_with_cpp(model_instance, audio_path, language):
    # CppModel logic
    # pywhispercpp transcribe returns segments
    print_step(f"Запуск транскрипции (C++)...")
    segments = model_instance.transcribe(audio_path, language=language if language else 'auto')
    # segments is a list of Segment objects usually, or text. 
    # Library specific: pywhispercpp usually returns segments list.
    full_text = ""
    for seg in segments:
        full_text += seg.text
    return full_text.strip()

def transcribe_with_python(model_instance, audio_path, language):
    print_step(f"Запуск транскрипции (Python)...")
    if language:
        result = model_instance.transcribe(audio_path, language=language)
    else:
        result = model_instance.transcribe(audio_path)
    return result['text'].strip()

def run_transcription_pass(audio_file, model_config, language, strict_mode):
    # model_config: {'engine': 'cpp'|'py', 'model': obj, 'name': str, 'suffix': str}
    engine = model_config['engine']
    model_obj = model_config['model']
    model_name = model_config['name']
    suffix = model_config.get('suffix', '')

    print_step(f"--- [Обработка: {model_config['desc']}] ---")

    # Audio prep
    active_audio_path, is_temp = convert_audio_if_needed(audio_file)
    if not active_audio_path:
        return False

    try:
        duration = get_audio_duration_seconds(active_audio_path)
        est = estimate_transcription_time(duration, model_name, engine)
        finish_time = datetime.now() + timedelta(seconds=est)
        print_step(f"Аудио: {duration/60:.2f} мин. Оценка: {est/60:.2f} мин (до {finish_time.strftime('%H:%M:%S')})")

        start_time = time.time()

        # Timer setup
        stop_timer = threading.Event()
        def timer_func():
            p_len = 30
            while not stop_timer.is_set():
                elapsed = time.time() - start_time
                prog = min(elapsed / est, 0.99) if est > 0 else 0
                filled = int(p_len * prog)
                bar = '█' * filled + '-' * (p_len - filled)
                print(f"\rПроцесс: {int(elapsed)} сек [{bar}]", end="", flush=True)
                time.sleep(1)
        
        t = threading.Thread(target=timer_func)
        t.start()
        
        text_result = ""
        try:
            if engine == 'cpp':
                text_result = transcribe_with_cpp(model_obj, active_audio_path, language)
            else:
                text_result = transcribe_with_python(model_obj, active_audio_path, language)
        except Exception as e:
            print(f"\nОшибка транскрипции: {e}")
            return False
        finally:
            stop_timer.set()
            t.join()
            print() # newline

        actual_dur = (time.time() - start_time) / 60
        print_step(f"Готово за {actual_dur:.2f} мин.")

        # Save
        if not text_result:
            print_step("Пустой результат транскрипции!")
            return False

        output_dir = os.path.abspath("files_done")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        base_name = os.path.basename(os.path.splitext(audio_file)[0])
        # Add suffix if provided
        final_file_name = f"{base_name}{suffix}_transcription.txt"
        out_path = os.path.join(output_dir, final_file_name)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text_result)
        
        print_step(f"Сохранено: {final_file_name}")
        print(f"\n--- Preview ({len(text_result)} chars total, showing first 1000) ---")
        print(text_result[:1000])
        print("--------------------------------------------------------------\n")
        return True

    finally:
        return audio_file, False

# --- WhisperX Logic (New) ---

def run_whisperx_pipeline(audio_file, hf_token=None, language=None):
    start_time_wx = time.time()
    print_step(f"Запуск WhisperX для {os.path.basename(audio_file)}...")
    try:
        import whisperx
        import gc
        import torch
        
        # --- FIX FOR PYTORCH 2.6+ ---
        try:
            from omegaconf import OmegaConf, DictConfig, ListConfig
            torch.serialization.add_safe_globals([DictConfig, ListConfig])
        except:
            pass

        if hasattr(torch, 'load'):
            _orig_load = torch.load
            def _unsafe_load(*args, **kwargs):
                # Force weights_only=False to support older checkpoints
                kwargs['weights_only'] = False 
                return _orig_load(*args, **kwargs)
            torch.load = _unsafe_load
        # ----------------------------

    except ImportError:
        print("\nОШИБКА: Библиотека whisperx не найдена.")
        print("Для использования опции 3 установите whisperx:")
        print("pip install git+https://github.com/m-bain/whisperx.git")
        print("Также убедитесь, что установлен ffmpeg.")
        return None

    # Настройки для Mac M1
    # WhisperX использует CTranslate2. На M1 лучше всего работает CPU + float32 или int8. 
    # MPS (Metal) поддержка в CTranslate2 экспериментальная.
    device = "cpu"
    compute_type = "float32" 

    print_step(f"Параметры: Device={device}, Type={compute_type}")
    
    try:
        # 1. Transcribe & VAD
        # batch_size reduced for M1 stability
        print_step("1. Загрузка модели и Транскрипция (с VAD)...")
        # Использование large-v2 как стандарта для whisperx
        model = whisperx.load_model("large-v2", device, compute_type=compute_type, language=language)
        
        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=4)
        
        # Очистка памяти модели транскрипции
        del model
        gc.collect()

        # 2. Alignment (Выравнивание)
        print_step("2. Выравнивание таймкодов (Alignment)...")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        
        del model_a
        gc.collect()

        # 3. Diarization (Спикеры)
        if hf_token:
            print_step("3. Диаризация (Определение спикеров)...")
            try:
                diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
                del diarize_model
                gc.collect()
            except Exception as e_dia:
                print(f"Ошибка диаризации (пропускаем): {e_dia}")
        else:
            print_step("Пропуск диаризации (нет HF токена).")

        # Форматирование результата
        print_step("Форматирование текста...")
        output_lines = []
        for segment in result["segments"]:
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            
            # Convert to HH:MM:SS
            start_str = str(timedelta(seconds=int(start)))
            end_str = str(timedelta(seconds=int(end)))
            
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment["text"].strip()
            
            # Format: [00:00:10 -> 00:00:15] [SPEAKER_01]: Text
            line = f"[{start_str} -> {end_str}] [{speaker}]: {text}"
            output_lines.append(line)
        
        # Stats
        total_time = time.time() - start_time_wx
        duration = get_audio_duration_seconds(audio_file)
        print_step(f"WhisperX завершен. Длительность аудио: {duration:.2f} сек ({duration/60:.2f} мин). Время конвертации: {total_time:.2f} сек ({total_time/60:.2f} мин).")

        return "\n".join(output_lines)

    except Exception as e:
        print(f"\nCRITICAL ERROR in WhisperX: {e}")
        import traceback
        traceback.print_exc()
        return None


# --- Startup Logic ---

def get_user_choice():
    print("\n" + "="*50)
    print("ВЫБОР РЕЖИМА РАБОТЫ:")
    print(" [1] Стандарт: Whisper CPP + Large-v3-Turbo (Быстро, Оптимально)")
    print(" [2] Тест-сравнение: Cpp-Turbo vs Py-Turbo vs Py-V2 (Медленно)")
    print(" [3] WhisperX: Диаризация, VAD, Точные тайминги (Требуется whisperx + HF Token)")
    print("="*50)
    
    timeout = 10
    print(f"Автоматический выбор [1] через {timeout} секунд...")

    start = time.time()
    while True:
        remaining = int(timeout - (time.time() - start))
        if remaining <= 0:
            print("\nВремя вышло. Выбран режим [1].")
            return "1"
        
        sys.stdout.write(f"\rВведите выбор [1/2/3] (осталось {remaining}с): ")
        sys.stdout.flush()
        
        # Check input
        ready, _, _ = select.select([sys.stdin], [], [], 1)
        if ready:
            inp = sys.stdin.readline().strip()
            if inp == '' or inp == '1':
                return '1'
            elif inp in ['2', '3']:
                return inp
            else:
                print("\nНеверный ввод. Введите 1, 2 или 3.")
                start = time.time() # Reset timer on interaction if desired, or let it flow. Let's just continue.

def get_language_priority():
    print("\n" + "="*50)
    print("Какой язык использовать приоритетным для транскрибации?")
    print(" [1] Автоопределение (Default)")
    print(" [2] Украинский")
    print(" [3] Русский")
    print(" [4] Английский")
    print("="*50)
    
    timeout = 7
    print(f"Автоматический выбор [1] через {timeout} секунд...")

    start = time.time()
    while True:
        remaining = int(timeout - (time.time() - start))
        if remaining <= 0:
            print("\nВремя вышло. Выбран язык [1] (Автоопределение).")
            return None
        
        sys.stdout.write(f"\rВведите выбор [1-4] (осталось {remaining}с): ")
        sys.stdout.flush()
        
        ready, _, _ = select.select([sys.stdin], [], [], 1)
        if ready:
            inp = sys.stdin.readline().strip()
            if inp == '' or inp == '1':
                return None
            elif inp == '2':
                return 'uk'
            elif inp == '3':
                return 'ru'
            elif inp == '4':
                return 'en'

def main():
    choice = get_user_choice()
    
    # Args parsing for language only
    parser = argparse.ArgumentParser()
    parser.add_argument("--language_pr", type=str, default=None)
    parser.add_argument("--language_only", type=str, default=None)
    args, unknown = parser.parse_known_args()

    # Language setup
    lang_map = {"-ukr": "uk", "-rus": "ru", "-eng": "en", "-pln": "pl"}
    language = None
    strict = False
    if args.language_pr:
        language = lang_map.get(args.language_pr)
    elif args.language_only:
        language = lang_map.get(args.language_only)
        strict = True

    if choice == "1":
        language = get_language_priority()
    
    # Find files
    input_dir = "MyWhisper_to_process"
    if not os.path.exists(input_dir):
        # Optional: create if not exists or just warn. 
        # User implies script takes from folder. Let's ensure it exists to avoid crash or just search empty.
        # But if we create it, glob will be empty anyway.
        pass

    exts = ['*.mp3', '*.MP3', '*.m4a', '*.M4A', '*.wav', '*.WAV', '*.ogg', '*.OGG', '*.opus', '*.OPUS', 
            '*.mp4', '*.MP4', '*.mov', '*.MOV', '*.avi', '*.AVI', '*.webm', '*.WEBM']
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
    files = sorted(list(set(files))) # Unique sorted
    
    if not files:
        print("Файлы не найдены.")
        return

    print(f"\nНайдено файлов: {len(files)}")
    for f in files: print(f" - {f}")

    # Prepare Configurations
    configs = []
    
    # Load Models based on choice
    try:
        if choice == "1":
            print_step("\nИнициализация Whisper C++ (large-v3-turbo)...")
            # Usually we pass model name. 'large-v3-turbo' might need manual download if library is old.
            # trying standard approach first.
            cpp_model = CppModel('large-v3-turbo', n_threads=6, print_realtime=False, print_progress=False)
            configs.append({'engine': 'cpp', 'model': cpp_model, 'name': 'large-v3-turbo', 'desc': 'CPP-Turbo (Standard)', 'suffix': ''})
            
        elif choice == "2":
            # 2.1 Cpp Turbo
            print_step("\nЗагрузка моделей для ТЕСТА...")
            print_step("1/3 Loading CPP large-v3-turbo...")
            cpp_model = CppModel('large-v3-turbo', n_threads=6, print_realtime=False, print_progress=False)
            configs.append({'engine': 'cpp', 'model': cpp_model, 'name': 'large-v3-turbo', 'desc': '2.1 CPP-Turbo', 'suffix': '_cpp_turbo'})
            
            # 2.2 Py Turbo
            print_step("2/3 Loading Python large-v3-turbo...")
            # Note: openai-whisper might call it "large-v3-turbo" or require a specific string if logic differs. 
            # Assuming "large-v3-turbo" works with updated package.
            py_turbo = whisper.load_model("large-v3-turbo") 
            configs.append({'engine': 'py', 'model': py_turbo, 'name': 'large-v3-turbo', 'desc': '2.2 PY-Turbo', 'suffix': '_py_turbo'})
            
            configs.append({'engine': 'py', 'model': py_v2, 'name': 'large-v2', 'desc': '2.3 PY-V2', 'suffix': '_py_v2'})

        elif choice == "3":
             print_step("\nРежим WhisperX выбран.")
             # WhisperX models are loaded inside the function process due to complexity
             configs.append({'engine': 'wx', 'model': None, 'name': 'whisperx', 'desc': 'WhisperX Pipeline', 'suffix': '_whisperx'})
             
             # Load token from .env
             env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
             hf_token = None
             if os.path.exists(env_path):
                 with open(env_path, "r") as f:
                     for line in f:
                         if line.startswith("HF_TOKEN="):
                             hf_token = line.split("=", 1)[1].strip()
                             break
             
             if hf_token:
                 print_step("HF Token найден в .env")
             else:
                 print("\nHF Token не найден в .env. Диаризация будет пропущена.")
                 hf_token = None

    except Exception as e:
        print(f"\nCRITICAL: Ошибка при загрузке моделей: {e}")
        return

    # Process Loop
    print("\n" + "="*50)
    print("НАЧАЛО ОБРАБОТКИ")
    print("="*50)

    for audio_file in files:
        print(f"\n=== Файл: {audio_file} ===")
        file_success = True
        
        for config in configs:
            if config['engine'] == 'wx':
                # WhisperX Path
                wx_text = run_whisperx_pipeline(audio_file, hf_token=hf_token if 'hf_token' in locals() else None, language=language)
                if wx_text:
                     output_dir = os.path.abspath("files_done")
                     if not os.path.exists(output_dir):
                         os.makedirs(output_dir)
                     base = os.path.basename(os.path.splitext(audio_file)[0])
                     out_path = os.path.join(output_dir, f"{base}_whisperx.txt")
                     with open(out_path, "w", encoding="utf-8") as f:
                         f.write(wx_text)
                     print_step(f"Сохранено: {out_path}")
                     print(f"\n--- Preview ({len(wx_text)} chars total, showing first 1000) ---")
                     print(wx_text[:1000])
                     print("--------------------------------------------------------------\n")
                     # Move only if success
                     if choice == "3":
                         move_to_remove_folder(audio_file)
            else:
                ok = run_transcription_pass(audio_file, config, language, strict)
                if not ok:
                    file_success = False
        
        if choice == "1" and file_success:
             move_to_remove_folder(audio_file)
        elif choice == "2" and file_success:
             move_to_remove_folder(audio_file)

    print("\nВсе задачи выполнены.")

if __name__ == "__main__":
    main()
