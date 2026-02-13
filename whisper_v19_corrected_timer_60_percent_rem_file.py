import os
import subprocess
import warnings
import sys
import select
try:
    import questionary
except ImportError:
    questionary = None

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.panel import Panel
    from rich.live import Live
    from rich.spinner import Spinner
    console = Console()
except ImportError:
    console = None
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
try:
    import mlx_whisper
except ImportError:
    mlx_whisper = None
from tqdm import tqdm
from datetime import datetime, timedelta
import argparse
import threading
import shutil

try:
    subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
except FileNotFoundError:
    print("FFmpeg не найден! Убедись, что путь указан верно.")

def print_step(message, style="bold blue"):
    if console:
        if not message.endswith('.'): message += '.'
        console.print(f"[{style}]▶ {message}[/ {style}]")
    else:
        if not message.endswith('.'): message += '.'
        print(message)

def is_apple_silicon():
    import platform
    return platform.system() == "Darwin" and platform.machine() == "arm64"

def get_key():
    import sys
    import select
    import platform
    if platform.system() != "Darwin":
        return sys.stdin.read(1)
    
    import termios
    import tty
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
        if ch == '\x1b':
            dr, _, _ = select.select([sys.stdin], [], [], 0.05)
            if dr:
                ch += sys.stdin.read(2)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def get_timed_choice(title, options, default_index, timeout=10):
    import time
    import sys
    import select
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TextColumn
    from rich.console import Group
    
    if not console:
        start = time.time()
        while time.time() - start < timeout:
            print(f"\r{title} [1-{len(options)}] (Автовыбор в {int(timeout - (time.time() - start))}с): ", end="", flush=True)
            r, _, _ = select.select([sys.stdin], [], [], 1)
            if r:
                res = sys.stdin.readline().strip()
                if res.isdigit() and 1 <= int(res) <= len(options):
                    return options[int(res)-1][1]
        return options[default_index][1]

    selected_index = default_index
    start_time = time.time()
    
    with Live(auto_refresh=True, console=console, transient=True) as live:
        while True:
            elapsed = time.time() - start_time
            remaining = max(0, timeout - elapsed)
            
            if remaining <= 0:
                break
            
            table = Table.grid(padding=(0, 1))
            for i, (label, value) in enumerate(options):
                prefix = "» " if i == selected_index else "  "
                style = "bold cyan" if i == selected_index else "white"
                
                if i == default_index:
                    display_label = f"{label} [dim yellow](ПО УМОЛЧАНИЮ)[/]"
                else:
                    display_label = label
                table.add_row(f"[{style}]{prefix}{display_label}[/]")
            
            progress = Progress(
                TextColumn("[bold yellow]Автовыбор через: {task.fields[remaining]:.1f}с"),
                BarColumn(bar_width=40, pulse_style="yellow"),
                console=console
            )
            progress.add_task("", total=timeout, completed=elapsed, remaining=remaining)
            
            panel = Panel(
                Group(table, "", progress),
                title=f"[bold blue]? {title}",
                subtitle="[dim]Стрелки для выбора, Enter - подтвердить, Цифры - быстрый выбор[/]",
                border_style="blue",
                padding=(1, 2)
            )
            live.update(panel)
            
            dr, _, _ = select.select([sys.stdin], [], [], 0.1)
            if dr:
                key = get_key()
                if key in ['\r', '\n']:
                    return options[selected_index][1]
                elif key == '\x1b[A': # Up
                    selected_index = (selected_index - 1) % len(options)
                elif key == '\x1b[B': # Down
                    selected_index = (selected_index + 1) % len(options)
                elif key.isdigit():
                    idx = int(key) - 1
                    if 0 <= idx < len(options):
                        return options[idx][1]
        return options[selected_index][1]

def print_preview(text, limit=1000):
    if not text: return
    preview = text[:limit]
    if len(text) > limit:
        preview += "\n..."
    if console:
        console.print(Panel(preview, title="[bold yellow]Предпросмотр (первые 1000 симв.)[/]", border_style="green", padding=(1, 2)))
    else:
        print("\n" + "="*20 + " ПРЕДПРОСМОТР " + "="*20)
        print(preview)
        print("="*54 + "\n")

def get_unique_path(path):
    """Добавляет _01, _02 и т.д., если файл уже существует."""
    if not os.path.exists(path):
        return path
    
    base, ext = os.path.splitext(path)
    counter = 1
    while True:
        new_path = f"{base}_{counter:02d}{ext}"
        if not os.path.exists(new_path):
            return new_path
        counter += 1

def move_to_remove_folder(audio_file):
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        remove_folder = os.path.join(script_dir, "files_to_remove")
        
        if not os.path.exists(remove_folder):
            os.makedirs(remove_folder)
            print_step(f"Создана папка: {remove_folder}")
        
        filename = os.path.basename(audio_file)
        destination = os.path.join(remove_folder, filename)
        destination = get_unique_path(destination) # Предотвращаем конфликт при перемещении
        
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
        elif engine == 'mlx':
            speed_factor *= 2.0 # MLX usually fastest on Apple Silicon

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

def transcribe_with_mlx(audio_path, language):
    print_step(f"Запуск транскрипции (MLX)...")
    result = mlx_whisper.transcribe(
        audio_path, 
        path_or_hf_repo="mlx-community/whisper-large-v3-turbo", 
        language=language if language else None
    )
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

        if console:
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task(f"[cyan]Транскрипция ({model_name})", total=int(est))
                
                text_result = ""
                stop_timer = threading.Event()
                
                def run():
                    nonlocal text_result
                    try:
                        if engine == 'cpp':
                            text_result = transcribe_with_cpp(model_obj, active_audio_path, language)
                        elif engine == 'mlx':
                            text_result = transcribe_with_mlx(active_audio_path, language)
                        else:
                            text_result = transcribe_with_python(model_obj, active_audio_path, language)
                    finally:
                        stop_timer.set()
                
                t = threading.Thread(target=run)
                t.start()
                
                while not stop_timer.is_set():
                    elapsed = time.time() - start_time
                    progress.update(task, completed=min(elapsed, est-1))
                    time.sleep(0.5)
                progress.update(task, completed=est)
                t.join()
        else:
            # Fallback old timer if no Rich
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
            if engine == 'cpp':
                text_result = transcribe_with_cpp(model_obj, active_audio_path, language)
            elif engine == 'mlx':
                text_result = transcribe_with_mlx(active_audio_path, language)
            else:
                text_result = transcribe_with_python(model_obj, active_audio_path, language)
            stop_timer.set()
            t.join()
            print()

        actual_dur = (time.time() - start_time) / 60
        print_step(f"Готово за {actual_dur:.2f} мин", style="bold green")

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
        out_path = get_unique_path(out_path) # Гарантируем уникальность

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text_result)
        
        print_preview(text_result)
        print_step(f"Сохранено: {os.path.basename(out_path)}", style="bold green")
        return True, time.time() - start_time

    except Exception as e:
        print_step(f"Ошибка в проходе: {e}")
        return False, 0
    finally:
        if is_temp and active_audio_path and os.path.exists(active_audio_path):
            try: os.remove(active_audio_path)
            except: pass

# --- WhisperX Logic (New) ---

def run_whisperx_pipeline(audio_file, hf_token=None, language=None, wx_mode='1'):
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
        return None

    device = "cpu"
    compute_type = "float32" 

    print_step(f"Параметры: Device={device}, Type={compute_type}")
    
    # Helper to run with a timer/spinner bar (Rich version)
    def run_with_timer_wx(desc, func, *args, **kwargs):
        if console:
            with Progress(
                SpinnerColumn(spinner_name="dots"),
                TextColumn(f"[yellow]└─ {desc}...[/]"),
                BarColumn(bar_width=20),
                TimeElapsedColumn(),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("", total=None) # indeterminate
                return func(*args, **kwargs)
        else:
            # Old fallback
            stop_timer = threading.Event()
            start_t = time.time()
            def timer_func():
                while not stop_timer.is_set():
                    elapsed = time.time() - start_t
                    sys.stdout.write(f"\r  └─ {desc}: {int(elapsed)}с... ")
                    sys.stdout.flush()
                    time.sleep(0.5)
            t = threading.Thread(target=timer_func)
            t.start()
            try: return func(*args, **kwargs)
            finally:
                stop_timer.set()
                t.join()
                sys.stdout.write("\r" + " " * 80 + "\r")

    try:
        # 1. Transcribe & VAD
        print_step("1/3 Транскрипция (с VAD)...")
        model = run_with_timer_wx("Транскрипция", lambda: whisperx.load_model("large-v2", device, compute_type=compute_type, language=language))
        audio = whisperx.load_audio(audio_file)
        result = run_with_timer_wx("Обработка VAD", lambda: model.transcribe(audio, batch_size=4))
        del model
        gc.collect()

        # 2. Alignment (Выравнивание)
        print_step("2/3 Выравнивание таймкодов...")
        model_a, metadata = run_with_timer_wx("Загрузка Alignment", lambda: whisperx.load_align_model(language_code=result["language"], device=device))
        result = run_with_timer_wx("Выравнивание", lambda: whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False))
        del model_a
        gc.collect()

        # 3. Diarization (Спикеры)
        should_diarize = (wx_mode in ['1', '3']) and hf_token
        if should_diarize:
            print_step("3/3 Диаризация (Спикеры)...")
            try:
                diarize_model = run_with_timer_wx("Загрузка Diarize", lambda: whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device))
                diarize_segments = run_with_timer_wx("Диаризация", lambda: diarize_model(audio))
                result = whisperx.assign_word_speakers(diarize_segments, result)
                del diarize_model
                gc.collect()
            except Exception as e_dia:
                print(f"  Ошибка диаризации: {e_dia}")
        elif wx_mode in ['1', '3'] and not hf_token:
            print_step("Пропуск диаризации (нет HF токена).")

        # Форматирование результата
        print_step("Форматирование текста...")
        output_lines = []
        for segment in result["segments"]:
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            start_str = str(timedelta(seconds=int(start)))
            end_str = str(timedelta(seconds=int(end)))
            
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment["text"].strip()
            
            if wx_mode == '1': # Time + Speaker
                line = f"[{start_str} -> {end_str}] [{speaker}]: {text}"
            elif wx_mode == '2': # Only Time
                line = f"[{start_str} -> {end_str}]: {text}"
            elif wx_mode == '3': # Only Speaker
                line = f"[{speaker}]: {text}"
            else:
                line = f"[{start_str} -> {end_str}] [{speaker}]: {text}"
                
            output_lines.append(line)
        
        total_time = time.time() - start_time_wx
        duration = get_audio_duration_seconds(audio_file)
        print_step(f"WhisperX завершен ({duration/60:.2f} мин за {total_time/60:.2f} мин).")

        return "\n".join(output_lines), total_time

    except Exception as e:
        print(f"\nCRITICAL ERROR in WhisperX: {e}")
        return None, 0


# --- Startup Logic ---

def get_wx_subchoice():
    if questionary:
        try:
            # Note: questionary doesn't have a built-in timeout, so we use a thread or just manual for now.
            # But let's keep it simple for interactive feel.
            choice = questionary.select(
                "НАСТРОЙКА WHISPERX:",
                choices=[
                    questionary.Choice("Время + Спикеры (Full)", value="1"),
                    questionary.Choice("Только Время (Быстрее)", value="2"),
                    questionary.Choice("Только Спикеры", value="3"),
                ],
                default="1"
            ).ask()
            return choice if choice else "1"
        except:
            return "1"
            
    # Fallback to old method if no questionary
    print("\n" + "-"*30)
    print("НАСТРОЙКА WHISPERX: [1] Full, [2] Time, [3] Speaker")
    timeout = 7
    start = time.time()
    while True:
        remaining = int(timeout - (time.time() - start))
        if remaining <= 0: return "1"
        sys.stdout.write(f"\rВыбор [1-3] ({remaining}с): ")
        sys.stdout.flush()
        ready, _, _ = select.select([sys.stdin], [], [], 1)
        if ready:
            inp = sys.stdin.readline().strip()
            return inp if inp in ['1', '2', '3'] else "1"

def get_user_choice():
    title = "ВЫБОР РЕЖИМА РАБОТЫ"
    choices = [
        ("Стандарт: Whisper CPP + Large-v3-Turbo", "1"),
        ("Тест-сравнение: 7 моделей/этапов", "2"),
        ("WhisperX: Диаризация и точные тайминги", "3"),
        ("MLX-Whisper: Максимальная скорость (Apple Silicon)", "4"),
    ]
    
    if is_apple_silicon():
        default_idx = 3 # Вариант 4
    else:
        default_idx = 0 # Вариант 1
        
    choice = get_timed_choice(title, choices, default_idx, timeout=10)
    
    sub = None
    if choice == "3":
        sub = get_wx_subchoice()
    return choice, sub

def get_language_priority():
    title = "Приоритетный язык для распознавания"
    choices = [
        ("Автоопределение (Auto)", "1"),
        ("Украинский (Ukrainian)", "uk"),
        ("Русский (Russian)", "ru"),
        ("Английский (English)", "en"),
    ]
    
    default_idx = 0 # Авто
    choice = get_timed_choice(title, choices, default_idx, timeout=10)
    return choice if choice != "1" else None

def main():
    choice, sub_choice = get_user_choice()
    
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

    if language is None:
        language = get_language_priority()
    
    # Find files (Recursive)
    input_dir = "MyWhisper_to_process"
    valid_exts = ('.mp3', '.m4a', '.wav', '.ogg', '.opus', '.mp4', '.mov', '.avi', '.webm')
    files = []
    if os.path.exists(input_dir):
        for root, dirs, filenames in os.walk(input_dir):
            for filename in filenames:
                if filename.lower().endswith(valid_exts):
                    files.append(os.path.join(root, filename))
    files = sorted(list(set(files))) # Unique sorted paths
    
    if not files:
        print(f"Файлы не найдены в папке {input_dir} и её подпапках.")
        return

    print(f"\nНайдено файлов: {len(files)}")
    for f in files: print(f" - {f}")

    # Prepare Configurations
    configs = []
    
    # Load Models based on choice
    # Load token from .env (Needed for Choice 2 and 3)
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    hf_token = None
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    hf_token = line.split("=", 1)[1].strip()
                    break

    # Prepare Configurations
    configs = []
    
    # Load Models based on choice
    try:
        if choice == "1":
            print_step("\nИнициализация Whisper C++ (large-v3-turbo)...")
            cpp_model = CppModel('large-v3-turbo', n_threads=6, print_realtime=False, print_progress=False)
            configs.append({'engine': 'cpp', 'model': cpp_model, 'name': 'large-v3-turbo', 'desc': 'CPP-Turbo (Standard)', 'suffix': ''})
            
        elif choice == "2":
            print_step("\nЗагрузка моделей для ТЕСТА (6 этапов)...")
            
            # 2.1 Cpp Turbo
            print_step("1/6 Loading CPP large-v3-turbo...")
            cpp_model = CppModel('large-v3-turbo', n_threads=6, print_realtime=False, print_progress=False)
            configs.append({'engine': 'cpp', 'model': cpp_model, 'name': 'large-v3-turbo', 'desc': '2.1 CPP-Turbo', 'suffix': '_cpp_turbo'})
            
            # 2.2 Py Turbo
            print_step("2/6 Loading Python large-v3-turbo...")
            py_turbo = whisper.load_model("large-v3-turbo") 
            configs.append({'engine': 'py', 'model': py_turbo, 'name': 'large-v3-turbo', 'desc': '2.2 PY-Turbo', 'suffix': '_py_turbo'})
            
            # 2.3 Py V2
            print_step("3/6 Loading Python large-v2...")
            py_v2 = whisper.load_model("large-v2")
            configs.append({'engine': 'py', 'model': py_v2, 'name': 'large-v2', 'desc': '2.3 PY-V2', 'suffix': '_py_v2'})

            # 2.4 WX Full
            configs.append({'engine': 'wx', 'model': None, 'name': 'whisperx', 'desc': '2.4 WX-Full', 'suffix': '_wx_full', 'wx_mode': '1'})
            
            # 2.5 WX Time
            configs.append({'engine': 'wx', 'model': None, 'name': 'whisperx', 'desc': '2.5 WX-Time', 'suffix': '_wx_time', 'wx_mode': '2'})
            
            # 2.6 WX Speaker
            configs.append({'engine': 'wx', 'model': None, 'name': 'whisperx', 'desc': '2.6 WX-Speaker', 'suffix': '_wx_speaker', 'wx_mode': '3'})

            # 2.7 MLX
            if mlx_whisper:
                configs.append({'engine': 'mlx', 'model': None, 'name': 'large-v3-turbo', 'desc': '2.7 MLX-Turbo', 'suffix': '_mlx_turbo'})

            if hf_token:
                print_step("HF Token найден, диаризация в тестах 2.4 и 2.6 будет работать.")
            else:
                print("\nВНИМАНИЕ: HF Token не найден. Диаризация в тестах будет пропущена.")

        elif choice == "3":
             print_step("\nРежим WhisperX выбран.")
             configs.append({'engine': 'wx', 'model': None, 'name': 'whisperx', 'desc': 'WhisperX Pipeline', 'suffix': '_whisperx', 'wx_mode': sub_choice})
             
             if hf_token:
                 print_step("HF Token найден в .env")
             else:
                 print("\nHF Token не найден в .env. Диаризация будет пропущена.")

        elif choice == "4":
            if mlx_whisper:
                print_step("\nИнициализация MLX-Whisper (large-v3-turbo)...")
                configs.append({'engine': 'mlx', 'model': None, 'name': 'large-v3-turbo', 'desc': 'MLX-Turbo', 'suffix': ''})
            else:
                print("\nОШИБКА: mlx-whisper не установлен. Используйте Вариант 1.")
                return

    except Exception as e:
        print(f"\nCRITICAL: Ошибка при загрузке моделей: {e}")
        return

    all_stats = []

    # Process Loop
    if console:
        console.print("\n" + "="*50)
        console.print("[bold white]НАЧАЛО ОБРАБОТКИ[/]", style="blue")
        console.print("="*50 + "\n")

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            console=console
        ) as global_progress:
            global_task = global_progress.add_task("ОБЩИЙ ПРОГРЕСС", total=len(files))
            
            for audio_file in files:
                fname = os.path.basename(audio_file)
                global_progress.update(global_task, description=f"Обработка: [bold cyan]{fname[:30]}...[/]")
                
                console.print(Panel(f"[bold white]Файл {global_progress.tasks[0].completed + 1} из {len(files)}[/]\n[cyan]{audio_file}[/]", title="Текущая задача", border_style="blue"))
                
                file_success = True
                file_stats = []
                total_configs = len(configs)
                
                for idx, config in enumerate(configs, 1):
                    if total_configs > 1:
                        console.print(f"  [bold yellow]Этап {idx}/{total_configs}: {config['desc']}[/]")
                    
                    if config['engine'] == 'wx':
                        mode_to_use = config.get('wx_mode', '1')
                        wx_text, elapsed = run_whisperx_pipeline(audio_file, hf_token=hf_token, language=language, wx_mode=mode_to_use)
                        if wx_text:
                            output_dir = os.path.abspath("files_done")
                            if not os.path.exists(output_dir): os.makedirs(output_dir)
                            base = os.path.basename(os.path.splitext(audio_file)[0])
                            suffix = config.get('suffix', '_whisperx')
                            out_path = get_unique_path(os.path.join(output_dir, f"{base}{suffix}.txt"))
                            with open(out_path, "w", encoding="utf-8") as f: f.write(wx_text)
                            print_preview(wx_text)
                            print_step(f"Сохранено: {os.path.basename(out_path)}", style="bold green")
                            file_stats.append((config['desc'], elapsed))
                        else: file_success = False
                    else:
                        ok, elapsed = run_transcription_pass(audio_file, config, language, strict)
                        if ok: file_stats.append((config['desc'], elapsed))
                        else: file_success = False
                
                all_stats.append((fname, file_stats))
                if file_success: move_to_remove_folder(audio_file)
                global_progress.advance(global_task)

    # --- FINAL SUMMARY FOR CHOICE 2 ---
    if choice == "2" and all_stats:
        if console:
            from rich.table import Table
            table = Table(title="ИТОГОВЫЙ ОТЧЕТ СРАВНЕНИЯ", show_header=True, header_style="bold magenta")
            table.add_column("Модель / Движок", style="dim", width=25)
            table.add_column("Время (сек)", justify="right")
            table.add_column("Время (мин)", justify="right")

            for fname, stats in all_stats:
                console.print(f"\n[bold cyan]Файл:[/] [white]{fname}[/]")
                file_table = table # Re-use or Create new per file
                # Better to create fresh table per file or one big one? Let's do fresh per file.
                file_table = Table(show_header=True, header_style="bold magenta")
                file_table.add_column("Конфигурация", style="cyan", width=25)
                file_table.add_column("Время (сек)", justify="right")
                file_table.add_column("Время (мин)", justify="right")
                
                for desc, duration in stats:
                    file_table.add_row(desc, f"{duration:.1f}", f"{duration/60:.2f}")
                console.print(file_table)
        else:
            # Old fallback report
            print("\n" + "="*60)
            print("ИТОГОВЫЙ ОТЧЕТ СРАВНЕНИЯ (ВРЕМЯ РАБОТЫ):")
            print("="*60)
            for fname, stats in all_stats:
                print(f"\nФайл: {fname}")
                print("-" * 40)
                for desc, duration in stats:
                    print(f" {desc:<25} : {duration:>6.1f} сек ({duration/60:>5.2f} мин)")
            print("="*60)

    # --- FINAL CLEANUP OF INPUT DIR ---
    try:
        if os.path.exists(input_dir):
            print_step("Очистка входящей папки от остатков и пустых подпапок...", style="cyan")
            for item in os.listdir(input_dir):
                item_path = os.path.join(input_dir, item)
                try:
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                except:
                    pass
            print_step("Входящая папка полностью очищена", style="bold green")
    except:
        pass

    print("\nВсе задачи выполнены.")

if __name__ == "__main__":
    main()
