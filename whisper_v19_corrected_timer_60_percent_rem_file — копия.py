import os
import subprocess
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# FFmpeg уже установлен в системе (Docker/Linux) или добавляем Windows путь
import platform
if platform.system() == "Windows":
    ffmpeg_path = r"C:\ffmpeg\bin"
    os.environ["PATH"] += os.pathsep + ffmpeg_path
# В Linux/Docker ffmpeg уже доступен через apt-get install

import whisper
import ffmpeg
from pydub import AudioSegment
import os
import glob
import time
from tqdm import tqdm
from datetime import datetime, timedelta
import argparse
import threading
import shutil

try:
    subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # print("FFmpeg доступен!")
except FileNotFoundError:
    print("FFmpeg не найден! Убедись, что путь указан верно.")


# Функция для отображения сообщений
def print_step(message, progress=None):
    # Добавляем точку в конце, если её нет
    if not message.endswith('.'):
        message += '.'
    if progress is not None:
        print(f"{message} - {progress}%")
    else:
        print(message)

# Функция для перемещения файла в папку "Files to Remove"
def move_to_remove_folder(audio_file):
    try:
        # Определяем директорию, где находится файл
        file_dir = os.path.dirname(os.path.abspath(audio_file))
        remove_folder = os.path.join(file_dir, "Files to Remove")
        
        # Создаем папку, если её нет
        if not os.path.exists(remove_folder):
            os.makedirs(remove_folder)
            print_step(f"Создана папка: {remove_folder}")
        
        # Получаем имя файла
        filename = os.path.basename(audio_file)
        destination = os.path.join(remove_folder, filename)
        
        # Перемещаем файл
        shutil.move(audio_file, destination)
        print_step(f"Файл {filename} перемещен в 'Files to Remove'")
        return True
    except Exception as e:
        print_step(f"Ошибка при перемещении файла {audio_file}: {e}")
        return False

# Функция для оценки времени транскрипции
def estimate_transcription_time(audio_length, model_name):
    # Скорректированные коэффициенты для более точной оценки
    model_speeds = {
        "tiny": 3.5,      # Быстрая модель
        "base": 2.5,      # Базовая модель  
        "small": 2.0,     # Небольшая модель
        "medium": 1.0,    # Средняя модель
        "large": 0.4,     # Большая модель
        "large-v2": 0.4   # Скорректировано: было 0.833, стало 0.4 (в 2 раза медленнее)
    }
    model_speed = model_speeds.get(model_name, 0.4)
    estimated_seconds = audio_length / model_speed
    return estimated_seconds

# Функция для обработки одного файла
def process_file(audio_file, model, model_name, language=None, strict_mode=False):
    print_step(f"Обработка файла: {audio_file}")
    
    file_format = audio_file.split('.')[-1].lower()
    
    if file_format in ['mp4', 'avi', 'mov', 'webm']:
        print_step(f"Извлечение аудио из {file_format} видеофайла с помощью ffmpeg")
        audio_output = "/tmp/temp_audio.wav"
        
        try:
            ffmpeg.input(audio_file).output(audio_output).run(quiet=True, overwrite_output=True)
            audio = AudioSegment.from_file(audio_output, format="wav")
            os.remove(audio_output)
        except ffmpeg.Error as e:
            print_step(f"Ошибка при извлечении аудио: {e}")
            return False
    else:
        print_step(f"Конвертация {file_format} в аудио сегмент")
        try:
            # Для .opus и .ogg (аудиосообщения Telegram/WhatsApp) пробуем разные подходы
            if file_format in ["opus", "ogg"]:
                try:
                    # Сначала пробуем открыть напрямую как ogg
                    audio = AudioSegment.from_file(audio_file, format="ogg")
                except:
                    # Если не получилось, используем ffmpeg для конвертации
                    print_step(f"Прямое открытие не удалось, использую ffmpeg для конвертации")
                    audio_output = "/tmp/temp_audio_ogg.wav"
                    ffmpeg.input(audio_file).output(audio_output, acodec='pcm_s16le', ac=1, ar='16000').run(quiet=True, overwrite_output=True)
                    audio = AudioSegment.from_file(audio_output, format="wav")
                    os.remove(audio_output)
            else:
                audio = AudioSegment.from_file(audio_file, format=file_format)
        except Exception as e:
            print_step(f"Ошибка при декодировании файла {audio_file}: {e}")
            print_step(f"Файл {audio_file} будет пропущен")
            return False
    
    audio_length_seconds = len(audio) / 1000
    print_step(f"Длина аудиофрагмента: {audio_length_seconds / 60:.2f} минут")
    
    estimated_time = estimate_transcription_time(audio_length_seconds, model_name)
    end_time_estimate = datetime.now() + timedelta(seconds=estimated_time)
    print_step(f"Оценка времени транскрипции: {estimated_time / 60:.2f} минут.\nЗакончится в {end_time_estimate.strftime('%H:%M:%S')}")
    
    start_time = time.time()
    start_time_str = datetime.now().strftime('%H:%M:%S')
    print_step(f"Время начала транскрипции: {start_time_str}")

    # Вывести режим и язык ДО прогресс-бара
    if strict_mode and language:
        print_step(f"Строгий режим транскрипции на языке: {language}")
    else:
        print_step(f"Приоритетный язык: {language if language else 'Авто'}")

    # Таймер для отображения прошедшего времени и прогресс-бара
    stop_timer = threading.Event()
    progress_bar_exception = None  # Для передачи исключения из потока
    def timer_func():
        try:
            bar_length = 30
            while not stop_timer.is_set():
                elapsed = int(time.time() - start_time)
                mins, secs = divmod(elapsed, 60)
                progress = min(elapsed / estimated_time, 1.0) if estimated_time > 0 else 1.0
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                percent = int(progress * 100)
                # Выводим время и прогресс-бар на одной строке, затирая предыдущий вывод
                print(f"\rПрошло времени: {mins:02d}:{secs:02d} [{bar}] {percent}%", end="", flush=True)
                time.sleep(1)
        except Exception as e:
            nonlocal progress_bar_exception
            progress_bar_exception = e
    timer_thread = threading.Thread(target=timer_func)
    timer_thread.start()

    try:
        if strict_mode and language:
            result = model.transcribe(audio_file, language=language, task='transcribe')
        else:
            result = model.transcribe(audio_file, language=language)
    finally:
        stop_timer.set()
        timer_thread.join()
        print()  # чтобы не было наложения \r
        if progress_bar_exception:
            print(f"\nОшибка в прогресс-баре: {progress_bar_exception}")

    end_time = time.time()
    actual_minutes = (end_time - start_time) / 60
    print_step(f"Транскрипция завершена за {actual_minutes:.2f} минут")

    # Сравнение с оценкой
    diff_minutes = actual_minutes - (estimated_time / 60)
    percent_diff = (diff_minutes / (estimated_time / 60)) * 100 if estimated_time != 0 else 0
    sign = '+' if diff_minutes > 0 else '-'
    print_step(f"Фактическое время отличается от оценки на {sign}{abs(diff_minutes):.2f} минут ({sign}{abs(percent_diff):.1f}%).")
    
    # Определяем папку для сохранения результатов
    output_dir = "/app/output" if os.path.exists("/app/output") else "."
    base_filename = os.path.basename(os.path.splitext(audio_file)[0])
    result_filename = os.path.join(output_dir, f"{base_filename}_transcription.txt")
    
    with open(result_filename, "w") as result_file:
        result_file.write(result['text'])
    print_step(f"Результат сохранен в файл: {result_filename}")

    # Вывод первых 1000 символов результата
    try:
        with open(result_filename, "r") as f:
            content = f.read(1000)
            print_step(f"Первые 1000 символов результата: {content}")
    except Exception as e:
        print_step(f"Ошибка при чтении результата: {e}")
    
    # Перемещаем успешно обработанный файл в папку "Files to Remove"
    move_to_remove_folder(audio_file)
    
    return True

# Функция для выбора языка
def select_language(language_param):
    language_mapping = {
        "-ukr": "uk",
        "-rus": "ru",
        "-eng": "en",
        "-pln": "pl"
    }
    return language_mapping.get(language_param, None)

# Основная функция
def main():
    parser = argparse.ArgumentParser(description="Скрипт для транскрипции аудио и видеофайлов.")
    parser.add_argument("--language_pr", type=str, help="Приоритетный язык: -ukr, -rus, -eng, -pln", default=None)
    parser.add_argument("--language_only", type=str, help="Фиксированный язык: -ukr, -rus, -eng, -pln", default=None)
    parser.add_argument("--model", type=int, choices=range(1, 6), default=5, help="Выбор модели: 1-tiny, 2-base, 3-small, 4-medium, 5-large-v2 (по умолчанию 5)")
    args = parser.parse_args()

    if args.language_pr and args.language_only:
        print("Ошибка: Нельзя использовать --language_pr и --language_only одновременно.")
        return

    language = None
    strict_mode = False

    if args.language_pr:
        language = select_language(args.language_pr)
        strict_mode = False
    elif args.language_only:
        language = select_language(args.language_only)
        strict_mode = True

    print_step(f"Выбранный язык: {language if language else 'Авто'}")
    print_step(f"Режим строгого выбора языка: {'Да' if strict_mode else 'Нет'}")

    # print_step("Поиск медиафайлов")
    audio_files = sorted(glob.glob("*.mp4") + glob.glob("*.MP4") +
                         glob.glob("*.mp3") + glob.glob("*.MP3") +
                         glob.glob("*.avi") + glob.glob("*.AVI") +
                         glob.glob("*.mov") + glob.glob("*.MOV") +
                         glob.glob("*.m4a") + glob.glob("*.M4A") +
                         glob.glob("*.webm") + glob.glob("*.WEBM") +
                         glob.glob("*.ogg") + glob.glob("*.OGG") +
                         glob.glob("*.wav") + glob.glob("*.WAV") +
                         glob.glob("*.opus") + glob.glob("*.OPUS"))  # Добавлена поддержка .wav и .opus
    
    if not audio_files:
        print("Нет файлов для обработки.")
        return


    print("Найдены файлы:")
    for idx, file in enumerate(audio_files, 1):
        print(f" {idx}. {file}")

    models = {
        1: "tiny",
        2: "base",
        3: "small",
        4: "medium",
        5: "large-v2"
    }
    model_name = models[args.model]
    print_step(f"Выбрана модель: {model_name}")
    model = whisper.load_model(model_name)
    print_step("Модель загружена")
    
    successful_count = 0
    failed_count = 0
    
    with tqdm(total=len(audio_files), desc="Общий прогресс", ncols=100) as pbar:
        for audio_file in audio_files:
            success = process_file(audio_file, model, model_name, language, strict_mode)
            if success:
                successful_count += 1
            else:
                failed_count += 1
            pbar.update(1)
    
    # Итоговая статистика
    print("\n" + "="*50)
    print_step(f"Обработка завершена!")
    print_step(f"Успешно обработано: {successful_count} файлов")
    if failed_count > 0:
        print_step(f"Не удалось обработать: {failed_count} файлов")

if __name__ == "__main__":
    main()
