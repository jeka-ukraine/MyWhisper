#!/bin/bash

# Определяем директорию скрипта (это корень проекта)
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Полный путь к виртуальному окружению
VENV_DIR="$PROJECT_DIR/myenv"

# Полный путь к Python скрипту
PYTHON_SCRIPT="$PROJECT_DIR/whisper_v19_corrected_timer_60_percent_rem_file.py"

# Переходим в директорию проекта
cd "$PROJECT_DIR" || {
    echo "Ошибка: не удалось перейти в директорию $PROJECT_DIR"
    read -p "Нажмите Enter для закрытия..."
    exit 1
}

# Проверяем наличие виртуального окружения
if [ ! -d "$VENV_DIR" ]; then
    echo "========================================"
    echo "ПЕРВЫЙ ЗАПУСК / FIRST RUN"
    echo "========================================"
    echo "Виртуальное окружение не найдено. Создаем..."
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    
    # Активируем
    source "$VENV_DIR/bin/activate"
    
    echo "Установка зависимостей (это может занять несколько минут)..."
    echo "Installing dependencies..."
    pip install --upgrade pip
    
    if [ -f "$PROJECT_DIR/requirements.txt" ]; then
        pip install -r "$PROJECT_DIR/requirements.txt"
    else
        echo "ВНИМАНИЕ: Файл requirements.txt не найден! Установка пакетов по умолчанию..."
        pip install openai-whisper pywhispercpp ffmpeg-python pydub tqdm omegaconf git+https://github.com/m-bain/whisperx.git
    fi
    
    echo "========================================"
    echo "Установка завершена! Запуск..."
    echo "Setup complete! Starting..."
    echo "========================================"
else
    # Активируем существующее
    echo "Активация виртуального окружения..."
    source "$VENV_DIR/bin/activate" || {
        echo "Ошибка: не удалось активировать виртуальное окружение $VENV_DIR"
        read -p "Нажмите Enter для закрытия..."
        exit 1
    }
fi

# Запускаем Python скрипт
echo "Запуск скрипта транскрипции..."
echo "========================================"
python "$PYTHON_SCRIPT"

# Сохраняем код выхода
EXIT_CODE=$?

# Выводим сообщение о завершении
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Скрипт успешно завершен!"
else
    echo "Скрипт завершен с ошибкой (код: $EXIT_CODE)"
fi

# Деактивируем виртуальное окружение
deactivate

# Оставляем терминал открытым
echo ""
read -p "Нажмите Enter для закрытия терминала..."
