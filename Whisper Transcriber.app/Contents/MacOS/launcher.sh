#!/bin/bash

# Определяем путь к корню проекта относительно этого скрипта (Contents/MacOS/ -> ProjectRoot)
DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$DIR/../../.." && pwd)"

# Создаем временный скрипт для выполнения
TEMP_SCRIPT=$(mktemp /tmp/whisper_launcher.XXXXXX.command)

# Используем cat с EOF без кавычек, чтобы переменная $PROJECT_ROOT подставилась,
# но экранируем переменные внутри скрипта (\$), чтобы они не вычислялись сейчас.
cat > "$TEMP_SCRIPT" << EOFSCRIPT
#!/bin/bash
cd "$PROJECT_ROOT"
source "myenv/bin/activate"
echo "Запуск Whisper Transcriber..."
echo "========================================"
"$PROJECT_ROOT/myenv/bin/python" "whisper_v19_corrected_timer_60_percent_rem_file.py"
EXIT_CODE=\$?
echo "========================================"
if [ \$EXIT_CODE -eq 0 ]; then
    echo "Скрипт успешно завершен!"
else
    echo "Скрипт завершен с ошибкой (код: \$EXIT_CODE)"
fi
deactivate
echo ""
read -p "Нажмите Enter для закрытия терминала..."
rm -f "\$0"  # Удаляем сам скрипт после завершения
EOFSCRIPT

chmod +x "$TEMP_SCRIPT"

# Открываем временный скрипт в Terminal
open -a Terminal.app "$TEMP_SCRIPT"
