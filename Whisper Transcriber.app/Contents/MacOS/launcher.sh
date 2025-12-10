#!/bin/bash

# Создаем временный скрипт для выполнения
TEMP_SCRIPT=$(mktemp /tmp/whisper_launcher.XXXXXX.command)

cat > "$TEMP_SCRIPT" << 'EOFSCRIPT'
#!/bin/bash
cd "/Users/eugeneprokopenko/Downloads/Converter"
source "myenv/bin/activate"
echo "Запуск Whisper Transcriber..."
echo "========================================"
python "whisper_v19_corrected_timer_60_percent_rem_file.py"
EXIT_CODE=$?
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Скрипт успешно завершен!"
else
    echo "Скрипт завершен с ошибкой (код: $EXIT_CODE)"
fi
deactivate
echo ""
read -p "Нажмите Enter для закрытия терминала..."
rm -f "$0"  # Удаляем сам скрипт после завершения
EOFSCRIPT

chmod +x "$TEMP_SCRIPT"

# Открываем временный скрипт в Terminal через open (это откроет ОДНО окно)
open -a Terminal.app "$TEMP_SCRIPT"



