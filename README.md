# AI-Meeting-Transcription-and-Analysis-Tool
# 🎙️ Meeting Intelligence: AI Transcription & Analysis Tool

**Meeting Intelligence** — это мощное десктопное приложение для автоматизации обработки совещаний. Оно извлекает текст из аудио и видео, разделяет голоса спикеров и генерирует интеллектуальное саммари с помощью LLM.

---

## ✨ Основные возможности
* **Транскрибация:** высокоточное распознавание речи с помощью `WhisperX`.
* **Диаризация:** определение того, кто именно произнес фразу (`Pyannote Audio 3.1`).
* **AI-Суммаризация:** автоматическое выделение ключевых моментов, решений и задач через `LangGraph` и модели `Qwen`.
* **Препроцессинг:** очистка аудио от шумов и нормализация громкости.
* **Экспорт:** Ссхранение результатов в форматах **PDF**, **DOCX** и **Markdown**.

---

## 🛠 Технологический стек
* **GUI:** PyQt6 (Modern Dark Theme)
* **ML Core:** WhisperX, Faster-Whisper, Pyannote.audio
* **Orchestration:** LangGraph, LangChain
* **LLM:** qwen3.6-plus (через OpenRouter API)
* **Audio:** FFmpeg, Librosa, Noisereduce

---

## ⚙️ Системные требования

### 1. Установка FFmpeg (Обязательно)
FFmpeg необходим для конвертации медиафайлов. Без него приложение не сможет прочитать видео или аудио.

* **Windows:**
    1. Скачайте сборку с [gyan.dev](https://www.gyan.dev/ffmpeg/builds/).
    2. Распакуйте архив и добавьте путь к папке `bin` в системную переменную **PATH**.
    3. Проверьте установку командой `ffmpeg -version` в терминале.
* **Linux (Ubuntu/Debian):**
    ```bash
    sudo apt update && sudo apt install ffmpeg
    ```
* **macOS:**
    ```bash
    brew install ffmpeg
    ```

### 2. Доступ к моделям Diarization (Hugging Face)
Для работы функции разделения спикеров (чтобы не было `SPEAKER_UNKNOWN`):
1.  Зарегистрируйтесь на [Hugging Face](https://huggingface.co/).
2.  Примите условия лицензии для моделей (нажать **Agree and access repository**):
    * [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
    * [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3.  Создайте токен доступа в [Settings -> Access Tokens](https://huggingface.co/settings/tokens) (роль: `Read`).

---

## 🚀 Установка и запуск

### 1. Клонирование и окружение
```bash
git clone <ссылка_на_репозиторий>
cd AI-Meeting-Transcription-and-Analysis-Tool
python -m venv venv
```
#### Активация окружения
#### Windows:
```bash
venv\Scripts\activate
```
#### Linux/macOS:
```bash
source venv/bin/activate
```
### 2. Установка зависимостей
```bash
# Установка PyTorch (выберите версию под свою ОС/CUDA на pytorch.org)
pip install torch torchvision torchaudio

# Установка WhisperX напрямую из репозитория
pip install git+https://github.com/m-bain/whisperX.git

# Установка остальных библиотек
pip install -r requirements.txt
```
### 3. Настройка переменных окружения
Создайте в корне проекта файл .env и добавьте в него свои ключи:
```
OPENROUTER_API_KEY=your_key_here
HF_TOKEN=your_huggingface_token_here
```
### 4. Запуск приложения
```bash
python app.py
```
## 📂 Структура проекта
* main.py — Точка входа, графический интерфейс.
* src/transcription.py — Логика распознавания и работы нейросетей.
* src/summarization.py — Граф обработки текста через LLM.
* src/utils.py — Обработка аудио и экспорт файлов.
* config.yaml — Настройки моделей.
## ⚠️ Решение проблем
* SPEAKER_UNKNOWN: Проверьте, приняли ли вы соглашения на Hugging Face для обеих моделей (diarization и segmentation).
* Низкая скорость на CPU: В config.yaml установите whisper: small или base. Убедитесь, что compute_type установлен в int8.