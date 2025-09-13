# Face Recognition Personnel Management System
# Система учета персонала с распознаванием лиц  
# Işgärleriň hasabat ulgamy - Ýüz tanamak

## Description / Описание / Düşündiriş

**English:**
A personnel management system with face recognition capabilities. The application uses computer vision to detect and recognize faces, track employee entry/exit times, and maintain a personnel database.

**Русский:**
Система учета персонала с возможностями распознавания лиц. Приложение использует компьютерное зрение для обнаружения и распознавания лиц, отслеживания времени входа/выхода сотрудников и ведения базы данных персонала.

**Türkmençe:**
Ýüz tanyş mümkinçilikleri bolan işgärleriň hasabat ulgamy. Programma ýüzleri tapnyp tanaw, işgärleriň giriş/çykyş waglaryny yzarlamak we işgärler maglumat bazasyny saklamak üçin kompýuter görüşini ulanýar.

## Features / Возможности / Aýratynlyklar

- **Dual Language Support** / **Поддержка двух языков** / **Iki dilde goldaw**
  - Russian (Русский) / Türkmen (Türkmençe)
  - **100% Complete Translation Coverage** / **100% полный перевод** / **100% doly terjime**

- **Face Recognition** / **Распознавание лиц** / **Ýüz tanamak**
  - Real-time face detection / Распознавание в реальном времени / Hakyky wagtda ýüz tanamak
  - Face database management / Управление базой лиц / Ýüzler bazasyny dolandyrmak
  - Confidence scoring / Оценка уверенности / Ynanyş bahasy

- **Personnel Management** / **Управление персоналом** / **Işgärleri dolandyrmak**
  - Employee registration / Регистрация сотрудников / Işgär hasaba almak
  - Entry/Exit tracking / Отслеживание входов/выходов / Giriş/çykyş yzarlamak
  - Personnel information storage / Хранение информации о персонале / Işgärler maglumaty saklamak

## Installation / Установка / Gurnamak

### Prerequisites / Требования / Talaplар

```bash
# Install Python dependencies / Установка зависимостей Python / Python baglylyklary gurnamak
pip install -r requirements.txt
```

### Requirements / Зависимости / Gereklilikler

- Python 3.7+
- OpenCV (cv2)
- NumPy
- scikit-learn
- Pillow (PIL)
- tkinter (usually included with Python)

## Usage / Использование / Ulanmak

### Starting the Application / Запуск приложения / Programmany başlatmak

```bash
python personnel_app.py
```

### Language Selection / Выбор языка / Dil saýlamak

The application supports two languages:
- **ru** - Russian / Русский  
- **tm** - Turkmen / Türkmençe

Use the language dropdown in the interface to switch between languages.
Используйте выпадающий список в интерфейсе для переключения языков.
Dilleri çalyşmak üçin interfeýsdäki saýlama menýusyny ulanyň.

### Main Functions / Основные функции / Esasy funksiýalar

1. **Camera Control** / **Управление камерой** / **Kamera dolandyryş**
   - Start/Stop camera / Включить/Выключить камеру / Kamera açmak/ýapmak
   - Select camera device / Выбор устройства камеры / Kamera enjamy saýlamak

2. **Face Recognition** / **Распознавание лиц** / **Ýüz tanamak**
   - Automatic face detection / Автоматическое обнаружение лиц / Awtomatik ýüz tanamak
   - Real-time recognition / Распознавание в реальном времени / Hakyky wagtda tanyş

3. **Personnel Management** / **Управление персоналом** / **Işgärleri dolandyrmak**
   - Add new employees / Добавление новых сотрудников / Täze işgärler goşmak
   - Edit employee information / Редактирование информации / Işgär maglumatyny üýtgetmek
   - View personnel list / Просмотр списка персонала / Işgärler sanawyny görmek

## Translation Status / Статус перевода / Terjime ýagdaýy

✅ **100% Complete** / **100% Готово** / **100% Tamamlandy**

- Total translation keys: **145**
- Russian translations: **145/145** (100%)
- Turkmen translations: **145/145** (100%)

### Recently Added Translations / Недавно добавленные переводы / Soňky goşulan terjimeler

| Key | Russian | Turkmen |
|-----|---------|---------|
| `name` | Имя | Ady |
| `procentage` | Уверенность распознавания | Ynanyş derejesi |
| `validality` | Уверенность | Ynamlylyk |

## File Structure / Структура файлов / Faýl gurluşy

```
Face_Recognation_Python/
├── personnel_app.py          # Main GUI application / Основное приложение / Esasy programma
├── simple_face_recognizer.py # Face recognition module / Модуль распознавания / Ýüz tanyş moduly
├── requirements.txt          # Python dependencies / Зависимости / Baglylyklar
├── .gitignore               # Git ignore file / Файл игнорирования / Git äsgermezlik
└── README.md               # This file / Этот файл / Bu faýl
```

## Technical Details / Технические детали / Tehniki jikme-jiklikler

### Face Recognition Algorithm / Алгоритм распознавания лиц / Ýüz tanyş algoritmi

- **LBPH** (Local Binary Patterns Histograms)
- **Haar Cascades** for face detection / для обнаружения лиц / ýüz tapma üçin
- **Cosine Similarity** for face matching / для сопоставления лиц / ýüzleri deňeşdirmek üçin

### Features Extracted / Извлекаемые признаки / Alynýan aýratynlyklar

- Local Binary Patterns (LBP)
- Intensity histograms / Гистограммы яркости / Ýagtylygyň gistogrammalary
- Gradient features (Sobel) / Градиентные признаки / Gradient aýratynlyklary
- Texture features / Текстурные признаки / Tekstura aýratynlyklary

## License / Лицензия / Ygtyýarnama

This project is open source. / Этот проект с открытым исходным кодом. / Bu taslama açyk çeşme.

## Contributing / Вклад в проект / Goşant goşmak

Contributions are welcome! / Вклады приветствуются! / Goşantlar garşylanýar!

## Contact / Контакты / Aragatnaşyk

For issues and questions, please use the GitHub issue tracker.
Для вопросов и проблем используйте систему отслеживания GitHub.
Soraglar we meseleler üçin GitHub yzarlaýyş ulgamyny ulanyň.