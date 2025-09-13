# -*- coding: utf-8 -*-
"""
Простой модуль распознавания лиц без использования нейронных сетей
Использует классические методы компьютерного зрения:
- LBPH (Local Binary Patterns Histograms)
- Haar Cascades для детекции лиц
- Сравнение гистограмм для распознавания
"""

import cv2
import numpy as np
import os
import time
from sklearn.metrics.pairwise import cosine_similarity
import pickle


class SimpleFaceRecognizer:
    def __init__(self):
        # Загружаем каскады Хаара для детекции лиц
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Простая система распознавания без LBPH (используем только признаки)
        
        # База данных лиц (словарь: имя -> данные о человеке)
        self.face_database = {}
        self.persons_data = {}  # Дополнительная информация о людях
        self.entry_history = []  # История входов/выходов
        self.label_counter = 0
        
        # Путь к файлу базы данных
        self.database_path = "database/face_database.pkl"
        
        # Загружаем существующую базу данных
        self.load_database()
    
    def detect_faces(self, frame):
        """Детекция лиц на изображении с обработкой ошибок"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Улучшаем контрастность для лучшей детекции
            gray = cv2.equalizeHist(gray)
            
            # Пробуем разные наборы параметров для надежности
            faces = []
            
            # Попытка 1: Стандартные параметры
            try:
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.3,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
            except cv2.error:
                # Попытка 2: Более простые параметры
                try:
                    faces = self.face_cascade.detectMultiScale(
                        gray, 
                        scaleFactor=1.1,
                        minNeighbors=3
                    )
                except cv2.error:
                    # Попытка 3: Минимальные параметры
                    try:
                        faces = self.face_cascade.detectMultiScale(gray)
                    except cv2.error:
                        print("Ошибка детекции лиц, возвращаем пустой результат")
                        return []
        
            detected_faces = []
            for (x, y, w, h) in faces:
                # Извлекаем область лица с небольшим расширением
                padding = int(min(w, h) * 0.1)
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(gray.shape[1], x + w + padding)
                y2 = min(gray.shape[0], y + h + padding)
                
                face_roi = gray[y1:y2, x1:x2]
                
                # Проверяем что ROI не пустой
                if face_roi.size == 0:
                    continue
                    
                # Нормализуем размер
                face_roi = cv2.resize(face_roi, (100, 100))
                
                detected_faces.append({
                    'face_roi': face_roi,
                    'coordinates': (x, y, w, h),
                    'features': self.extract_features(face_roi)
                })
            
            return detected_faces
            
        except Exception as e:
            print(f"Общая ошибка при детекции лиц: {e}")
            return []
    def detect_faces_from_image(self, image_path):
        """Детекция лиц из файла изображения с улучшенными параметрами"""
        try:
            # Загружаем изображение
            frame = cv2.imread(image_path)
            if frame is None:
                return []
            
            # Масштабируем изображение если оно очень большое
            height, width = frame.shape[:2]
            max_dimension = 1000
            if max(height, width) > max_dimension:
                if width > height:
                    new_width = max_dimension
                    new_height = int(height * max_dimension / width)
                else:
                    new_height = max_dimension
                    new_width = int(width * max_dimension / height)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Преобразуем в градации серого
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Улучшаем контрастность
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # Пробуем разные наборы параметров
            all_faces = []
            
            # Набор 1: Стандартные параметры
            faces1 = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )
            all_faces.extend(faces1)
            
            # Набор 2: Более чувствительные параметры
            faces2 = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)
            )
            all_faces.extend(faces2)
            
            # Набор 3: Для больших лиц
            faces3 = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=3, minSize=(100, 100)
            )
            all_faces.extend(faces3)
            
            # Убираем дубликаты
            unique_faces = []
            for face in all_faces:
                is_duplicate = False
                for existing in unique_faces:
                    # Проверяем перекрытие
                    x1, y1, w1, h1 = face
                    x2, y2, w2, h2 = existing
                    if abs(x1 - x2) < 50 and abs(y1 - y2) < 50:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_faces.append(face)
            
            detected_faces = []
            for (x, y, w, h) in unique_faces:
                # Извлекаем область лица с отступами
                margin = 10
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(gray.shape[1], x + w + margin)
                y2 = min(gray.shape[0], y + h + margin)
                
                face_roi = gray[y1:y2, x1:x2]
                face_roi = cv2.resize(face_roi, (100, 100))
                
                detected_faces.append({
                    'face_roi': face_roi,
                    'coordinates': (x, y, w, h),
                    'features': self.extract_features(face_roi)
                })
            
            return detected_faces
            
        except Exception as e:
            print(f"Ошибка при обработке изображения {image_path}: {e}")
            return []
    
    def extract_features(self, face_roi):
        """Извлечение признаков из области лица"""
        features = {}
        
        # 1. LBP (Local Binary Patterns)
        lbp = self.calculate_lbp(face_roi)
        features['lbp_hist'] = cv2.calcHist([lbp], [0], None, [256], [0, 256]).flatten()
        
        # 2. Гистограмма яркости
        features['intensity_hist'] = cv2.calcHist([face_roi], [0], None, [256], [0, 256]).flatten()
        
        # 3. Градиенты (Sobel)
        sobel_x = cv2.Sobel(face_roi, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(face_roi, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        features['gradient_hist'] = cv2.calcHist([gradient_magnitude.astype(np.uint8)], [0], None, [256], [0, 256]).flatten()
        
        # 4. Текстурные признаки (стандартное отклонение в окнах)
        features['texture'] = self.calculate_texture_features(face_roi)
        
        # Объединяем все признаки
        combined_features = np.concatenate([
            features['lbp_hist'],
            features['intensity_hist'],
            features['gradient_hist'],
            features['texture']
        ])
        
        # Нормализация
        combined_features = combined_features / (np.linalg.norm(combined_features) + 1e-7)
        
        return combined_features
    
    def calculate_lbp(self, image):
        """Вычисление Local Binary Patterns"""
        lbp = np.zeros_like(image)
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                center = image[i, j]
                code = 0
                
                # 8 соседей
                neighbors = [
                    image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                    image[i, j+1], image[i+1, j+1], image[i+1, j],
                    image[i+1, j-1], image[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code += 2**k
                
                lbp[i, j] = code
        
        return lbp
    
    def calculate_texture_features(self, image):
        """Вычисление текстурных признаков с фиксированной размерностью"""
        texture_features = []
        
        # Фиксированный размер блока и количество блоков для оптимизации
        h, w = image.shape
        block_size = 10
        target_blocks = 64  # Уменьшаем для производительности
        
        # Вычисляем шаг для получения фиксированного количества блоков
        step_h = max(1, (h - block_size) // 8) if h > block_size else 1
        step_w = max(1, (w - block_size) // 8) if w > block_size else 1
        
        block_count = 0
        for i in range(0, h - block_size, step_h):
            for j in range(0, w - block_size, step_w):
                if block_count >= target_blocks:
                    break
                block = image[i:i+block_size, j:j+block_size]
                if block.size > 0:
                    texture_features.append(np.std(block))
                    block_count += 1
            if block_count >= target_blocks:
                break
        
        # Дополняем до целевого размера нулями или обрезаем
        while len(texture_features) < target_blocks:
            texture_features.append(0.0)
        texture_features = texture_features[:target_blocks]
        
        return np.array(texture_features)
    
    def add_person(self, face_roi, person_info, original_frame=None):
        """Добавление нового человека в базу данных с сохранением фотографии и оригинального кадра"""
        name = person_info['name']
        features = self.extract_features(face_roi)
        
        if name not in self.face_database:
            self.face_database[name] = []
            self.label_counter += 1
        
        self.face_database[name].append(features)
        
        # Создаем папку для конкретного человека
        person_dir = os.path.join("faces", name.replace(" ", "_").replace("/", "_"))
        os.makedirs(person_dir, exist_ok=True)
        
        timestamp = int(time.time())
        
        # Сохраняем оригинальный кадр в высоком качестве если он предоставлен
        original_path = None
        if original_frame is not None:
            original_filename = f"original_{timestamp}.jpg"
            original_path = os.path.join(person_dir, original_filename)
            cv2.imwrite(original_path, original_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Сохраняем фотографию лица с высоким качеством
        face_filename = f"face_{timestamp}.jpg"
        face_path = os.path.join(person_dir, face_filename)
        
        # Увеличиваем размер изображения для лучшего качества
        enlarged_face = cv2.resize(face_roi, (400, 400), interpolation=cv2.INTER_CUBIC)
        
        # Улучшаем качество изображения
        # Применяем билатеральный фильтр для уменьшения шума
        denoised = cv2.bilateralFilter(enlarged_face, 9, 75, 75)
        
        # Повышаем резкость
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # Улучшаем контрастность
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(sharpened)
        
        # Сохраняем с максимальным качеством JPEG
        cv2.imwrite(face_path, enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Сохраняем информацию о человеке
        person_data = {
            'info': {
                'name': person_info['name'],
                'position': person_info['position'],
                'age': person_info['age'],
                'rank': person_info['rank']
            },
            'face_photo_path': face_path,      # Путь к обработанному лицу
            'original_photo_path': original_path,  # Путь к оригинальному кадру
            'entry_time': None,
            'exit_time': None,
            'status': 'Вышел',  # По умолчанию считаем, что человек не на работе
            'face_images': [face_roi.copy()]  # Сохраняем изображение лица
        }
        
        # Добавляем путь к оригинальной фотографии если есть
        if 'original_photo_path' in person_info:
            person_data['original_photo_path'] = person_info['original_photo_path']
        
        self.persons_data[name] = person_data
        
        # Сохраняем базу данных
        self.save_database()
        
        return True
    
    def add_person_from_image(self, image_path, person_info):
        """Добавление человека из файла изображения"""
        faces = self.detect_faces_from_image(image_path)
        
        if not faces:
            return False, "Лицо не найдено на изображении"
        
        if len(faces) > 1:
            return False, "На изображении найдено несколько лиц. Используйте изображение с одним лицом."
        
        # Используем первое найденное лицо
        face_roi = faces[0]['face_roi']
        
        # Сохраняем оригинальную фотографию в папку faces
        name = person_info['name']
        os.makedirs("faces", exist_ok=True)
        
        timestamp = int(time.time())
        original_photo_filename = f"{name}_original_{timestamp}.jpg"
        original_photo_path = os.path.join("faces", original_photo_filename)
        
        # Сохраняем оригинальную фотографию с улучшенным качеством
        import shutil
        original_image = cv2.imread(image_path)
        if original_image is not None:
            # Улучшаем качество оригинального изображения
            # Применяем денойзинг для цветного изображения
            denoised = cv2.fastNlMeansDenoisingColored(original_image, None, 10, 10, 7, 21)
            
            # Повышаем резкость
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            
            # Сохраняем с высоким качеством
            cv2.imwrite(original_photo_path, sharpened, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            # Если не удалось улучшить, просто копируем
            shutil.copy2(image_path, original_photo_path)
        
        # Добавляем информацию об оригинальной фотографии
        person_info['original_photo_path'] = original_photo_path
        
        success = self.add_person(face_roi, person_info)
        
        if success:
            return True, "Человек успешно добавлен из изображения"
        else:
            return False, "Ошибка при добавлении человека"
    
    def add_multiple_samples(self, face_samples, person_info):
        """Добавление нескольких образцов лица для повышения точности"""
        name = person_info['name']
        
        if name not in self.face_database:
            self.face_database[name] = []
            self.label_counter += 1
            
            # Сохраняем информацию о человеке только при первом добавлении
            self.persons_data[name] = {
                'name': person_info['name'],
                'position': person_info['position'],
                'age': person_info['age'],
                'rank': person_info['rank'],
                'entry_time': None,
                'exit_time': None,
                'status': 'Вышел'
            }
        
        # Добавляем все образцы
        for face_roi in face_samples:
            features = self.extract_features(face_roi)
            self.face_database[name].append(features)
        
        # Сохраняем базу данных
        self.save_database()
        
        return True
    
    def recognize_face(self, face_roi):
        """Улучшенное распознавание лица с проверкой размерностей"""
        if not self.face_database:
            return None, 0.0
        
        features = self.extract_features(face_roi)
        
        best_match = None
        best_similarity = 0.0
        threshold = 0.65  # Понижаем порог для лучшего распознавания
        
        # Сравниваем с каждым человеком в базе данных
        for name, feature_list in self.face_database.items():
            similarities = []
            
            for stored_features in feature_list:
                try:
                    # Проверяем размерности
                    if len(features) != len(stored_features):
                        print(f"Предупреждение: Несовместимые размерности для {name}: {len(features)} vs {len(stored_features)}")
                        continue
                    
                    # Косинусное сходство
                    similarity = cosine_similarity([features], [stored_features])[0, 0]
                    similarities.append(similarity)
                    
                except Exception as e:
                    print(f"Ошибка при сравнении с {name}: {e}")
                    continue
            
            if similarities:
                # Берем среднее из топ-3 сходств для большей надежности
                similarities.sort(reverse=True)
                top_similarities = similarities[:min(3, len(similarities))]
                avg_similarity = np.mean(top_similarities)
                
                if avg_similarity > best_similarity and avg_similarity > threshold:
                    best_similarity = avg_similarity
                    best_match = name
        
        return best_match, best_similarity
    
    def record_entry_exit(self, name, action):
        """Записать вход/выход сотрудника"""
        import datetime
        
        if name not in self.persons_data:
            return False
        
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if action == "Вошел":
            self.persons_data[name]['entry_time'] = current_time
            self.persons_data[name]['status'] = 'На работе'
        else:  # Вышел
            self.persons_data[name]['exit_time'] = current_time
            self.persons_data[name]['status'] = 'Вышел'
        
        # Добавляем в историю
        self.entry_history.append({
            'name': name,
            'action': action,
            'time': current_time
        })
        
        # Сохраняем изменения
        self.save_database()
        
        return True
    
    def get_person_info(self, name):
        """Получить информацию о человеке"""
        if name in self.persons_data:
            person_data = self.persons_data[name]
            # Объединяем информацию для обратной совместимости
            if 'info' in person_data:
                result = person_data['info'].copy()
                result.update({
                    'entry_time': person_data.get('entry_time'),
                    'exit_time': person_data.get('exit_time'),
                    'status': person_data.get('status', 'Вышел')
                })
                return result
            else:
                # Старый формат данных
                return person_data
        return None
    
    def get_entry_history(self, name=None):
        """Получить историю входов/выходов"""
        if name:
            return [entry for entry in self.entry_history if entry['name'] == name]
        return self.entry_history
    
    def save_database(self):
        """Сохранение базы данных"""
        os.makedirs("database", exist_ok=True)
        
        with open(self.database_path, 'wb') as f:
            pickle.dump({
                'face_database': self.face_database,
                'persons_data': self.persons_data,
                'entry_history': self.entry_history,
                'label_counter': self.label_counter
            }, f)
    
    def load_database(self):
        """Загрузка базы данных"""
        if os.path.exists(self.database_path):
            try:
                with open(self.database_path, 'rb') as f:
                    data = pickle.load(f)
                    self.face_database = data.get('face_database', {})
                    self.persons_data = data.get('persons_data', {})
                    self.entry_history = data.get('entry_history', [])
                    self.label_counter = data.get('label_counter', 0)
                
                # Очистка базы данных от несовместимых записей
                self.cleanup_database()
                
            except Exception as e:
                print(f"Ошибка при загрузке базы данных: {e}")
                self.face_database = {}
                self.persons_data = {}
                self.entry_history = []
                self.label_counter = 0
    
    def cleanup_database(self):
        """Очистка базы данных от записей с несовместимыми размерностями"""
        print("Проверка совместимости записей в базе данных...")
        
        # Получаем эталонную размерность из нового образца
        test_image = np.zeros((120, 120), dtype=np.uint8)
        reference_features = self.extract_features(test_image)
        reference_size = len(reference_features)
        
        cleaned_database = {}
        removed_count = 0
        
        for name, feature_list in self.face_database.items():
            cleaned_features = []
            
            for features in feature_list:
                if len(features) == reference_size:
                    cleaned_features.append(features)
                else:
                    removed_count += 1
                    print(f"Удалена несовместимая запись для {name}: размер {len(features)} vs {reference_size}")
            
            if cleaned_features:
                cleaned_database[name] = cleaned_features
            else:
                print(f"Удален пользователь {name} - не осталось совместимых записей")
                if name in self.persons_data:
                    del self.persons_data[name]
        
        self.face_database = cleaned_database
        
        if removed_count > 0:
            print(f"Удалено {removed_count} несовместимых записей")
            self.save_database()
        else:
            print("Все записи совместимы")
    
    def get_all_persons(self):
        """Получение списка всех людей в базе данных"""
        return list(self.face_database.keys())
    
    def delete_person(self, name):
        """Удаление человека из базы данных"""
        if name in self.face_database:
            del self.face_database[name]
            
        if name in self.persons_data:
            del self.persons_data[name]
            
        # Сохраняем изменения
        self.save_database()
        
        return True
    
    def get_stored_face_images(self, name):
        """Получение сохраненных изображений лиц для человека"""
        if name in self.persons_data and 'face_images' in self.persons_data[name]:
            return self.persons_data[name]['face_images']
        return []
    
    def update_person_info(self, old_name, new_info):
        """Обновление информации о человеке"""
        if old_name in self.face_database and old_name in self.persons_data:
            # Сохраняем старые данные лица
            old_features = self.face_database[old_name].copy()
            old_data = self.persons_data[old_name].copy()
            
            # Удаляем старые записи если имя изменилось
            if old_name != new_info['name']:
                del self.face_database[old_name]
                del self.persons_data[old_name]
            
            # Создаем новые записи
            self.face_database[new_info['name']] = old_features
            self.persons_data[new_info['name']] = old_data
            
            # Обновляем информацию
            self.persons_data[new_info['name']]['info'].update({
                'name': new_info['name'],
                'position': new_info['position'],
                'age': new_info['age'],
                'rank': new_info['rank']
            })
            
            # Сохраняем изменения
            self.save_database()
            return True
        return False
