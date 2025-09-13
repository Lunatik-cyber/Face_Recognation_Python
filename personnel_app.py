# -*- coding: utf-8 -*-
"""
Система учета персонала с распознаванием лиц
Включает полную информацию о сотрудниках и историю входов/выходов
Оптимизирована для работы на слабых процессорах
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import cv2
import numpy as np
import os
from PIL import Image, ImageTk
import threading
import time
from simple_face_recognizer import SimpleFaceRecognizer


class PersonnelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Система учета персонала - Распознавание лиц")
        self.root.geometry("1200x800")
        
        # Инициализация распознавателя лиц
        self.face_recognizer = SimpleFaceRecognizer()
        
        # Настройки языка
        self.current_language = "ru"  # По умолчанию русский
        
        # Словари переводов
        self.translations = {
            "ru": {
                # Заголовки окон
                "main_title": "Система учета персонала - Распознавание лиц",
                "add_person_title": "Добавить сотрудника",
                "add_from_photo_title": "Добавить из фото",
                "identify_title": "Определение лица",
                "edit_person_title": "Редактировать сотрудника",
                "person_info_title": "Информация о сотруднике",
                "history_title": "История входов/выходов",
                
                # Основные элементы интерфейса
                "camera_control": "Управление камерой",
                "start_camera": "Включить камеру",
                "stop_camera": "Выключить камеру",
                "camera": "Камера:",
                "recognition_on": "🔍 Распознавание ВКЛ",
                "recognition_off": "🔍 Распознавание ВЫКЛ",
                "language": "Язык:",
                
                # Основные действия
                "main_actions": "Основные действия",
                "add_face": "➕ Добавить лицо",
                "identify_face": "👤 Определить лицо",
                "add_from_photo": "📷 Добавить из фото",
                "update_list": "🔄 Обновить список",
                "edit_person": "✏️ Редактировать",
                "person_info": "ℹ️ Инфо о пользователе",
                "delete_employee": "🗑️ Удалить сотрудника",
                "show_history": "📊 История входов/выходов",
                
                # Информация и списки
                "current_face_info": "Информация о распознанном лице",
                "personnel_list": "Список сотрудников",
                "full_name": "ФИО",
                "position": "Должность",
                "age": "Возраст",
                "rank": "Звание",
                "status": "Статус",
                "procentage": "Уверенность распознавания",
                "validality": "Уверенность",
                "name": "Имя",
                
                # Формы ввода
                "enter_full_name": "ФИО:",
                "enter_position": "Должность:",
                "enter_age": "Возраст:",
                "enter_rank": "Звание:",
                "save": "Сохранить",
                "cancel": "Отмена",
                "select_photo": "Выбрать фото",
                
                # Статусы и действия
                "entered": "Вошел",
                "exited": "Вышел",
                "entry_time": "Время входа",
                "exit_time": "Время выхода",
                "unknown": "Неизвестный",
                "face_detected": "Лицо обнаружено",
                "person_detected": "Человек обнаружен",
                
                # Сообщения
                "success": "Успех",
                "error": "Ошибка",
                "warning": "Предупреждение",
                "information": "Информация",
                "camera_on": "Камера включена",
                "camera_off": "Камера выключена",
                "camera_not_active": "Камера не активна",
                "no_face_detected": "Лицо не обнаружено! Посмотрите в камеру.",
                "person_added": "Сотрудник добавлен успешно!",
                "person_updated": "Информация обновлена!",
                "select_person": "Выберите сотрудника из списка",
                "confirm_delete": "Вы уверены что хотите удалить этого сотрудника?",
                
                # Дополнительные переводы
                "choose_photo": "Выберите фотографию",
                "face_not_found": "Лицо не найдено на фотографии",
                "confidence": "Уверенность",
                "time": "Время",
                "date": "Дата",
                "last_seen": "Последний раз виден",
                "total_entries": "Всего входов",
                "current_status": "Текущий статус",
                "photo": "Фото",
                "no_photo": "Нет фото",
                "employee_info": "Информация о сотруднике",
                "entry_history": "История входов",
                "edit": "Редактировать",
                "delete": "Удалить",
                "close": "Закрыть",
                "apply": "Применить",
                "refresh": "Обновить",
                "loading": "Загрузка...",
                "please_wait": "Пожалуйста, подождите",
                "not_found": "Не найдено",
                "invalid_data": "Неверные данные",
                "operation_completed": "Операция завершена",
                "operation_failed": "Операция не удалась",
                "person_added_db": "добавлен в базу данных!",
                "person_added_photo": "добавлен из фотографии!",
                "entry_marked": "Вход отмечен!",
                "exit_marked": "Выход отмечен!",
                "person_deleted": "удален!",
                "info_updated": "Информация о сотруднике обновлена!",
                "fill_all_fields": "Заполните все поля!",
                "age_must_be_number": "Возраст должен быть числом!",
                "failed_add_person": "Не удалось добавить сотрудника!",
                "no_faces_found": "На изображении не найдено лиц!",
                "failed_get_frame": "Не удалось получить кадр с камеры!",
                "failed_record_entry": "Не удалось записать вход!",
                "failed_record_exit": "Не удалось записать выход!",
                "failed_open_camera": "Не удалось открыть камеру",
                "select_employee_delete": "Выберите сотрудника для удаления!",
                "select_employee_edit": "Выберите сотрудника для редактирования!",
                "select_employee_info": "Выберите сотрудника для просмотра информации!",
                "employee_info_not_found": "Информация о сотруднике не найдена!",
                "failed_update_info": "Не удалось обновить информацию!",
                "confirmation": "Подтверждение",
                "delete_employee_confirm": "Удалить сотрудника",
                "failed_add_employee": "Не удалось добавить сотрудника:",
                "camera_inactive": "Камера не активна!",
                "face_recognition_failed": "Не удалось распознать лицо или человек не найден в базе данных.",
                "choose_action": "Выберите действие",
                "position_label": "Должность",
                "age_label": "Возраст",
                "years_old": "лет",
                "rank_label": "Звание", 
                "current_status": "Текущий статус",
                "file_images": "Изображения",
                "file_jpeg": "JPEG файлы",
                "file_png": "PNG файлы",
                "file_all": "Все файлы",
                "detected_face": "Обнаруженное лицо:",
                "save_button": "Сохранить",
                "cancel_button": "Отмена",
                "multiple_faces_warning": "На изображении найдено",
                "faces_found": "лиц. Будет использовано первое обнаруженное лицо.",
                "edit_title": "Редактировать:",
                "edit_data": "Редактирование данных:",
                "apply_button": "Применить",
                "close_button": "Закрыть",
                "info_title": "Информация о сотруднике:",
                "main_info": "Основная информация",
                "photo_from_db": "Фотография из базы данных",
                "last_entries": "Последние входы/выходы",
                "no_history": "История не найдена",
                "entry_action": "Вошел",
                "exit_action": "Вышел",
                "identification": "Идентификация:",
                "employee": "Сотрудник:",
                "confidence_label": "Уверенность:",
                "photo_from_database": "Фото из базы данных",
                "employee_info": "Информация о сотруднике",
                "action_buttons": "Выберите действие",
                "entry_button": "🟢 Вошел",
                "exit_button": "🔴 Вышел",
                "cancel_action": "❌ Отмена",
                "history_window_title": "История входов/выходов",
                "column_name": "ФИО",
                "column_action": "Действие", 
                "column_time": "Время"
            },
            
            "tm": {
                # Заголовки окон
                "main_title": "Işgärleriň hasabat ulgamy - Ýüz tanamak",
                "add_person_title": "Işgär goşmak",
                "add_from_photo_title": "Suratdan goşmak",
                "identify_title": "Ýüz kesgitlemek",
                "edit_person_title": "Işgäri üýtgetmek",
                "person_info_title": "Işgär barada maglumat",
                "history_title": "Giriş/çykyş taryhy",
                
                # Основные элементы интерфейса
                "camera_control": "Kamera dolandyryş",
                "start_camera": "Kamera açmak",
                "stop_camera": "Kamera ýapmak",
                "camera": "Kamera:",
                "recognition_on": "🔍 Ýüz tanamak AÇYK",
                "recognition_off": "🔍 Ýüz tanamak ÝAPYK",
                "language": "Dil:",
                
                # Основные действия
                "main_actions": "Esasy hereketler",
                "add_face": "➕ Ýüz goşmak",
                "identify_face": "👤 Ýüz kesgitlemek",
                "add_from_photo": "📷 Suratdan goşmak",
                "update_list": "🔄 Sanawy täzelemek",
                "edit_person": "✏️ Üýtgetmek",
                "person_info": "ℹ️ Ulanyjy barada",
                "delete_employee": "🗑️ Işgäri pozmak",
                "show_history": "📊 Giriş/çykyş taryhy",
                
                # Информация и списки
                "current_face_info": "Tanalanan ýüz barada maglumat",
                "personnel_list": "Işgärler sanawy",
                "full_name": "Ady-familiýasy",
                "position": "Wezipesi",
                "age": "Ýaşy",
                "rank": "Derejesi",
                "status": "Ýagdaýy",
                "procentage": "Ynanyş derejesi",
                "validality": "Ynamlylyk",
                "name": "Ady",
                
                # Формы ввода
                "enter_full_name": "Ady-familiýasy:",
                "enter_position": "Wezipesi:",
                "enter_age": "Ýaşy:",
                "enter_rank": "Derejesi:",
                "save": "Ýatda saklamak",
                "cancel": "Ýatyrmak",
                "select_photo": "Surat saýlamak",
                
                # Статусы и действия
                "entered": "Girdi",
                "exited": "Çykdy",
                "entry_time": "Giriş wagty",
                "exit_time": "Çykyş wagty",
                "unknown": "Näbelli",
                "face_detected": "Ýüz tapyldy",
                "person_detected": "Adam tapyldy",
                
                # Сообщения
                "success": "Üstünlik",
                "error": "Ýalňyşlyk",
                "warning": "Duýduryş",
                "information": "Maglumat",
                "camera_on": "Kamera açyldy",
                "camera_off": "Kamera ýapyldy",
                "camera_not_active": "Kamera işlemeýär",
                "no_face_detected": "Ýüz tapylmady! Kamera tarap serediň.",
                "person_added": "Işgär üstünlikli goşuldy!",
                "person_updated": "Maglumat täzelendi!",
                "select_person": "Sanawdan işgär saýlaň",
                "confirm_delete": "Bu işgäri pozmak isleýärsiňizmi?",
                
                # Дополнительные переводы
                "choose_photo": "Surat saýlaň",
                "face_not_found": "Suratda ýüz tapylmady",
                "confidence": "Ynam",
                "time": "Wagt",
                "date": "Sene",
                "last_seen": "Soňky görlen",
                "total_entries": "Jemi girişler",
                "current_status": "Häzirki ýagdaý",
                "photo": "Surat",
                "no_photo": "Surat ýok",
                "employee_info": "Işgär barada maglumat",
                "entry_history": "Giriş taryhy",
                "edit": "Üýtgetmek",
                "delete": "Pozmak",
                "close": "Ýapmak",
                "apply": "Sakla",
                "refresh": "Täzelemek",
                "loading": "Ýüklenýär...",
                "please_wait": "Garaşyň",
                "not_found": "Tapylmady",
                "invalid_data": "Nädogry maglumat",
                "operation_completed": "Amal tamamlandy",
                "operation_failed": "Amal şowsuz",
                "person_added_db": "maglumat bazasyna goşuldy!",
                "person_added_photo": "suratdan goşuldy!",
                "entry_marked": "Giriş bellenildi!",
                "exit_marked": "Çykyş bellenildi!",
                "person_deleted": "pozuldy!",
                "info_updated": "Işgär barada maglumat täzelendi!",
                "fill_all_fields": "Ähli meýdanlary dolduryň!",
                "age_must_be_number": "Ýaş san bolmaly!",
                "failed_add_person": "Işgär goşup bolmady!",
                "no_faces_found": "Suratda ýüz tapylmady!",
                "failed_get_frame": "Kameradan kadr alyp bolmady!",
                "failed_record_entry": "Girişi ýazmak bolmady!",
                "failed_record_exit": "Çykyşy ýazmak bolmady!",
                "failed_open_camera": "Kamera açyp bolmady",
                "select_employee_delete": "Pozmak üçin işgär saýlaň!",
                "select_employee_edit": "Üýtgetmek üçin işgär saýlaň!",
                "select_employee_info": "Maglumat görmek üçin işgär saýlaň!",
                "employee_info_not_found": "Işgär barada maglumat tapylmady!",
                "failed_update_info": "Maglumat täzeläp bolmady!",
                "confirmation": "Tassyklamak",
                "delete_employee_confirm": "Işgäri pozmak",
                "failed_add_employee": "Işgär goşup bolmady:",
                "camera_inactive": "Kamera işlemeýär!",
                "face_recognition_failed": "Ýüz tanamak bolmady ýa-da adam maglumat bazasynda tapylmady.",
                "choose_action": "Hereketini saýlaň",
                "position_label": "Wezipesi",
                "age_label": "Ýaşy",
                "years_old": "ýaş",
                "rank_label": "Derejesi",
                "current_status": "Häzirki ýagdaýy",
                "file_images": "Suratlar",
                "file_jpeg": "JPEG faýllar",
                "file_png": "PNG faýllar",
                "file_all": "Ähli faýllar",
                "detected_face": "Tapylan ýüz:",
                "save_button": "Ýatda saklamak",
                "cancel_button": "Ýatyrmak",
                "multiple_faces_warning": "Suratda tapyldy",
                "faces_found": "ýüz. Ilkinji tapylan ýüz ulanylar.",
                "edit_title": "Üýtgetmek:",
                "edit_data": "Maglumatlary üýtgetmek:",
                "apply_button": "Ulanmak",
                "close_button": "Ýapmak",
                "info_title": "Işgär barada maglumat:",
                "main_info": "Esasy maglumat",
                "photo_from_db": "Maglumat bazasyndan surat",
                "last_entries": "Soňky girişler/çykyşlar",
                "no_history": "Taryh tapylmady",
                "entry_action": "Girdi",
                "exit_action": "Çykdy",
                "identification": "Kesgitlemek:",
                "employee": "Işgär:",
                "confidence_label": "Ynam:",
                "photo_from_database": "Maglumat bazasyndan surat",
                "employee_info": "Işgär barada maglumat",
                "action_buttons": "Hereketini saýlaň",
                "entry_button": "🟢 Girdi",
                "exit_button": "🔴 Çykdy",
                "cancel_action": "❌ Ýatyrmak",
                "history_window_title": "Girişler/çykyşlar taryhy",
                "column_name": "Ady-familiýasy",
                "column_action": "Hereket",
                "column_time": "Wagt"
            }
        }
        
        # Переменные для камеры
        self.cap = None
        self.camera_active = False
        self.current_camera = 0
        self.detected_faces = []
        self.recognition_active = True
        
        # Создание интерфейса
        self.create_widgets()
        
        # Запуск камеры по умолчанию
        self.start_camera()
        
        # Запуск потока обновления видео
        self.update_thread = threading.Thread(target=self.update_frame, daemon=True)
        self.update_thread.start()
    
    def get_text(self, key):
        """Получить перевод для текущего языка"""
        return self.translations[self.current_language].get(key, key)
    
    def change_language(self, language_code):
        """Смена языка интерфейса"""
        if language_code in self.translations:
            self.current_language = language_code
            self.refresh_interface()
    
    def refresh_interface(self):
        """Полное обновление интерфейса после смены языка"""
        # Обновляем заголовок окна
        self.root.title(self.get_text("main_title"))
        
        # Останавливаем камеру для безопасности
        was_camera_active = self.cap and self.cap.isOpened()
        if was_camera_active:
            self.stop_camera()
        
        # Очищаем текущий интерфейс
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Пересоздаем интерфейс
        self.create_widgets()
        
        # Восстанавливаем камеру если была активна
        if was_camera_active:
            self.start_camera()
    

    def create_widgets(self):
        """Создание улучшенного интерфейса"""
        
        # Основные фреймы
        left_frame = ttk.Frame(self.root, width=600)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        right_frame = ttk.Frame(self.root, width=600)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # === ЛЕВАЯ ЧАСТЬ - ВИДЕО И УПРАВЛЕНИЕ ===
        
        # Панель управления камерой
        camera_frame = ttk.LabelFrame(left_frame, text=self.get_text("camera_control"))
        camera_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Первая строка кнопок камеры
        camera_row1 = ttk.Frame(camera_frame)
        camera_row1.pack(fill=tk.X, pady=5)
        
        ttk.Button(camera_row1, text=self.get_text("start_camera"), 
                  command=self.start_camera).pack(side=tk.LEFT, padx=5)
        ttk.Button(camera_row1, text=self.get_text("stop_camera"), 
                  command=self.stop_camera).pack(side=tk.LEFT, padx=5)
        
        # Выбор камеры
        ttk.Label(camera_row1, text=self.get_text("camera")).pack(side=tk.LEFT, padx=(10, 5))
        self.camera_var = tk.StringVar(value="0")
        camera_combo = ttk.Combobox(camera_row1, textvariable=self.camera_var, 
                                   values=["0", "1", "2", "3"], width=5)
        camera_combo.pack(side=tk.LEFT, padx=5)
        camera_combo.bind('<<ComboboxSelected>>', self.change_camera)
        
        # Выбор языка
        ttk.Label(camera_row1, text=self.get_text("language")).pack(side=tk.LEFT, padx=(10, 5))
        self.language_var = tk.StringVar(value="ru")
        language_combo = ttk.Combobox(camera_row1, textvariable=self.language_var, 
                                     values=["ru", "tm"], width=5)
        language_combo.pack(side=tk.LEFT, padx=5)
        language_combo.bind('<<ComboboxSelected>>', self.on_language_change)
        
        # Переключатели (Toggle кнопки)
        toggles_frame = ttk.Frame(camera_row1)
        toggles_frame.pack(side=tk.RIGHT, padx=10)
        
        # Toggle для автоматического распознавания
        self.recognition_var = tk.BooleanVar(value=True)
        self.recognition_toggle_btn = ttk.Button(toggles_frame, text=self.get_text("recognition_on"), 
                                               command=self.toggle_recognition, width=20)
        self.recognition_toggle_btn.pack(side=tk.LEFT, padx=2)
        
        # Видео область
        self.video_label = ttk.Label(left_frame, text=self.get_text("camera_not_active"), 
                                    background="black", foreground="white")
        self.video_label.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # === ГЛАВНЫЕ КНОПКИ ДЕЙСТВИЙ (перенесены наверх правой части) ===
        main_actions_frame = ttk.LabelFrame(right_frame, text=self.get_text("main_actions"))
        main_actions_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Первая строка основных действий
        actions_row1 = ttk.Frame(main_actions_frame)
        actions_row1.pack(fill=tk.X, pady=2)

        btn_width = 22  # Одинаковая ширина для всех кнопок

        ttk.Button(actions_row1, text=self.get_text("add_face"),
              command=self.add_person_dialog, style="Action.TButton", width=btn_width).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(actions_row1, text=self.get_text("identify_face"),
              command=self.identify_person_dialog, style="Action.TButton", width=btn_width).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Вторая строка основных действий
        actions_row2 = ttk.Frame(main_actions_frame)
        actions_row2.pack(fill=tk.X, pady=2)

        ttk.Button(actions_row2, text=self.get_text("add_from_photo"),
              command=self.add_from_photo_dialog, style="Action.TButton", width=btn_width).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(actions_row2, text=self.get_text("update_list"),
              command=self.update_personnel_list, style="Action.TButton", width=btn_width).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Третья строка основных действий
        actions_row3 = ttk.Frame(main_actions_frame)
        actions_row3.pack(fill=tk.X, pady=2)

        ttk.Button(actions_row3, text=self.get_text("edit_person"),
              command=self.edit_person_dialog, style="Action.TButton", width=btn_width).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(actions_row3, text=self.get_text("person_info"),
              command=self.show_person_info, style="Action.TButton", width=btn_width).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Четвертая строка основных действий
        actions_row4 = ttk.Frame(main_actions_frame)
        actions_row4.pack(fill=tk.X, pady=2)

        ttk.Button(actions_row4, text=self.get_text("delete_employee"),
              command=self.delete_person, style="Delete.TButton", width=btn_width).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(actions_row4, text=self.get_text("show_history"),
              command=self.show_history, style="Action.TButton", width=btn_width).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        # Информация о текущем лице
        current_info_frame = ttk.LabelFrame(right_frame, text=self.get_text("current_face_info"))
        current_info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.current_info_text = tk.Text(current_info_frame, height=10, state=tk.DISABLED, 
                                        font=("Arial", 9))
        self.current_info_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Список всех сотрудников
        personnel_frame = ttk.LabelFrame(right_frame, text=self.get_text("personnel_list"))
        personnel_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview для отображения сотрудников
        columns = (self.get_text("full_name"), self.get_text("position"), 
                  self.get_text("age"), self.get_text("rank"), self.get_text("status"))
        self.personnel_tree = ttk.Treeview(personnel_frame, columns=columns, show="headings", height=6)
        
        for col in columns:
            self.personnel_tree.heading(col, text=col)
            self.personnel_tree.column(col, width=100)
        
        personnel_scrollbar = ttk.Scrollbar(personnel_frame, orient=tk.VERTICAL, 
                                          command=self.personnel_tree.yview)
        self.personnel_tree.configure(yscrollcommand=personnel_scrollbar.set)
        
        self.personnel_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        personnel_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Настройка стилей кнопок
        style = ttk.Style()
        style.configure("Action.TButton", font=("Arial", 9, "bold"))
        style.configure("Delete.TButton", font=("Arial", 9, "bold"), foreground="red")
        style.configure("Success.TButton", font=("Arial", 9, "bold"), foreground="green")
        style.configure("Warning.TButton", font=("Arial", 9, "bold"), foreground="orange")
        
        # Обновляем список сотрудников
        self.update_personnel_list()
        
        # Инициализируем стили toggle кнопок
        self.recognition_toggle_btn.configure(style="Success.TButton")
    
    def center_window(self, window, width, height):
        """Центрирование окна относительно главного окна"""
        # Получаем размеры главного окна
        self.root.update_idletasks()
        main_x = self.root.winfo_x()
        main_y = self.root.winfo_y()
        main_width = self.root.winfo_width()
        main_height = self.root.winfo_height()
        
        # Вычисляем позицию для центрирования
        x = main_x + (main_width - width) // 2
        y = main_y + (main_height - height) // 2
        
        # Устанавливаем размер и позицию
        window.geometry(f"{width}x{height}+{x}+{y}")
    
    def start_camera(self):
        """Запуск камеры"""
        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.current_camera)
            if self.cap.isOpened():
                # Камера уже запущена
                messagebox.showinfo(self.get_text("success"), self.get_text("camera_on"))
            else:
                messagebox.showerror(self.get_text("error"), f"{self.get_text('failed_open_camera')} {self.current_camera}")
    
    def stop_camera(self):
        """Остановка камеры"""
        if self.cap:
            # Камера остановлена
            self.cap.release()
            self.video_label.configure(image="", text=self.get_text("camera_not_active"))
            messagebox.showinfo(self.get_text("information"), self.get_text("camera_off"))
    
    def toggle_recognition(self):
        """Переключатель автоматического распознавания"""
        current_state = self.recognition_var.get()
        self.recognition_var.set(not current_state)
        
        if self.recognition_var.get():
            # Включаем распознавание
            self.recognition_toggle_btn.configure(text=self.get_text("recognition_on"), style="Success.TButton")
        else:
            # Выключаем распознавание
            self.recognition_toggle_btn.configure(text=self.get_text("recognition_off"), style="Warning.TButton")
            self.clear_current_info()
    
    def on_recognition_toggle(self, *args):
        """Вызывается при изменении статуса автоматического распознавания"""
        if not self.recognition_var.get():
            # При отключении автоматического распознавания очищаем информацию
            self.clear_current_info()
    
    def change_camera(self, event=None):
        """Смена камеры"""
        if self.cap and self.cap.isOpened():
            self.stop_camera()
        
        self.current_camera = int(self.camera_var.get())
        
        # Автоматически запускаем камеру после смены
        self.start_camera()
    
    def on_language_change(self, event=None):
        """Обработка смены языка"""
        new_language = self.language_var.get()
        if new_language != self.current_language:
            self.change_language(new_language)
    
    def put_text_utf8(self, image, text, position, font_scale=2, color=(255, 255, 255), thickness=2):
        """Безопасное отображение UTF-8 текста на изображении с поддержкой кириллицы"""
        try:
            # Пытаемся создать изображение с текстом через PIL
            from PIL import Image as PILImage, ImageDraw, ImageFont
            import numpy as np
            
            # Конвертируем BGR в RGB
            if len(image.shape) == 3:
                pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = PILImage.fromarray(image)
            
            draw = ImageDraw.Draw(pil_image)
            
            # Пытаемся загрузить системный шрифт
            try:
                # Для Windows
                font = ImageFont.truetype("arial.ttf", int(38 * font_scale))
            except:
                try:
                    # Альтернативный шрифт для Windows
                    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", int(38 * font_scale))
                except:
                    # Стандартный шрифт PIL
                    font = ImageFont.load_default()
            
            # Рисуем текст
            x, y = position
            draw.text((x, y - 20), text, font=font, fill=color[::-1])  # BGR -> RGB
            
            # Конвертируем обратно в OpenCV формат
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Копируем результат обратно в оригинальное изображение
            image[:] = cv_image[:]
            
        except Exception as e:
            # Если PIL не работает, используем транслитерацию
            try:
                # Простая транслитерация русских символов
                transliteration = {
                    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo',
                    'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
                    'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
                    'ф': 'f', 'х': 'h', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'sch',
                    'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
                    'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': 'YO',
                    'Ж': 'ZH', 'З': 'Z', 'И': 'I', 'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M',
                    'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U',
                    'Ф': 'F', 'Х': 'H', 'Ц': 'TS', 'Ч': 'CH', 'Ш': 'SH', 'Щ': 'SCH',
                    'Ъ': '', 'Ы': 'Y', 'Ь': '', 'Э': 'E', 'Ю': 'YU', 'Я': 'YA',
                    ' ': ' ', '.': '.', ',': ',', '!': '!', '?': '?', ':': ':', ';': ';',
                    '(': '(', ')': ')', '-': '-', '_': '_'
                }
                
                transliterated_text = ''
                for char in text:
                    if char in transliteration:
                        transliterated_text += transliteration[char]
                    elif char.isascii():
                        transliterated_text += char
                    else:
                        transliterated_text += '?'
                
                cv2.putText(image, transliterated_text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
                
            except:
                # В крайнем случае выводим заглушку
                cv2.putText(image, "Person detected", position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    def update_frame(self):
        """Оптимизированное обновление кадра видео для слабых процессоров"""
        frame_count = 0
        recognition_interval = 5  # Распознавание каждый 5-й кадр для производительности
        
        while True:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Отражаем изображение
                    frame = cv2.flip(frame, 1)
                    
                    # Уменьшаем разрешение для обработки
                    processing_frame = cv2.resize(frame, (320, 240))
                    
                    # Детекция и распознавание только если включено автоматическое распознавание
                    if self.recognition_var.get() and frame_count % recognition_interval == 0:
                        self.detected_faces = self.face_recognizer.detect_faces(processing_frame)
                        
                        # Масштабируем координаты обратно к оригинальному размеру
                        scale_x = frame.shape[1] / 320
                        scale_y = frame.shape[0] / 240
                        
                        for face in self.detected_faces:
                            x, y, w, h = face['coordinates']
                            x = int(x * scale_x)
                            y = int(y * scale_y)
                            w = int(w * scale_x)
                            h = int(h * scale_y)
                            face['coordinates'] = (x, y, w, h)
                            
                            # Распознавание лица
                            name, confidence = self.face_recognizer.recognize_face(face['face_roi'])
                            face['name'] = name
                            face['confidence'] = confidence
                    
                    # Если автоматическое распознавание отключено, только детектируем лица
                    elif not self.recognition_var.get() and frame_count % recognition_interval == 0:
                        self.detected_faces = self.face_recognizer.detect_faces(processing_frame)
                        
                        # Масштабируем координаты обратно к оригинальному размеру
                        scale_x = frame.shape[1] / 320
                        scale_y = frame.shape[0] / 240
                        
                        for face in self.detected_faces:
                            x, y, w, h = face['coordinates']
                            x = int(x * scale_x)
                            y = int(y * scale_y)
                            w = int(w * scale_x)
                            h = int(h * scale_y)
                            face['coordinates'] = (x, y, w, h)
                            # Без распознавания - только детекция
                            face['name'] = None
                            face['confidence'] = 0.0
                    
                    # Очищаем детекцию если автоматическое распознавание отключено и время прошло
                    elif not self.recognition_var.get():
                        if hasattr(self, 'detected_faces'):
                            # Очищаем старые результаты распознавания
                            for face in self.detected_faces:
                                face['name'] = None
                                face['confidence'] = 0.0
                    
                    # Рисуем прямоугольники и информацию
                    if hasattr(self, 'detected_faces') and self.detected_faces:
                        for face in self.detected_faces:
                            x, y, w, h = face['coordinates']
                            
                            # Цвет рамки зависит от статуса распознавания
                            if self.recognition_var.get():
                                color = (0, 255, 0) if face.get('name') else (0, 0, 255)
                            else:
                                color = (255, 255, 0)  # Желтая рамка когда распознавание отключено
                            
                            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                            
                            # Показываем информацию только если автоматическое распознавание включено
                            if self.recognition_var.get():
                                name = face.get('name')
                                confidence = face.get('confidence', 0.0)
                                
                                if name and confidence > 0.65:
                                    # Отображаем имя и уверенность
                                    self.put_text_utf8(frame, f"{name} ({confidence:.2f})", 
                                                     (x, y - 10), 0.6, (0, 255, 0), 2)
                                    
                                    # Обновляем информацию о текущем лице
                                    self.update_current_info(name, confidence)
                                else:
                                    self.put_text_utf8(frame, self.get_text("unknown"), 
                                                     (x, y - 10), 0.6, (0, 0, 255), 2)
                            else:
                                # Показываем только что лицо обнаружено
                                self.put_text_utf8(frame, self.get_text("face_detected"), 
                                                 (x, y - 10), 0.6, (255, 255, 0), 2)
                        
                        # Очищаем информацию периодически если автораспознавание выключено
                        if not self.recognition_var.get():
                            if frame_count % recognition_interval == 0:
                                self.clear_current_info()
                    
                    # Конвертируем для отображения в tkinter с пониженным качеством для производительности
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (580, 400))
                    img = Image.fromarray(frame_resized)
                    photo = ImageTk.PhotoImage(image=img)
                    
                    # Обновляем изображение
                    self.video_label.configure(image=photo, text="")
                    self.video_label.image = photo
                    
                    frame_count += 1
            
            # Увеличиваем задержку для снижения нагрузки на процессор
            time.sleep(0.05)  # ~20 FPS вместо 30 для экономии ресурсов
    
    def add_person_dialog(self):
        """Диалог добавления нового сотрудника"""
        if not self.detected_faces:
            messagebox.showwarning(self.get_text("warning"), self.get_text("no_face_detected"))
            return
        
        # Создаем диалоговое окно
        dialog = tk.Toplevel(self.root)
        dialog.title(self.get_text("add_person_title"))
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Центрируем окно
        self.center_window(dialog, 500, 500)
        
        # Поля ввода
        ttk.Label(dialog, text=self.get_text("enter_full_name")).pack(pady=5)
        name_entry = ttk.Entry(dialog, width=40)
        name_entry.pack(pady=5)
        
        ttk.Label(dialog, text=self.get_text("enter_position")).pack(pady=5)
        position_entry = ttk.Entry(dialog, width=40)
        position_entry.pack(pady=5)
        
        ttk.Label(dialog, text=self.get_text("enter_age")).pack(pady=5)
        age_entry = ttk.Entry(dialog, width=40)
        age_entry.pack(pady=5)
        
        ttk.Label(dialog, text=self.get_text("enter_rank")).pack(pady=5)
        rank_entry = ttk.Entry(dialog, width=40)
        rank_entry.pack(pady=5)
        
        def save_person():
            name = name_entry.get().strip()
            position = position_entry.get().strip()
            age = age_entry.get().strip()
            rank = rank_entry.get().strip()
            
            if not all([name, position, age, rank]):
                messagebox.showerror(self.get_text("error"), self.get_text("fill_all_fields"))
                return
            
            try:
                age = int(age)
            except ValueError:
                messagebox.showerror(self.get_text("error"), self.get_text("age_must_be_number"))
                return
            
            # Добавляем сотрудника
            person_info = {
                'name': name,
                'position': position,
                'age': age,
                'rank': rank
            }
            
            face_roi = self.detected_faces[0]['face_roi']
            
            # Получаем текущий кадр для сохранения оригинального изображения
            original_frame = None
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    original_frame = cv2.flip(frame, 1)  # Отражаем как в интерфейсе
            
            success = self.face_recognizer.add_person(face_roi, person_info, original_frame)
            
            if success:
                messagebox.showinfo(self.get_text("success"), f"{name} {self.get_text('person_added_db')}")
                self.update_personnel_list()
                dialog.destroy()
            else:
                messagebox.showerror(self.get_text("error"), self.get_text("failed_add_person"))
        
        # Кнопки
        buttons_frame = ttk.Frame(dialog)
        buttons_frame.pack(pady=20)
        
        ttk.Button(buttons_frame, text=self.get_text("save"), command=save_person).pack(side=tk.LEFT, padx=10)
        ttk.Button(buttons_frame, text=self.get_text("cancel"), command=dialog.destroy).pack(side=tk.LEFT, padx=10)
    
    def add_from_photo_dialog(self):
        """Диалог добавления сотрудника из фотографии"""
        # Выбираем файл изображения
        file_path = filedialog.askopenfilename(
            title=self.get_text("choose_photo"),
            filetypes=[
                (self.get_text("file_images"), "*.jpg *.jpeg *.png *.bmp *.tiff"),
                (self.get_text("file_jpeg"), "*.jpg *.jpeg"),
                (self.get_text("file_png"), "*.png"),
                (self.get_text("file_all"), "*.*")
            ]
        )
        
        if not file_path:
            return
        
        # Проверяем, есть ли лица на изображении
        faces = self.face_recognizer.detect_faces_from_image(file_path)
        
        if not faces:
            messagebox.showerror(self.get_text("error"), self.get_text("no_faces_found"))
            return
        
        if len(faces) > 1:
            messagebox.showwarning(self.get_text("warning"), 
                                 f"{self.get_text('multiple_faces_warning')} {len(faces)} {self.get_text('faces_found')}")
        
        # Создаем диалоговое окно для ввода информации
        dialog = tk.Toplevel(self.root)
        dialog.title(self.get_text("add_from_photo_title"))
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Центрируем окно
        self.center_window(dialog, 400, 600)
        
        # Показываем миниатюру обнаруженного лица
        face_roi = faces[0]['face_roi']
        face_image = Image.fromarray(face_roi)
        face_image = face_image.resize((100, 100))
        face_photo = ImageTk.PhotoImage(face_image)
        
        ttk.Label(dialog, text=self.get_text("detected_face")).pack(pady=5)
        face_label = ttk.Label(dialog, image=face_photo)
        face_label.image = face_photo  # Сохраняем ссылку
        face_label.pack(pady=5)
        
        # Поля ввода
        ttk.Label(dialog, text=self.get_text("enter_full_name")).pack(pady=5)
        name_entry = ttk.Entry(dialog, width=40)
        name_entry.pack(pady=5)
        
        ttk.Label(dialog, text=self.get_text("enter_position")).pack(pady=5)
        position_entry = ttk.Entry(dialog, width=40)
        position_entry.pack(pady=5)
        
        ttk.Label(dialog, text=self.get_text("enter_age")).pack(pady=5)
        age_entry = ttk.Entry(dialog, width=40)
        age_entry.pack(pady=5)
        
        ttk.Label(dialog, text=self.get_text("enter_rank")).pack(pady=5)
        rank_entry = ttk.Entry(dialog, width=40)
        rank_entry.pack(pady=5)
        
        def save_person_from_photo():
            name = name_entry.get().strip()
            position = position_entry.get().strip()
            age = age_entry.get().strip()
            rank = rank_entry.get().strip()
            
            if not all([name, position, age, rank]):
                messagebox.showerror(self.get_text("error"), self.get_text("fill_all_fields"))
                return
            
            try:
                age = int(age)
            except ValueError:
                messagebox.showerror(self.get_text("error"), self.get_text("age_must_be_number"))
                return
            
            # Добавляем сотрудника
            person_info = {
                'name': name,
                'position': position,
                'age': age,
                'rank': rank
            }
            
            success, message = self.face_recognizer.add_person_from_image(file_path, person_info)
            
            if success:
                messagebox.showinfo(self.get_text("success"), f"{name} {self.get_text('person_added_photo')}")
                self.update_personnel_list()
                dialog.destroy()
            else:
                messagebox.showerror(self.get_text("error"), f"{self.get_text('failed_add_employee')} {message}")
        
        # Кнопки
        buttons_frame = ttk.Frame(dialog)
        buttons_frame.pack(pady=20)
        
        ttk.Button(buttons_frame, text=self.get_text("save_button"), command=save_person_from_photo).pack(side=tk.LEFT, padx=10)
        ttk.Button(buttons_frame, text=self.get_text("cancel_button"), command=dialog.destroy).pack(side=tk.LEFT, padx=10)
    
    def identify_person_dialog(self):
        """Улучшенный диалог идентификации лица для отметки входа/выхода"""
        # Получаем текущий кадр для распознавания
        if not self.cap or not self.cap.isOpened():
            messagebox.showwarning(self.get_text("warning"), self.get_text("camera_inactive"))
            return
            
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showwarning(self.get_text("warning"), self.get_text("failed_get_frame"))
            return
        
        # Отражаем изображение
        frame = cv2.flip(frame, 1)
        
        # Детектируем лица на текущем кадре
        detected_faces = self.face_recognizer.detect_faces(frame)
        
        if not detected_faces:
            messagebox.showwarning(self.get_text("warning"), self.get_text("no_face_detected"))
            return
        
        # Берем первое обнаруженное лицо
        face_roi = detected_faces[0]['face_roi']
        
        # Распознаем лицо с теми же параметрами что и в автоматическом режиме
        name, confidence = self.face_recognizer.recognize_face(face_roi)
        
        if not name or confidence < 0.65:  # Тот же порог что и в автоматическом режиме
            messagebox.showwarning(self.get_text("warning"), 
                                 f"{self.get_text('face_recognition_failed')} ({self.get_text('confidence')}: {confidence:.2%})")
            return
        
        # Диалог с фотографией и кнопками действий
        dialog = tk.Toplevel(self.root)
        dialog.title(f"{self.get_text('identification')} {name}")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Центрируем окно
        self.center_window(dialog, 400, 700)
        
        # Заголовок
        title_frame = ttk.Frame(dialog)
        title_frame.pack(pady=10)
        
        ttk.Label(title_frame, text=f"{self.get_text('employee')} {name}", font=("Arial", 14, "bold")).pack()
        ttk.Label(title_frame, text=f"{self.get_text('confidence_label')} {confidence:.2%}", font=("Arial", 10)).pack()
        
        # Фотография из базы данных
        photo_frame = ttk.LabelFrame(dialog, text=self.get_text("photo_from_database"))
        photo_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # Получаем сохраненное изображение лица из базы данных
        person_info = self.face_recognizer.get_person_info(name)
        photo_shown = False
        
        # Пытаемся показать сохраненную фотографию из файла
        if person_info and 'photo_path' in person_info:
            try:
                photo_path = person_info['photo_path']
                if os.path.exists(photo_path):
                    face_image = cv2.imread(photo_path)
                    if face_image is not None:
                        face_image = cv2.resize(face_image, (150, 150))
                        face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                        
                        face_pil = Image.fromarray(face_image_rgb)
                        face_photo = ImageTk.PhotoImage(face_pil)
                        
                        photo_label = ttk.Label(photo_frame, image=face_photo)
                        photo_label.image = face_photo  # Сохраняем ссылку
                        photo_label.pack(pady=10)
                        photo_shown = True
            except Exception as e:
                print(f"Ошибка загрузки фотографии: {e}")
        
        # Если не удалось показать фотографию из файла, пытаемся показать из памяти
        if not photo_shown:
            stored_faces = self.face_recognizer.get_stored_face_images(name)
            if stored_faces:
                # Берем первое сохраненное изображение
                face_image = stored_faces[0]
                face_image = cv2.resize(face_image, (150, 150))
                face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB) if len(face_image.shape) == 2 else face_image
                
                face_pil = Image.fromarray(face_image_rgb)
                face_photo = ImageTk.PhotoImage(face_pil)
                
                photo_label = ttk.Label(photo_frame, image=face_photo)
                photo_label.image = face_photo  # Сохраняем ссылку
                photo_label.pack(pady=10)
                photo_shown = True
        
        if not photo_shown:
            ttk.Label(photo_frame, text="Фото не найдено", font=("Arial", 10, "italic")).pack(pady=20)
        
        # Информация о сотруднике
        info_frame = ttk.LabelFrame(dialog, text=self.get_text("employee_info"))
        info_frame.pack(pady=10, padx=20, fill=tk.X)
        
        person_info = self.face_recognizer.get_person_info(name)
        if person_info:
            info_text = f"""{self.get_text('position_label')}: {person_info['position']}
{self.get_text('age_label')}: {person_info['age']} {self.get_text('years_old')}
{self.get_text('rank_label')}: {person_info['rank']}
{self.get_text('current_status')}: {person_info['status']}"""
            
            ttk.Label(info_frame, text=info_text, font=("Arial", 9), justify=tk.LEFT).pack(pady=10)
        
        # Кнопки действий
        action_frame = ttk.LabelFrame(dialog, text=self.get_text("choose_action"))
        action_frame.pack(pady=20, padx=20, fill=tk.X)
        
        # Переменные для результата
        result = {'action': None}
        
        def record_entry():
            result['action'] = 'Вошел'
            success = self.face_recognizer.record_entry_exit(name, 'Вошел')
            if success:
                messagebox.showinfo(self.get_text("success"), f"{name} - {self.get_text('entry_marked')}")
                self.update_personnel_list()
                self.update_current_info(name, confidence)
                dialog.destroy()
            else:
                messagebox.showerror(self.get_text("error"), self.get_text("failed_record_entry"))
        
        def record_exit():
            result['action'] = 'Вышел'
            success = self.face_recognizer.record_entry_exit(name, 'Вышел')
            if success:
                messagebox.showinfo(self.get_text("success"), f"{name} - {self.get_text('exit_marked')}")
                self.update_personnel_list()
                self.update_current_info(name, confidence)
                dialog.destroy()
            else:
                messagebox.showerror(self.get_text("error"), self.get_text("failed_record_exit"))
        
        def cancel_action():
            result['action'] = 'Отмена'
            dialog.destroy()
        
        # Размещение кнопок
        buttons_frame = ttk.Frame(action_frame)
        buttons_frame.pack(pady=15)
        
        ttk.Button(buttons_frame, text=self.get_text("entry_button"), command=record_entry, 
                  style="Success.TButton", width=12).pack(side=tk.TOP, pady=5, fill=tk.X)
        ttk.Button(buttons_frame, text=self.get_text("exit_button"), command=record_exit, 
                  style="Danger.TButton", width=12).pack(side=tk.TOP, pady=5, fill=tk.X)
        ttk.Button(buttons_frame, text=self.get_text("cancel_action"), command=cancel_action, 
                  style="Default.TButton", width=12).pack(side=tk.TOP, pady=5, fill=tk.X)
        
        # Настройка стилей кнопок
        style = ttk.Style()
        style.configure("Success.TButton", font=("Arial", 10, "bold"))
        style.configure("Danger.TButton", font=("Arial", 10, "bold"))
        style.configure("Default.TButton", font=("Arial", 10))
    
    def update_current_info(self, name, confidence):
        """Обновление информации о текущем лице"""
        person_info = self.face_recognizer.get_person_info(name)
        
        if person_info:
            info_text = f"""
{self.get_text("full_name")}: {person_info['name']}
{self.get_text("position")}: {person_info['position']}
{self.get_text("age")}: {person_info['age']} лет
{self.get_text("rank")}: {person_info['rank']}
{self.get_text("status")}: {person_info['status']}
{self.get_text("entry_time")}: {person_info['entry_time'] or 'Не отмечено'}
{self.get_text("exit_time")}: {person_info['exit_time'] or 'Не отмечено'}
{self.get_text("procentage")}: {confidence:.2%}
"""
        else:
            info_text = f"{self.get_text('name')}: {name}\n{self.get_text('validality')}: {confidence:.2%}\n{self.get_text('employee_info_not_found')}"
        
        self.current_info_text.config(state=tk.NORMAL)
        self.current_info_text.delete(1.0, tk.END)
        self.current_info_text.insert(1.0, info_text)
        self.current_info_text.config(state=tk.DISABLED)
    
    def clear_current_info(self):
        """Очистка информации о текущем лице"""
        self.current_info_text.config(state=tk.NORMAL)
        self.current_info_text.delete(1.0, tk.END)
        self.current_info_text.insert(1.0, f"{self.get_text("unknown")}")
        self.current_info_text.config(state=tk.DISABLED)
    
    def update_personnel_list(self):
        """Обновление списка сотрудников"""
        # Очищаем список
        for item in self.personnel_tree.get_children():
            self.personnel_tree.delete(item)
        
        # Добавляем сотрудников
        for name in self.face_recognizer.get_all_persons():
            person_info = self.face_recognizer.get_person_info(name)
            if person_info:
                self.personnel_tree.insert("", tk.END, values=(
                    person_info['name'],
                    person_info['position'],
                    person_info['age'],
                    person_info['rank'],
                    person_info['status']
                ))
    
    def delete_person(self):
        """Удаление выбранного сотрудника"""
        selected = self.personnel_tree.selection()
        if not selected:
            messagebox.showwarning(self.get_text("warning"), self.get_text("select_employee_delete"))
            return
        
        item = self.personnel_tree.item(selected[0])
        name = item['values'][0]
        
        if messagebox.askyesno(self.get_text("confirmation"), f"{self.get_text('delete_employee_confirm')} {name}?"):
            self.face_recognizer.delete_person(name)
            self.update_personnel_list()
            messagebox.showinfo(self.get_text("success"), f"{name} {self.get_text('person_deleted')}")
    
    def show_history(self):
        """Показ истории входов/выходов"""
        history_window = tk.Toplevel(self.root)
        history_window.title(self.get_text("history_window_title"))
        history_window.transient(self.root)
        
        # Центрируем окно
        self.center_window(history_window, 700, 500)
        
        # Treeview для истории
        columns = (self.get_text("column_name"), self.get_text("column_action"), self.get_text("column_time"))
        history_tree = ttk.Treeview(history_window, columns=columns, show="headings")
        
        for col in columns:
            history_tree.heading(col, text=col)
            history_tree.column(col, width=200)
        
        # Добавляем записи истории
        history = self.face_recognizer.get_entry_history()
        for entry in reversed(history):  # Показываем последние записи сверху
            history_tree.insert("", tk.END, values=(
                entry['name'],
                entry['action'],
                entry['time']
            ))
        
        history_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Кнопка закрытия
        ttk.Button(history_window, text=self.get_text("close_button"), 
                  command=history_window.destroy).pack(pady=10)
    
    def edit_person_dialog(self):
        """Диалог редактирования информации о сотруднике"""
        selected = self.personnel_tree.selection()
        if not selected:
            messagebox.showwarning(self.get_text("warning"), self.get_text("select_employee_edit"))
            return
        
        item = self.personnel_tree.item(selected[0])
        name = item['values'][0]
        person_info = self.face_recognizer.get_person_info(name)
        
        if not person_info:
            messagebox.showerror(self.get_text("error"), self.get_text("employee_info_not_found"))
            return
        
        # Создаем диалоговое окно редактирования
        dialog = tk.Toplevel(self.root)
        dialog.title(f"{self.get_text('edit_title')} {name}")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Центрируем окно
        self.center_window(dialog, 450, 350)
        
        # Заголовок
        ttk.Label(dialog, text=f"{self.get_text('edit_data')} {name}", 
                 font=("Arial", 12, "bold")).pack(pady=10)
        
        # Поля ввода с текущими значениями
        ttk.Label(dialog, text=self.get_text("enter_full_name")).pack(pady=5)
        name_entry = ttk.Entry(dialog, width=40)
        name_entry.insert(0, person_info['name'])
        name_entry.pack(pady=5)
        
        ttk.Label(dialog, text=self.get_text("enter_position")).pack(pady=5)
        position_entry = ttk.Entry(dialog, width=40)
        position_entry.insert(0, person_info['position'])
        position_entry.pack(pady=5)
        
        ttk.Label(dialog, text=self.get_text("enter_age")).pack(pady=5)
        age_entry = ttk.Entry(dialog, width=40)
        age_entry.insert(0, str(person_info['age']))
        age_entry.pack(pady=5)
        
        ttk.Label(dialog, text=self.get_text("enter_rank")).pack(pady=5)
        rank_entry = ttk.Entry(dialog, width=40)
        rank_entry.insert(0, person_info['rank'])
        rank_entry.pack(pady=5)
        
        def save_changes():
            new_name = name_entry.get().strip()
            new_position = position_entry.get().strip()
            new_age = age_entry.get().strip()
            new_rank = rank_entry.get().strip()
            
            if not all([new_name, new_position, new_age, new_rank]):
                messagebox.showerror("Ошибка", "Заполните все поля!")
                return
            
            try:
                new_age = int(new_age)
            except ValueError:
                messagebox.showerror("Ошибка", "Возраст должен быть числом!")
                return
            
            # Обновляем информацию
            updated_info = {
                'name': new_name,
                'position': new_position,
                'age': new_age,
                'rank': new_rank
            }
            
            success = self.face_recognizer.update_person_info(name, updated_info)
            
            if success:
                messagebox.showinfo(self.get_text("success"), self.get_text("info_updated"))
                self.update_personnel_list()
                dialog.destroy()
            else:
                messagebox.showerror(self.get_text("error"), self.get_text("failed_update_info"))
        
        # Кнопки
        buttons_frame = ttk.Frame(dialog)
        buttons_frame.pack(pady=20)
        
        ttk.Button(buttons_frame, text=self.get_text("apply_button"), command=save_changes).pack(side=tk.LEFT, padx=10)
        ttk.Button(buttons_frame, text=self.get_text("cancel_button"), command=dialog.destroy).pack(side=tk.LEFT, padx=10)
    
    def show_person_info(self):
        """Показ подробной информации о выбранном сотруднике"""
        selected = self.personnel_tree.selection()
        if not selected:
            messagebox.showwarning(self.get_text("warning"), self.get_text("select_employee_info"))
            return
        
        item = self.personnel_tree.item(selected[0])
        name = item['values'][0]
        person_info = self.face_recognizer.get_person_info(name)
        
        if not person_info:
            messagebox.showerror(self.get_text("error"), self.get_text("employee_info_not_found"))
            return
        
        # Создаем окно с информацией
        info_window = tk.Toplevel(self.root)
        info_window.title(f"{self.get_text('info_title')} {name}")
        info_window.transient(self.root)
        
        # Центрируем окно
        self.center_window(info_window, 500, 600)
        
        # Основная информация
        main_frame = ttk.LabelFrame(info_window, text=self.get_text("main_info"))
        main_frame.pack(pady=10, padx=20, fill=tk.X)
        
        info_text = f"""{self.get_text('full_name')}: {person_info['name']}
{self.get_text('position')}: {person_info['position']}
{self.get_text('age')}: {person_info['age']} {self.get_text('years_old')}
{self.get_text('rank')}: {person_info['rank']}
{self.get_text('current_status')}: {person_info['status']}
{self.get_text('entry_time')}: {person_info.get('entry_time', self.get_text('not_found'))}
{self.get_text('exit_time')}: {person_info.get('exit_time', self.get_text('not_found'))}"""
        
        ttk.Label(main_frame, text=info_text, font=("Arial", 10), justify=tk.LEFT).pack(pady=10)
        
        # Фотография
        photo_frame = ttk.LabelFrame(info_window, text="Фотография из базы данных")
        photo_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # Пытаемся загрузить оригинальную фотографию
        original_photo_path = person_info.get('original_photo_path')
        face_photo_path = person_info.get('face_photo_path')
        
        if original_photo_path and os.path.exists(original_photo_path):
            # Показываем оригинальную фотографию
            try:
                original_image = cv2.imread(original_photo_path)
                if original_image is not None:
                    # Масштабируем изображение для отображения
                    height, width = original_image.shape[:2]
                    max_size = 250
                    if max(height, width) > max_size:
                        scale = max_size / max(height, width)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        original_image = cv2.resize(original_image, (new_width, new_height))
                    
                    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                    original_pil = Image.fromarray(original_image_rgb)
                    original_photo = ImageTk.PhotoImage(original_pil)
                    
                    photo_label = ttk.Label(photo_frame, image=original_photo)
                    photo_label.image = original_photo  # Сохраняем ссылку
                    photo_label.pack(pady=10)
                else:
                    raise Exception("Не удалось загрузить изображение")
            except Exception as e:
                ttk.Label(photo_frame, text=f"Ошибка загрузки оригинального фото: {e}").pack(pady=10)
        elif face_photo_path and os.path.exists(face_photo_path):
            # Показываем обработанное лицо
            try:
                face_image = cv2.imread(face_photo_path, cv2.IMREAD_GRAYSCALE)
                if face_image is not None:
                    face_image = cv2.resize(face_image, (200, 200))
                    face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
                    face_pil = Image.fromarray(face_image_rgb)
                    face_photo = ImageTk.PhotoImage(face_pil)
                    
                    photo_label = ttk.Label(photo_frame, image=face_photo)
                    photo_label.image = face_photo  # Сохраняем ссылку
                    photo_label.pack(pady=10)
                else:
                    raise Exception("Не удалось загрузить изображение лица")
            except Exception as e:
                ttk.Label(photo_frame, text=f"Ошибка загрузки фото лица: {e}").pack(pady=10)
        else:
            # Показываем сохраненное лицо из базы данных
            stored_faces = self.face_recognizer.get_stored_face_images(name)
            if stored_faces:
                face_image = stored_faces[0]
                face_image = cv2.resize(face_image, (200, 200))
                face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB) if len(face_image.shape) == 2 else face_image
                
                face_pil = Image.fromarray(face_image_rgb)
                face_photo = ImageTk.PhotoImage(face_pil)
                
                photo_label = ttk.Label(photo_frame, image=face_photo)
                photo_label.image = face_photo
                photo_label.pack(pady=10)
            else:
                ttk.Label(photo_frame, text="Фото не найдено", font=("Arial", 10, "italic")).pack(pady=20)
        
        # История входов/выходов
        history_frame = ttk.LabelFrame(info_window, text="Последние входы/выходы")
        history_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        # Получаем историю для этого сотрудника
        all_history = self.face_recognizer.get_entry_history()
        person_history = [entry for entry in all_history if entry['name'] == name]
        
        if person_history:
            history_text = tk.Text(history_frame, height=6, state=tk.DISABLED, font=("Arial", 9))
            history_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            history_text.config(state=tk.NORMAL)
            for entry in reversed(person_history[-10:]):  # Последние 10 записей
                history_text.insert(tk.END, f"{entry['action']} - {entry['time']}\n")
            history_text.config(state=tk.DISABLED)
        else:
            ttk.Label(history_frame, text="История не найдена", font=("Arial", 10, "italic")).pack(pady=20)
        
        # Кнопка закрытия
        ttk.Button(info_window, text="Закрыть", 
                  command=info_window.destroy).pack(pady=10)
    
    def __del__(self):
        """Деструктор - освобождение ресурсов"""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()


def main():
    root = tk.Tk()
    app = PersonnelApp(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Приложение закрыто пользователем")
    finally:
        if hasattr(app, 'cap') and app.cap:
            app.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
