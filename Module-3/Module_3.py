# #Промежуточная аттестация по Модулю 3.
# #=====================================

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import cv2 as cv
import nltk
import sklearn 
import re
import os

# # Задача 1. Используя методы NumPy, преобразуйте столбец цен билетов fare в массив, 
# # очистите от пропусков, затем рассчитайте среднее, медиану и стандартное отклонение 
# # стоимости билета.
# # https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

#file = 'titanic.csv'
#df = pd.read_csv(file)
# print(f'Loaded titanic.csv:') 

# print("Work array:")
# print(f'Тип df: {type(df)}') #pandas data frame
# fare = df['Fare'].to_numpy(dtype=float)  #numpy array
# print(f"Тип fare: {type(fare)}\n")

# print(f"Size: {fare.shape}")
# print(f"Data type: {fare.dtype}")
# print(f"First 10 values: {fare[:10]}\n")

# print("TASK 1")
# print(f"Ticket cost analyze:")
# print("=" * 20)
# fare_clean = fare[~np.isnan(fare)]  #check missed values
# cleaned = len(fare) - len(fare_clean) 
# print(f"Check missed values = {cleaned}")
# print(f"Total tickets: {len(fare_clean)}")
# print(f"Mean={fare_clean.mean():.2f}")
# print(f"Median={np.median(fare_clean):.2f}")
# print(f"STD={np.std(fare_clean):.2f}\n")


# #Задача 2. При помощи Pandas определите, сколько мужчин и сколько женщин было в 
# #каждой из трёх пассажирских классов, и выведите полученную таблицу с числами для 
# #каждой комбинации пола и класса.

# print("TASK 2")
# print(f"Total of man and female by pclass:")
# print("=" * 20)
# gender_class_table = pd.crosstab(df['Pclass'], df['Sex'])
# print(gender_class_table, "\n")


# # Задача 3. С помощью Pandas заполните пропуски в столбце возраста age средним 
# # возрастом по группе пассажиров того же класса (pclass), а затем создайте новый столбец 
# # age_group, где каждому пассажиру присвоена категория «ребёнок» (до 18 лет), 
# # «взрослый» (18–60) или «пожилой» (старше 60).

# print('TASK 3')
# print("Fill missed values for Age by mean value, create age_group:")
# print("=" * 20)
# missed = df['Age'].isnull().sum()
# print(f"Missed age = {missed} values")

# mean_age_by_class = df.groupby('Pclass')['Age'].mean()
# print("Avarage age by pclass:")
# for pclass, mean_age in mean_age_by_class.items():
#      print(f"Pclass {pclass}: {mean_age:.1f} y.")

# #Создаем копию DataFrame чтобы не изменять оригинальные данные
# df_filled = df.copy()
# # #Заполняем пропуски в возрасте средним значением по классу 
# # #в новой колонке Age_filled
# df_filled['Age_filled'] = df_filled.groupby('Pclass')['Age'].transform(
#      lambda x: x.fillna(x.mean())
# )

# missed = df_filled['Age_filled'].isnull().sum()
# print(f"After filling the missed ages = {missed}")

# # in case of someone is left, fill by common mean 
# pass_mean_age = df_filled['Age_filled'].mean()
# print(f"Passangers age mean = {pass_mean_age:.1f} y.")
# if(missed != 0):
#     df_filled['Age_filled'] = df_filled['Age_filled'].fillna(pass_mean_age)
    
# #age category
# def get_age_group(age):
#     if pd.isna(age):
#         return "unknown"
#     elif age < 18:
#         return "child"
#     elif age <= 60:
#         return "adult"
#     else:
#         return "senior"

#create a new column Age_group
#df_filled['Age_group'] = df_filled['Age_filled'].apply(get_age_group)
# print("First 10 records with a new column Age_group:")
# print(df_filled[['Pclass', 'Age', 'Age_filled', 'Age_group']].head(10), '\n')


# # Задача 4. Постройте при помощи Matplotlib три графика в одной фигуре: на первом — 
# # гистограмму распределения возраста пассажиров, на втором — ящичную диаграмму 
# # (boxplot) цен билетов, на третьем — столбчатую диаграмму выживаемости (долю 
# # выживших) по классам обслуживания. Не забудьте подписать оси и добавить заголовки 
# # каждого подграфика.

# # Создаем фигуру с 3 подграфиками
# fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# # 4.1 Гистограмма (Histogram) распределения возраста пассажиров - axes[0]
# axes[0].hist(df_filled['Age_filled'], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
# axes[0].set_xlabel('Age')
# axes[0].set_ylabel('Passengers')
# axes[0].set_title('Passengers age distribution (Histogram)')
# axes[0].grid(alpha=0.3)

# # 4.2 Ящичная диаграмма (Boxplot), стоимости билетов - axes[1]
# df_filled.boxplot(column='Fare', by='Pclass', ax=axes[1])
# axes[1].set_xlabel('Passsenger class')
# axes[1].set_ylabel('Ticket fare')
# axes[1].set_title('Ticket fare by passenger class')

# # 4.3. Cтолбчатая диаграмм (Bar), выживаемость по классам - axes[2]
# survival_rate = df_filled.groupby('Pclass')['Survived'].mean()
# axes[2].bar(survival_rate.index, survival_rate.values * 100, color=['red', 'blue', 'green'], alpha=0.7)
# axes[2].set_xlabel('Passsenger class')
# axes[2].set_ylabel('Survived (%)')
# axes[2].set_title('Survived by class')
# axes[2].set_xticks([1, 2, 3])

#plt.show()


# # Дополнительная задача  5. Загрузите произвольное изображение с диска через OpenCV, 
# # преобразуйте его в оттенки серого и отобразите исходник и результат рядом.

# img_path = "cat.jpg"
# #load an image 
# image = cv.imread(img_path)

# if img_path is np.nan:
#     raise "filenotfounderror('file not found!')"

# (b, g, r) = image[0, 0] 
# print("loaded image(BGR), pixel[0, 0], Blue: {}, Green: {}, Red: {}".format(b, g, r)) 

# image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
# (r, g, b) = image_rgb[0, 0] 
# print("converted image(RGB), pixel[0, 0], Red: {}, Green: {}, Blue: {}".format(r, g, b),'\n') 

# #convert to gray
# image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# print("Gray image (2D)")
# print("Height:"+str(image_gray.shape[0])) 
# print("Width:" + str(image_gray.shape[1]),'\n') 

# print("RGB image (3D)")
# print("Height:"+str(image_rgb.shape[0])) 
# print("Width:" + str(image_rgb.shape[1])) 
# print("N_cannels:" + str(image_rgb.shape[2]))

# # images output
# plt.figure(figsize=(15, 6))

# plt.subplot(1, 2, 1)
# plt.imshow(image_rgb)
# plt.title('rgb image')
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(image_gray, cmap='gray')
# plt.title('gray shades')
# plt.axis('off')

# plt.tight_layout()
# plt.show()


# Дополнительная задача  6. Создайте небольшой корпус из трёх–пяти предложений, 
# выполните для него токенизацию, удаление стоп-слов и векторизацию с помощью 
# CountVectorizer из scikit-learn, а затем выведите полученную матрицу признаков. 

# скачиваем необходимые данные NLTK
# from sklearn.feature_extraction.text import CountVectorizer
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

# print('TASK 6')
# print("Векторизация с помощью CountVectorizer")
# print("=" * 20)

# # Корпус
# corpus = [
#     "Елки-палки, большинство башен(Towers) деревянных крепостей были четырехугольными в плане, или, как ещё писали в летописях, рублены в четыре стены.",
#     "Круглые, или многоугольные, башни хотя и были менее распространенными, но почти всегда им отводилась роль главных проездных башен.",
#     "В зависимости от размеров и значимости крепости варьировались количество башен и их размеры.",
#     "Немаловажным достоинством многоугольных башен было то, что они выступали за линию городовой стены тремя, четырьмя или пятью стенами.",
#     "Характерной чертой башен некоторых крепостей было наличие в них навесных балконов-часовен над въездными воротами."
# ]

# print("Corpus:")
# print("=" * 50)
# for i, sentence in enumerate(corpus, 1):    
#     print(f"{i}. {sentence}")

# # уберем русские стоп-слова
# russian_stopwords = set(stopwords.words('russian'))
# print(f'First ten stopwords: {list(russian_stopwords)[:10]}')
# print(f'Stopwords total = {len(russian_stopwords)}')

# #функция для предобработки текста
# def preprocess_text(review_text):
#     review_text = review_text.replace('ё','е')
#     review_text = review_text.replace('Ё','Е')
#     # Удаляем все, кроме букв
#     review_text = re.sub("[^а-яА-Я]", " ", review_text)    
#     # токенизация
#     tokens = word_tokenize(review_text.lower(), language='russian')
#     # удаление стоп-слов и пунктуации
#     filtered_tokens = [token for token in tokens
#                       if token.isalpha() and token not in russian_stopwords]
#     return filtered_tokens

# # применяем предобработку ко всему корпусу
# processed_corpus = []
# print("\nPre-processing:")
# print("=" * 50)
# for i, sentence in enumerate(corpus, 1):
#     tokens = preprocess_text(sentence)
#     processed_corpus.append(' '.join(tokens))
#     print(f"Doc {i}:")
#     print(f"Before: {sentence}")
#     print(f"After: {' '.join(tokens)}")
#     print("-" * 30)

# # Векторизация
# vectorizer = CountVectorizer(stop_words=stopwords.words('russian'))
# X = vectorizer.fit_transform(processed_corpus)

# # Результат
# df = pd.DataFrame(X.toarray(),
#                  columns=vectorizer.get_feature_names_out(),
#                  index=[f'Doc{i+1}' for i in range(len(processed_corpus))])
# print("\nFeature matrix:")
# print(df)


# Дополнительная задача 7. Откройте короткий видеофайл или поток с веб-камеры через 
# OpenCV, посчитайте и выведите общее число кадров в видео, затем сохраните каждый 
# десятый кадр в отдельный файл.

print('TASK 7')
print("Обработка видео")
print("=" * 20)

# Укажите путь к видео 
video_source = 'test.mp4' 
output_folder = 'frames'

os.makedirs(output_folder, exist_ok = True)
print(f"\nСоздана папка для кадров: {output_folder}")

cap = cv.VideoCapture(video_source)
if not cap.isOpened():
    raise RuntimeError('Не удалось открыть видео/камеру. Проверьте путь или доступ к устройству.')
else:
    print(f"Видеофайл {video_source} успешно открыт!")

# Получаем основные характеристики видео
total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv.CAP_PROP_FPS)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
duration = total_frames / fps

print("\nИНФОРМАЦИЯ О ВИДЕО:")
print("=" * 40)
print(f"Общее количество кадров: {total_frames}")
print(f"Частота кадров в секунду: {fps:.1f}")
print(f"Размеры кадра, ШxВ: {width}x{height}")
print(f"Время видео, сек.: {duration:.1f}")

# извлечение кадров
saved_frames = 0
current_frame = 0

print("\nИЗВЛЕЧЕНИЕ КАДРОВ:")
print("=" * 40)

while True:
    # Читаем кадр
    ret, frame = cap.read()

    # Если кадр не прочитан (конец видео)
    if not ret:
        break

    # Сохраняем каждый 10-й кадр
    if current_frame % 10 == 0:
        # Создаем имя файла для кадра
        frame_filename = os.path.join(output_folder, f"frame_{current_frame:05d}.jpg")

        # Сохраняем кадр
        cv.imwrite(frame_filename, frame)
        saved_frames += 1

        # Выводим прогресс каждые 50 сохраненных кадров
        if saved_frames % 50 == 0:
            print(f"Сохранено кадров: {saved_frames}")

    current_frame += 1

# Закрываем видеофайл
cap.release()

print(f"\nИзвлечение завершено!")
print(f"Всего сохранено кадров: {saved_frames}")
