# Core Pkgs
import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import io
import os
from io import StringIO
from models import get_fork_status_from_image, get_result
import pandas as pd

@st.cache
def load_image(img):
    im = Image.open(img)
    return im

@st.cache(allow_output_mutation=True)
def get_cap(location):
    print("Loading in function", str(location))
    video_stream = cv2.VideoCapture(str(location))

    # Check if camera opened successfully
    if (video_stream.isOpened() == False):
        print("Error opening video  file")
    return video_stream




def main():
    """Inspections App"""

    st.title("Автоматизация осмотров")


    activities = ["Изображение", "Видео"]
    choice = st.sidebar.selectbox("Выбор типа файла", activities)

    if choice == 'Изображение':
        #st.subheader("Turn on/off Detection")

        image_file = st.file_uploader("Загрузка изображения", type=['jpg', 'png', 'jpeg'], encoding=None, key='a')

        if image_file is not None:
            our_image = Image.open(image_file)
            st.text("Оригинальное изображение")
            # st.write(type(our_image))
            our_image_2 = np.array(our_image)
            resized_image = cv2.resize(our_image_2, (400, 300))
            st.image(resized_image)


            enhance_type = st.sidebar.radio("Модели",
                                            ["Оригинал",   "Круг",
                                             "Вилки"])




            if enhance_type == 'Вилки':

                image = np.array(our_image)

                img_width = 800
                img_height = 600

                #image_bgr = cv2.imread(image)
                print("ok")

                image_rgb_res = cv2.resize(image, (img_width, img_height))


                status,fork_image = get_fork_status_from_image(image_rgb_res)
                fork_resized = cv2.resize(fork_image, (400, 300))




                st.image(fork_resized)
                if status["fork_0"] == True and status["fork_1"] == True and status["fork_1"] == True:
                    #st.text("Визуальное состояние оборудования в порядке")
                    df_status = {'Адрес': ['ш. Энтузиастов, 56, Москва'],
                            'Проблема': ["Выявлены дефекты"],
                            "Ответственный": ['Петров'],
                            "Решено": ["Да"]

                            }
                    df = pd.DataFrame(df_status, columns=['Адрес', 'Проблема', "Ответственный", "Решено"], )


                    df.reset_index(drop=True, inplace=True)  # Resets the index, makes factor a column
                    #df.drop("index", axis=1, inplace=True)


                    st.table(df)


                    

            elif enhance_type == 'Круг':


                new_img = np.array(our_image.convert('RGB'))
                image = np.array(our_image)


                circle_status = get_result(new_img)
                print(circle_status)
                if circle_status is None:

                    circle_text = 'Датчик не распознан'
                elif circle_status:
                         circle_text = 'Включен'
                else:
                         circle_text = 'Отключен'
                # if circle_status == "False":
                #     circle_text = "Отключен"
                # elif circle_status == "True":
                #     circle_text = "Включен"
                # elif circle_status == "None":
                #     circle_text = "Не распознан"

                print(circle_status)
                # print(circle_text)
                st.markdown('Элегазовый силовой выключатель **{}**.'.format(circle_text))

                df_status = {'Адрес': ['ш. Энтузиастов, 56, Москва'],
                             'Проблема': ["Выявлены дефекты"],
                             "Ответственный": ['Петров'],
                             "Решено": ["Нет"]

                             }
                df = pd.DataFrame(df_status, columns=['Адрес', 'Проблема', "Ответственный", "Решено"], )
                st.table(df)

            elif enhance_type == 'Оригинал':
                image = np.array(our_image)
                resized_image = cv2.resize(image, (400, 300))
                st.image(resized_image, width=None)
                df_status = {'Адрес': ['ш. Энтузиастов, 56, Москва'],
                             'Проблема': ["Выявлены дефекты"],
                             "Ответственный": ['Петров'],
                             "Решено": ["Нет"]

                             }
                df = pd.DataFrame(df_status, columns=['Адрес', 'Проблема', "Ответственный", "Решено"], )

                st.table(df)
            else:
                image = np.array(our_image)
                resized_image = cv2.resize(image, (400, 300))
                st.image(resized_image, width=None)






    elif choice == 'Video':

        video_file = st.file_uploader("Upload Video", type=['mp4'], encoding="auto")
        temporary_location = False

        if video_file is not None:
            g = io.BytesIO(video_file.read())  ## BytesIO Object
            #temporary_location = "testout_simple.mp4"

            with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
                out.write(g.read())  ## Read bytes into file
                st.video(g)
                vidcap = cv2.VideoCapture(video_file)
                st.video(vidcap)

            # close file
            out.close()



            enhance_type = st.sidebar.radio("Модели",
                                            ["Оригинал",  "Круг",
                                             "Вилка"])






if __name__ == '__main__':
    main()