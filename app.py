# Core Pkgs
import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import io
import os
from io import StringIO
from models import get_fork_status_from_image
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
                #
                hsv_min = np.array((2, 28, 65), np.uint8)
                hsv_max = np.array((26, 238, 255), np.uint8)

                #new_img = np.array(our_image.convert('RGB'))
                image = np.array(our_image)

                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                thresh = cv2.inRange(hsv, hsv_min, hsv_max)

                contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                circle_img = cv2.drawContours(image, contours, -1, (255, 0, 0), 3, cv2.LINE_AA, hierarchy, 1)

                gray = cv2.cvtColor(circle_img, cv2.COLOR_BGR2GRAY)
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.05, 100)
                if circles is not None:
                    # convert the (x, y) coordinates and radius of the circles to integers
                    circles = np.round(circles[0, :]).astype("int")
                    # loop over the (x, y) coordinates and radius of the circles
                    for (x, y, r) in circles:
                        # draw the circle in the output image, then draw a rectangle
                        # corresponding to the center of the circle

                        circle_img=cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

                        circle_resized_image = cv2.resize(circle_img, (400, 300))


                        st.image(circle_resized_image)

            elif enhance_type == 'Оригинал':
                image = np.array(our_image)
                resized_image = cv2.resize(image, (400, 300))
                st.image(resized_image, width=None)
            else:
                image = np.array(our_image)
                resized_image = cv2.resize(image, (400, 300))
                st.image(resized_image, width=None)






    elif choice == 'Video':

        video_file = st.file_uploader("Upload Video", type=['mp4'], encoding="auto")
        temporary_location = False

        if video_file is not None:
            g = io.BytesIO(video_file.read())  ## BytesIO Object
            temporary_location = "testout_simple.mp4"

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