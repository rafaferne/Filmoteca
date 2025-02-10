from tkinter import *
from moviepy.editor import VideoFileClip
import cv2
import concurrent.futures
import face_recognition
import os
import uuid
import time
import numpy as np
import speech_recognition as sr
import tkinter as tk
from tkinter import simpledialog
from tkinter import messagebox
from PIL import Image, ImageTk
from pynput.keyboard import Key, Listener
import json

img_actual = None
cambiar_edad = False
nombre_actual = "NA"

def on_press(key):
    global cambiar_edad
    if str(key) == "'v'":
        cambiar_edad = True

# Asocia la función con la pulsación de una tecla
listener = Listener(on_press=on_press)
listener.start()

#Funcion para guardar la imagen del usuario
def guardar_imagen_usuario(frame):
    filename = str(uuid.uuid4()) + ".jpg"
    cv2.imwrite(filename, frame)
    return filename

def registro(frame):
    # Crea una ventana para la entrada del nombre del usuario
    root = tk.Tk()
    root.withdraw()
    nombre_usuario = simpledialog.askstring("Registro", "Introduce tu nombre:")
    
    # Comprueba si existe la subcarpeta, sino la crea
    if not os.path.exists('imagenes'):
        os.makedirs('imagenes')

    # Guarda el frame actual en la subcarpeta
    cv2.imwrite(f"imagenes/{nombre_usuario}.jpg", frame)
    
    # Crea la nueva entrada para el usuario
    nuevo_usuario = {
        "name": nombre_usuario,
        "foto": f"imagenes/{nombre_usuario}.jpg",
        "peliculas": funcionEdad()
    }
    
    # Carga los datos de usuarios existentes
    with open('usuarios.json', 'r') as f:
        data = json.load(f)
    
    # Añade el nuevo usuario
    data['usuarios'].append(nuevo_usuario)
    
    # Guarda los datos actualizados
    with open('usuarios.json', 'w') as f:
        json.dump(data, f, indent=4)
        
    return nuevo_usuario

#Función para elegir el tipo de publico
def funcionEdad():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("¿Que tipo de publico eres?: Infantil o adulto.")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio, language='es-ES')
            print(f"Has dicho: {text}")
            if "infantil" in text.lower():
                return {
                    "uno": {"nombre": "coco.jpeg", "trailer": "coco_trailer.mp4", "sinopsis": "coco_sinopsis.jpeg"},
                    "dos": {"nombre": "del_reves.jpeg", "trailer": "del_reves_trailer.mp4", "sinopsis": "del_reves_sinopsis.jpeg"},
                    "tres": {"nombre": "vengadores.jpeg", "trailer": "vengadores_trailer.mp4", "sinopsis": "vengadores_sinopsis.jpeg"}
                    }
            elif "adulto" in text.lower():
                return {
                    "uno": {"nombre": "titanic.jpeg", "trailer": "titanic_trailer.mp4", "sinopsis": "titanic_sinopsis.jpeg"},
                    "dos": {"nombre": "intocable.jpeg", "trailer": "intocable_trailer.mp4", "sinopsis": "intocable_sinopsis.jpeg"},
                    "tres": {"nombre": "interstellar.jpeg", "trailer": "interstellar_trailer.mp4", "sinopsis": "interstellar_sinopsis.jpeg"}
                }
            else:
                print("No reconocí tu edad. Intenta de nuevo.")
                return funcionEdad()
        except sr.UnknownValueError:
            print("No entendí lo que dijiste. Intenta de nuevo.")
            return funcionEdad()

def login():
    global nombre_actual
    # Carga una imagen de referencia y aprende a reconocerla.

    with open('usuarios.json','r') as f:
        data = json.load(f)

    usuarios = {}
    for usuario in data['usuarios']:
        fotoUsuario = face_recognition.load_image_file(usuario['foto'])
        usuarioEncode = face_recognition.face_encodings(fotoUsuario)[0]
        usuarios[usuario['name']] = {
            'encoding': usuarioEncode,
            'peliculas': usuario['peliculas']
        }

    # Obtiene la referencia de captura de video
    video_capture = cv2.VideoCapture(0)

    frame_count = 0
    process_every_n_frames = 10

    usuarioLogeado = None
    login = False

    while True:
        # Captura un único frame de video
        ret, frame = video_capture.read()

        # Redimensionamiento del frame a 1/4 para mayor rapidez en el procesamiento
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Procesamiento cada N frames del video para ahorrar tiempo
        if frame_count % process_every_n_frames == 0:
            # Encuentra todas las caras y los encodings de las caras en el frame actual del video
            face_locations = face_recognition.face_locations(small_frame, model='hog')
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)

            face_names = []
            for (x, y, w, h), face_encoding in zip(face_locations, face_encodings):
                login = False
                for name, usuario in usuarios.items():
                    match = face_recognition.compare_faces([usuario['encoding']], face_encoding)

                    if match[0]:
                        cv2.rectangle(small_frame, (h, x), (y, w), (0, 0, 255), 2)
                        nombre_actual = name
                        usuarioLogeado = usuario
                        login = True
                        break

                if not login:
                    cv2.rectangle(small_frame, (h, x), (y, w), (0, 0, 255), 2)
                    name = "Desconocido"
                    usuarioLogeado = registro(frame)
                    login = True
                    break

                face_names.append(name)

            if login:
                break

        frame_count += 1

        # Mostramos los resultados
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Redimensionamiento de las caras a su tamaño original
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Dibujamos un recuadro alrededor de las caras
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Escribimos los nombres de las caras reconocidas (si no anonimo)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        # Mostramos el frame resultante
        cv2.imshow('Video', frame)

        # Se puede presionar SPACE en el teclado para terminar el loop
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    # Liberamos la webcam
    video_capture.release()
    cv2.destroyAllWindows()

    # Devolvemos True si es un usuario conocido o False si no lo es
    return usuarioLogeado if login else None

def funcionPeliculas(user):
    global img_actual, trailer, sinopsis
    peliculas = user["peliculas"]
    
    r = sr.Recognizer()
    MIC_INDEX = 0  # Índice del micrófono
    
    bucle = False
    while not bucle:
        with sr.Microphone(device_index=MIC_INDEX) as source:
            print("Diga algo...")
            audio = r.listen(source, timeout=5)
            try:
                texto = r.recognize_google(audio, language='es-ES')
                print("Usted dijo: " + texto)

                if "uno" in texto or "1" in texto:
                    img_actual = cv2.imread(peliculas['uno']['nombre'])
                    trailer = peliculas['uno']['trailer']
                    sinopsis = peliculas['uno']['sinopsis']
                    bucle = True
                elif "dos" in texto or "2" in texto:
                    img_actual = cv2.imread(peliculas['dos']['nombre'])
                    trailer = peliculas['dos']['trailer']
                    sinopsis = peliculas['dos']['sinopsis']
                    bucle = True
                elif "tres" in texto or "3" in texto:
                    img_actual = cv2.imread(peliculas['tres']['nombre'])
                    trailer = peliculas['tres']['trailer']
                    sinopsis = peliculas['tres']['sinopsis']
                    bucle = True
                elif "tráiler" in texto.lower():
                    if trailer:
                        play_trailer(trailer)
                    else:
                        print("No hay tráiler asignado.")
                elif "sinopsis" in texto.lower():
                    if sinopsis:
                        img_actual = cv2.imread(sinopsis)
                    else:
                        print("No hay sinopsis asignada.")
            except sr.UnknownValueError:
                print("No se pudo entender el audio.")
            except sr.RequestError as e:
                print(f"No se pudo solicitar resultados; {e}")
            
            time.sleep(2)

def play_trailer(file_path):
    print("Intentando reproducir el tráiler en:", file_path)
    try:
        clip = VideoFileClip(file_path)
        clip.preview(fps=24, audio=True)
        print("Video reproducido")
    except Exception as e:
        print("Error durante la reproducción del video:", e)
        
def funcionArucos(user):
    global img_actual
    global cambiar_edad

    if os.path.exists('camara.py'):
        import camara
    else:
        print("Es necesario realizar la calibración de la cámara")
        exit()

    # Configuración de Aruco
    DIC = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    parametros = cv2.aruco.DetectorParameters()

    cap = cv2.VideoCapture(0)

    if cap.isOpened():
        hframe = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        wframe = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        matrix, roi = cv2.getOptimalNewCameraMatrix(camara.cameraMatrix, camara.distCoeffs, (wframe,hframe), 1, (wframe,hframe))
        roi_x, roi_y, roi_w, roi_h = roi

        # Iniciar el hilo para funcionPeliculas
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        executor.submit(funcionPeliculas, user)

        finalizado = False
        while not finalizado:
            if cambiar_edad:
                with open('usuarios.json','r') as f:
                    data = json.load(f)
                
                peliculas_nuevas = funcionEdad()
                # Actualizar peliculas en los datos del usuario
                for usuario in data['usuarios']:
                    if usuario['name'] == nombre_actual:
                        usuario['peliculas'] = peliculas_nuevas
                        break

                # Actualizar el archivo JSON
                with open('usuarios.json', 'w') as f:
                    json.dump(data, f)

                # Actualiza peliculas en la variable user
                user['peliculas'] = peliculas_nuevas
                # Reinicia la función de voz con las nuevos peliculas
                executor.submit(funcionPeliculas, user)

                cambiar_edad = False

            ret, framebgr = cap.read()
            if ret:
                framerectificado = cv2.undistort(framebgr, camara.cameraMatrix, camara.distCoeffs, None, matrix)
                framerecortado = framerectificado[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]

                (corners, ids, rejected) = cv2.aruco.detectMarkers(framerecortado, DIC, parameters=parametros)
                if len(corners)>0:
                    for i in range(len(corners)):
                        cv2.polylines(framerecortado, [corners[i].astype(int)], True, (0,255,0), 4)
                        
                        if img_actual is not None:
                            # Calcula el tamaño del aruco
                            p1, p2, p3, p4 = corners[i][0]
                            dist1 = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                            dist2 = np.sqrt((p2[0] - p3[0])**2 + (p2[1] - p3[1])**2)
                            marker_size = (dist1 + dist2) / 2

                            imagen_resized = cv2.resize(img_actual, (int(marker_size), int(marker_size)))

                            pts1 = np.float32(corners[i][0])
                            pts2 = np.float32([[0,0],[marker_size,0],[marker_size,marker_size],[0,marker_size]])
                            M = cv2.getPerspectiveTransform(pts2,pts1)
                            dst = cv2.warpPerspective(imagen_resized,M,(roi_w,roi_h))

                            mask = np.zeros(framerecortado.shape, dtype=np.uint8)
                            roi_corners = np.int32(corners[i][0])
                            channel_count = framerecortado.shape[2]
                            ignore_mask_color = (255,)*channel_count
                            cv2.fillConvexPoly(mask, roi_corners, ignore_mask_color)
                            mask_inverse = np.full(mask.shape, 255, dtype=np.uint8)
                            cv2.fillConvexPoly(mask_inverse, roi_corners, (0,0,0))
                            im_masked = cv2.bitwise_and(framerecortado, mask_inverse)
                            dst_masked = cv2.bitwise_and(dst, mask)

                            combined = cv2.add(im_masked, dst_masked)
                            framerecortado = combined

                            # Reinicia la función de voz para actualizar img_actual
                            executor.submit(funcionPeliculas, user)
                        
                cv2.imshow('Pelicula', framerecortado)
                key = cv2.waitKey(1)
                if key == 27: # tecla Esc para salir
                    finalizado = True

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("No se ha detectado ninguna cámara")
    cap.release()
    cv2.destroyAllWindows()



def execute():
    user = login()

    if user is not None:
        funcionArucos(user)
    else:
        print("No estás resgitrado")


#Codigo de la "interfaz"
ventana=Tk()
ventana.title("AR Filmin")

ventana.resizable(True, True)

framePrincipal=Frame()

framePrincipal.pack(fill="both", expand="True")

framePrincipal.config(width="1024", height="720")

imagenFondo = Image.open('fondo.jpeg')
imagenFondo = imagenFondo.resize((1024, 768), Image.LANCZOS)
photo = ImageTk.PhotoImage(imagenFondo)

labelImagen = Label(framePrincipal, image=photo)
labelImagen.place(x=0, y=0, relwidth=1, relheight=1)

labelAutor = Label(framePrincipal, text="Autor: Rafael Fernández Nevado", font=(14), bg="White", fg="Black")
labelAutor.place(x=380, y=200)

botonInicio=Button(ventana, text="Iniciar", command=execute, font=("Helvetica", 18))
botonInicio.pack()
botonInicio.place(relx=0.5, rely=0.5, anchor='center')

labelImagen.image = photo

ventana.mainloop()