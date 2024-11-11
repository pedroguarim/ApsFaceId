import cv2
import face_recognition

# Carregar e codificar as imagens pré-salvas
image1 = face_recognition.load_image_file("fotos/matheus.jpg")
image2 = face_recognition.load_image_file("fotos/lucas.jpg")
image3 = face_recognition.load_image_file("fotos/pedro.jpg")

encoding_image1 = face_recognition.face_encodings(image1)[0]
encoding_image2 = face_recognition.face_encodings(image2)[0]
encoding_image3 = face_recognition.face_encodings(image3)[0]

# Lista de encodings e nomes correspondentes
encodings_known = [encoding_image1, encoding_image2, encoding_image3]
names = ["Seja bem vindo", "Seja bem vindo Diretor", "Seja bem vindo Ministro"]

# Captura de imagem ao vivo da webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    
    # Redimensionar o frame para acelerar o processo de reconhecimento
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # Encontrar todas as faces e seus encodings no frame atual da webcam
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)
    
    for face_encoding in face_encodings:
        # Comparar a face da webcam com as imagens pré-salvas
        matches = face_recognition.compare_faces(encodings_known, face_encoding)
        
        # Exibir a mensagem de identificação
        if True in matches:
            match_index = matches.index(True)
            cv2.putText(frame, names[match_index], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Acesso negado", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Exibir a imagem com a mensagem na tela
    cv2.imshow('Video', frame)

    # Pressione 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere a captura e feche a janela
video_capture.release()
cv2.destroyAllWindows()