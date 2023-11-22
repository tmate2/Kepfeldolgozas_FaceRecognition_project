import cv2
import face_recognition as fr  # https://github.com/ageitgey/face_recognition
import numpy as np

# hozzáadott modulok "python-opencv", "face-recognition", "dlib"


# Kamera: 0 laptop, 1 webcam
KAMERA = 1


def face_detect(frame, known_faces_enc, names):
    # arcok helyének elmentése az aktuális képkocáról
    faces = fr.face_locations(frame)

    # ha nem talál arcot, akkor a frame-et visszaadjuk
    if faces.__str__() == "[]":
        return frame

    # talált arcok előkészítése vizsgálatra
    faces_enc = fr.face_encodings(frame, faces)

    # két lista előkészítése, hogy a felismert arcokat
    # össze tudjuk párosítani a nevekkel
    known_faces_index = list()
    known_faces_name = list()

    # talált és ismert arcok összehasonlítása, egyezés esetén pedig az indexek mentése
    for i, face in enumerate(faces_enc):
        for j, kf in enumerate(known_faces_enc):
            kff = np.asarray([kf])  # kf átalakítás np.ndarray típusra az összehasonlításhoz
            result = fr.compare_faces(kff, face)
            if True in result:
                known_faces_index.append(i)
                known_faces_name.append(j)
    known_faces_name.append(-1)     # hiba megkerülése, hogy ne crasheljen a program
    # hátránya: néha eltűnik a név az ismert arc alol

    # TODO: Hibajavitás
    # Ha két arc közel van egymáshoz vagy összeérnek a koordinátái, "IndexError"-t dob
    # a 'name = names[known_faces_name[index]]' sorba és leáll a program (exit code 1)
    handle_index_error(known_faces_index, known_faces_name, faces, frame,
                       names)  # az alatti resz kod majd ebben a metodusban lesz

    return frame


def handle_index_error(known_faces_index: list, known_faces_name: list, faces: list, frame, names):
    # arc körberajzolása, ismeretlen arc pirossal,
    # ismert arcokat zölddel, alattuk a hozzá tartozó névvel
    for index, (top, right, bottom, left) in enumerate(faces):
        if index in known_faces_index:
            # körberajzolás
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 4)
            # névmező létrehozása
            cv2.rectangle(frame, (left - 2, bottom), (right + 2, bottom + 23), (0, 255, 0), cv2.FILLED)

            # szöveg méretének igazítása az arc széllességéhez
            font_scale = right - left
            if font_scale >= 200:
                font_scale = 1
            else:
                font_scale /= 200
            # archoz tartozó név kiírása a névmezőbe
            name = names[known_faces_name[index]]
            cv2.putText(frame, name, (left + 6, bottom + 21), cv2.FONT_HERSHEY_TRIPLEX, font_scale, (0, 0, 0), 1)
        else:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 4)


def main():
    capture = cv2.VideoCapture(KAMERA)

    # képek előkészítése az arcok felismeréséhez
    face1 = fr.load_image_file("known_faces/MateThold.jpg")
    face1_enc = fr.face_encodings(face1)[0]

    face2 = fr.load_image_file("known_faces/TerryDavis.jpg")
    face2_enc = fr.face_encodings(face2)[0]

    face3 = fr.load_image_file("known_faces/JoeBiden.jpg")
    face3_enc = fr.face_encodings(face3)[0]

    face4 = fr.load_image_file("known_faces/AronMarton.jpg")
    face4_enc = fr.face_encodings(face4)[0]

    known_faces_enc = [
        face1_enc,
        face2_enc,
        face3_enc,
        face4_enc
    ]

    # az arcokhoz tartozó nevek eltárolása
    names = [
        "Mate Thold",
        "Terry Davis",
        "Joe Biden",
        "Aron Marton"
    ]

    # a kamera képének a megjelenítéséhez egy végtelenített függvényt használunk
    while True:
        is_true, frame = capture.read()

        # az aktuális framen arcokat keresünk majd annak az így kapott és esetlegesen megjelölt képet megjelenítjük
        frame = face_detect(frame, known_faces_enc, names)
        cv2.imshow("Face detector", frame)

        # a 'd' billentyű lenyomásával megszakítjuk a végtelenciklust, ezáltal "leállítjuk a felvételt"
        if cv2.waitKey(20) & 0xFF == ord('d'):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
