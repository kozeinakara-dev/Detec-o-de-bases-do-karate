import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Função para calcular o ângulo entre três pontos
def calculate_angle(a, b, c):
    a = np.array(a)  # Primeiro ponto
    b = np.array(b)  # Ponto central
    c = np.array(c)  # Último ponto
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Captura de vídeo
cap = cv2.VideoCapture(0)  # Use 0 para webcam ou passe o caminho de um vídeo

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Converter a imagem para RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Voltar para BGR para exibir
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Inicializar mensagem padrão
    pose_text = "Nenhuma base detectada"

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Coordenadas de pontos-chave
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

        # Calcular ângulos dos joelhos e quadris
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)

        # Exibir ângulos para depuração
        cv2.putText(image, f"Joelho Esq: {left_knee_angle:.2f}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, f"Joelho Dir: {right_knee_angle:.2f}", (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, f"Quadril Esq: {left_hip_angle:.2f}", (50, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, f"Quadril Dir: {right_hip_angle:.2f}", (50, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Calcular a distância entre os tornozelos para verificar largura da base
        ankle_distance = abs(left_ankle[0] - right_ankle[0])

        # Lógica para detectar bases
        # Zenkutsu Dachi: Perna da frente dobrada, perna de trás esticada
        if 90 < left_knee_angle < 140 and right_knee_angle > 150 and 100 < left_hip_angle < 150:
            pose_text = "Zenkutsu Dachi"
        # Kiba Dachi: Pernas afastadas, joelhos dobrados simetricamente
        elif 80 < left_knee_angle < 130 and 80 < right_knee_angle < 130 and ankle_distance > 0.3:
            pose_text = "Kiba Dachi"
        # Kokutsu Dachi: Perna de trás dobrada, perna da frente esticada
        elif 90 < right_knee_angle < 140 and left_knee_angle > 150 and 100 < right_hip_angle < 150:
            pose_text = "Kokutsu Dachi"
        # Shiko Dachi: Pernas bem afastadas, joelhos dobrados, pés abertos
        elif 80 < left_knee_angle < 130 and 80 < right_knee_angle < 130 and ankle_distance > 0.4:
            pose_text = "Shiko Dachi"
        # Nekoashi Dachi: Perna de trás dobrada, perna da frente com pouco peso
        elif 90 < right_knee_angle < 140 and 140 < left_knee_angle < 170 and left_ankle[1] < left_knee[1]:
            pose_text = "Nekoashi Dachi"
        # Fudo Dachi: Pernas moderadamente afastadas, joelhos levemente dobrados
        elif 100 < left_knee_angle < 150 and 100 < right_knee_angle < 150 and 0.2 < ankle_distance < 0.4:
            pose_text = "Fudo Dachi"
        # Heiko Dachi: Pernas retas, pés juntos
        elif left_knee_angle > 160 and right_knee_angle > 160 and ankle_distance < 0.2:
            pose_text = "Heiko Dachi"

        # Exibir a base detectada
        cv2.putText(image, pose_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Desenhar landmarks no corpo
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Exibir a imagem
    cv2.imshow('Karate Pose Detection', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
pose.close()