import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import base64
from pathlib import Path
from PIL import Image

try :
  # Configuration de la page Streamlit
  st.set_page_config(
      page_title="Détecteur de Posture",
      page_icon="🧘",
      layout="wide"
  )
  
  
  # Fonctions utilitaires
  def get_coordinates(landmarks, landmark_id, image_width, image_height):
      """Récupère les coordonnées (x, y) d'un landmark spécifique."""
      landmark = landmarks[landmark_id]
      x = int(landmark.x * image_width)
      y = int(landmark.y * image_height)
      return (x, y)
  
  def average_point(point1, point2):
      """Calcule le point moyen entre deux points."""
      x = (point1[0] + point2[0]) // 2
      y = (point1[1] + point2[1]) // 2
      return (x, y)
  
  def calculate_angle(point1, point2):
      """Calcule l'angle en degrés entre la verticale et la ligne formée par deux points."""
      deltaY = point2[1] - point1[1]
      deltaX = point2[0] - point1[0]
      angle_radians = np.arctan2(deltaX, deltaY)
      angle_degrees = abs(np.degrees(angle_radians))
      return angle_degrees
  
  def get_base64_audio(file_path):
      """Convertit un fichier audio en base64 pour l'utilisation dans HTML"""
      if Path(file_path).exists():
          with open(file_path, "rb") as audio_file:
              return base64.b64encode(audio_file.read()).decode()
      return None
  
  def get_alert_html(play=False):
      """Génère le HTML pour l'alerte audio"""
      alert_audio = get_base64_audio("alert.wav")
      
      if alert_audio is None:
          return "<!-- Fichier audio non trouvé -->"
          
      if play:
          html = f"""
          <audio autoplay loop>
              <source src="data:audio/wav;base64,{alert_audio}" type="audio/wav">
          </audio>
          """
      else:
          html = f"""
          <audio id="alert_sound">
              <source src="data:audio/wav;base64,{alert_audio}" type="audio/wav">
          </audio>
          """
      return html
  
  # Interface principale
  def main():
      # Chargement de MediaPipe
      mp_pose = mp.solutions.pose
      pose = mp_pose.Pose()
      
      # Titre de l'application
      st.title("Détecteur de Posture")
      
      # Paramètres de l'application dans la barre latérale
      st.sidebar.header("Paramètres")
      
      # Bouton pour démarrer/arrêter la webcam
      if 'running' not in st.session_state:
          st.session_state.running = False
      
      if st.sidebar.button('Démarrer' if not st.session_state.running else 'Arrêter'):
          st.session_state.running = not st.session_state.running
          # Réinitialiser les minuteries lors du démarrage
          if st.session_state.running:
              st.session_state.head_timer = None
              st.session_state.back_timer = None
              st.session_state.neck_timer = None
              st.session_state.alert_playing = False
      
      # Activer/désactiver les alertes sonores
      if 'alert_enabled' not in st.session_state:
          st.session_state.alert_enabled = True
      
      if st.sidebar.button('Alerte Off' if st.session_state.alert_enabled else 'Alerte On'):
          st.session_state.alert_enabled = not st.session_state.alert_enabled
          # Arrêter l'alerte si elle est en cours et qu'on désactive les alertes
          if not st.session_state.alert_enabled:
              st.session_state.alert_playing = False
      
      # Définir le délai d'alerte
      if 'ALERT_DELAY' not in st.session_state:
          st.session_state.ALERT_DELAY = 5  # délai par défaut
      
      st.session_state.ALERT_DELAY = st.sidebar.slider(
          "Délai d'alerte (secondes)", 
          min_value=1, 
          max_value=10, 
          value=st.session_state.ALERT_DELAY
      )
      
      # Initialiser les timers s'ils n'existent pas
      if 'head_timer' not in st.session_state:
          st.session_state.head_timer = None
      if 'back_timer' not in st.session_state:
          st.session_state.back_timer = None
      if 'neck_timer' not in st.session_state:
          st.session_state.neck_timer = None
      if 'alert_playing' not in st.session_state:
          st.session_state.alert_playing = False
      
      # Afficher les indicateurs de posture
      st.sidebar.header("Indicateurs de posture")
      head_indicator = st.sidebar.empty()
      back_indicator = st.sidebar.empty()
      neck_indicator = st.sidebar.empty()
      
      # Zone pour l'alerte audio HTML
      audio_placeholder = st.sidebar.empty()
      
      # Zone principale pour la vidéo
      video_placeholder = st.empty()
  
      # Contrôle du framerate
      fps_limit = 10
      prev_time = 0
      
      # Si la webcam est active
      if st.session_state.running:
          cap = cv2.VideoCapture(0)
  
          # Réduction de la taille pour alléger le traitement
          frame_width = 480
          frame_height = 360
          cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
          cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
          
          # Vérifier si la webcam est ouverte correctement
          if not cap.isOpened():
              st.error("Impossible d'accéder à la webcam. Vérifiez les permissions.")
              st.session_state.running = False
          else:
              stframe = video_placeholder.empty()
              
              try:
                  while st.session_state.running:
                      success, frame = cap.read()
                      if not success:
                          st.error("Erreur lors de la lecture du flux vidéo.")
                          break
  
                      # Limiter le nombre de FPS
                      current_time = time.time()
                      if (current_time - prev_time) < 1 / fps_limit:
                          continue
                      prev_time = current_time
                      
                      # Traiter l'image pour MediaPipe (qui attend RGB)
                      image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                      results = pose.process(image)
                      
                      # Préparer l'image pour l'affichage
                      image_with_results = image.copy()
                      
                      bad_posture_detected = False
                      now = time.time()
                      
                      if results.pose_landmarks:
                          image_height, image_width, _ = image.shape
                          landmarks = results.pose_landmarks.landmark
                          
                          # Récupération des coordonnées clés
                          nose = get_coordinates(landmarks, mp_pose.PoseLandmark.NOSE, image_width, image_height)
                          left_ear = get_coordinates(landmarks, mp_pose.PoseLandmark.LEFT_EAR, image_width, image_height)
                          right_ear = get_coordinates(landmarks, mp_pose.PoseLandmark.RIGHT_EAR, image_width, image_height)
                          left_shoulder = get_coordinates(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, image_width, image_height)
                          right_shoulder = get_coordinates(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, image_width, image_height)
                          left_hip = get_coordinates(landmarks, mp_pose.PoseLandmark.LEFT_HIP, image_width, image_height)
                          
                          mid_shoulder = average_point(left_shoulder, right_shoulder)
                          
                          # Calculs pour évaluer la posture
                          vertical_distance = mid_shoulder[1] - nose[1]
                          back_angle = calculate_angle(left_shoulder, left_hip)
                          chin_drop_left = nose[1] - left_ear[1]
                          chin_drop_right = nose[1] - right_ear[1]
                          
                          # 1. Détection tête penchée
                          if vertical_distance > 220:
                              head_posture = "Tête penchée"
                              head_color = (255, 0, 0)  # Rouge
                              if st.session_state.head_timer is None:
                                  st.session_state.head_timer = now
                              elif now - st.session_state.head_timer >= st.session_state.ALERT_DELAY:
                                  bad_posture_detected = True
                          else:
                              head_posture = "Tête droite"
                              head_color = (0, 255, 0)  # Vert
                              st.session_state.head_timer = None
                          
                          # Afficher sur l'image
                          cv2.putText(image_with_results, f"{head_posture} ({vertical_distance:.0f}px)", 
                                     (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, head_color, 2)
                          
                          # Mise à jour de l'indicateur
                          head_indicator.markdown(f"**Tête:** {head_posture} - {vertical_distance:.0f}px")
                          
                          # 2. Détection mauvaise posture dos
                          if back_angle >= 15:
                              posture = "Mauvaise posture"
                              color = (255, 0, 0)  # Rouge
                              if st.session_state.back_timer is None:
                                  st.session_state.back_timer = now
                              elif now - st.session_state.back_timer >= st.session_state.ALERT_DELAY:
                                  bad_posture_detected = True
                          else:
                              posture = "Bonne posture"
                              color = (0, 255, 0)  # Vert
                              st.session_state.back_timer = None
                          
                          # Afficher sur l'image
                          cv2.putText(image_with_results, f"{posture} (angle: {int(back_angle)}°)", 
                                     (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                          
                          # Mise à jour de l'indicateur
                          back_indicator.markdown(f"**Dos:** {posture} - {int(back_angle)}°")
                          
                          # 3. Détection cou sous tension
                          if chin_drop_left > 30 or chin_drop_right > 30:
                              neck_state = "Cou sous tension"
                              neck_color = (255, 0, 0)  # Rouge
                              if st.session_state.neck_timer is None:
                                  st.session_state.neck_timer = now
                              elif now - st.session_state.neck_timer >= st.session_state.ALERT_DELAY:
                                  bad_posture_detected = True
                          else:
                              neck_state = "Tête bien alignée"
                              neck_color = (0, 255, 0)  # Vert
                              st.session_state.neck_timer = None
                          
                          # Afficher sur l'image
                          cv2.putText(image_with_results, f"{neck_state} ({chin_drop_left:.0f}px | {chin_drop_right:.0f}px)", 
                                     (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, neck_color, 2)
                          
                          # Mise à jour de l'indicateur
                          neck_indicator.markdown(f"**Cou:** {neck_state} - G:{chin_drop_left:.0f}px | D:{chin_drop_right:.0f}px")
                          
                          # Gérer l'alerte sonore
                          if bad_posture_detected and st.session_state.alert_enabled:
                              if not st.session_state.alert_playing:
                                  audio_placeholder.markdown(get_alert_html(play=True), unsafe_allow_html=True)
                                  st.session_state.alert_playing = True
                          elif st.session_state.alert_playing and (not bad_posture_detected or not st.session_state.alert_enabled):
                              audio_placeholder.markdown(get_alert_html(play=False), unsafe_allow_html=True)
                              st.session_state.alert_playing = False
  
                          # Gérer les alertes sonores
                          if bad_posture_detected and st.session_state.alert_enabled:
                              if not st.session_state.alert_playing:
                                  audio_placeholder.markdown(get_alert_html(play=True), unsafe_allow_html=True)
                                  st.session_state.alert_playing = True
                          else:
                              if st.session_state.alert_playing:
                                  # Replacer l'audio HTML sans autoplay pour arrêter le son
                                  audio_placeholder.markdown(get_alert_html(play=False), unsafe_allow_html=True)
                                  st.session_state.alert_playing = False
  
                          
                          # Dessiner les landmarks MediaPipe
                          mp_drawing = mp.solutions.drawing_utils
                          mp_drawing.draw_landmarks(image_with_results, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                      
                      # Afficher l'image résultante
                      stframe.image(image_with_results, channels="RGB", use_container_width=True)
                      
                      # Petite pause pour réduire l'utilisation CPU
                      time.sleep(0.03)
              
              finally:
                  cap.release()
      else:
          # Afficher un message quand la webcam est inactive
          video_placeholder.markdown("""
          <div style="display: flex; justify-content: center; align-items: center; height: 400px; 
                      background-color: #f0f2f6; border-radius: 10px; flex-direction: column;">
              <h2 style="color: #4c566a;">Cliquez sur 'Démarrer' pour activer la webcam</h2>
              <p style="color: #4c566a;">Le détecteur de posture analysera votre position en temps réel</p>
          </div>
          """, unsafe_allow_html=True)
      
      # Information supplémentaire
      st.markdown("""
      ### Comment utiliser l'application
      1. Cliquez sur **Démarrer** pour lancer l'analyse de posture
      2. L'application détecte trois éléments clés:
         - **Posture du dos**: Reste droit et bien aligné
         - **Position de la tête**: Évite de pencher la tête vers l'avant
         - **Tension du cou**: Garde la tête alignée avec les épaules
      3. Une alerte sonore retentit après 5 secondes de mauvaise posture
      4. Vous pouvez ajuster le délai d'alerte et activer/désactiver le son
      """)
      
      # Notes au bas de page
      st.sidebar.markdown("---")
      st.sidebar.info("⚠️ Pour que l'application fonctionne correctement, assurez-vous que votre webcam est visible et que vous êtes bien éclairé.")
      st.sidebar.info("📝 Cette application utilise MediaPipe pour la détection de posture. Aucune donnée n'est enregistrée ou partagée.")
except Exception as e:
    st.error(f"Une erreur est survenue : {e}")

if __name__ == "__main__":
    main()