import cv2
import numpy as np
import pyvirtualcam
import threading
import tkinter as tk
from tkinter import ttk



# Effets disponibles

EFFECTS = {
	"Glitch RGB": False,
	"Flou": False,
	"Inversion des couleurs": False,
	"Effet dessin": False,
	"Effet pixel": False,
	"Tremblement": False,
	"Effet vieux film": False,
	"Cartoon": False,
	"Distorsion (Vague)": False,
	"Effet Inversion Radiale": False,
	"Rotation": False,
}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")



# Fonctions effets standards

def apply_glitch(frame):
	b, g, r = cv2.split(frame)
	r = np.roll(r, 5, axis=1)
	b = np.roll(b, -5, axis=0)
	return cv2.merge([b, g, r])

def apply_blur(frame):
	return cv2.GaussianBlur(frame, (15, 15), 0)

def apply_invert(frame):
	return cv2.bitwise_not(frame)

def apply_sketch(frame):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	inv = cv2.bitwise_not(gray)
	blur = cv2.GaussianBlur(inv, (21, 21), 0)
	sketch = cv2.divide(gray, 255 - blur, scale=256)
	return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

def apply_pixelate(frame, pixel_size=10):
	height, width = frame.shape[:2]
	small = cv2.resize(frame, (width // pixel_size, height // pixel_size), interpolation=cv2.INTER_LINEAR)
	return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)

def apply_distortion(frame):
	rows, cols = frame.shape[:2]
	M = np.float32([[1, 0, np.random.randint(-10, 10)], [0, 1, np.random.randint(-10, 10)]])
	distorted_frame = cv2.warpAffine(frame, M, (cols, rows))
	return distorted_frame

def apply_sepia(frame):
	kernel = np.array([[0.393, 0.769, 0.189],
					   [0.349, 0.686, 0.168],
					   [0.272, 0.534, 0.131]])
	sepia_frame = cv2.transform(frame, kernel)
	sepia_frame = np.clip(sepia_frame, 0, 255)
	return sepia_frame.astype(np.uint8)

def add_noise(frame):
	row, col, ch = frame.shape
	mean = 0
	sigma = 25
	gauss = np.random.normal(mean, sigma, (row, col, ch))
	noisy_frame = np.array(frame, dtype=float) + gauss
	noisy_frame = np.clip(noisy_frame, 0, 255)
	return noisy_frame.astype(np.uint8)

def apply_old_film_effect(frame):
	frame = apply_sepia(frame)
	frame = add_noise(frame)
	return frame

def apply_cartoon(frame):
	blurred = cv2.bilateralFilter(frame, 9, 75, 75)
	edges = cv2.adaptiveThreshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 255,
								  cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
	cartoon_frame = cv2.bitwise_and(blurred, blurred, mask=edges)
	return cartoon_frame

def apply_wave_distortion(frame, intensity=10):
	rows, cols, _ = frame.shape
	for i in range(rows):
		offset = int(intensity * np.sin(i / 10))
		frame[i] = np.roll(frame[i], offset, axis=0)
	return frame

rotation_angle = 0
def apply_rotation(frame):
	global rotation_angle, rotation_speed
	rows, cols, _ = frame.shape
	rotation_angle += rotation_speed
	
	if rotation_angle >= 360:
		rotation_angle = 0
	
	matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
	
	rotated_frame = cv2.warpAffine(frame, matrix, (cols, rows))
	return rotated_frame

def apply_radial_inversion_on_face(frame, scale=0.5):
	rows, cols = frame.shape[:2]
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

	if len(faces) == 0:
		return frame

	(x, y, w, h) = faces[0]
	center_x = x + w // 2
	center_y = y + h // 2
	max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
	X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
	distance = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
	max_radius = max_distance * scale
	norm_distance = (distance / max_radius)
	result = frame.copy()
	
	for i in range(rows):
		for j in range(cols):
			if norm_distance[i, j] < 1:
				factor = norm_distance[i, j]
				x = int(center_x + factor * (j - center_x))
				y = int(center_y + factor * (i - center_y))
				
				if 0 <= x < cols and 0 <= y < rows:
					result[i, j] = frame[y, x]

	return result



# Application des effets actifs

def apply_effects(frame):
	if EFFECTS["Glitch RGB"]:
		frame = apply_glitch(frame)
	if EFFECTS["Flou"]:
		frame = apply_blur(frame)
	if EFFECTS["Inversion des couleurs"]:
		frame = apply_invert(frame)
	if EFFECTS["Effet dessin"]:
		frame = apply_sketch(frame)
	if EFFECTS["Effet pixel"]:
		frame = apply_pixelate(frame)
	if EFFECTS["Tremblement"]:
		frame = apply_distortion(frame)
	if EFFECTS["Effet vieux film"]:
		frame = apply_old_film_effect(frame)
	if EFFECTS["Cartoon"]:
		frame = apply_cartoon(frame)
	if EFFECTS["Distorsion (Vague)"]:
		frame = apply_wave_distortion(frame)
	if EFFECTS["Effet Inversion Radiale"]:
		frame = apply_radial_inversion_on_face(frame, scale=0.2)
	if EFFECTS["Rotation"]:
		frame = apply_rotation(frame)
	return frame

rotation_speed = 1



# Interface Tkinter

def start_gui():
	root = tk.Tk()
	root.title("🎛️ Contrôle des filtres caméra")
	root.geometry("300x500")
	root.resizable(False, True)

	ttk.Label(root, text="Active / désactive les effets :", font=("Arial", 12, "bold")).pack(pady=10)

	for name in EFFECTS.keys():
		var = tk.BooleanVar(value=False)
		def toggle_effect(effect_name=name, var=var):
			EFFECTS[effect_name] = var.get()
		chk = ttk.Checkbutton(root, text=name, variable=var, command=toggle_effect)
		chk.pack(anchor="w", padx=20, pady=5)

	ttk.Label(root, text="Vitesse de rotation :").pack(pady=10)
	rotation_slider = tk.Scale(root, from_=1, to=10, orient="horizontal", command=lambda val: update_rotation_speed(val))
	rotation_slider.set(rotation_speed)
	rotation_slider.pack(pady=10)

	ttk.Label(root, text="Ferme la fenêtre pour arrêter le flux.", font=("Arial", 9)).pack(side="bottom", pady=10)

	def update_rotation_speed(val):
		global rotation_speed
		rotation_speed = int(val)

	def on_close():
		global running
		running = False
		root.destroy()
	root.protocol("WM_DELETE_WINDOW", on_close)

	root.mainloop()



# Capture caméra

cap = cv2.VideoCapture(1)  # valeur à modifier pour une autre caméra
if not cap.isOpened():
	raise RuntimeError("Impossible d'accéder à la caméra")

running = True
threading.Thread(target=start_gui, daemon=True).start()



# Envoi à Discord

with pyvirtualcam.Camera(width=640, height=480, fps=20) as cam:
	print("🎥 Caméra virtuelle prête ! (Sélectionne-la dans Discord)")
	while running:
		ret, frame = cap.read()
		if not ret:
			break

		frame = apply_effects(frame)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		cam.send(frame)
		cam.sleep_until_next_frame()

		cv2.imshow("Caméra avec filtres", frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			running = False
			break

cap.release()
cv2.destroyAllWindows()
print("🛑 Caméra arrêtée.")
