import cv2
import numpy as np
import pyvirtualcam
import threading
import tkinter as tk
from tkinter import ttk
import time
from queue import Queue

# ================================== #
#        Effets disponibles          #
# ================================== #

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
	"Mirroir": False,
	"Disco": False,
}

frame_queue = Queue(maxsize=2)
output_queue = Queue(maxsize=2)

glitch_intensity = 5
blur_intensity = 15
pixel_size = 10
shake_intensity = 10
radial_scale = 0.2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ================================== #
#     Fonctions effets standards     #
# ================================== #

def apply_glitch(frame):
	global glitch_intensity
	b, g, r = cv2.split(frame)
	r = np.roll(r, glitch_intensity, axis=1)
	b = np.roll(b, -glitch_intensity, axis=0)
	return cv2.merge([b, g, r])

def apply_blur(frame):
	global blur_intensity
	k = int(blur_intensity)
	if k % 2 == 0:
		k += 1
	if k < 1:
		k = 1
	return cv2.GaussianBlur(frame, (k, k), 0)

def apply_invert(frame):
	return cv2.bitwise_not(frame)

def apply_sketch(frame):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	inv = cv2.bitwise_not(gray)
	blur = cv2.GaussianBlur(inv, (21, 21), 0)
	sketch = cv2.divide(gray, 255 - blur, scale=256)
	return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

def apply_pixelate(frame):
	global pixel_size
	p = max(1, int(pixel_size))
	h, w = frame.shape[:2]
	small = cv2.resize(frame, (max(1, w // p), max(1, h // p)), interpolation=cv2.INTER_LINEAR)
	return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def apply_distortion(frame):
	global shake_intensity
	s = int(shake_intensity)
	if s < 1:
		return frame
	rows, cols = frame.shape[:2]
	dx = np.random.randint(-s, s + 1)
	dy = np.random.randint(-s, s + 1)
	M = np.float32([[1, 0, dx], [0, 1, dy]])
	return cv2.warpAffine(frame, M, (cols, rows))

def apply_sepia(frame):
	kernel = np.array([[0.393, 0.769, 0.189],
										 [0.349, 0.686, 0.168],
										 [0.272, 0.534, 0.131]])
	sepia = cv2.transform(frame, kernel)
	return np.clip(sepia, 0, 255).astype(np.uint8)

def add_noise(frame):
	noise = np.random.normal(0, 25, frame.shape)
	return np.clip(frame + noise, 0, 255).astype(np.uint8)

def apply_old_film_effect(frame):
	return add_noise(apply_sepia(frame))

def apply_cartoon(frame):
	blurred = cv2.bilateralFilter(frame, 9, 75, 75)
	edges = cv2.adaptiveThreshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 255,	cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
	return cv2.bitwise_and(blurred, blurred, mask=edges)

def apply_wave_distortion(frame, intensity=10):
	rows, cols, _ = frame.shape
	for i in range(rows):
		offset = int(intensity * np.sin(i / 10))
		frame[i] = np.roll(frame[i], offset, axis=0)
	return frame

def apply_radial_inversion_on_face(frame):
	global radial_scale
	rows, cols = frame.shape[:2]
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
	if len(faces) == 0:
		return frame
	x, y, w, h = faces[0]
	cx = x + w // 2
	cy = y + h // 2
	X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
	dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
	max_r = np.sqrt(cx**2 + cy**2) * radial_scale
	norm = dist / max_r
	mask = norm < 1
	fx = cx + norm * (X - cx)
	fy = cy + norm * (Y - cy)
	fx = fx.astype(np.int32)
	fy = fy.astype(np.int32)
	fx = np.clip(fx, 0, cols - 1)
	fy = np.clip(fy, 0, rows - 1)
	result = frame.copy()
	result[mask] = frame[fy[mask], fx[mask]]
	return result

rotation_angle = 0
rotation_speed = 1

def apply_rotation(frame):
	global rotation_angle, rotation_speed
	rows, cols = frame.shape[:2]
	rotation_angle = (rotation_angle + rotation_speed) % 360
	M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
	return cv2.warpAffine(frame, M, (cols, rows))

def apply_mirror(frame):
	return cv2.flip(frame, 1)

def apply_disco(frame):
	overlay = np.zeros_like(frame)
	color = np.random.randint(0, 255, size=(1,1,3), dtype=np.uint8)
	overlay[:] = color
	return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)


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
		frame = apply_radial_inversion_on_face(frame)
	if EFFECTS["Rotation"]:
		frame = apply_rotation(frame)
	if EFFECTS["Mirroir"]:
		frame = apply_mirror(frame)
	if EFFECTS["Disco"]:
		frame = apply_disco(frame)
	return frame

# ================================== #
#          Gestion caméra            #
# ================================== #

selected_camera_index = 0
cap = None
running = True

def change_camera(new_index):
	global cap
	if cap is not None and cap.isOpened():
		cap.release()
	cap = cv2.VideoCapture(new_index)
	if not cap.isOpened():
		print(f"❌ Impossible d'accéder à la caméra {new_index}")
	else:
		print(f"✅ Caméra {new_index} activée")

# ================================== #
#          Interface Tkinter         #
# ================================== #

def detect_cameras(max_tested=5):
	available = []
	for i in range(max_tested):
		test = cv2.VideoCapture(i)
		if test.isOpened():
			available.append(f"Caméra {i}")
			test.release()
	return available

def start_gui():
	global selected_camera_index, running
	global glitch_intensity, blur_intensity, pixel_size, shake_intensity, radial_scale

	root = tk.Tk()
	root.title("🎛️ Contrôle des filtres caméra")
	root.geometry("300x600")
	root.resizable(False, True)

	ttk.Label(root, text="Sélectionne la caméra :", font=("Arial", 11, "bold")).pack(pady=10)
	cameras = detect_cameras()
	if not cameras:
		cameras = ["Aucune caméra détectée"]

	selected_camera = tk.StringVar(value=cameras[0])
	combo = ttk.Combobox(root, textvariable=selected_camera, values=cameras, state="readonly")
	combo.pack(pady=5)

	def update_camera(event=None):
		global selected_camera_index
		try:
			idx = int(selected_camera.get().split()[-1])
			selected_camera_index = idx
			change_camera(idx)
			print(f"🎥 Caméra sélectionnée : {idx}")
		except:
			pass

	combo.bind("<<ComboboxSelected>>", update_camera)

	ttk.Separator(root, orient="horizontal").pack(fill="x", pady=10)
	ttk.Label(root, text="Active / désactive les effets :", font=("Arial", 12, "bold")).pack(pady=5)

	for name in EFFECTS:
		var = tk.BooleanVar(value=False)
		chk = ttk.Checkbutton(root, text=name, variable=var, command=lambda n=name, v=var: EFFECTS.__setitem__(n, v.get()))
		chk.pack(anchor="w", padx=20, pady=3)

	def make_slider(label, from_, to_, varname):
		ttk.Label(root, text=label).pack(pady=5)
		slider = tk.Scale(root, from_=from_, to=to_, orient="horizontal", command=lambda val: globals().__setitem__(varname, float(val)))
		slider.set(globals()[varname])
		slider.pack()
	make_slider("Intensité glitch", 0, 50, "glitch_intensity")
	make_slider("Intensité flou", 1, 50, "blur_intensity")
	make_slider("Taille pixelisation", 1, 50, "pixel_size")
	make_slider("Vitesse tremblement", 0, 30, "shake_intensity")
	make_slider("Zoom visage (Radial)", 0.05, 1.0, "radial_scale")

	ttk.Label(root, text="Vitesse rotation").pack(pady=10)
	rot_slider = tk.Scale(root, from_=1, to=10, orient="horizontal", command=lambda val: globals().__setitem__("rotation_speed", int(val)))
	rot_slider.set(rotation_speed)
	rot_slider.pack()

	ttk.Label(root, text="Ferme la fenêtre pour arrêter le flux.", font=("Arial", 9)).pack(side="bottom", pady=10)

	def on_close():
		global running
		running = False
		root.destroy()

	root.protocol("WM_DELETE_WINDOW", on_close)
	root.mainloop()

# ================================== #
#      Nouveau système multi-thread   #
# ================================== #

def capture_thread():
	global running, cap
	while running:
			if cap is None or not cap.isOpened():
				time.sleep(0.05)
				continue
			ret, frame = cap.read()
			if not ret:
				continue
			if frame_queue.full():
				frame_queue.get_nowait()
			frame_queue.put(frame)

def processing_thread():
	global running
	while running:
		try:
			frame = frame_queue.get(timeout=0.1)
		except:
			continue
		processed = apply_effects(frame)
		if output_queue.full():
			output_queue.get_nowait()
		output_queue.put(processed)

def virtualcam_thread():
	global running
	with pyvirtualcam.Camera(width=640, height=480, fps=20) as cam:
		print("🎥 Caméra virtuelle prête !")
		while running:
			try:
				processed = output_queue.get(timeout=0.1)
			except:
				continue
			frame_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
			cam.send(frame_rgb)
			cam.sleep_until_next_frame()
			cv2.imshow("Caméra avec filtres", frame_rgb)
			if cv2.waitKey(1) & 0xFF == ord("q"):
				running = False
				break
	cv2.destroyAllWindows()
	print("🛑 Flux terminé.")

# ================================== #
#             Lancement              #
# ================================== #

change_camera(selected_camera_index)

threading.Thread(target=capture_thread, daemon=True).start()
threading.Thread(target=processing_thread, daemon=True).start()
threading.Thread(target=virtualcam_thread, daemon=True).start()

start_gui()
