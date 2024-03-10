import cv2
import tkinter as tk
from PIL import Image, ImageTk
import os
from tkinter import filedialog
from tkinter import Tk, Label, Frame, Button, Checkbutton, IntVar
from tkinter import TOP
import test
import tensorflow as tf
import numpy as np
from skimage import transform

FILE_HASH = {"yolov5s_face_dynamic": "e7854a5cae48ded05b3b31aa93765f0d"}
DEFAULT_DETECTOR = "https://github.com/leondgarse/Keras_insightface/releases/download/v1.0.0/yolov5s_face_dynamic.h5"
DEFAULT_ANCHORS = np.array(
    [
        [[0.5, 0.625], [1.0, 1.25], [1.625, 2.0]],
        [[1.4375, 1.8125], [2.6875, 3.4375], [4.5625, 6.5625]],
        [[4.5625, 6.781199932098389], [7.218800067901611, 9.375], [10.468999862670898, 13.531000137329102]],
    ],
    dtype="float32",
)
DEFAULT_STRIDES = np.array([8, 16, 32], dtype="float32")
recognition_result = None
c=0
recognize_model = tf.keras.models.load_model('Model\GhostFaceNet_W1.3_S1_ArcFace.h5', compile=False)
embeddings = np.load('embs.npy')
image_classes = np.load('embs_class.npy')

def recognition(face_image, embeddings = embeddings, image_classes = image_classes):
    emb_unk = recognize_model((face_image - 127.5) * 0.0078125).numpy()
    emb_unk = emb_unk / np.linalg.norm(emb_unk, axis=-1, keepdims=True)
    cosine_similarities = emb_unk @ embeddings.T
    recognition_indexes = cosine_similarities.argmax(-1)
    recognition_similarities = [cosine_similarities[id, ii] for id, ii in enumerate(recognition_indexes)]
    recognition_classes = [image_classes[ii] for ii in recognition_indexes]
    for i in range(len(recognition_similarities)):
        if recognition_similarities[i] < 0.5:
            recognition_classes[i] = None
            return('Not found')
        return(os.listdir('Data_after_augumentation')[int(recognition_classes[0])])

def face_align_landmarks(img, landmarks, image_size=(112, 112), method="similar"):
    tform = transform.AffineTransform() if method == "affine" else transform.SimilarityTransform()
    src = np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.729904, 92.2041]],
        dtype=np.float32,
    )
    ret = []
    landmarks = landmarks if landmarks.shape[1] == 5 else tf.reshape(landmarks, [-1, 5, 2]).numpy()
    for landmark in landmarks:
        tform.estimate(landmark, src)
        ndimage = transform.warp(img, tform.inverse, output_shape=image_size)
        if len(ndimage.shape) == 2:
            ndimage = np.stack([ndimage, ndimage, ndimage], -1)
        ret.append(ndimage)
    return (np.array(ret) * 255).astype(np.uint8)

current_labels = []
captured_image_paths = {}

def detect_in_image(model, master, image, max_output_size=50, iou_threshold=0.3, score_threshold=0.7, image_format="RGB"):
    global current_labels
    global captured_image_paths

    for label in current_labels:
        label.destroy()
    current_labels = []

    for button_info in getattr(master, "select_buttons", []):
        button_info["button"].destroy()

    master.select_buttons = []

    bbs, pps, ccs = model(image, max_output_size, iou_threshold, score_threshold, image_format)

    if len(bbs) == 0:
        label = Label(master, text="No human face", font=("Helvetica", 20))
        label.grid(row=2, column=2, padx=5, pady=5)
        current_labels.append(label)
    else:
        frame = Frame(master)
        frame.grid(row=2, column=3, columnspan=4, padx=5, pady=5)

        for i, face_image in enumerate(face_align_landmarks(image, pps)):
            sub_frame = Frame(frame)
            sub_frame.grid(row=i // 6, column=i % 6, padx=5, pady=5)

            pil_image = Image.fromarray(face_image)
            pil_image = pil_image.resize((42, 42))
            img_tk = ImageTk.PhotoImage(pil_image)
            label = Label(sub_frame, image=img_tk)
            label.image = img_tk
            label.pack(side=TOP)
            current_labels.append(label)

            select_button = Button(sub_frame, text="Select", command=lambda i=i, face_image=face_image: result(i, face_image, master))
            select_button.pack(side=TOP, padx=5, pady=5)

            master.select_buttons.append({"index": i, "button": select_button})
            

def result(index, face_image, master):
    global recognition_result
    tmp = face_image.reshape(1, 112, 112, 3)
    recognition_result =  recognition(tmp)
    annotation_file = ["Anno1\C1.txt", "Anno1\C2.txt", "Anno1\C3.txt", "Anno1\C4.txt", "Anno1\C5.txt", "Anno1\C6.txt", "Anno1\C7.txt"]
    tmp = []
    for i, cap in enumerate(annotation_file):
        f = open(cap, "r")
        for j in f.readlines():
            if (c-5 < int(j.split(',')[0])//5 < c+1) and j.split(',')[1] == recognition_result:
                tmp.append(i+1)
    lb = Label(master, text=f"                                                        ", font=("Helvetica", 20))
    lb.grid(row=2, column=2, padx=5, pady=5)
    if len(tmp) == 0:
        label = Label(master, text="Not found", font=("Helvetica", 20))
    else:
        label = Label(master, text=f"Camera: {set(tmp)}", font=("Helvetica", 20))
    label.grid(row=2, column=2, padx=5, pady=5)
    current_labels.append(lb)
    current_labels.append(label)


    
class VideoPlayer:
    def __init__(self, master, video_files, model=None):
        self.master = master
        self.video_files = video_files
        self.model = model
        self.video_caps = [cv2.VideoCapture(file) for file in video_files]
        self.frames = []
        self.labels = []
        self.frame_counters = [0] * len(video_files)
        self.show_frames()
        self.captured_image_label = None

        self.cap = cv2.VideoCapture(0)

        self.popup_button_title = tk.Label(self.master, text="Search Human", font=("Helvetica", 20))
        self.popup_button_title.grid(row=0, column=3, sticky="nw")

        self.popup_button = tk.Button(self.master, text="Option load image", command=self.load_image_popup, font=("Helvetica", 12))
        self.popup_button.grid(row=0, column=3, padx=(130, 100), pady=(10, 10), sticky="ew")

        self.camera_frame = tk.Frame(self.master)

        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        self.master.geometry(f"{screen_width}x{screen_height}")

    def show_frames(self):
        for cap in self.video_caps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for i, cap in enumerate(self.video_caps):
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.resize_frame(frame)
                img = Image.fromarray(frame)
                img = ImageTk.PhotoImage(image=img)
                label = tk.Label(self.master, image=img)
                label.image = img
                if i < 9:  
                    label.grid(row=i // 3, column=i % 3, padx=5, pady=5)
                else:
                    label.grid_forget()
                self.labels.append(label)
                self.frames.append(frame)

                if i == 7:
                    video_name = "Ground Plane"
                else:
                    video_name = f"Camera {i+1}"
                
                video_label = tk.Label(self.master, text=video_name)
                video_label.grid(row=i // 3, column=i % 3, padx=5, pady=5, sticky='nw')

    def update_frames(self):
        global c
        for i, cap in enumerate(self.video_caps):
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.resize_frame(frame)
                self.frames[i] = frame
        for i, frame in enumerate(self.frames):
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=img)
            self.labels[i].config(image=img)
            self.labels[i].image = img
        self.master.after(200, self.update_frames)
        c = c + 1

    def resize_frame(self, frame):
        window_width = self.master.winfo_width()
        window_height = self.master.winfo_height()
        col_count = 4
        row_count = 3

        cell_width = max(1, (window_width - 20) // col_count - 10)
        cell_height = max(1, window_height // row_count - 10)

        resized_frame = cv2.resize(frame, (cell_width, cell_height))
        return resized_frame


    def load_image(self, popup_window):
        file_path = filedialog.askopenfilename()
        if file_path:
            img = Image.open(file_path).convert("RGB")

            img_width, img_height = 356, 254
            img = img.resize((img_width, img_height))
            
            img_np = np.asarray(img)
            detect_in_image(self.model, self.master, img_np)

            img = self.resize_image(img)
            img = ImageTk.PhotoImage(img)

            if self.captured_image_label:
                self.captured_image_label.destroy()

            label = tk.Label(self.master, image=img)
            label.image = img
            label.grid(row=1, column=3, padx=5, pady=5)
            popup_window.destroy()  
            
            self.captured_image_label = label

    def load_image_popup(self):
        load_image_window = tk.Toplevel(self.master)
        load_image_window.title("Load Image")

        load_button = tk.Button(load_image_window, text="Load Image", command=lambda: self.load_image(load_image_window), font=("Helvetica", 12))
        load_button.pack(pady=5)

        capture_button = tk.Button(load_image_window, text="Capture from Camera", command=self.show_capture_popup, font=("Helvetica", 12))
        capture_button.pack(pady=5)

    def show_capture_popup(self):
        capture_popup = tk.Toplevel(self.master)
        capture_popup.title("Capture from Camera")

        capture_button = tk.Button(capture_popup, text="Capture", command=lambda: self.capture_and_display(capture_popup), font=("Helvetica", 12))
        capture_button.pack(pady=5)

        captured_label = tk.Label(capture_popup)
        captured_label.pack()

        def display_camera_popup():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.resize_frame(frame)
                img = Image.fromarray(frame)
                img = ImageTk.PhotoImage(image=img)
                captured_label.config(image=img)
                captured_label.image = img
            capture_popup.after(10, display_camera_popup)

        display_camera_popup()

    def capture_and_display(self, popup_window):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)

            img_np = np.asarray(img)
            detect_in_image(self.model, self.master, img_np)

            img = self.resize_image(img)
            img = ImageTk.PhotoImage(img)

            if self.captured_image_label:
                self.captured_image_label.destroy()

            label = tk.Label(self.master, image=img)
            label.image = img
            label.grid(row=1, column=3, padx=5, pady=5)
            popup_window.destroy()  
            
            self.captured_image_label = label

    def resize_image(self, img):
        window_width = self.master.winfo_width()
        window_height = self.master.winfo_height()
        col_count = 4
        row_count = 3

        cell_width = max(1, (window_width - 20) // col_count - 10)
        cell_height = max(1, window_height // row_count - 10)

        img.thumbnail((cell_width, cell_height))
        return img

    def display_camera(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.resize_frame(frame)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=img)
            if hasattr(self, 'camera_label'):
                self.camera_label.config(image=img)
                self.camera_label.image = img
            else:
                self.camera_label = tk.Label(self.camera_frame, image=img)
                self.camera_label.image = img
                self.camera_label.pack()
        self.master.after(100, self.display_camera)
    

class YoloV5FaceDetector():
    def __init__(self, model_path=DEFAULT_DETECTOR, anchors=DEFAULT_ANCHORS, strides=DEFAULT_STRIDES):
        if isinstance(model_path, str) and model_path.startswith("http"):
            file_name = os.path.basename(model_path)
            file_hash = FILE_HASH.get(os.path.splitext(file_name)[0], None)
            model_path = tf.keras.utils.get_file(file_name, model_path, cache_subdir="models", file_hash=file_hash)
            self.model = tf.keras.models.load_model(model_path)
        elif isinstance(model_path, str) and model_path.endswith(".h5"):
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = model_path

        self.anchors, self.strides = anchors, strides
        self.num_anchors = anchors.shape[1]
        self.anchor_grids = tf.math.ceil((anchors * strides[:, tf.newaxis, tf.newaxis])[:, tf.newaxis, :, tf.newaxis, :])

    def make_grid(self, nx=20, ny=20, dtype=tf.float32):
        xv, yv = tf.meshgrid(tf.range(nx), tf.range(ny))
        return tf.cast(tf.reshape(tf.stack([xv, yv], 2), [1, 1, -1, 2]), dtype=dtype)

    def pre_process_32(self, image):
        hh, ww, _ = image.shape
        pad_hh = (32 - hh % 32) % 32  # int(tf.math.ceil(hh / 32) * 32) - hh
        pad_ww = (32 - ww % 32) % 32  # int(tf.math.ceil(ww / 32) * 32) - ww
        if pad_ww != 0 or pad_hh != 0:
            image = tf.pad(image, [[0, pad_hh], [0, pad_ww], [0, 0]])
        return tf.expand_dims(image, 0)

    def post_process(self, outputs, image_height, image_width):
        post_outputs = []
        for output, stride, anchor, anchor_grid in zip(outputs, self.strides, self.anchors, self.anchor_grids):
            hh, ww = image_height // stride, image_width // stride
            anchor_width = output.shape[-1] // self.num_anchors
            output = tf.reshape(output, [-1, output.shape[1] * output.shape[2], self.num_anchors, anchor_width])
            output = tf.transpose(output, [0, 2, 1, 3])

            cls = tf.sigmoid(output[:, :, :, :5])
            cur_grid = self.make_grid(ww, hh, dtype=output.dtype) * stride
            xy = cls[:, :, :, 0:2] * (2 * stride) - 0.5 * stride + cur_grid
            wh = (cls[:, :, :, 2:4] * 2) ** 2 * anchor_grid

            mm = [1, 1, 1, 5]
            landmarks = output[:, :, :, 5:15] * tf.tile(anchor_grid, mm) + tf.tile(cur_grid, mm)

            post_out = tf.concat([xy, wh, landmarks, cls[:, :, :, 4:]], axis=-1)
            post_outputs.append(tf.reshape(post_out, [-1, output.shape[1] * output.shape[2], anchor_width - 1]))
        return tf.concat(post_outputs, axis=1)

    def yolo_nms(self, inputs, max_output_size=15, iou_threshold=0.35, score_threshold=0.25):
        inputs = inputs[0][inputs[0, :, -1] > score_threshold]
        xy_center, wh, ppt, cct = inputs[:, :2], inputs[:, 2:4], inputs[:, 4:14], inputs[:, 14]
        xy_start = xy_center - wh / 2
        xy_end = xy_start + wh
        bbt = tf.concat([xy_start, xy_end], axis=-1)
        rr = tf.image.non_max_suppression(bbt, cct, max_output_size=max_output_size, iou_threshold=iou_threshold, score_threshold=0.0)
        bbs, pps, ccs = tf.gather(bbt, rr, axis=0), tf.gather(ppt, rr, axis=0), tf.gather(cct, rr, axis=0)
        pps = tf.reshape(pps, [-1, 5, 2])
        return bbs.numpy(), pps.numpy(), ccs.numpy()

    def __call__(self, image, max_output_size=15, iou_threshold=0.45, score_threshold=0.25, image_format="RGB"):
        imm_RGB = image if image_format == "RGB" else image[:, :, ::-1]
        imm_RGB = self.pre_process_32(imm_RGB)
        outputs = self.model(imm_RGB)
        post_outputs = self.post_process(outputs, imm_RGB.shape[1], imm_RGB.shape[2])
        return self.yolo_nms(post_outputs, max_output_size, iou_threshold, score_threshold)


def main():
    model = YoloV5FaceDetector()
    root = tk.Tk()
    root.title("Supervision by Multicamera")
    root.geometry("1000x600")  

    video_files = ["c0.mp4", "c1.mp4", "c2.mp4", "c3.mp4", "c4.mp4", "c5.mp4", "c6.mp4", "c7.mp4"]
    player = VideoPlayer(root, video_files, model=model)
    player.update_frames()
    player.display_camera()

    root.mainloop()

if __name__ == "__main__":
    main()
