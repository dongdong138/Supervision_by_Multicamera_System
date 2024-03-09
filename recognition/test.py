import os
import cv2 as cv
import glob2
import pandas as pd
# import insightface
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import normalize
from tqdm import tqdm
from face_detector import YoloV5FaceDetector
import GhostFaceNets

class FaceDetectorAndRecognizer:
    def __init__(self, recognize_model="ghostnetv1", recognize_model_path = 'Model/ghostnetv1_basic_model_latest.h5', detect_model_path = 'Model\yolov5s_face_dynamic.h5', known_users=None, batch_size=32, force_reload=False):
        #create recognize model and load weight
        self.recognize_model = GhostFaceNets.buildin_models(recognize_model, dropout=0, emb_shape=512, output_layer='GDC', bn_momentum=0.9, bn_epsilon=1e-5)
        self.recognize_model = GhostFaceNets.add_l2_regularizer_2_model(self.recognize_model, weight_decay=5e-4, apply_to_batch_normal=False)
        self.recognize_model = GhostFaceNets.replace_ReLU_with_PReLU(self.recognize_model)

        self.recognize_model.load_weights(recognize_model_path)
        # recognize_model = tf.keras.models.load_model('Model\ghostnetv1.h5', compile=False)
        #load detect model
        self.detect_model = YoloV5FaceDetector(model_path=detect_model_path)
        if known_users is not None:
            assert self.recognize_model is not None, "recognize_model is not provided while initializing this instance."
            self.known_image_classes, self.known_embeddings = self.process_known_user_dataset(known_users, batch_size, force_reload=force_reload)
        else:
            self.known_image_classes, self.known_embeddings = None, None
    
    # Detect face and collect features off it
    def image_detect_and_embedding(self, image, image_format="RGB"):
        bbs, _, ccs, nimgs = self.detect_model.detect_in_image(image, max_output_size=15, iou_threshold=0.85, score_threshold=0.7, image_format=image_format)
        if len(bbs) == 0:
            return np.array([]), [], [], None
        emb_unk = self.recognize_model((nimgs - 127.5) * 0.0078125).numpy()
        emb_unk = emb_unk / np.linalg.norm(emb_unk, axis=-1, keepdims=True)
        return emb_unk, bbs, ccs, nimgs

    # Compare all images to each other
    def compare_images(self, images, image_format="RGB"):
        gathered_emb, gathered_bbs, gathered_ccs = [], [], []
        for id, image in enumerate(images):
            emb_unk, bbs, ccs, nimgs= self.image_detect_and_embedding(image, image_format=image_format)
            gathered_emb.append(emb_unk)
            gathered_bbs.append(bbs)
            gathered_ccs.append(ccs)
            print(">>>> image_path: {}, faces count: {}".format(image if isinstance(image, str) else id, emb_unk.shape[0]))
        gathered_emb = np.concatenate(gathered_emb, axis=0)
        cosine_similarities = gathered_emb @ gathered_emb.T
        return cosine_similarities, gathered_emb, gathered_bbs, gathered_ccs

    # Process all user dataset
    def process_known_user_dataset(self, known_users_path, batch_size=32, img_shape=(112, 112), force_reload=False):
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        img_gen = ImageDataGenerator().flow_from_directory(known_users_path, class_mode='binary', target_size=img_shape, batch_size=batch_size, shuffle=False)
        model_interf = lambda imms: self.recognize_model((imms - 127.5) * 0.0078125).numpy()
        
        steps = int(np.ceil(img_gen.classes.shape[0] / img_gen.batch_size))
        filenames = np.array(img_gen.filenames)

        embs, imm_classes = [], []
        for _ in tqdm(range(steps), "Embedding"):
            imm, imm_class = img_gen.next()
            emb = model_interf(imm)
            embs.extend(emb)
            imm_classes.extend(imm_class)
        embeddings, image_classes = normalize(np.array(embs).astype("float32")), np.array(imm_classes).astype("int")

        return image_classes, embeddings

    # Collect features off multifaces in a new image to compare with all user dataset
    def search_in_known_users(self, image, known_users_path, threshold=0.90, image_format="RGB"):
        emb_unk, bbs, ccs, nimgs = self.image_detect_and_embedding(image, image_format=image_format)
        if len(emb_unk) != 0:
            known_image_classes, known_embeddings = self.process_known_user_dataset(known_users_path)
            cosine_similarities = emb_unk @ known_embeddings.T
            recognition_indexes = cosine_similarities.argmax(-1)
            recognition_similarities = [cosine_similarities[id, ii] for id, ii in enumerate(recognition_indexes)]
            recognition_classes = [known_image_classes[ii] for ii in recognition_indexes]
            for i in range(len(recognition_similarities)):
                if recognition_similarities[i] < threshold:
                    recognition_classes[i] = None
            return recognition_similarities, recognition_classes, bbs, ccs
        else: return None, None, None, None
    
    # compare_multi_images_of_a_person and face dataset
    def compare_multi_images_of_a_person(self, images, known_users_path, image_format="RGB"):
        gathered_emb, gathered_bbs, gathered_ccs,  gathered_nings= [], [], [], []
        for image in images:
            emb_unk, bbs, ccs, nimgs = self.image_detect_and_embedding(image, image_format=image_format)
            if nimgs is not None:
                gathered_emb.append(emb_unk)
                gathered_bbs.append(bbs)
                gathered_ccs.append(ccs)
                gathered_nings.append(nimgs)
            # print(">>>> image_path: {}, faces count: {}".format(image if isinstance(image, str) else id, emb_unk.shape[0]))
        if len(gathered_emb) != 0:
            gathered_emb = np.concatenate(gathered_emb, axis=0)
        
            known_image_classes, known_embeddings = self.process_known_user_dataset(known_users_path)

            cosine_similarities = gathered_emb @ known_embeddings.T
        
            return cosine_similarities, known_image_classes, bbs, ccs, gathered_nings
        else: return None, None, None, None, None