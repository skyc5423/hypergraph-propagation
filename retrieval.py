from re import I

from idna import valid_contextj
from feature_extractor import FeatureExtractor
from utils import cosine_similarity, sample_max_similarity
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image, ImageDraw
import pickle


class ImageRetrieval:
    def __init__(self, model, query_image_path, prototype_image_list, intra_threshold, threshold):
        self.feature_extractor = FeatureExtractor(model=model)
        self.query_image_path = query_image_path
        self.prototype_image_list = prototype_image_list
        self.intra_threshold = intra_threshold
        self.threshold = threshold
        self._extract_query_feature()
        self._load_global_features_all()

    def _load_global_features_all(self):
        with open("total_features_seg.pkl", "rb") as f:
            total_features = pickle.load(f)
            self.global_features_all = np.concatenate([total_features['features'][total_features['image_list'].index(
                db_image_path)] for db_image_path in self.prototype_image_list], axis=0)

    def _visualize_image(self, image_path, valid_local_indices):
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        for valid_index in valid_local_indices:
            y = valid_index // 16
            x = valid_index % 16
            draw.rectangle([x * 14, y * 14, (x + 1) * 14, (y + 1) * 14],
                           outline='red', width=2)
        return img

    def visualize_query_image(self):
        return self._visualize_image(self.query_image_path, self.query_valid_local_indices)

    def get_salient_indices(self, features_dict, N=50):
        local_feature = features_dict['local'].squeeze(0)
        global_feature = features_dict['global']
        sim_matrix_local_global = cosine_similarity(
            local_feature, global_feature).squeeze(1)
        # valid_local_indices = np.where(
        # sim_matrix_local_global > self.intra_threshold)[0]
        valid_local_indices = np.argsort(sim_matrix_local_global)[-N:]
        return valid_local_indices

    def _extract_activated_local_features(self, image_path, N=50):
        features_dict = self.feature_extractor.extract_features(image_path)
        valid_local_indices = self.get_salient_indices(features_dict, N)
        return features_dict['local'].squeeze(0)[valid_local_indices], valid_local_indices, features_dict['global']

    def _extract_query_feature(self):
        self.query_activated_local_features, self.query_valid_local_indices, self.query_global_feature = self._extract_activated_local_features(
            self.query_image_path, N=50)

    def _get_nn_by_global_feature(self, global_feature, n=20):
        cos_sim = cosine_similarity(
            global_feature, self.global_features_all)[0]
        return np.argsort(cos_sim)[-n:]

    def get_topk_images(self, k):
        if self.query_activated_local_features is None:
            raise ValueError(
                "Query features not extracted. Call extract_query_feature() first")

        most_retrieved_num_list = []
        most_retrieved_image_list = []
        matches_list = []
        valid_local_indices_list = []

        nn_indices = self._get_nn_by_global_feature(self.query_global_feature)

        for idx in tqdm(nn_indices):
            db_image_path = self.prototype_image_list[idx]
            db_features, valid_local_indices, global_features = self._extract_activated_local_features(
                db_image_path, N=100)
            cos_sim = cosine_similarity(
                self.query_activated_local_features, db_features)
            matches = self.match_features(cos_sim)

            if len(matches) < 4:
                continue

            kp1 = [cv2.KeyPoint(x=14*float(i % 16), y=14*float(i//16), size=1)
                   for i in self.query_valid_local_indices]
            kp2 = [cv2.KeyPoint(x=14*float(i % 16), y=14*float(i//16), size=1)
                   for i in valid_local_indices]

            H, num_inliers, inliers = self.find_homography_and_inliers(
                kp1, kp2, matches, 5.)
            # print(db_image_path, ': ', num_inliers)

            most_retrieved_image_list.append(db_image_path)
            most_retrieved_num_list.append(sum([m.distance for m in matches]))
            matches_list.append((matches, kp1, kp2, inliers))
            valid_local_indices_list.append(valid_local_indices)

        if len(most_retrieved_num_list) == 0:
            return [self.prototype_image_list[i] for i in nn_indices[:k]], [None for _ in nn_indices[:k]], [None for _ in nn_indices[:k]]

        indices = np.argsort(np.array(most_retrieved_num_list))[-k:]
        return [most_retrieved_image_list[i] for i in indices], [matches_list[i] for i in indices], [valid_local_indices_list[i] for i in indices]

    def match_features(self, similarity, max_num=20):
        matches = []
        for y, x in sample_max_similarity(similarity, max_num):
            if similarity[y, x] > self.threshold:
                matches.append(cv2.DMatch(y, x, similarity[y, x]))

        return matches

    def find_homography_and_inliers(self, kp1, kp2, matches, ransac_threshold=10.0):
        # Convert keypoints to numpy arrays
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find homography using RANSAC
        H, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, ransac_threshold)

        # Count inliers
        inliers = mask.ravel().tolist()
        num_inliers = sum(inliers)

        return H, num_inliers, inliers

    @staticmethod
    def visualize_matches(query_image_path, most_retrieved_file_path, most_retrieved_matches):
        img1 = cv2.imread(query_image_path)
        img2 = cv2.imread(most_retrieved_file_path)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        matches, kp1, kp2, inliers = most_retrieved_matches
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, [
                               m for m, i in zip(matches, inliers)], None, flags=2)
        return img3
