from email.mime import image
import numpy as np
from glob import glob
from feature_extractor import FeatureExtractor
from retrieval import ImageRetrieval
from utils import cosine_similarity, visualize_salient_patch
from PIL import Image
import os
from pathlib import Path
import shutil
import random
from tqdm import tqdm
import json


def main(i):
    # Assuming you have a folder with 10 images
    intra_threshold = 0.5
    threshold = 0.1
    model = 'vit'

    query_image_list = glob("testset1/*/*.png")
    prototype_image_list = []
    for c in os.listdir("testset1"):
        class_image_list = glob(f"testset1/{c}/*.png")
        if len(class_image_list) < 2:
            continue
        prototyp_image = random.sample(class_image_list, 1)[0]
        query_image_list.remove(prototyp_image)
        prototype_image_list.append(prototyp_image)

    cnt = 0
    correct = {}
    wrong = {}
    with tqdm(query_image_list) as pbar:
        for query_image_path in pbar:
            query_class = query_image_path.split("/")[1]
            image_retrieval = ImageRetrieval(model=model,
                                             query_image_path=query_image_path,
                                             prototype_image_list=prototype_image_list,
                                             intra_threshold=intra_threshold,
                                             threshold=threshold)
            query_image = image_retrieval.visualize_query_image()

            Path(f"results_{i}/{query_image_path.split('/')[-1].split('.')[0]}/{query_image_path.split('/')[-1]}").parent.mkdir(
                parents=True, exist_ok=True)
            query_image.save(
                f"results_{i}/{query_image_path.split('/')[-1].split('.')[0]}/{query_image_path.split('/')[-1]}")

            most_retrieved_file_path_list, most_retrieved_matches_list, most_retrieved_local_indices_list = image_retrieval.get_topk_images(
                1)
            retrieved_class = most_retrieved_file_path_list[0].split("/")[1]

            if query_class in correct:
                correct[query_class] += 1 if query_class == retrieved_class else 0
                wrong[query_class] += 0 if query_class == retrieved_class else 1
            else:
                correct[query_class] = 1 if query_class == retrieved_class else 0
                wrong[query_class] = 0 if query_class == retrieved_class else 1

            if 'total' in correct:
                correct['total'] += 1 if query_class == retrieved_class else 0
                wrong['total'] += 0 if query_class == retrieved_class else 1
            else:
                correct['total'] = 1 if query_class == retrieved_class else 0
                wrong['total'] = 0 if query_class == retrieved_class else 1

            for most_retrieved_file_path, most_retrieved_matches, most_retrieved_local_indices in zip(most_retrieved_file_path_list, most_retrieved_matches_list, most_retrieved_local_indices_list):
                is_correct = 'O' if query_class == retrieved_class else 'X'
                p = Path(
                    f"results_{i}/{query_image_path.split('/')[-1].split('.')[0]}/", most_retrieved_file_path.split('/')[-1])
                # shutil.copy(most_retrieved_file_path, p)
                if most_retrieved_local_indices is not None:
                    retrieved_image = image_retrieval._visualize_image(
                        most_retrieved_file_path, most_retrieved_local_indices)
                    retrieved_image.save(p)
                else:
                    Image.open(most_retrieved_file_path).save(p)

                if most_retrieved_matches is None:
                    continue

                match_viz = image_retrieval.visualize_matches(
                    query_image_path, most_retrieved_file_path, most_retrieved_matches)
                Image.fromarray(match_viz).save(
                    str(p).replace(".png", f"_{is_correct}.png"))
            cnt += 1
            pbar.set_postfix(acc=correct['total']/cnt)

    with open(f"results_{i}.json", "w") as f:
        json.dump({"correct": correct, "wrong": wrong}, f)


if __name__ == '__main__':
    for i in range(100, 101):
        main(i)
