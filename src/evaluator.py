import gc
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import plotly.express as px
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.spatial.distance import euclidean
import seaborn as sns

from .utils import configure_logger
from .dataset import ClassficationDataset as EvaluationDataset,load_from_json


def mean_euclidean_distance(embeddings1: List, embeddings2: List):
    """
    Given two lists of embeddings, find the mean distance between all embeddings.
    """
    distances = []
    for e1 in embeddings1:
        for e2 in embeddings2:
            distances.append(euclidean(e1[0], e2[0]))
    return np.mean(distances)


class Evaluator:
    """Evaluator definition"""

    def __init__(self, out_dir, device, conf):
        self.out_dir = out_dir
        self.device = device
        self.conf = conf

        self.logger = configure_logger(out_dir, __name__)
        self.logger.info("Initializing Evaluator")

        std_ds = EvaluationDataset(self.conf.data.data_dir, conf.data.splits.test)
        self.std_dl = DataLoader(std_ds, batch_size=16, shuffle=False, pin_memory=True)

        if "pet" in conf.data.splits.test:
            self.label_map = load_from_json(f"{self.conf.data.data_dir}/pet_labels.json")
        else:
            self.label_map = load_from_json(f"{self.conf.data.data_dir}/micro_labels.json")
        self.label_map  = {value: key for key, value in self.label_map.items()}

    def __call__(self, enc, conf, epoch):

        enc.eval()
        self.logger.info(f"Evaluation done at {epoch} epoch.......")

        embs = []
        labels = []
        # std_embs = {}

        for _, batch in tqdm(
            enumerate(self.std_dl), desc="Evaluation", total=len(self.std_dl)
        ):
            emb = enc(batch[0])
            label = batch[1][0]

            for i, e in enumerate(emb):
                embs.append(e.cpu().detach().numpy().reshape(-1))
                labels.append(label)
                # if label not in std_embs:
                #     std_embs[label] = [
                #         [e.cpu().detach().numpy().reshape(-1)]
                #     ]
                # else:
                #     std_embs[label].append(
                #         [e.cpu().detach().numpy().reshape(-1)]
                # )

        labels = [self.label_map[i.item()] for i in labels]
        embed = np.array(embs)
        labels = np.array(labels)

        ## Clustering
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embed)

        df = pd.DataFrame(embeddings_2d, columns=["Dim1", "Dim2"])
        df["Label"] = labels

        fig = px.scatter(df, x="Dim1", y="Dim2", color="Label", hover_data=["Label"])
        fig.update_layout(
            title="t-SNE visualization of Embeddings",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.05),
        )

        fig.write_html(f"{self.out_dir}/outputs/epoch_{epoch}_clsuters.html")

        # ## Distance plot
        # labels = list(std_embs.keys())
        # distance_df = pd.DataFrame(index=labels, columns=labels)

        # for label1 in labels:
        #     for label2 in labels:
        #         distance_df.loc[label1, label2] = mean_euclidean_distance(
        #             std_embs[label1], std_embs[label2]
        #         )

        # distance_df = distance_df.astype(float)
        # plt.figure(figsize=(100, 100))
        # sns.heatmap(
        #     distance_df,
        #     annot=True,
        #     fmt=".2f",
        #     cmap="viridis",
        #     cbar=True,
        #     square=True,
        #     linewidths=0.5,
        # )
        # plt.title("Mean Euclidean Distance Between Labels")
        # plt.xlabel("STDs")
        # plt.ylabel("STDs")

        # output_path = f"{self.out_dir}/outputs/epoch_{epoch}_mean_dist.png"
        # plt.savefig(output_path)
        # plt.close()

        # del distance_df, tsne, embeddings_2d, df, fig, embs, labels, embed
        # gc.collect()
        # torch.cuda.empty_cache()
