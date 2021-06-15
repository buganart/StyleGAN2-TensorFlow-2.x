import os
import io
import sys
import subprocess

import click
import time

# import torch
from flask import Flask, jsonify, request, Response
import requests

from argparse import Namespace

import numpy as np

from pathlib import Path

# change path inside stylegan2 repo
import sys

# sys.path.append("./stylegan2-pytorch")

import tensorflow as tf
from tqdm import tqdm
from PIL import Image


from stylegan2_generator import StyleGan2Generator
from utils.utils_stylegan2 import convert_images_to_uint8


# global ckpt_dir
# ckpt_dir = "./checkpoint"

# global result_dir
# result_dir = "./checkpoint/results"

app = Flask(__name__)
app.config["SERVER_NAME"] = os.environ.get("SERVER_NAME")

# mjpeg functions
class Camera(object):
    def __init__(self, generator, walk_vectors):
        self.generator = generator
        self.walk_vectors = walk_vectors
        # self.frames = [open(f + '.jpg', 'rb').read() for f in ['1', '2', '3']]
        self.walk_index = 0
        self.vector_index = 0

    def get_frame(self):
        # return self.frames[int(time()) % 3]
        print("start generate image", self.walk_index, self.vector_index)
        latent_z = self.walk_vectors[self.walk_index].astype("float32")
        vector = latent_z[self.vector_index].reshape((1, -1))
        out = self.generator(vector)
        # converting image to uint8
        out_image = convert_images_to_uint8(out, nchw_to_nhwc=True, uint8_cast=True)
        out_image = out_image.numpy()[0]
        # save image
        im = Image.fromarray(out_image)
        tempImage = io.BytesIO()
        im.save(tempImage, format="JPEG")

        # update walk_index and vector_index
        self.vector_index = (self.vector_index + 1) % latent_z.shape[0]
        if self.vector_index == 0:
            # update walk_index
            self.walk_index = (self.walk_index + 1) % len(self.walk_vectors)
        print("return image", self.walk_index, self.vector_index)
        # load jpg in binary
        # im = open("./temp.jpeg", "rb").read()
        return tempImage.getvalue()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/generate-latent-walk", methods=["post"])
def generate():
    # TODO: install dependency
    # python 3.7
    # pip install opensimplex, click, flask, requests, Pillow, tqdm, scipy
    # pip install tensorflow==1.15

    req = request.get_json(force=True)
    print("req", req)
    ckpt_file = req.get("ckpt_file", "ffhq")
    seeds = req.get("seeds", "3,7")
    frames = int(req.get("frames", 10))
    # psi = float(req.get("psi", 0.7))

    # ckpt_file = "ffhq"
    # seeds = "3,7"
    # frames = 10

    # validate seeds
    seeds = seeds.split(",")
    seeds = [int(s) for s in seeds]
    # validate ckpt_file
    ckpt_file_name = str(Path(ckpt_file).stem)
    # print("ckpt_dir", ckpt_dir)

    # ckptfile_list = Path(ckpt_dir).rglob("*.*")
    # target_ckpt = None
    # for ckpt in ckptfile_list:
    #     print("file:", ckpt)
    #     ckpt_name = str(Path(ckpt).stem)
    #     if ckpt_name == ckpt_file_name:
    #         target_ckpt = str(ckpt)
    #         break
    weights_name = ckpt_file

    impl = "ref"  # 'ref' if cuda is not available in your machine or 'cuda'
    gpu = False  # False if tensorflow cpu is used

    # modify package functions for our needs
    # setup generator and parameters
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    size = 1024  # image height/width
    latent = 512

    generator = StyleGan2Generator(weights=weights_name, impl=impl, gpu=gpu)

    # seed to latent vectors
    zs = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        z = rng.randn(1, latent)
        zs.append(z)

    # convert to linspace
    num_walk = len(zs) - 1
    walk_vectors = []
    sample = int(frames / num_walk) + 1
    for nw in range(num_walk):
        z0, z1 = zs[nw + 0], zs[nw + 1]
        walk_vector = np.linspace(z0, z1, sample)
        # append walk_vector shape (sample, 1, latent), so remove 1 dim to make shape (sample, latent)
        walk_vectors.append(walk_vector[:, 0, :])

    # load jpeg to build mjpeg stream

    return Response(
        gen(Camera(generator, walk_vectors)),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status", methods=["GET"])
def status():
    return "ok"


def setup(cli_checkpoint_dir="./checkpoint/", cli_result_dir="./checkpoint/results"):
    # global checkpoint_dir, result_dir
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR") or cli_checkpoint_dir
    # result_dir = cli_result_dir
    # Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    # Path(result_dir).mkdir(parents=True, exist_ok=True)
    return app


@click.command()
@click.option("--debug", "-d", is_flag=True)
@click.option("--checkpoint-dir", "-cp", default="./checkpoint/")
@click.option("--result-dir", "-cp", default="./checkpoint/results")
def api_run(debug, checkpoint_dir, result_dir):
    app = setup(checkpoint_dir, result_dir)
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))


if __name__ == "__main__":
    api_run()
