import os
import json
import pandas as pd
import yaml
from flask import Flask, request, render_template, redirect, send_from_directory
import matplotlib.image as mpimage

from utils import PredictionEngine, INPUT_WIDTH, INPUT_HEIGHT


CONFIG_FILE = "config.yaml"


def create_app():
    app = Flask(__name__)
    with open(CONFIG_FILE, "rt") as fl:
        config = yaml.safe_load(fl)
        app.config.update(config)
    app.prediction_engine = PredictionEngine(db_path=app.config["DB_PATH"],
                                             model_path=app.config["MODEL_PATH"])
    app.image = None
    return app

app = create_app()

@app.route("/", methods=['GET'])
def recognize_get():
    return render_template("crop.html",
                           cropData="''", img_width="''", img_height="''",
                           x_coords="''", y_coords="''",
                           pred_files="''", pred_names="''", probs="''",
                           img_path="'" + app.config["img_path"] + "'",
                           )


@app.route("/", methods=['POST'])
def recognize():

    if app.image is None:
        return redirect("/")

    data = request.form.get("secret")
    if data is not None:
        coords = json.loads(data)
        print(type(coords))
        print(coords)
        x0 = int(coords["x"])
        x1 = int(coords["x"] + coords["width"])
        y0 = int(coords["y"])
        y1 = int(coords["y"] + coords["height"])

        x0 = max(min(x0, app.image.shape[1]), 0)
        x1 = max(min(x1, app.image.shape[1]), 0)
        y0 = max(min(y0, app.image.shape[0]), 0)
        y1 = max(min(y1, app.image.shape[0]), 0)

        print(app.image[y0: y1, x0: x1].shape)

        if (y1 > y0) and (x1 > x0):

            preds, coords = app.prediction_engine.preprocess_pred(app.image[y0: y1, x0: x1])
            print(coords)
            y_coords = [int(x) for x in (coords[::2] * INPUT_WIDTH).astype(int)]
            x_coords = [int(y) for y in (coords[1::2] * INPUT_HEIGHT).astype(int)]
            print(x_coords)
            print(y_coords)
            pred_images = pd.merge(preds.reset_index(), app.prediction_engine.df[["flag_128", "name"]], on="name", how="left")

            num_preds = app.config["num_preds"]
            pred_files = json.dumps(list(pred_images["flag_128"].values[:num_preds]))
            pred_names = json.dumps(list(pred_images["name"].values[:num_preds]))
            probs = (100 * preds).round(2).astype(str)
            probs = json.dumps(list(probs.values[:num_preds]))
            print(pred_files)
            return render_template("crop.html", preds=preds.to_frame().to_html(),
                                   cropData=data,
                                   x_coords=json.dumps(x_coords), y_coords=json.dumps(y_coords),
                                   img_width=app.image.shape[1],
                                   img_height=app.image.shape[0],
                                   pred_files=pred_files,
                                   pred_names=pred_names,
                                   probs=probs,
                                   img_path="'" + app.config["img_path"] + "'",
                                   )
        else:
            return redirect("/")


@app.route("/upload", methods=['POST'])
def upload_image():
    print("upload_image")
    if "file" not in request.files:
        return "no file"
    print("found a file")
    print(request.files)
    file = request.files["file"]
    print("saving")
    os.makedirs("uploads", exist_ok=True)
    file.save("uploads/example1.jpeg")
    app.image = mpimage.imread(file)
    print("saved")
    return "", 204

@app.route("/uploads/<filename>")
def upload(filename):
    print("enter get file")
    return send_from_directory("uploads", filename)


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-store"
    return response

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")





