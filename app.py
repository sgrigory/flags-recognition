import os
import json
import pandas as pd
import yaml
from flask import Flask, request, render_template, redirect, send_from_directory
import matplotlib.image as mpimage
import logging

from utils import PredictionEngine


CONFIG_FILE = "config.yaml"


def create_app():
    """
    Prepare the app: load configuration file and create the prediction engine
    """
    logging.getLogger().setLevel(logging.DEBUG)

    app = Flask(__name__)
    with open(CONFIG_FILE, "rt") as fl:
        config = yaml.safe_load(fl)
        logging.debug(config)
        app.config.update(config)
    app.prediction_engine = PredictionEngine(db_path=app.config["DB_PATH"],
                                             model_path=app.config["MODEL_PATH"])
    app.image = None

    return app

app = create_app()

@app.route("/", methods=['GET'])
def recognize_get():
    """
    Render the page before predictions are made - with or without uploaded image
    """
    if app.image is None:
        visibility = "hidden"
    else:
        visibility = "visible"

    return render_template("crop.html",
                           cropData="''", img_width="''", img_height="''",
                           x_coords="''", y_coords="''",
                           pred_files="''", pred_names="''", probs="''",
                           img_path="'" + app.config["img_path"] + "'",
                           visibility=visibility,
                           uploaded_img_path=app.config["uploaded_img_path"],
                           input_width=app.prediction_engine.input_width,
                           input_height=app.prediction_engine.input_height,
                           )


@app.route("/", methods=['POST'])
def recognize():
    """
    Render the page after predictions are made
    """
    if app.image is None:
        logging.debug("app.image is None, redirecting")
        return redirect("/")

    # Get cropper data
    data = request.form.get("secret")
    # If cropper data is available
    if data is not None:
        coords = json.loads(data)
        # Get coordinates of the cropper
        x0 = int(coords["x"])
        x1 = int(coords["x"] + coords["width"])
        y0 = int(coords["y"])
        y1 = int(coords["y"] + coords["height"])

        # Fit cropper coordinates into the image frame
        x0 = max(min(x0, app.image.shape[1]), 0)
        x1 = max(min(x1, app.image.shape[1]), 0)
        y0 = max(min(y0, app.image.shape[0]), 0)
        y1 = max(min(y1, app.image.shape[0]), 0)

        # If cropper box is not empty
        if (y1 > y0) and (x1 > x0):

            # Get predictions for classes and coordinates of the identified area
            preds, coords = app.prediction_engine.preprocess_pred(app.image[y0: y1, x0: x1])
            # Rescale the coordinates from fractions of 1 to pixels of original image
            y_coords = [int(x) for x in (coords[::2] * app.prediction_engine.input_width).astype(int)]
            x_coords = [int(y) for y in (coords[1::2] * app.prediction_engine.input_height).astype(int)]
            # Attach to predicted classes names of countries
            pred_images = pd.merge(preds.reset_index(), app.prediction_engine.df[["flag_128", "name"]],
                                   on="name", how="left")

            # Format predictions for passing to the FE
            num_preds = app.config["num_preds"]
            pred_files = json.dumps(list(pred_images["flag_128"].values[:num_preds]))
            pred_names = json.dumps(list(pred_images["name"].values[:num_preds]))
            probs = (100 * preds).round(2).astype(str)
            probs = json.dumps(list(probs.values[:num_preds]))

            # Render the FE
            return render_template("crop.html", preds=preds.to_frame().to_html(),
                                   cropData=data,
                                   x_coords=json.dumps(x_coords), y_coords=json.dumps(y_coords),
                                   img_width=app.image.shape[1],
                                   img_height=app.image.shape[0],
                                   pred_files=pred_files,
                                   pred_names=pred_names,
                                   probs=probs,
                                   img_path="'" + app.config["img_path"] + "'",
                                   visibility="visible",
                                   uploaded_img_path=app.config["uploaded_img_path"],
                                   input_width=app.prediction_engine.input_width,
                                   input_height=app.prediction_engine.input_height,
                                   )
        else:
            logging.debug("cropper is invalid, redirecting")
            return redirect("/")


@app.route("/upload", methods=['POST'])
def upload_image():
    """
    Get uploaded image and save it to a file
    """
    logging.debug("upload_image")
    if "file" not in request.files:
        return "no file"
        logging.debug("found a file")
    file = request.files["file"]
    logging.debug("saving")
    os.makedirs("uploads", exist_ok=True)
    file.save(app.config["uploaded_img_path"])
    app.image = mpimage.imread(file)
    logging.debug("saved")
    return "", 204

@app.route("/uploads/<filename>")
def upload(filename):
    logging.debug("enter get file")
    return send_from_directory("uploads", filename)


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-store"
    return response

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")





