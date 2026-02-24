from flask import Flask, render_template, request
import os
from detector.predict import predict_image
from video_detect import predict_video

app = Flask(__name__)

# -----------------------------
# Upload folder setup (AUTO CREATE)
# -----------------------------
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# -----------------------------
# HOME ROUTE
# -----------------------------
@app.route("/", methods=["GET","POST"])
def home():
    result = None
    filename = None

    if request.method == "POST":

        # check file exists
        if "file" not in request.files:
            result = "No file selected"
            return render_template("index.html", result=result)

        file = request.files["file"]

        if file.filename == "":
            result = "No file selected"
            return render_template("index.html", result=result)

        # save file
        filename = file.filename
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(path)

        # ---------------- IMAGE DETECTION ----------------
        if filename.lower().endswith((".png",".jpg",".jpeg")):
            label, confidence = predict_image(path)
            result = f"{label} ({confidence}% confidence)"

        # ---------------- VIDEO DETECTION ----------------
        elif filename.lower().endswith((".mp4",".avi",".mov")):
            label, confidence = predict_video(path)
            result = f"{label} ({confidence}% confidence)"

        else:
            result = "Unsupported file format"

    return render_template("index.html", result=result, filename=filename)


# -----------------------------
# RUN SERVER (Render + Local)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render provides PORT
    app.run(host="0.0.0.0", port=port, debug=False)