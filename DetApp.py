from flask import Flask, render_template, request, flash, url_for
from modules import make_result
import os
import urllib.request


if not os.path.exists('static/Results'):
    os.mkdir('static/Results')

if not os.path.exists('static/Others/pretrained_weights.pth'):
    destination = 'static/Others/pretrained_weights.pth'
    url = 'https://drive.google.com/u/2/uc?id=1RY2Hv2juE3LptIU7sxpjWjXt23KseW2l&export=download&confirm=t&uuid=85cb0d70-25a8-4610-b13c-0feb3c55b141&at=AHV7M3eOFWFJubKsKwf8W-sExiK2:1669572420250'
    urllib.request.urlretrieve(url, destination)

app = Flask(__name__)
app.config['SECRET_KEY'] = '21n223j4m43n23m'

menu = ["Insert a link to an image(only links from the internet in '.jpg' format)",
        "Click the 'detect' button",
        "Enjoy:)"]

if os.path.exists('static/Others/num_detection.txt'):
    with open('static/Others/num_detection.txt', "r") as file:
        num_detection = file.read()
else:
    num_detection = 0
    with open('static/Others/num_detection.txt', "w") as file:
        file.write(str(num_detection))


@app.route("/", methods=["POST", "GET"])
def site():
    global num_detection
    if request.method == "POST":
        num_detection = int(num_detection) + 1
        with open('static/Others/num_detection.txt', "w") as file:
            file.write(str(num_detection))

        if len(request.form['link']) == 0:
            flash('Attempt to submit an empty form !', category='error')
        if len(request.form['link']) > 0:
            flash('Good link', category='success')
            make_result(str(request.form['link']), num_detection)
        print(request.form['link'])
    return render_template('body.html',
                           title="Fruit detection (bananas, oranges, apples)",
                           dir="static/Results/" + str(num_detection) + "_picture_with_det.jpg",
                           menu=menu)


@app.errorhandler(500)
def pageNotFount(error):
    return render_template('page500.html', title="Fruit detection (bananas, oranges, apples)")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1234, debug=False)