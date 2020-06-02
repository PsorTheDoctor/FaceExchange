from flask import Flask, request, render_template

from inference import generate_image

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        gen_img = generate_image(img_bytes)

        return render_template('result.html', gen_img=gen_img)


if __name__ == '__main__':
    app.run(debug=True)
