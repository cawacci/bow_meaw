

# import modules
import torch
from animal import transform, Net # animal ファイルからモデルをインポートする
from flask import Flask, request, render_template, redirect
import io
from PIL import Image
import base64

# prediction プログラム
def predict(img):
    # net のインスタンス生成
    net = Net().cpu().eval()
    # load weights
    net.load_state_dict(torch.load('./k-fold_dog_cat.pt', map_location=torch.device('cpu')))

    # preprocess
    img = transform(img).unsqueeze(0)
    # predict
    label = torch.argmax(net(img), dim=1).cpu().detach().numpy()

    if label == 0:
        return '猫'
    else:
        return '犬'
    
# Flask のインスタンス生成
app = Flask(__name__)

# restrict uploadable images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# URLにアクセスしたときの処理
@app.route('/', methods=['GET', 'POST'])
def predicts():
    if request.method == 'POST':
        # ファイルがなかったときの処理
        if 'filename' not in request.files:
            return redirect(request.url)
        
        # データの取り出し
        file = request.files['filename']
        # ファイルのチェック
        if file and allowed_file(file.filename):

            # 画像ファイルに対する処理
            # 画像書き込み用バッファを確保
            buf = io.BytesIO()
            # 画像ファイルを読み込み、RGB（３チャンネル）に変換
            img = Image.open(file).convert('RGB')
            # 画像ファイルをバッファに書き込み
            img.save(buf, format='png')

            # バイナリデータをbase64エンコード, utf8に変換
            base64_str = base64.b64encode(buf.getvalue()).decode('utf-8')

            # HTML側の src の記述に合わせて付帯情報を追加
            base64_data = 'data:image/png;base64,' + base64_str

            # 入力された画像に対して推論を実行
            result = predict(img)

            return render_template('result.html', animalName=result, img=base64_data)
        return redirect(request.url)
    
    return render_template('index.html')

# プログラムの実行
if __name__ == '__main__':
    app.run(debug=True)