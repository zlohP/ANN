import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from simple_convnet import SimpleConvNet

# --- 모델 생성 및 파라미터 로드 ---
model = SimpleConvNet(input_dim=(1, 28, 28),
                      conv_param={'filter_num1':16, 'filter_num2':32, 'filter_size':3, 'pad':1, 'stride':1},
                      hidden_size=100, output_size=10, weight_init_std=0.01)
model.load_params("params.pkl")

# --- 이미지 경로 목록 (여러 이미지 처리 가능) ---
image_paths = [
    "myDataset/new_1.jpg",
    "myDataset/new_2.jpg",
    "myDataset/new_3.jpg",
    "myDataset/new_4.jpg",
    "myDataset/new_5.jpg",
    "myDataset/new_6.jpg",
    "myDataset/image_4135680927.jpg",
    "myDataset/image_5790142683 (2).jpg",
    "myDataset/image_5790142683.jpg",
    "myDataset/image_7162035984.jpg",
    "myDataset/image_9746130528.jpg"
   ]

# --- 전체 데이터 저장용 ---
x_data = []
t_data = []

# --- 이미지별 처리 루프 ---
for image_path in image_paths:
    img = cv2.imread(image_path)

    img_copy = img.copy()

    rects = []
    drawing = False
    ix, iy = -1, -1


    def draw_rectangle(event, x, y, flags, param):
        global ix, iy, drawing, rects, img_copy

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            img_copy = img.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            rect = (min(ix, x), min(iy, y), abs(x - ix), abs(y - iy))
            rects.append(rect)
            print(f"사각형 추가됨: {rect}")

    cv2.namedWindow("Select digits")
    cv2.setMouseCallback("Select digits", draw_rectangle)

    print(f"\n🖱️ {image_path}에서 숫자 네모로 감싸고 Enter (↩️) 키로 종료")
    while True:
        cv2.imshow("Select digits", img_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter
            break
    cv2.destroyAllWindows()

    print(f"\n총 {len(rects)}개의 사각형이 선택되었습니다.")

    for i, (x, y, w, h) in enumerate(rects):
        cropped = img[max(y, 0):y+h, max(x, 0):x+w]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_NEAREST)
        norm = resized.astype('float32') / 255.0
        norm = 1.0 - norm  # 흰 배경, 검정 숫자

        test_input = norm.reshape(1, 1, 28, 28)
        pred = model.predict(test_input)
        pred_label = int(np.argmax(pred))

        # 사용자 확인
        plt.imshow(norm, cmap='gray')
        plt.title(f"예측값: {pred_label}")
        plt.axis('off')
        plt.show()

        true_label = input(f"[{i}] 정답 라벨 입력 (Enter=예측값 {pred_label}): ").strip()
        label = pred_label if true_label == "" else int(true_label)

        x_data.append(norm)
        t_data.append(label)

# --- 저장 ---
x_data = np.array(x_data).reshape(-1, 28, 28)
t_data = np.array(t_data)

with open("Dataset3D_2.pkl", "wb") as f:
    pickle.dump((x_data, t_data), f)

print("저장 완료! Dataset3D_2.pkl 생성됨")
print(f"\n✅ 총 {len(x_data)}개의 숫자 이미지가 저장되었습니다.")