import sys
import os
import numpy as np
import cv2
import numpy as np
from keras.models import load_model
img_width = 150
img_height = 150

cap = cv2.VideoCapture(0)
c=0
res, score = '', 0.0
i = 0
mem = ''
consecutive = 0
sequence = ''
model = load_model('model.hdf5')
counter = 1
current_state = 5
current_value = -1
while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    if ret:
        a = cv2.waitKey(1) # waits to see if `esc` is pressed
        if counter % 5 == 0:
            x1, y1, x2, y2 = 400, 100, 600, 300
            img_cropped = img[y1:y2, x1:x2]
            img1 = cv2.resize(img_cropped, (img_height, img_width))
            img1 = np.expand_dims(img1, axis=0)
            x = model.predict_classes(img1)[0]
	    if current_value == x:
                if current_state == 5:
                    print(x)
                    current_value = -1
                    current_state = 1
                else:
                    current_state = current_state + 1
            else:
                current_state = 1
                current_value = x
            # c += 1
            # image_data = cv2.imencode('.jpg', img_cropped)[1].tostring()
            
            # """
            # if i == 4:
            #     res_tmp, score = predict(image_data)
            #     res = res_tmp
            #     i = 0
            #     if mem == res:
            #         consecutive += 1
            #     else:
            #         consecutive = 0
            #     if consecutive == 2 and res not in ['nothing']:
            #         if res == 'space':
            #             sequence += ' '
            #         elif res == 'del':
            #             sequence = sequence[:-1]
            #         else:
            #             sequence += res
            #         consecutive = 0
            # i += 1
            # """
            # cv2.putText(img, '%s' % (res.upper()), (100,400), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 4)
            # #cv2.putText(img, '(score = %.5f)' % (float(score)), (100,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
            # mem = res
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
            # img_sequence = np.zeros((200,1200,3), np.uint8)
            # cv2.putText(img_sequence, '%s' % (sequence.upper()), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            # #cv2.imshow('sequence', img_sequence)
            cv2.imshow("img", img)
            counter = 0
        counter = counter + 1;
        if a == 27: # when `esc` is pressed
            break
