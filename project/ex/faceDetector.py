import cv2
import matplotlib.pyplot as plt
import cvlib as cv

image = "face.png"
im = cv2.imread(image)
plt.imshow(im)
plt.show

faces, confidences =  cv.detect_face(im)

for face in faces:
  (startX, startY) = face[0], face[1]
  (endX, endY) = face[2], face[3]
  cv2.rectangle(im, (startX, startY), (endX, endY), (0, 255, 0), 2)

plt.imshow(im)
plt.show()

cv2.imwrite('result.jpg', im)

import numpy as np 

faces, confidences =  cv.detect_face(im)

for face in faces:
  (startX, startY) = face[0], face[1]
  (endX, endY) = face[2], face[3]
  cv2.rectangle(im, (startX, startY), (endX, endY), (0, 255, 0), 2)

  face_crop = np.copy(im[startY:endY, startX:endX])

  (label, confidences) = cv.detect_gender(face_crop)

  print(confidences)
  print(label)

  idx = np.argmax(confidences)
  label = label[idx]

  label = "{}: {:.2f}%".format(label, confidences[idx]*100)

  Y = startY - 10 if startY - 10 > 10 else startY + 10

  cv2.putText(im, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

plt.imshow(im)
plt.show()
cv2.imwrite("result2.jpg", im) 