import imageio
import matplotlib.pyplot as plt
%matplotlib inline

from mlxtend.image import extract_face_landmarks

img = imageio.imread('m5.jpg')
landmarks = extract_face_landmarks(img)
print(landmarks.shape)
print('\n\n landmarks:\n', landmarks)

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(1, 3, 1)
ax.imshow(img)
ax = fig.add_subplot(1, 3, 2)
ax.scatter(landmarks[:, 0], -landmarks[:, 1], alpha=0.8)
ax = fig.add_subplot(1, 3, 3)
img2 = img.copy()
for p in landmarks:
    img2[p[1]-3:p[1]+3,p[0]-3:p[0]+3,:] = (255, 255, 255)
ax.imshow(img2)
plt.show()

# Feature indexes
import numpy as np
jaw = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
left_eyebrow = np.array([17,18,19,20,21])
right_eyebrow = np.array([22,23,24,25,26]) 
nose = np.array([27,28,29,30,31,32,33,34,35]) 
left_eye = np.array([36, 37, 38, 39, 40, 41])
right_eye = np.array([42, 43, 44, 45, 46, 47])
mouth = np.array([48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67])

#mean of the features
jaw = np.mean(landmarks[jaw], axis=0,dtype='int')
left_eyebrow = np.mean(landmarks[left_eyebrow], axis=0,dtype='int')
right_eyebrow = np.mean(landmarks[right_eyebrow], axis=0,dtype='int')
nose = np.mean(landmarks[nose], axis=0,dtype='int')
left_eye = np.mean(landmarks[left_eye], axis=0,dtype='int')
right_eye = np.mean(landmarks[right_eye], axis=0,dtype='int')
mouth = np.mean(landmarks[mouth], axis=0,dtype='int')
print('Jaw:',jaw)
print('Right Eyebrow:',right_eyebrow)
print('Left Eyebrow:',left_eyebrow)
print('Nose:',nose)
print('Left Eye:', left_eye)
print('Right Eye:', right_eye)
print('Mouth:',mouth)

#landmark plotting with numbers
#fig = plt.figure(figsize=(10,10))
#plt.plot(landmarks[:,0], -landmarks[:,1], 'ro', markersize=8, alpha = 0.5)
#for i in range(landmarks.shape[0]):
#    plt.text(landmarks[i,0]+1, -landmarks[i,1], str(i), size=14)
#Mean value of 7 landmarks
plt.plot([left_eye[0]], [-left_eye[1]], 
            marker='+', color='blue', markersize=20, mew=4)
plt.plot([right_eye[0]], [-right_eye[1]], 
            marker='+', color='blue', markersize=20, mew=4)
plt.plot([left_eyebrow[0]], [-left_eyebrow[1]], 
            marker='+', color='red', markersize=20, mew=4)
plt.plot([right_eyebrow[0]], [-right_eyebrow[1]], 
            marker='+', color='red', markersize=20, mew=4)
plt.plot([jaw[0]], [-jaw[1]], 
            marker='+', color='green', markersize=20, mew=4)
plt.plot([mouth[0]], [-mouth[1]], 
            marker='+', color='purple', markersize=20, mew=4)
plt.plot([nose[0]], [-nose[1]], 
            marker='+', color='yellow', markersize=20, mew=4)
plt.show()