import cv2
import numpy as np
left_img = cv2.imread(&quot;left_image.jpg&quot;, cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread(&quot;right_image.jpg&quot;, cv2.IMREAD_GRAYSCALE)
if left_img is None or right_img is None:
print(&quot;Error: Stereo images not loaded&quot;)
exit()
stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=15)
disparity = stereo.compute(left_img, right_img)
disparity_normalized = cv2.normalize(
disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
)
disparity_normalized = np.uint8(disparity_normalized)
cv2.imshow(&quot;Left Image&quot;, left_img)
cv2.imshow(&quot;Right Image&quot;, right_img)
cv2.imshow(&quot;Disparity Map&quot;, disparity_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()
